#include <cassert>
#include <complex>
#include <algorithm>
#include <libgnme/utils/eri_ao2mo.h>
#include "wick_pair.h"
#include "wick_kernels.h"

namespace libgnme {

namespace {

/* Singular values of the occupied orbital overlap up to this value are zero-overlap pairs */
const double zero_thresh = 1e-8;

/* Nonzero singular values below this value are near-zero. They would introduce large 1/S_k
   terms that cancel in the matrix elements. Each matrix element is linear in 1/S_k, so it is
   evaluated from the corners 1/S_k -> 0 and 1/S_k -> 1 as S_k F(1/S_k) = (S_k - 1) F(0) + F(1). */
const double near_thresh = 1e-1;

/* Symmetry-unique (m1, m2, m3, m4) assignments of the same-spin J intermediate */
const int jslots[10][4] = {
    {0,0,0,0}, {0,0,0,1}, {0,0,1,0}, {0,0,1,1}, {0,1,0,1},
    {0,1,1,0}, {0,1,1,1}, {1,0,1,0}, {1,0,1,1}, {1,1,1,1} };

double conj2(double x) { return x; }
std::complex<double> conj2(std::complex<double> x) { return std::conj(x); }

/** \brief Coulomb and exchange matrices J_st = (st|uv) M_uv and K_st = (su|vt) M_uv **/
template<typename Tc, typename Tb>
void coulomb_exchange(
    const arma::Mat<Tb> &V, const arma::Mat<Tc> &M, arma::Mat<Tc> &J, arma::Mat<Tc> &K)
{
    size_t n = M.n_rows;
    J.zeros(n, n);
    K.zeros(n, n);
    for(size_t u=0; u < n; u++)
    for(size_t v=0; v < n; v++)
    {
        const Tc muv = M(u,v);
        for(size_t s=0; s < n; s++)
        for(size_t t=0; t < n; t++)
            J(s,t) += V(s*n+t, u*n+v) * muv;
    }
    for(size_t v=0; v < n; v++)
    for(size_t t=0; t < n; t++)
    for(size_t s=0; s < n; s++)
    for(size_t u=0; u < n; u++)
        K(s,t) += V(s*n+u, v*n+t) * M(u,v);
}

/** \brief Copy a matrix into the slab in row-major order **/
template<typename Tc>
void put_rows(Tc *slab, int64_t off, const arma::Mat<Tc> &A)
{
    arma::Mat<Tc> At = A.st();
    std::copy(At.memptr(), At.memptr() + At.n_elem, slab + off);
}

/** \brief Copy a matrix into the slab in column-major order **/
template<typename Tc>
void put_cols(Tc *slab, int64_t off, const arma::Mat<Tc> &A)
{
    std::copy(A.memptr(), A.memptr() + A.n_elem, slab + off);
}

/** \brief Environment of the kernels for the intermediates of one pair **/
template<typename Tc>
wick_kernels::wick_env environment(
    const std::vector<int64_t> &info, const std::vector<Tc> &slab, const std::vector<Tc> &scal,
    double Vc, int kmax)
{
    wick_kernels::wick_env env;
    env.nref = 1;
    env.kmax = kmax;
    env.dstride = WICK_DHEAD + 4 * kmax;
    env.simd = 0;
    env.enuc = Vc;
    env.pairinfo = info.data();
    env.slab = slab.data();
    env.scal = scal.data();
    return env;
}

} // unnamed namespace


template<typename Tc, typename Tb>
wick_pair<Tc,Tb>::wick_pair(
    const arma::Mat<Tb> &metric,
    const reference_state<Tc> &xa, const reference_state<Tc> &wa,
    const reference_state<Tc> &xb, const reference_state<Tc> &wb,
    double Vc) :
    m_Vc(Vc)
{
    const arma::Mat<Tc> S = arma::conv_to<arma::Mat<Tc> >::from(metric);
    const reference_state<Tc> *refs[2][2] = { {&xa, &wa}, {&xb, &wb} };

    // Active occupied and virtual orbitals of each spin, padded to the same size in the bra
    // and ket, with the same number of labels for both spins
    size_t nvir[2];
    for(size_t spin=0; spin < 2; spin++)
    {
        const reference_state<Tc> &x = *refs[spin][0], &w = *refs[spin][1];
        assert(x.m_nelec == w.m_nelec);
        assert(x.m_ncore <= x.m_nelec and x.m_nelec <= x.m_ncore + x.m_nact);
        assert(w.m_ncore <= w.m_nelec and w.m_nelec <= w.m_ncore + w.m_nact);
        m_nox[spin] = x.m_nelec - x.m_ncore;
        m_now[spin] = w.m_nelec - w.m_ncore;
        m_nocc[spin] = std::max(m_nox[spin], m_now[spin]);
        nvir[spin] = std::max(x.m_ncore + x.m_nact, w.m_ncore + w.m_nact) - x.m_nelec;
    }
    m_n = std::max(m_nocc[0] + nvir[0], m_nocc[1] + nvir[1]);
    assert(m_n <= WICK_MAXMO);

    for(size_t spin=0; spin < 2; spin++)
    {
        const reference_state<Tc> &x = *refs[spin][0], &w = *refs[spin][1];
        const size_t nbsf = x.m_C.n_rows, nocc = x.m_nelec, n = m_n, k = m_nocc[spin];
        const arma::Mat<Tc> &Cx = x.m_C, &Cw = w.m_C;

        // Rows V_x + O_w and columns O_x + V_w. The bra virtual and ket occupied orbitals are
        // also kept separately for the direct orbital contribution.
        size_t nvx = x.m_ncore + x.m_nact - nocc, nvw = w.m_ncore + w.m_nact - nocc;
        arma::Mat<Tc> rowv(nbsf, n, arma::fill::zeros), rowo(nbsf, n, arma::fill::zeros);
        arma::Mat<Tc> colo(nbsf, n, arma::fill::zeros), colv(nbsf, n, arma::fill::zeros);
        if(nvx > 0) rowv.cols(0, nvx-1) = Cx.cols(nocc, nocc+nvx-1);
        if(m_now[spin] > 0) rowo.cols(n-k, n-k+m_now[spin]-1) = Cw.cols(w.m_ncore, nocc-1);
        if(m_nox[spin] > 0) colo.cols(0, m_nox[spin]-1) = Cx.cols(x.m_ncore, nocc-1);
        if(nvw > 0) colv.cols(k, k+nvw-1) = Cw.cols(nocc, nocc+nvw-1);
        arma::Mat<Tc> rowc = rowv + rowo, colc = colo + colv;

        // Lowdin pair the occupied orbitals
        arma::Mat<Tc> cxt = Cx.head_cols(nocc), cwt = Cw.head_cols(nocc);
        arma::vec s(nocc, arma::fill::zeros);
        Tc phase = 1.0;
        if(nocc > 0)
        {
            arma::Mat<Tc> U, V;
            arma::svd(U, s, V, arma::Mat<Tc>(cxt.t() * S * cwt));
            phase = arma::det(U) * conj2(arma::det(V));
            cxt = cxt * U;
            cwt = cwt * V;
        }
        size_t nnear = 0;
        for(size_t i=0; i < nocc; i++)
            if(s(i) > zero_thresh and s(i) < near_thresh) nnear++;

        // M^{(0)} = W + P^{xw} + P^{ww}, M^{(1)} = P^{xw}. Near-zero pairs are removed from W
        // with weight S_k - 1 in the reduced overlap, or enter W with unit weight.
        m_pair[spin].resize((size_t) 1 << nnear);
        for(size_t corner=0; corner < m_pair[spin].size(); corner++)
        {
            spin_pairing &p = m_pair[spin][corner];
            p.phase = phase;
            p.redS = 1.0; p.m = 0;
            p.M[0].zeros(nbsf, nbsf);
            p.M[1].zeros(nbsf, nbsf);
            size_t inear = 0;
            for(size_t i=0; i < nocc; i++)
            {
                if(s(i) <= zero_thresh)
                {
                    p.M[1] += cwt.col(i) * cxt.col(i).t();
                    p.M[0] += cwt.col(i) * cwt.col(i).t();
                    p.m++;
                }
                else if(s(i) < near_thresh)
                {
                    if((corner >> inear) & 1)
                        p.M[0] += cwt.col(i) * cxt.col(i).t();
                    else
                        p.redS *= (s(i) - 1.0);
                    inear++;
                }
                else
                {
                    p.M[0] += cwt.col(i) * cxt.col(i).t() / s(i);
                    p.redS *= s(i);
                }
            }
            p.M[0] += p.M[1];

            // Fundamental contractions. For m_i = 0 the direct orbital contribution is removed.
            for(size_t i=0; i < 2; i++)
            {
                arma::Mat<Tc> xao = S * p.M[i] * S;
                p.X[i] = rowc.t() * xao * colc;
                p.Y[i] = (i == 0) ? arma::Mat<Tc>(rowc.t() * (xao - S) * colc) : p.X[i];
                double f = (i == 0) ? 1.0 : 0.0;
                p.CX[i] = p.M[i].t() * S * rowc - f * rowv;
                p.XC[i] = p.M[i] * S * colc - f * colv;
            }
        }
    }

    // Offsets of each corner combination, with the other combinations following the first
    const size_t nc = ncorner();
    m_info.resize(nc * WICK_PAIRLEN);
    m_scal.assign(nc * WICK_NSCAL, Tc(0.0));
    size_t end = 0;
    for(size_t c=0; c < nc; c++)
        end = layout(m_info.data() + c * WICK_PAIRLEN, c, end);
    m_info[WICK_CSTART] = 1;
    m_slab.assign(end, Tc(0.0));

    // Fundamental contractions and overlap scalars
    for(size_t c=0; c < nc; c++)
    {
        const size_t cs[2] = {c / m_pair[1].size(), c % m_pair[1].size()};
        const int64_t *info = m_info.data() + c * WICK_PAIRLEN;
        Tc *scal = m_scal.data() + c * WICK_NSCAL;
        for(size_t spin=0; spin < 2; spin++)
        {
            const spin_pairing &p = m_pair[spin][cs[spin]];
            const int64_t *off = info + (spin == 0 ? WICK_AA : WICK_BB);
            for(size_t i=0; i < 2; i++)
            {
                put_rows(m_slab.data(), off[WICK_X+i], p.X[i]);
                put_rows(m_slab.data(), off[WICK_Y+i], p.Y[i]);
            }
            scal[(spin == 0 ? WICK_SAA : WICK_SBB) + WICK_SPHASE] = p.phase;
            scal[(spin == 0 ? WICK_SAA : WICK_SBB) + WICK_STSP] = p.redS;
        }
    }
}

template<typename Tc, typename Tb>
void wick_pair<Tc,Tb>::record(
    int *d, int kmax, const arma::umat &ahp, const arma::umat &bhp, bool ket) const
{
    assert(ahp.n_rows <= (size_t) kmax and bhp.n_rows <= (size_t) kmax);
    d[WICK_DPARENT] = 0;
    d[WICK_DRA] = ahp.n_rows;
    d[WICK_DRB] = bhp.n_rows;
    d[WICK_DSIGN] = 1;
    d[WICK_DOSIGN] = 1;
    for(size_t spin=0; spin < 2; spin++)
    {
        const arma::umat &hp = (spin == 0) ? ahp : bhp;
        int *holes = d + WICK_DHEAD + 2 * kmax * spin, *parts = holes + kmax;
        for(size_t k=0; k < hp.n_rows; k++)
        {
            holes[k] = label(hp(k,0), ket, spin == 0);
            parts[k] = label(hp(k,1), ket, spin == 0);
        }
    }
}

template<typename Tc, typename Tb>
size_t wick_pair<Tc,Tb>::layout(int64_t *info, size_t c, size_t start) const
{
    const size_t n2 = m_n * m_n, n4 = n2 * n2;
    const size_t ma = m_pair[0][c / m_pair[1].size()].m, mb = m_pair[1][c % m_pair[1].size()].m;
    size_t i = start;
    for(int k=0; k < WICK_PAIRLEN; k++) info[k] = -1;
    info[WICK_MA] = ma;
    info[WICK_MB] = mb;
    info[WICK_NOCCA] = m_nocc[0];
    info[WICK_NOCCB] = m_nocc[1];
    info[WICK_NMO] = m_n;
    info[WICK_NCORNER] = (c == 0) ? ncorner() : 1;

    const size_t blk[2] = {WICK_AA, WICK_BB}, m[2] = {ma, mb};
    for(size_t spin=0; spin < 2; spin++)
    {
        // X, Y, F and V
        for(size_t k=0; k < WICK_FH + 4 + 8; k++) { info[blk[spin]+k] = i; i += n2; }
        for(size_t slot=0; slot < 10; slot++)
        {
            size_t sum = jslots[slot][0] + jslots[slot][1] + jslots[slot][2] + jslots[slot][3];
            if(sum <= m[spin]) { info[blk[spin]+WICK_J+slot] = i; i += n4; }
        }
    }
    // Different-spin one-column intermediates
    for(size_t k=0; k < 16; k++) { info[WICK_AB+k] = i; i += n2; }
    // Precombined one-column intermediates without zero-overlap pairs
    if(ma == 0 and mb == 0)
    {
        info[WICK_AA+WICK_HCOL0] = i; i += n2;
        info[WICK_BB+WICK_HCOL0] = i; i += n2;
    }
    // Different-spin two-column intermediates
    for(size_t ma0=0; ma0 < 2; ma0++)
    for(size_t maz=0; maz < 2; maz++)
    for(size_t mb0=0; mb0 < 2; mb0++)
    for(size_t mby=0; mby < 2; mby++)
    {
        if(ma0 + maz <= ma and mb0 + mby <= mb)
        {
            info[WICK_AB+WICK_IIAB+ma0*8+maz*4+mb0*2+mby] = i;
            i += n4;
        }
    }
    return i;
}

template<typename Tc, typename Tb>
void wick_pair<Tc,Tb>::add_one_body(const arma::Mat<Tc> &Fa, const arma::Mat<Tc> &Fb)
{
    const arma::Mat<Tc> *F[2] = {&Fa, &Fb};
    for(size_t c=0; c < ncorner(); c++)
    {
        const size_t cs[2] = {c / m_pair[1].size(), c % m_pair[1].size()};
        const int64_t *info = m_info.data() + c * WICK_PAIRLEN;
        Tc *scal = m_scal.data() + c * WICK_NSCAL;
        for(size_t spin=0; spin < 2; spin++)
        {
            const spin_pairing &p = m_pair[spin][cs[spin]];
            const int64_t *off = info + (spin == 0 ? WICK_AA : WICK_BB);
            Tc *sc = scal + (spin == 0 ? WICK_SAA : WICK_SBB);

            // Zeroth-order and one-column terms
            for(size_t i=0; i < 2; i++)
            {
                sc[WICK_SF0H+i] = arma::accu(p.M[i] % F[spin]->st());
                for(size_t j=0; j < 2; j++)
                    put_cols(m_slab.data(), off[WICK_FH+i*2+j], 
                             arma::Mat<Tc>(p.CX[i].t() * (*F[spin]) * p.XC[j]));
            }
        }
        combine(info);
    }
}

template<typename Tc, typename Tb>
void wick_pair<Tc,Tb>::add_two_body(arma::Mat<Tb> &V)
{
    for(size_t c=0; c < ncorner(); c++)
    {
        write_two_body(m_slab.data(), m_scal.data() + c * WICK_NSCAL,
                       m_info.data() + c * WICK_PAIRLEN, c, V);
        combine(m_info.data() + c * WICK_PAIRLEN);
    }
}

template<typename Tc, typename Tb>
void wick_pair<Tc,Tb>::write_two_body(
    Tc *slab, Tc *scal, const int64_t *info, size_t c, arma::Mat<Tb> &V)
{
    const size_t n = m_n;
    const size_t cs[2] = {c / m_pair[1].size(), c % m_pair[1].size()};
    const size_t blk[2] = {WICK_AA, WICK_BB}, sblk[2] = {WICK_SAA, WICK_SBB};

    // Coulomb matrices of both spins
    arma::Mat<Tc> J[2][2], K;
    for(size_t spin=0; spin < 2; spin++)
    {
        spin_pairing &p = m_pair[spin][cs[spin]];
        const int64_t *off = info + blk[spin];
        Tc *sc = scal + sblk[spin];

        // V_0^{(m1,m2)} is symmetric in (m1,m2) and indexed by m1 + m2. The evaluator visits
        // (0,1) and (1,0) separately, so the middle entry holds one assignment only.
        arma::Mat<Tc> JK[2];
        for(size_t i=0; i < 2; i++)
        {
            coulomb_exchange(V, p.M[i], J[spin][i], K);
            JK[i] = J[spin][i] - K;
        }
        sc[WICK_SV0+0] = arma::accu(JK[0] % p.M[0].st());
        sc[WICK_SV0+1] = (p.m >= 1) ? Tc(arma::accu(JK[0] % p.M[1].st())) : Tc(0.0);
        sc[WICK_SV0+2] = (p.m >= 2) ? Tc(arma::accu(JK[1] % p.M[1].st())) : Tc(0.0);

        // One-column terms v[m1][m3][m2] = CX[m1]^+ (J-K)[m2] XC[m3]
        for(size_t i=0; i < 2; i++)
        for(size_t j=0; j < 2; j++)
        for(size_t k=0; k < 2; k++)
            put_cols(slab, off[WICK_V+i*4+j*2+k], arma::Mat<Tc>(p.CX[i].t() * JK[k] * p.XC[j]));

        // Two-column terms J_{pq,rs} = (pq|rs) - (pr|qs) stored with the pair (r,s) first. The
        // row coefficients CX are conjugated and the column coefficients XC are not.
        for(size_t slot=0; slot < 10; slot++)
        {
            if(off[WICK_J+slot] < 0) continue;
            const int mi = jslots[slot][0], mj = jslots[slot][1];
            const int mk = jslots[slot][2], ml = jslots[slot][3];
            arma::Mat<Tc> D, E;
            arma::Mat<Tc> CXk = arma::conj(p.CX[mk]), XCj = arma::conj(p.XC[mj]);
            eri_ao2mo(p.CX[mi], p.XC[mj], p.CX[mk], p.XC[ml], V, D, false);
            eri_ao2mo(p.CX[mi], CXk, XCj, p.XC[ml], V, E, false);
            const Tc *d = D.memptr(), *e = E.memptr();
            Tc *t = slab + off[WICK_J+slot];
            for(size_t r=0; r < n; r++)
            for(size_t s=0; s < n; s++)
            for(size_t pp=0; pp < n; pp++)
            for(size_t q=0; q < n; q++)
                t[((r*n+s)*n+pp)*n+q] = d[((r*n+s)*n+pp)*n+q] - e[((q*n+s)*n+pp)*n+r];
        }
    }

    // Different-spin terms
    spin_pairing &pa = m_pair[0][cs[0]], &pb = m_pair[1][cs[1]];
    const int64_t *off = info + WICK_AB;
    for(size_t i=0; i < 2; i++)
    for(size_t j=0; j < 2; j++)
    {
        scal[WICK_SVAB0+i*2+j] = arma::accu(J[0][i] % pb.M[j].st());
        for(size_t k=0; k < 2; k++)
        {
            // V^a: ma0*4 + mb0*2 + maz, and V^b: mb0*4 + ma0*2 + mby
            put_cols(slab, off[WICK_VAB+i*4+j*2+k], arma::Mat<Tc>(pa.CX[i].t() * J[1][j] * pa.XC[k]));
            put_cols(slab, off[WICK_VBA+i*4+j*2+k], arma::Mat<Tc>(pb.CX[i].t() * J[0][j] * pb.XC[k]));
        }
    }
    for(size_t ma0=0; ma0 < 2; ma0++)
    for(size_t maz=0; maz < 2; maz++)
    for(size_t mb0=0; mb0 < 2; mb0++)
    for(size_t mby=0; mby < 2; mby++)
    {
        int64_t o = off[WICK_IIAB+ma0*8+maz*4+mb0*2+mby];
        if(o < 0) continue;
        arma::Mat<Tc> II;
        eri_ao2mo(pa.CX[ma0], pa.XC[maz], pb.CX[mb0], pb.XC[mby], V, II, false);
        put_rows(slab, o, II);
    }
}

template<typename Tc, typename Tb>
void wick_pair<Tc,Tb>::combine(const int64_t *info)
{
    Tc *slab = m_slab.data();
    if(info[WICK_AA+WICK_HCOL0] < 0) return;
    const size_t n2 = info[WICK_NMO] * info[WICK_NMO];
    const int64_t *aa = info + WICK_AA, *bb = info + WICK_BB, *ab = info + WICK_AB;
    for(size_t k=0; k < n2; k++)
    {
        slab[aa[WICK_HCOL0]+k] = slab[aa[WICK_FH]+k] + slab[aa[WICK_V]+k] + slab[ab[WICK_VAB]+k];
        slab[bb[WICK_HCOL0]+k] = slab[bb[WICK_FH]+k] + slab[bb[WICK_V]+k] + slab[ab[WICK_VBA]+k];
    }
}

template<typename Tc, typename Tb>
void wick_pair<Tc,Tb>::evaluate(
    const arma::umat &xahp, const arma::umat &xbhp,
    const arma::umat &wahp, const arma::umat &wbhp,
    Tc &S, Tc &V) const
{
    const int kmax = std::max(std::max(xahp.n_rows, xbhp.n_rows), 
                              std::max(std::max(wahp.n_rows, wbhp.n_rows), (arma::uword) 1));
    int dx[WICK_DHEAD + 4 * WICK_MAXMO], dw[WICK_DHEAD + 4 * WICK_MAXMO];
    record(dx, kmax, xahp, xbhp, false);
    record(dw, kmax, wahp, wbhp, true);
    wick_kernels::wick_env env = environment(m_info, m_slab, m_scal, m_Vc, kmax);

    S = 0.0; V = 0.0;
    for(size_t c=0; c < ncorner(); c++)
    {
        Tc h, s;
        wick_kernels::hamiltonian_corner<Tc>(&h, &s, &env, info(c), scal(c), dx, dw);
        S += s; V += h;
    }
}

template<typename Tc, typename Tb>
void wick_pair<Tc,Tb>::evaluate_overlap(
    const arma::umat &xahp, const arma::umat &xbhp,
    const arma::umat &wahp, const arma::umat &wbhp,
    Tc &S) const
{
    const int kmax = std::max(std::max(xahp.n_rows, xbhp.n_rows), 
                              std::max(std::max(wahp.n_rows, wbhp.n_rows), (arma::uword) 1));
    int dx[WICK_DHEAD + 4 * WICK_MAXMO], dw[WICK_DHEAD + 4 * WICK_MAXMO];
    record(dx, kmax, xahp, xbhp, false);
    record(dw, kmax, wahp, wbhp, true);
    wick_kernels::wick_env env = environment(m_info, m_slab, m_scal, m_Vc, kmax);

    S = 0.0;
    for(size_t c=0; c < ncorner(); c++)
        S += wick_kernels::overlap_corner<Tc>(&env, info(c), scal(c), dx, dw);
}

template<typename Tc, typename Tb>
void wick_pair<Tc,Tb>::evaluate_one_body_spin(
    const arma::umat &xhp, const arma::umat &whp, Tc &S, Tc &V, bool alpha) const
{
    const int kmax = std::max(std::max(xhp.n_rows, whp.n_rows), (arma::uword) 1);
    const arma::umat none(0, 2);
    int dx[WICK_DHEAD + 4 * WICK_MAXMO], dw[WICK_DHEAD + 4 * WICK_MAXMO];
    record(dx, kmax, alpha ? xhp : none, alpha ? none : xhp, false);
    record(dw, kmax, alpha ? whp : none, alpha ? none : whp, true);
    wick_kernels::wick_env env = environment(m_info, m_slab, m_scal, m_Vc, kmax);

    // Corner combinations of this spin only
    const size_t ncb = m_pair[1].size(), nc = alpha ? m_pair[0].size() : ncb;
    S = 0.0; V = 0.0;
    for(size_t k=0; k < nc; k++)
    {
        size_t c = alpha ? k * ncb : k;
        Tc h, s;
        wick_kernels::spin_one_body_corner<Tc>(&h, &s, &env, info(c), scal(c), dx, dw, alpha);
        S += s; V += h;
    }
}

template class wick_pair<double, double>;
template class wick_pair<std::complex<double>, double>;
template class wick_pair<std::complex<double>, std::complex<double> >;

} // namespace libgnme
