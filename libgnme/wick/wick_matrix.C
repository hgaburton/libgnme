#include <cassert>
#include <complex>
#include <algorithm>
#include <libgnme/utils/eri_ao2mo.h>
#include "wick_matrix.h"
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

/** \brief Lowdin-paired orbitals and fundamental contractions of one spin of a pair of
           references at one near-zero corner **/
template<typename Tc>
struct spin_pairing
{
    Tc phase;  //!< Phase of the orbital pairing
    Tc redS;   //!< Reduced overlap, including the near-zero corner weights
    size_t m;  //!< Number of zero-overlap orbital pairs
    arma::Mat<Tc> M[2];  //!< Co-density matrices
    arma::Mat<Tc> X[2];  //!< X fundamental contractions, rows V_x + O_w, columns O_x + V_w
    arma::Mat<Tc> Y[2];  //!< Y fundamental contractions
    arma::Mat<Tc> CX[2]; //!< Row contraction coefficients
    arma::Mat<Tc> XC[2]; //!< Column contraction coefficients
};

/** \brief Number of zero-overlap and near-zero occupied orbital pairs **/
template<typename Tc>
void count_pairs(
    const arma::Mat<Tc> &S, const arma::Mat<Tc> &Cx, const arma::Mat<Tc> &Cw, size_t nocc,
    size_t &nzero, size_t &nnear)
{
    nzero = 0; nnear = 0;
    if(nocc == 0) return;
    arma::vec s = arma::svd(arma::Mat<Tc>(Cx.head_cols(nocc).t() * S * Cw.head_cols(nocc)));
    for(size_t k=0; k < nocc; k++)
    {
        if(s(k) <= zero_thresh) nzero++;
        else if(s(k) < near_thresh) nnear++;
    }
}

/** \brief Lowdin pairing and fundamental contractions for one near-zero corner
    \param corner Bit mask of near-zero pairs entering with unit weight in place of 1/S_k
 **/
template<typename Tc>
void pair_orbitals(
    spin_pairing<Tc> &p, const arma::Mat<Tc> &S,
    const arma::Mat<Tc> &Cx, const arma::Mat<Tc> &Cw, size_t nocc, size_t corner)
{
    size_t nbsf = Cx.n_rows, nmo = Cx.n_cols;
    arma::Mat<Tc> cxo = Cx.head_cols(nocc), cwo = Cw.head_cols(nocc);
    arma::Mat<Tc> cxv = Cx.tail_cols(nmo-nocc), cwv = Cw.tail_cols(nmo-nocc);

    // Lowdin pair the occupied orbitals
    arma::Mat<Tc> cxt = cxo, cwt = cwo;
    arma::vec s(nocc, arma::fill::zeros);
    p.phase = 1.0;
    if(nocc > 0)
    {
        arma::Mat<Tc> U, V;
        arma::svd(U, s, V, arma::Mat<Tc>(cxo.t() * S * cwo));
        p.phase = arma::det(U) * conj2(arma::det(V));
        cxt = cxo * U;
        cwt = cwo * V;
    }

    // M^{(0)} = W + P^{xw} + P^{ww}, M^{(1)} = P^{xw}. Near-zero pairs are removed from W with
    // weight S_k - 1 in the reduced overlap, or enter W with unit weight.
    p.redS = 1.0; p.m = 0;
    p.M[0].zeros(nbsf, nbsf);
    p.M[1].zeros(nbsf, nbsf);
    size_t inear = 0;
    for(size_t k=0; k < nocc; k++)
    {
        if(s(k) <= zero_thresh)
        {
            p.M[1] += cwt.col(k) * cxt.col(k).t();
            p.M[0] += cwt.col(k) * cwt.col(k).t();
            p.m++;
        }
        else if(s(k) < near_thresh)
        {
            if((corner >> inear) & 1)
                p.M[0] += cwt.col(k) * cxt.col(k).t();
            else
                p.redS *= (s(k) - 1.0);
            inear++;
        }
        else
        {
            p.M[0] += cwt.col(k) * cxt.col(k).t() / s(k);
            p.redS *= s(k);
        }
    }
    p.M[0] += p.M[1];

    // Fundamental contractions with rows V_x + O_w and columns O_x + V_w
    arma::Mat<Tc> rowc = arma::join_rows(cxv, cwo);
    arma::Mat<Tc> colc = arma::join_rows(cxo, cwv);
    for(size_t i=0; i < 2; i++)
    {
        arma::Mat<Tc> xao = S * p.M[i] * S;
        p.X[i] = rowc.t() * xao * colc;
        p.Y[i] = (i == 0) ? arma::Mat<Tc>(rowc.t() * (xao - S) * colc) : p.X[i];

        // For m_i = 0 the direct orbital contribution is removed
        double f = (i == 0) ? 1.0 : 0.0;
        p.CX[i] = arma::join_rows(p.M[i].t() * S * cxv - f * cxv, p.M[i].t() * S * cwo);
        p.XC[i] = arma::join_rows(p.M[i] * S * cxo, p.M[i] * S * cwv - f * cwv);
    }
}

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

/** \brief Offsets of the intermediates of one pair, adapted to its zero-overlap pairs **/
size_t pair_layout(int64_t *info, size_t ma, size_t mb, size_t nmo, size_t start)
{
    const size_t n2 = nmo * nmo, n4 = n2 * n2;
    size_t i = start;
    for(int k=0; k < WICK_PAIRLEN; k++) info[k] = -1;
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

/** \brief Offsets of the same-reference MO integrals of a diagonal pair **/
size_t orthogonal_layout(int64_t *info, size_t nmo, size_t start)
{
    const size_t n2 = nmo * nmo, n4 = n2 * n2;
    size_t i = start;
    info[WICK_OH_A] = i; i += n2;
    info[WICK_OH_B] = i; i += n2;
    info[WICK_OAA] = i; i += n4;
    info[WICK_OBB] = i; i += n4;
    info[WICK_OAB] = i; i += n4;
    return i;
}

/** \brief Copy a matrix into the slab in row-major order **/
template<typename Tc>
void put_rows(std::vector<Tc> &slab, int64_t off, const arma::Mat<Tc> &A)
{
    arma::Mat<Tc> At = A.st();
    std::copy(At.memptr(), At.memptr() + At.n_elem, slab.begin() + off);
}

/** \brief Copy a matrix into the slab in column-major order **/
template<typename Tc>
void put_cols(std::vector<Tc> &slab, int64_t off, const arma::Mat<Tc> &A)
{
    std::copy(A.memptr(), A.memptr() + A.n_elem, slab.begin() + off);
}

/** \brief One-body and same-spin two-body intermediates of one spin **/
template<typename Tc, typename Tb>
void write_same_spin(
    std::vector<Tc> &slab, Tc *scal, const int64_t *off, spin_pairing<Tc> &p,
    const arma::Mat<Tc> &F, arma::Mat<Tb> &V,
    arma::Mat<Tc> (&fh)[2][2], arma::Mat<Tc> (&v)[2][2][2])
{
    const size_t n = p.X[0].n_rows;

    // Zeroth-order and one-column one-body terms
    for(size_t i=0; i < 2; i++)
    {
        scal[WICK_SF0H+i] = arma::accu(p.M[i] % F.st());
        for(size_t j=0; j < 2; j++)
            fh[i][j] = p.CX[i].t() * F * p.XC[j];
    }

    // V_0^{(m1,m2)} is symmetric in (m1,m2) and indexed by m1 + m2. The evaluator visits
    // (0,1) and (1,0) separately, so the middle entry holds one assignment only.
    arma::Mat<Tc> JK[2], J, K;
    for(size_t i=0; i < 2; i++)
    {
        coulomb_exchange(V, p.M[i], J, K);
        JK[i] = J - K;
    }
    scal[WICK_SV0+0] = arma::accu(JK[0] % p.M[0].st());
    scal[WICK_SV0+1] = (p.m >= 1) ? Tc(arma::accu(JK[0] % p.M[1].st())) : Tc(0.0);
    scal[WICK_SV0+2] = (p.m >= 2) ? Tc(arma::accu(JK[1] % p.M[1].st())) : Tc(0.0);

    // One-column two-body terms v[m1][m3][m2] = CX[m1]^+ (J-K)[m2] XC[m3]
    for(size_t i=0; i < 2; i++)
    for(size_t j=0; j < 2; j++)
    for(size_t k=0; k < 2; k++)
        v[i][j][k] = p.CX[i].t() * JK[k] * p.XC[j];

    // Store the fundamental contractions and one-column terms
    for(size_t i=0; i < 2; i++)
    {
        put_rows(slab, off[WICK_X+i], p.X[i]);
        put_rows(slab, off[WICK_Y+i], p.Y[i]);
        for(size_t j=0; j < 2; j++)
        {
            put_cols(slab, off[WICK_FH+i*2+j], fh[i][j]);
            for(size_t k=0; k < 2; k++)
                put_cols(slab, off[WICK_V+i*4+j*2+k], v[i][j][k]);
        }
    }

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
        Tc *t = slab.data() + off[WICK_J+slot];
        for(size_t r=0; r < n; r++)
        for(size_t s=0; s < n; s++)
        for(size_t pp=0; pp < n; pp++)
        for(size_t q=0; q < n; q++)
            t[((r*n+s)*n+pp)*n+q] = d[((r*n+s)*n+pp)*n+q] - e[((q*n+s)*n+pp)*n+r];
    }

    // Scalars
    scal[WICK_SPHASE] = p.phase;
    scal[WICK_STSP] = p.redS;
}

/** \brief Different-spin two-body intermediates **/
template<typename Tc, typename Tb>
void write_diff_spin(
    std::vector<Tc> &slab, Tc *scal, const int64_t *info,
    spin_pairing<Tc> &pa, spin_pairing<Tc> &pb, arma::Mat<Tb> &V,
    arma::Mat<Tc> (&vab)[2][2][2], arma::Mat<Tc> (&vba)[2][2][2])
{
    const int64_t *off = info + WICK_AB;
    arma::Mat<Tc> Ja[2], Jb[2], K;
    for(size_t i=0; i < 2; i++)
    {
        coulomb_exchange(V, pa.M[i], Ja[i], K);
        coulomb_exchange(V, pb.M[i], Jb[i], K);
    }
    for(size_t i=0; i < 2; i++)
    for(size_t j=0; j < 2; j++)
    {
        scal[WICK_SVAB0+i*2+j] = arma::accu(Ja[i] % pb.M[j].st());
        for(size_t k=0; k < 2; k++)
        {
            // V^a: ma0*4 + mb0*2 + maz, and V^b: mb0*4 + ma0*2 + mby
            vab[i][j][k] = pa.CX[i].t() * Jb[j] * pa.XC[k];
            vba[i][j][k] = pb.CX[i].t() * Ja[j] * pb.XC[k];
            put_cols(slab, off[WICK_VAB+i*4+j*2+k], vab[i][j][k]);
            put_cols(slab, off[WICK_VBA+i*4+j*2+k], vba[i][j][k]);
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

/** \brief MO integrals of one reference for the same-reference Slater-Condon rules **/
template<typename Tc, typename Tb>
void write_orthogonal(
    std::vector<Tc> &slab, const int64_t *info,
    const arma::Mat<Tc> &Fa, const arma::Mat<Tc> &Fb, arma::Mat<Tb> &V,
    arma::Mat<Tc> Ca, arma::Mat<Tc> Cb)
{
    const size_t n = Ca.n_cols;
    put_rows(slab, info[WICK_OH_A], arma::Mat<Tc>(Ca.t() * Fa * Ca));
    put_rows(slab, info[WICK_OH_B], arma::Mat<Tc>(Cb.t() * Fb * Cb));
    // Same-spin (pq|rs) - (ps|rq)
    for(size_t spin=0; spin < 2; spin++)
    {
        arma::Mat<Tc> &C = (spin == 0) ? Ca : Cb;
        arma::Mat<Tc> G;
        eri_ao2mo(C, C, C, C, V, G, false);
        const Tc *g = G.memptr();
        Tc *t = slab.data() + info[spin == 0 ? WICK_OAA : WICK_OBB];
        for(size_t p=0; p < n; p++)
        for(size_t q=0; q < n; q++)
        for(size_t r=0; r < n; r++)
        for(size_t s=0; s < n; s++)
            t[((p*n+q)*n+r)*n+s] = g[((r*n+s)*n+p)*n+q] - g[((r*n+q)*n+p)*n+s];
    }
    // Different-spin (pq|rs)
    arma::Mat<Tc> G;
    eri_ao2mo(Ca, Ca, Cb, Cb, V, G, false);
    put_rows(slab, info[WICK_OAB], G);
}

/** \brief Fill the rows of a matrix block between bra dx and the kets dws of one reference. 
           Excitations of the same reference use the Slater-Condon rules if orth is true. 
 **/
template<typename Tc>
void hs_segment(
    Tc *h, Tc *s, const wick_kernels::wick_env *env, const int *dx, const uint64_t *sx,
    const int *dws, const uint64_t *sws, int nw, bool orth)
{
    if(dx[WICK_DPARENT] != dws[WICK_DPARENT])
    {
        for(int j=0; j < nw; j++)
            wick_kernels::hamiltonian<Tc>(h+j, s+j, env, dx, dws + (size_t) j * env->dstride);
    }
    else if(orth)
    {
        wick_kernels::hs_orth_row<Tc>(h, s, env, dx, sx, dws, sws, nw);
    }
    else
    {
        // A pair of identical references has no near-zero corners
        const size_t k = (size_t) dx[WICK_DPARENT] * env->nref + dx[WICK_DPARENT];
        const int64_t *info = env->pairinfo + k * WICK_PAIRLEN;
        const Tc *scal = (const Tc *) env->scal + k * WICK_NSCAL;
        for(int j=0; j < nw; j++)
            wick_kernels::hamiltonian_corner<Tc>(h+j, s+j, env, info, scal, dx, 
                                                 dws + (size_t) j * env->dstride);
    }
}

} // unnamed namespace


template<typename Tc, typename Tf, typename Tb>
void wick_matrix<Tc,Tf,Tb>::add_reference(const arma::Mat<Tc> &Ca, const arma::Mat<Tc> &Cb)
{
    assert(Ca.n_rows == m_nbsf and Ca.n_cols == m_nmo);
    assert(Cb.n_rows == m_nbsf and Cb.n_cols == m_nmo);
    m_Ca.push_back(Ca);
    m_Cb.push_back(Cb);
    arma::Mat<Tc> S = arma::conv_to<arma::Mat<Tc> >::from(m_metric);
    double err = std::max(
        arma::abs(arma::Mat<Tc>(Ca.t() * S * Ca) - arma::eye<arma::Mat<Tc> >(m_nmo, m_nmo)).max(),
        arma::abs(arma::Mat<Tc>(Cb.t() * S * Cb) - arma::eye<arma::Mat<Tc> >(m_nmo, m_nmo)).max());
    m_orthonormal.push_back(err < 1e-10);
    m_built = false;
}

template<typename Tc, typename Tf, typename Tb>
void wick_matrix<Tc,Tf,Tb>::add_one_body(arma::Mat<Tf> &F)
{
    add_one_body(F, F);
}

template<typename Tc, typename Tf, typename Tb>
void wick_matrix<Tc,Tf,Tb>::add_one_body(arma::Mat<Tf> &Fa, arma::Mat<Tf> &Fb)
{
    m_Fa = arma::conv_to<arma::Mat<Tc> >::from(Fa);
    m_Fb = arma::conv_to<arma::Mat<Tc> >::from(Fb);
    m_built = false;
}

template<typename Tc, typename Tf, typename Tb>
void wick_matrix<Tc,Tf,Tb>::add_two_body(arma::Mat<Tb> &V)
{
    assert(V.n_rows == m_nbsf * m_nbsf);
    assert(V.n_cols == m_nbsf * m_nbsf);
    m_V = &V;
    m_built = false;
}

template<typename Tc, typename Tf, typename Tb>
void wick_matrix<Tc,Tf,Tb>::build()
{
    const size_t nref = m_Ca.size();
    const size_t nmo = m_nmo;
    const arma::Mat<Tc> S = arma::conv_to<arma::Mat<Tc> >::from(m_metric);

    // Operators that were not added are zero
    if(m_Fa.n_elem == 0) { m_Fa.zeros(m_nbsf, m_nbsf); m_Fb.zeros(m_nbsf, m_nbsf); }
    arma::Mat<Tb> Vzero;
    if(m_V == nullptr) Vzero.zeros(m_nbsf*m_nbsf, m_nbsf*m_nbsf);
    arma::Mat<Tb> &V = (m_V == nullptr) ? Vzero : *m_V;

    // Layout of the primary corner of every ordered pair
    std::vector<int64_t> info((size_t) nref * nref * WICK_PAIRLEN);
    std::vector<size_t> nca(nref*nref), ncb(nref*nref);
    size_t end = 0;
    for(size_t x=0; x < nref; x++)
    for(size_t w=0; w < nref; w++)
    {
        size_t k = x * nref + w;
        size_t ma, mb, na, nb;
        count_pairs(S, m_Ca[x], m_Ca[w], m_nalpha, ma, na);
        count_pairs(S, m_Cb[x], m_Cb[w], m_nbeta, mb, nb);
        int64_t *row = info.data() + k * WICK_PAIRLEN;
        end = pair_layout(row, ma, mb, nmo, end);
        if(x == w) end = orthogonal_layout(row, nmo, end);
        row[WICK_MA] = ma;
        row[WICK_MB] = mb;
        row[WICK_NOCCA] = m_nalpha;
        row[WICK_NOCCB] = m_nbeta;
        row[WICK_NMO] = nmo;
        nca[k] = (size_t) 1 << na;
        ncb[k] = (size_t) 1 << nb;
        row[WICK_NCORNER] = nca[k] * ncb[k];
    }

    // The other corner combinations follow all primary rows
    size_t nrow = nref * nref;
    for(size_t k=0; k < nref*nref; k++)
    {
        int64_t ncorner = info[k*WICK_PAIRLEN+WICK_NCORNER];
        info[k*WICK_PAIRLEN+WICK_CSTART] = nrow;
        for(int64_t c=1; c < ncorner; c++)
        {
            std::vector<int64_t> row(WICK_PAIRLEN);
            end = pair_layout(row.data(), info[k*WICK_PAIRLEN+WICK_MA],
                              info[k*WICK_PAIRLEN+WICK_MB], nmo, end);
            for(int i=WICK_MA; i <= WICK_NMO; i++) row[i] = info[k*WICK_PAIRLEN+i];
            row[WICK_NCORNER] = 1;
            info.insert(info.end(), row.begin(), row.end());
            nrow++;
        }
    }
    m_pairinfo.swap(info);
    m_slab.assign(end, Tc(0.0));
    m_scal.assign(nrow * WICK_NSCAL, Tc(0.0));

    // Intermediates of each pair and corner combination
    for(size_t x=0; x < nref; x++)
    for(size_t w=0; w < nref; w++)
    {
        size_t k = x * nref + w;
        std::vector<spin_pairing<Tc> > pa(nca[k]), pb(ncb[k]);
        std::vector<arma::Mat<Tc> > hcola(nca[k]), hcolb(ncb[k]);
        for(size_t ca=0; ca < nca[k]; ca++)
            pair_orbitals(pa[ca], S, m_Ca[x], m_Ca[w], m_nalpha, ca);
        for(size_t cb=0; cb < ncb[k]; cb++)
            pair_orbitals(pb[cb], S, m_Cb[x], m_Cb[w], m_nbeta, cb);

        for(size_t ca=0; ca < nca[k]; ca++)
        for(size_t cb=0; cb < ncb[k]; cb++)
        {
            size_t c = ca * ncb[k] + cb;
            size_t row = (c == 0) ? k : m_pairinfo[k*WICK_PAIRLEN+WICK_CSTART] + c - 1;
            const int64_t *off = m_pairinfo.data() + row * WICK_PAIRLEN;
            Tc *scal = m_scal.data() + row * WICK_NSCAL;
            assert((size_t) off[WICK_MA] == pa[ca].m and (size_t) off[WICK_MB] == pb[cb].m);

            arma::Mat<Tc> fha[2][2], fhb[2][2], va[2][2][2], vb[2][2][2];
            arma::Mat<Tc> vab[2][2][2], vba[2][2][2];
            write_same_spin(m_slab, scal + WICK_SAA, off + WICK_AA, pa[ca], m_Fa, V, fha, va);
            write_same_spin(m_slab, scal + WICK_SBB, off + WICK_BB, pb[cb], m_Fb, V, fhb, vb);
            write_diff_spin(m_slab, scal, off, pa[ca], pb[cb], V, vab, vba);

            // Precombined one-column intermediates F + V + V^{ab} without zero-overlap pairs
            if(off[WICK_AA+WICK_HCOL0] >= 0)
            {
                put_cols(m_slab, off[WICK_AA+WICK_HCOL0],
                         arma::Mat<Tc>(fha[0][0] + va[0][0][0] + vab[0][0][0]));
                put_cols(m_slab, off[WICK_BB+WICK_HCOL0],
                         arma::Mat<Tc>(fhb[0][0] + vb[0][0][0] + vba[0][0][0]));
            }
        }
        if(x == w)
            write_orthogonal(m_slab, m_pairinfo.data() + k * WICK_PAIRLEN, m_Fa, m_Fb, V,
                             m_Ca[x], m_Cb[x]);
    }
    m_built = true;
}

template<typename Tc, typename Tf, typename Tb>
void wick_matrix<Tc,Tf,Tb>::evaluate(
    const std::vector<size_t> &ref,
    std::vector<bitset> &ba, std::vector<bitset> &bb,
    arma::Mat<Tc> &S, arma::Mat<Tc> &H)
{
    const size_t ndet = ref.size();
    assert(ba.size() == ndet and bb.size() == ndet);
    if(not m_built) build();

    // Evaluate the determinants grouped by reference
    std::vector<size_t> order(ndet);
    for(size_t i=0; i < ndet; i++) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&ref](size_t i, size_t j) { return ref[i] < ref[j]; });

    // Determinant records of nowick_kernels.h with holes and particles relative to the
    // reference, which has its occupied orbitals first
    int kmax = 1;
    std::vector<arma::uvec> occa(ndet), occb(ndet);
    for(size_t i=0; i < ndet; i++)
    {
        occa[i] = ba[i].occ();
        occb[i] = bb[i].occ();
        assert(occa[i].n_elem == m_nalpha and occb[i].n_elem == m_nbeta);
        int ra = 0, rb = 0;
        for(size_t k=0; k < m_nalpha; k++) ra += (occa[i](k) >= m_nalpha);
        for(size_t k=0; k < m_nbeta; k++) rb += (occb[i](k) >= m_nbeta);
        kmax = std::max(kmax, std::max(ra, rb));
    }
    const int dstride = WICK_DHEAD + 4 * kmax;
    std::vector<int> rec(ndet * dstride, 0);
    for(size_t ii=0; ii < ndet; ii++)
    {
        size_t i = order[ii];
        assert(ref[i] < m_Ca.size());
        int *d = rec.data() + ii * dstride;
        d[WICK_DPARENT] = ref[i];
        int odd = 0;
        for(size_t spin=0; spin < 2; spin++)
        {
            const arma::uvec &occ = (spin == 0) ? occa[i] : occb[i];
            const int nocc = (spin == 0) ? m_nalpha : m_nbeta;
            int *holes = d + WICK_DHEAD + 2 * kmax * spin;
            int *parts = holes + kmax;
            std::vector<bool> inref(nocc, false);
            int r = 0;
            for(size_t k=0; k < occ.n_elem; k++)
            {
                if((int) occ(k) < nocc) inref[occ(k)] = true;
                else parts[r++] = occ(k);
            }
            int nh = 0, hsum = 0;
            for(int k=0; k < nocc; k++)
            {
                if(not inref[k]) { holes[nh++] = k; hsum += k; }
            }
            assert(nh == r);
            d[spin == 0 ? WICK_DRA : WICK_DRB] = r;
            // Annihilating the holes (descending) and creating the particles (ascending)
            // on the reference string
            odd += hsum + r * (nocc - r) + r * (r - 1) / 2;
        }
        d[WICK_DSIGN] = (odd & 1) ? -1 : 1;
        d[WICK_DOSIGN] = 1;
    }

    // Occupation strings of every determinant
    std::vector<uint64_t> strs(ndet * WICK_NSTR);
    for(size_t i=0; i < ndet; i++)
        wick_kernels::wick_det_strings(strs.data() + i * WICK_NSTR, rec.data() + i * dstride,
                                       kmax, m_nalpha, m_nbeta);

    wick_kernels::wick_env env;
    env.nref = m_Ca.size();
    env.kmax = kmax;
    env.dstride = dstride;
    env.simd = 0;
    env.enuc = m_Vc;
    env.pairinfo = m_pairinfo.data();
    env.slab = m_slab.data();
    env.scal = m_scal.data();

    // Determinants already grouped by reference are evaluated directly into the output
    bool sorted = true;
    for(size_t i=0; i < ndet; i++) sorted = sorted and (order[i] == i);
    arma::Mat<Tc> Hs, Ss;
    arma::Mat<Tc> &Hm = sorted ? H : Hs;
    arma::Mat<Tc> &Sm = sorted ? S : Ss;
    Hm.set_size(ndet, ndet);
    Sm.set_size(ndet, ndet);
    Tc *hp = Hm.memptr(), *sp = Sm.memptr();

    // Upper triangle by rows, in segments of kets with the same reference. The rows are stored
    // contiguously, which is the transpose of the column-major matrix.
    #pragma omp parallel for schedule(dynamic, 1)
    for(size_t i=0; i < ndet; i++)
    {
        const int *dx = rec.data() + i * dstride;
        size_t j1;
        for(size_t j=i; j < ndet; j=j1)
        {
            int pw = rec[j * dstride + WICK_DPARENT];
            for(j1 = j+1; j1 < ndet; j1++)
                if(rec[j1 * dstride + WICK_DPARENT] != pw) break;
            hs_segment<Tc>(hp + i * ndet + j, sp + i * ndet + j, &env, 
                           dx, strs.data() + i * WICK_NSTR, rec.data() + j * dstride, 
                           strs.data() + j * WICK_NSTR, j1 - j, m_orthonormal[pw]);
        }
    }

    // Lower triangle by Hermiticity. The column-major matrix then holds the transpose, which 
    // is the complex conjugate.
    #pragma omp parallel for schedule(static)
    for(size_t i=0; i < ndet; i++)
    for(size_t j=0; j < i; j++)
    {
        hp[i * ndet + j] = conj2(hp[j * ndet + i]);
        sp[i * ndet + j] = conj2(sp[j * ndet + i]);
    }
    #pragma omp parallel for schedule(static)
    for(size_t k=0; k < ndet * ndet; k++)
    {
        hp[k] = conj2(hp[k]);
        sp[k] = conj2(sp[k]);
    }

    // Original determinant order
    if(not sorted)
    {
        H.set_size(ndet, ndet);
        S.set_size(ndet, ndet);
        for(size_t jj=0; jj < ndet; jj++)
        for(size_t ii=0; ii < ndet; ii++)
        {
            H(order[ii],order[jj]) = Hs(ii,jj);
            S(order[ii],order[jj]) = Ss(ii,jj);
        }
    }
}

template class wick_matrix<double, double, double>;
template class wick_matrix<std::complex<double>, double, double>;
template class wick_matrix<std::complex<double>, std::complex<double>, double>;
template class wick_matrix<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
