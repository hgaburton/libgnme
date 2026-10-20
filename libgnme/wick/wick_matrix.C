#include <cassert>
#include <complex>
#include <algorithm>
#include <libgnme/utils/eri_ao2mo.h>
#include "wick_matrix.h"
#include "wick_kernels.h"
#include "wick_pair.h"

namespace libgnme {

#if defined(LIBGNME_X86_SIMD)
/* Packed kernels of wick_kernels_avx2.C and wick_kernels_avx512.C */
namespace wick_avx2 {
template<typename DT>
void hs_row(DT *hrow, DT *srow, const wick_kernels::wick_env *env, const int64_t *info,
            const DT *scal, const int *dx, const int *dws, int nw);
}
namespace wick_avx512 {
template<typename DT>
void hs_row(DT *hrow, DT *srow, const wick_kernels::wick_env *env, const int64_t *info,
            const DT *scal, const int *dx, const int *dws, int nw);
}
#endif

namespace {

#if defined(LIBGNME_X86_SIMD)
/* SIMD levels of the packed kernels */
const int simd_none = 0, simd_avx2 = 1, simd_avx512 = 2;

/** \brief Packed kernels supported by the CPU **/
int simd_level()
{
    __builtin_cpu_init();
    if(__builtin_cpu_supports("avx512f")) return simd_avx512;
    if(__builtin_cpu_supports("avx2") and __builtin_cpu_supports("fma")) return simd_avx2;
    return simd_none;
}
#endif

double conj2(double x) { return x; }
std::complex<double> conj2(std::complex<double> x) { return std::conj(x); }

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
#if defined(LIBGNME_X86_SIMD)
    // Packed rows for pairs without zero-overlap orbitals, summed over near-zero corners
    const size_t k = (size_t) dx[WICK_DPARENT] * env->nref + dws[WICK_DPARENT];
    const int64_t *info = env->pairinfo + k * WICK_PAIRLEN;
    if(dx[WICK_DPARENT] != dws[WICK_DPARENT] and env->simd != simd_none 
       and info[WICK_MA] == 0 and info[WICK_MB] == 0)
    {
        std::vector<Tc> buf(info[WICK_NCORNER] > 1 ? 2 * nw : 0);
        for(int64_t c=0; c < info[WICK_NCORNER]; c++)
        {
            size_t kc = (c == 0) ? k : info[WICK_CSTART] + c - 1;
            const int64_t *infoc = env->pairinfo + kc * WICK_PAIRLEN;
            const Tc *scal = (const Tc *) env->scal + kc * WICK_NSCAL;
            Tc *hc = (c == 0) ? h : buf.data();
            Tc *sc = (c == 0) ? s : buf.data() + nw;
            if(env->simd == simd_avx512)
                wick_avx512::hs_row<Tc>(hc, sc, env, infoc, scal, dx, dws, nw);
            else
                wick_avx2::hs_row<Tc>(hc, sc, env, infoc, scal, dx, dws, nw);
            if(c == 0) continue;
            for(int j=0; j < nw; j++) { h[j] += hc[j]; s[j] += sc[j]; }
        }
        return;
    }
#endif
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

    // Operators that were not added are zero
    if(m_Fa.n_elem == 0) { m_Fa.zeros(m_nbsf, m_nbsf); m_Fb.zeros(m_nbsf, m_nbsf); }
    arma::Mat<Tb> Vzero;
    if(m_V == nullptr) Vzero.zeros(m_nbsf*m_nbsf, m_nbsf*m_nbsf);
    arma::Mat<Tb> &V = (m_V == nullptr) ? Vzero : *m_V;

    // The primary corner of every ordered pair, followed by the other corner combinations
    m_pairinfo.assign(nref * nref * WICK_PAIRLEN, -1);
    m_scal.assign(nref * nref * WICK_NSCAL, Tc(0.0));
    m_slab.clear();
    size_t end = 0;
    for(size_t x=0; x < nref; x++)
    for(size_t w=0; w < nref; w++)
    {
        // Lowdin pairing of the two references
        const size_t k = x * nref + w;
        reference_state<Tc> xa(m_nbsf, m_nmo, m_nalpha, m_Ca[x]);
        reference_state<Tc> wa(m_nbsf, m_nmo, m_nalpha, m_Ca[w]);
        reference_state<Tc> xb(m_nbsf, m_nmo, m_nbeta, m_Cb[x]);
        reference_state<Tc> wb(m_nbsf, m_nmo, m_nbeta, m_Cb[w]);
        wick_pair<Tc,Tb> pair(m_metric, xa, wa, xb, wb);
        pair.add_one_body(m_Fa, m_Fb);
        pair.add_two_body(V);

        // Intermediates of each corner combination, with the offsets moved to the slab
        const size_t cstart = m_pairinfo.size() / WICK_PAIRLEN;
        for(size_t c=0; c < pair.ncorner(); c++)
        {
            const size_t row = (c == 0) ? k : cstart + c - 1;
            if(c > 0)
            {
                m_pairinfo.resize(m_pairinfo.size() + WICK_PAIRLEN);
                m_scal.resize(m_scal.size() + WICK_NSCAL);
            }
            int64_t *info = m_pairinfo.data() + row * WICK_PAIRLEN;
            std::copy(pair.info(c), pair.info(c) + WICK_PAIRLEN, info);
            for(int i=WICK_AA; i < WICK_OH_A; i++)
                if(info[i] >= 0) info[i] += end;
            std::copy(pair.scal(c), pair.scal(c) + WICK_NSCAL, m_scal.data() + row * WICK_NSCAL);
        }
        m_pairinfo[k * WICK_PAIRLEN + WICK_CSTART] = cstart;
        m_slab.insert(m_slab.end(), pair.slab().begin(), pair.slab().end());
        end = m_slab.size();
        if(x == w)
        {
            end = orthogonal_layout(m_pairinfo.data() + k * WICK_PAIRLEN, m_nmo, end);
            m_slab.resize(end, Tc(0.0));
            write_orthogonal(m_slab, m_pairinfo.data() + k * WICK_PAIRLEN, m_Fa, m_Fb, V,
                             m_Ca[x], m_Cb[x]);
        }
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
#if defined(LIBGNME_X86_SIMD)
    env.simd = simd_level();
#else
    env.simd = 0;
#endif
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
