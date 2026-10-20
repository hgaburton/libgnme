#ifndef LIBGNME_WICK_KERNELS_H
#define LIBGNME_WICK_KERNELS_H

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <complex>

namespace libgnme {

/** \brief Kernels for extended nonorthogonal Wick theorem matrix elements between excitations
           of many reference determinants, used by wick_matrix.

    The intermediates of every ordered reference pair (x,w) are stored in one slab, with the
    offsets of each pair in a row of pairinfo and its scalars in a row of scal. For a bra
    excitation of x with rank L_x and a ket excitation of w with rank L_w, the contraction
    determinant D has dimension L = L_x + L_w. Its row labels lie in V_x + O_w and its column
    labels in O_x + V_w,
        (r_k, c_k) = (a_k - nocc, i_k)          k < L_x
        (r_k, c_k) = (nvir + j_k, b_k)          L_x <= k < L
    with D_{ij} = X^{(m_j)}_{r_i c_j} for i >= j and Y^{(m_j)}_{r_i c_j} for i < j. When the
    pair has m zero-overlap orbital pairs, matrix elements are sums over the binary
    distributions sum_k m_k = m.

    References: H. G. A. Burton, J. Chem. Phys. 154, 144109 (2021); 157, 204109 (2022)
    \ingroup gnme_wick
 **/
namespace wick_kernels {

/* Largest L_a + L_b with a fixed-rank m_a = m_b = 0 kernel.  This covers
 * determinant pairs with up to quadruple excitations on each side. */
#define WICK_MAXL 8

/* Apply F(LA, LB) to every (L_a, L_b) with L_a + L_b <= WICK_MAXL */
#define WICK_LALB_CASES(F) \
    F(0,0) F(0,1) F(0,2) F(0,3) F(0,4) F(0,5) F(0,6) F(0,7) F(0,8) \
    F(1,0) F(1,1) F(1,2) F(1,3) F(1,4) F(1,5) F(1,6) F(1,7) \
    F(2,0) F(2,1) F(2,2) F(2,3) F(2,4) F(2,5) F(2,6) \
    F(3,0) F(3,1) F(3,2) F(3,3) F(3,4) F(3,5) \
    F(4,0) F(4,1) F(4,2) F(4,3) F(4,4) \
    F(5,0) F(5,1) F(5,2) F(5,3) \
    F(6,0) F(6,1) F(6,2) \
    F(7,0) F(7,1) \
    F(8,0)

/* pairinfo layout */
#define WICK_PAIRLEN 98
#define WICK_MA      0
#define WICK_MB      1
#define WICK_NOCCA   2
#define WICK_NOCCB   3
#define WICK_NMO     4
#define WICK_AA      5
#define WICK_BB      32
#define WICK_AB      59
/* orthogonal (same-parent) MO integrals, diagonal pairs (x,x) only */
#define WICK_OH_A 91 /* C_a^+ h C_a */
#define WICK_OH_B 92
#define WICK_OAA  93 /* (pq|rs) - (ps|rq), alpha */
#define WICK_OBB  94
#define WICK_OAB  95 /* (pq|rs), alpha pq, beta rs */
/* near-zero corners, primary rows only */
#define WICK_CSTART  96 /* row of the first other corner combination */
#define WICK_NCORNER 97 /* number of corner combinations */
/* offsets inside one same-spin block */
#define WICK_X     0  /* X^{(mi)} */
#define WICK_Y     2  /* Y^{(mi)} */
#define WICK_FH    4  /* F^{(mi,mj)}, [z,r] order, mi*2+mj */
#define WICK_V     8  /* V^{(m1,m3,m2)}, [z,r] order, m1*4+m3*2+m2 */
#define WICK_HCOL0 16 /* F+V+V^{ab} for m_a=m_b=0, [z,r] order */
#define WICK_J     17 /* ten symmetry-unique J, [i,j,r,c] order */
/* offsets inside the different-spin block */
#define WICK_VAB  0  /* V^a, [z,r] order, ma0*4+mb0*2+maz */
#define WICK_VBA  8  /* V^b, [y,r] order, mb0*4+ma0*2+mby */
#define WICK_IIAB 16 /* II, ma0*8+maz*4+mb0*2+mby */

/* scalar layout */
#define WICK_NSCAL 24
#define WICK_SAA   0
#define WICK_SBB   8
#define WICK_SVAB0 16 /* V_ab0^{(ma0,mb0)}, ma0*2+mb0 */
/* inside one same-spin scalar block */
#define WICK_SPHASE 0 /* orbital pairing phase */
#define WICK_STSP   1 /* product of nonzero singular values */
#define WICK_SF0H   2 /* F_0^{(m)} */
#define WICK_SV0    4 /* V_0 by m1+m2 */

/* determinant record: header followed by
 * holes_a[kmax] parts_a[kmax] holes_b[kmax] parts_b[kmax]
 * holes and particles are indices in the occupied-first orbital order of
 * the parent, sorted ascending.  sign converts the excitation operator
 * string applied to the parent into the determinant (Wick evaluators);
 * osign converts the canonical determinant in the occupied-first order
 * into the original MO order (Slater-Condon evaluator). */
#define WICK_DPARENT 0
#define WICK_DRA     1
#define WICK_DRB     2
#define WICK_DSIGN   3
#define WICK_DOSIGN  4
#define WICK_DHEAD   5

/* Occupation strings of a record in the parent's occupied-first order:
 * alpha words 0, 1 and beta words 2, 3, for up to WICK_MAXMO MOs */
#define WICK_NSTR  4
#define WICK_MAXMO 128
/* Largest number of contraction labels of one spin */
#define WICK_MAXREC (2 * WICK_MAXMO + 1)

static inline void wick_det_strings(uint64_t *str, const int *d, int kmax, int nocca, int noccb)
{
    const int nocc[2] = { nocca, noccb };
    const int *h, *p;
    uint64_t *w;
    int spin, i, r;
    for(spin = 0; spin < 2; spin++)
    {
        w = str + 2 * spin;
        w[0] = nocc[spin] >= 64 ? ~0ULL : (1ULL << nocc[spin]) - 1;
        w[1] = nocc[spin] <= 64 ? 0 : nocc[spin] >= 128 ? ~0ULL : (1ULL << (nocc[spin] - 64)) - 1;
        r = d[spin ? WICK_DRB : WICK_DRA];
        h = d + WICK_DHEAD + 2 * kmax * spin;
        p = h + kmax;
        for(i = 0; i < r; i++)
        {
            w[h[i] >> 6] ^= 1ULL << (h[i] & 63);
            w[p[i] >> 6] ^= 1ULL << (p[i] & 63);
        }
    }
}

struct wick_env
{
    int nref;
    int kmax;    /* largest per-spin excitation rank of a record */
    int dstride; /* WICK_DHEAD + 4*kmax */
    int simd;    /* 0: scalar only; 1: allow SIMD packets */
    double enuc;
    const int64_t *pairinfo;
    const void *slab;
    const void *scal;
};

#define WICK_SIMD_NONE   0
#define WICK_SIMD_AVX2   1
#define WICK_SIMD_AVX512 2

#define ALWAYS_INLINE static inline __attribute__((always_inline))

/*
 * Faddeev-LeVerrier recurrence for det(A) and, optionally, the cofactor
 * matrix cof[r*n+c] = adj(A)[c*n+r].  It is a polynomial in the entries of
 * A, so the cofactors are exact for singular A.  Inlined with a constant n,
 * the loops are fully unrolled.
 */
template<typename DT>
ALWAYS_INLINE DT fl_det_cof(DT *cof, const DT *a, const int n, const int want_cof)
{
    DT b[WICK_MAXL * WICK_MAXL];
    DT p[WICK_MAXL * WICK_MAXL];
    DT c, v;
    int i, j, k, l;
    if(n == 0)
    {
        return 1;
    }
    if(n == 1)
    {
        if(want_cof)
        {
            cof[0] = 1;
        }
        return a[0];
    }
    if(n == 2)
    {
        if(want_cof)
        {
            cof[0] = a[3];
            cof[1] = -a[2];
            cof[2] = -a[1];
            cof[3] = a[0];
        }
        return a[0] * a[3] - a[1] * a[2];
    }
    if(n == 3 && !want_cof)
    {
        return a[0] * (a[4] * a[8] - a[5] * a[7]) - a[1] * (a[3] * a[8] - a[5] * a[6]) +
               a[2] * (a[3] * a[7] - a[4] * a[6]);
    }
    /* B_1 = A + c_1 I with c_1 = -tr(A) */
    c = 0;
    for(i = 0; i < n; i++)
    {
        c -= a[i * n + i];
    }
    for(i = 0; i < n * n; i++)
    {
        b[i] = a[i];
    }
    for(i = 0; i < n; i++)
    {
        b[i * n + i] += c;
    }
    for(k = 2; k <= n; k++)
    {
        if(want_cof && k == n)
        {
            /* adj(A) = (-1)^{n-1} B_{n-1} */
            for(i = 0; i < n; i++)
            {
                for(j = 0; j < n; j++)
                {
                    cof[i * n + j] = (n & 1) ? b[j * n + i] : -b[j * n + i];
                }
            }
        }
        if(k == n)
        {
            /* only the trace of A B_{n-1} is needed */
            c = 0;
            for(i = 0; i < n; i++)
            {
                for(l = 0; l < n; l++)
                {
                    c += a[i * n + l] * b[l * n + i];
                }
            }
            c *= -1. / n;
            return (n & 1) ? -c : c;
        }
        for(i = 0; i < n; i++)
        {
            for(j = 0; j < n; j++)
            {
                v = 0;
                for(l = 0; l < n; l++)
                {
                    v += a[i * n + l] * b[l * n + j];
                }
                p[i * n + j] = v;
            }
        }
        c = 0;
        for(i = 0; i < n; i++)
        {
            c += p[i * n + i];
        }
        c *= -1. / k;
        for(i = 0; i < n * n; i++)
        {
            b[i] = p[i];
        }
        for(i = 0; i < n; i++)
        {
            b[i * n + i] += c;
        }
    }
    return 0;
}

/* Partial-pivot LU determinant, used above WICK_MAXL */
template<typename DT>
static DT det_lu(const DT *a, int n)
{
    DT *lu = (DT *) malloc(sizeof(DT) * n * n);
    DT det = 1;
    DT f, t;
    int i, j, k, piv;
    double amax, aval;
    for(i = 0; i < n * n; i++)
    {
        lu[i] = a[i];
    }
    for(k = 0; k < n; k++)
    {
        piv = k;
        amax = std::abs(lu[k * n + k]);
        for(i = k + 1; i < n; i++)
        {
            aval = std::abs(lu[i * n + k]);
            if(aval > amax)
            {
                amax = aval;
                piv = i;
            }
        }
        if(amax == 0)
        {
            free(lu);
            return 0;
        }
        if(piv != k)
        {
            for(j = 0; j < n; j++)
            {
                t = lu[k * n + j];
                lu[k * n + j] = lu[piv * n + j];
                lu[piv * n + j] = t;
            }
            det = -det;
        }
        det *= lu[k * n + k];
        for(i = k + 1; i < n; i++)
        {
            f = lu[i * n + k] / lu[k * n + k];
            for(j = k + 1; j < n; j++)
            {
                lu[i * n + j] -= f * lu[k * n + j];
            }
        }
    }
    free(lu);
    return det;
}

template<typename DT>
static void cof_minor(DT *out, const DT *a, int n, int r0, int c0)
{
    int i, j, k = 0;
    for(i = 0; i < n; i++)
    {
        if(i == r0) continue;
        for(j = 0; j < n; j++)
        {
            if(j == c0) continue;
            out[k++] = a[i * n + j];
        }
    }
}

#define FL_CASES(F) F(0) F(1) F(2) F(3) F(4) F(5) F(6) F(7) F(8)

template<typename DT>
static DT det(const DT *a, int n)
{
    switch(n)
    {
#define CASE(N) case N: return fl_det_cof<DT>(NULL, a, N, 0);
        FL_CASES(CASE)
#undef CASE
    }
    return det_lu<DT>(a, n);
}

/* Determinant and cofactor matrix for arbitrary n */
template<typename DT>
static DT det_cof(DT *cof, const DT *a, int n)
{
    DT *mnr;
    int i, j;
    switch(n)
    {
#define CASE(N) case N: return fl_det_cof<DT>(cof, a, N, 1);
        FL_CASES(CASE)
#undef CASE
    }
    /* explicit minors keep the cofactors exact for singular a */
    mnr = (DT *) malloc(sizeof(DT) * (n - 1) * (n - 1));
    for(i = 0; i < n; i++)
    {
        for(j = 0; j < n; j++)
        {
            cof_minor<DT>(mnr, a, n, i, j);
            cof[i * n + j] = ((i + j) & 1) ? -det_lu<DT>(mnr, n - 1) : det_lu<DT>(mnr, n - 1);
        }
    }
    free(mnr);
    return det_lu<DT>(a, n);
}

/* Determinant of a with rows r0 < r1 and columns c0 < c1 removed */
template<typename DT>
ALWAYS_INLINE DT second_minor_det(DT *buf, const DT *a, const int n, int r0, int r1, int c0, int c1)
{
    int i, j, k = 0;
    for(i = 0; i < n; i++)
    {
        if(i == r0 || i == r1) continue;
        for(j = 0; j < n; j++)
        {
            if(j == c0 || j == c1) continue;
            buf[k++] = a[i * n + j];
        }
    }
    if(n - 2 <= WICK_MAXL)
    {
        switch(n - 2)
        {
#define CASE(N) case N: return fl_det_cof<DT>(NULL, buf, N, 0);
            FL_CASES(CASE)
#undef CASE
        }
    }
    return det_lu<DT>(buf, n - 2);
}

/* Map the excitations of one spin to contraction-determinant labels */
template<typename DT>
ALWAYS_INLINE void labels(int *rows, int *cols, const int *hx, const int *px, int rx, const int *hw,
                          const int *pw, int rw, int nocc, int nmo)
{
    int i;
    int nvir = nmo - nocc;
    for(i = 0; i < rx; i++)
    {
        rows[i] = px[i] - nocc;
        cols[i] = hx[i];
    }
    for(i = 0; i < rw; i++)
    {
        rows[rx + i] = nvir + hw[i];
        cols[rx + i] = pw[i];
    }
}

template<typename DT>
ALWAYS_INLINE void build_d(DT *d, const DT *x, const DT *y, const int *rows, const int *cols,
                           const int l, int n)
{
    int i, j;
    for(i = 0; i < l; i++)
    {
        const DT *xr = x + (size_t) rows[i] * n;
        const DT *yr = y + (size_t) rows[i] * n;
        for(j = 0; j <= i; j++)
        {
            d[i * l + j] = xr[cols[j]];
        }
        for(j = i + 1; j < l; j++)
        {
            d[i * l + j] = yr[cols[j]];
        }
    }
}

/* Select column k of mix from d1 when bit k of bits is set, else from d0 */
template<typename DT>
static void mix_columns(DT *mix, const DT *d0, const DT *d1, int l, uint64_t bits)
{
    int i, j;
    for(i = 0; i < l; i++)
    {
        for(j = 0; j < l; j++)
        {
            mix[i * l + j] = ((bits >> j) & 1) ? d1[i * l + j] : d0[i * l + j];
        }
    }
}

/* Next integer with the same popcount (Gosper) */
static inline uint64_t next_combination(uint64_t x)
{
    uint64_t c = x & -x;
    uint64_t r = x + c;
    return (((r ^ x) >> 2) / c) | r;
}

/* number of binary distributions of m over l contractions */
static inline int64_t ncombinations(int l, int m)
{
    int64_t c = 1;
    int i;
    if(m < 0 || m > l) return 0;
    for(i = 0; i < m; i++)
    {
        c = c * (l - i) / (i + 1);
    }
    return c;
}

template<typename DT>
struct spin_t
{
    const DT *slab;
    const int64_t *off; /* same-spin offset block */
    const DT *scal;     /* same-spin scalar block */
    int m;
    int nocc;
    int nmo;
};

template<typename DT>
static void spin_view(spin_t<DT> *sp, const wick_env *env, const int64_t *info, const DT *scal,
                      int alpha)
{
    sp->slab = (const DT *) env->slab;
    sp->off = info + (alpha ? WICK_AA : WICK_BB);
    sp->scal = scal + (alpha ? WICK_SAA : WICK_SBB);
    sp->m = info[alpha ? WICK_MA : WICK_MB];
    sp->nocc = info[alpha ? WICK_NOCCA : WICK_NOCCB];
    sp->nmo = info[WICK_NMO];
}

#define SLAB(sp, k) ((sp)->slab + (sp)->off[k])

/*
 * Same-spin overlap sum_{m_1..m_L} det D(m_1..m_L), without the reduced
 * overlap prefactor.
 */
template<typename DT>
static DT spin_overlap(const spin_t<DT> *sp, const int *rows, const int *cols, int l)
{
    const int n = sp->nmo;
    DT dbuf[WICK_MAXL * WICK_MAXL];
    DT *d0, *d1, *mix;
    DT acc;
    uint64_t bits, last;
    if(sp->m > l)
    {
        return 0;
    }
    if(l == 0)
    {
        return 1;
    }
    if(sp->m == 0 && l <= WICK_MAXL)
    {
        build_d<DT>(dbuf, SLAB(sp, WICK_X), SLAB(sp, WICK_Y), rows, cols, l, n);
        return det<DT>(dbuf, l);
    }
    d0 = (DT *) malloc(sizeof(DT) * l * l * 3);
    d1 = d0 + l * l;
    mix = d1 + l * l;
    build_d<DT>(d0, SLAB(sp, WICK_X), SLAB(sp, WICK_Y), rows, cols, l, n);
    build_d<DT>(d1, SLAB(sp, WICK_X + 1), SLAB(sp, WICK_Y + 1), rows, cols, l, n);
    acc = 0;
    last = (l == 64) ? ~0ULL : (1ULL << l);
    for(bits = (1ULL << sp->m) - 1; bits < last; bits = next_combination(bits))
    {
        mix_columns<DT>(mix, d0, d1, l, bits);
        acc += det<DT>(mix, l);
        if(sp->m == 0) break;
    }
    free(d0);
    return acc;
}

template<typename DT>
static inline DT spin_pref(const spin_t<DT> *sp)
{
    return sp->scal[WICK_SPHASE] * sp->scal[WICK_STSP];
}

#define PAIRINFO(env, k) ((env)->pairinfo + (size_t) (k) * WICK_PAIRLEN)
#define PAIRSCAL(env, k) ((const DT *) (env)->scal + (size_t) (k) * WICK_NSCAL)

/* Row of corner combination c of the pair whose primary row is info */
static inline size_t corner_row(const int64_t *info, size_t k, int64_t c)
{
    return c == 0 ? k : (size_t) (info[WICK_CSTART] + c - 1);
}

/* Overlap between two determinant records at one corner combination */
template<typename DT>
static DT overlap_corner(const wick_env *env, const int64_t *info, const DT *scal, const int *dx,
                         const int *dw)
{
    const int kmax = env->kmax;
    int rxa = dx[WICK_DRA], rxb = dx[WICK_DRB];
    int rwa = dw[WICK_DRA], rwb = dw[WICK_DRB];
    int la = rxa + rwa;
    int lb = rxb + rwb;
    int rows[WICK_MAXREC], cols[WICK_MAXREC];
    spin_t<DT> sa, sb;
    DT ova, ovb;
    if(info[WICK_MA] > la || info[WICK_MB] > lb)
    {
        return 0;
    }
    spin_view<DT>(&sa, env, info, scal, 1);
    spin_view<DT>(&sb, env, info, scal, 0);
    labels<DT>(rows, cols, dx + WICK_DHEAD, dx + WICK_DHEAD + kmax, rxa, dw + WICK_DHEAD,
               dw + WICK_DHEAD + kmax, rwa, sa.nocc, sa.nmo);
    ova = spin_overlap<DT>(&sa, rows, cols, la);
    if(ova == 0.0)
    {
        return 0;
    }
    labels<DT>(rows, cols, dx + WICK_DHEAD + 2 * kmax, dx + WICK_DHEAD + 3 * kmax, rxb,
               dw + WICK_DHEAD + 2 * kmax, dw + WICK_DHEAD + 3 * kmax, rwb, sb.nocc, sb.nmo);
    ovb = spin_overlap<DT>(&sb, rows, cols, lb);
    return (double) (dx[WICK_DSIGN] * dw[WICK_DSIGN]) * spin_pref<DT>(&sa) * spin_pref<DT>(&sb) *
           ova * ovb;
}

/* Overlap between two determinant records */
template<typename DT>
DT overlap(const wick_env *env, const int *dx, const int *dw)
{
    const size_t k = (size_t) dx[WICK_DPARENT] * env->nref + dw[WICK_DPARENT];
    const int64_t *info = PAIRINFO(env, k);
    DT out = 0;
    int64_t c;
    size_t kc;
    for(c = 0; c < info[WICK_NCORNER]; c++)
    {
        kc = corner_row(info, k, c);
        out += overlap_corner<DT>(env, PAIRINFO(env, kc), PAIRSCAL(env, kc), dx, dw);
    }
    return out;
}

/*
 * Map the four contraction assignments of J_{eta z, xi y} to one of the ten
 * stored symmetry-unique tensors.  swap marks the pair-exchanged access
 * J_{xi y, eta z}.
 */
static const int8_t JSLOT[16] = {
    0, 1, 2, 3, /* (0,0,0,0) (0,0,0,1) (0,0,1,0) (0,0,1,1) */
    1, 4, 5, 6, /* (0,1,0,0) (0,1,0,1) (0,1,1,0) (0,1,1,1) */
    2, 5, 7, 8, /* (1,0,0,0) (1,0,0,1) (1,0,1,0) (1,0,1,1) */
    3, 6, 8, 9, /* (1,1,0,0) (1,1,0,1) (1,1,1,0) (1,1,1,1) */
};
static const int8_t JSWAP[16] = {
    0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0,
};

/*
 * J entry for the fixed pair (a,b) and varying pair (c,d) of determinant
 * positions.  bits holds (m1, m2, m_col0, m_col1, ...).
 */
template<typename DT>
ALWAYS_INLINE DT jget(const spin_t<DT> *sp, const int *rows, const int *cols, uint64_t bits, int a,
                      int b, int c, int d)
{
    const size_t n = sp->nmo;
    int mi = bits & 1;
    int mj = (bits >> 1) & 1;
    int mk = (bits >> (d + 2)) & 1;
    int ml = (bits >> (b + 2)) & 1;
    int key = mi * 8 + mj * 4 + mk * 2 + ml;
    const DT *t = sp->slab + sp->off[WICK_J + JSLOT[key]];
    size_t ri = rows[a], cj = cols[b], rr = rows[c], cc = cols[d];
    if(JSWAP[key])
    {
        return t[((rr * n + cc) * n + ri) * n + cj];
    }
    else
    {
        return t[((ri * n + cj) * n + rr) * n + cc];
    }
}

/*
 * Same-spin two-column J contribution for one distribution:
 *   sum_{eta<xi, z<y} (-1)^{eta+xi+z+y} det D[eta,xi|z,y]
 *     * (J(eta z, xi y) + J(xi y, eta z) - J(eta y, xi z) - J(xi z, eta y))
 * The m = 0 J tensor is pair symmetric, so the bracket is twice the
 * direct-minus-exchange difference.
 */
template<typename DT>
static DT jterm(const spin_t<DT> *sp, const DT *d, const int *rows, const int *cols, int l,
                uint64_t bits)
{
    DT *buf = (DT *) malloc(sizeof(DT) * (l > 2 ? (l - 2) * (l - 2) : 1));
    DT acc = 0;
    DT m2, val;
    int eta, xi, z, y;
    for(eta = 0; eta < l; eta++)
    {
        for(xi = eta + 1; xi < l; xi++)
        {
            for(z = 0; z < l; z++)
            {
                for(y = z + 1; y < l; y++)
                {
                    m2 = second_minor_det<DT>(buf, d, l, eta, xi, z, y);
                    if(m2 == 0.0) continue;
                    val = jget<DT>(sp, rows, cols, bits, eta, z, xi, y) +
                          jget<DT>(sp, rows, cols, bits, xi, y, eta, z) -
                          jget<DT>(sp, rows, cols, bits, eta, y, xi, z) -
                          jget<DT>(sp, rows, cols, bits, xi, z, eta, y);
                    if((eta + xi + z + y) & 1)
                    {
                        acc -= m2 * val;
                    }
                    else
                    {
                        acc += m2 * val;
                    }
                }
            }
        }
    }
    free(buf);
    return acc;
}

/* sum_{k,r} col_t[cols[k]*n + rows[r]] cof[r*l+k] for the column matrix col_t
 * chosen by the assignment of column k */
template<typename DT>
ALWAYS_INLINE DT column_sum(const DT *t0, const DT *t1, const DT *cof, const int *rows,
                            const int *cols, const int l, size_t n, uint64_t colbits)
{
    DT acc = 0;
    int k, r;
    for(k = 0; k < l; k++)
    {
        const DT *t = (((colbits >> k) & 1) ? t1 : t0) + cols[k] * n;
        for(r = 0; r < l; r++)
        {
            acc += t[rows[r]] * cof[r * l + k];
        }
    }
    return acc;
}

template<typename DT>
struct spinterms_t
{
    DT s;  /* sum of overlap distributions */
    DT h1; /* one-electron */
    DT h2; /* same-spin two-electron, without the 1/2 */
};

/*
 * General same-spin pass.  Enumerates sum_{k=1}^{L+2} m_k = m with m_1 and
 * m_2 the operator contractions.  m_2 = 0 selects the one-body terms and
 * m_1 = m_2 = 0 the overlap terms.
 */
template<typename DT>
static void spin_terms(spinterms_t<DT> *out, const spin_t<DT> *sp, const int *rows, const int *cols,
                       int l)
{
    const size_t n = sp->nmo;
    const int m = sp->m;
    DT *d0, *d1, *mix, *cof;
    DT det, h;
    uint64_t bits, last, cb;
    int m1, m2;
    out->s = 0;
    out->h1 = 0;
    out->h2 = 0;
    if(m > l + 2)
    {
        return;
    }
    d0 = (DT *) malloc(sizeof(DT) * (l * l * 4 + 1));
    d1 = d0 + l * l;
    mix = d1 + l * l;
    cof = mix + l * l;
    build_d<DT>(d0, SLAB(sp, WICK_X), SLAB(sp, WICK_Y), rows, cols, l, n);
    build_d<DT>(d1, SLAB(sp, WICK_X + 1), SLAB(sp, WICK_Y + 1), rows, cols, l, n);

    last = 1ULL << (l + 2);
    for(bits = (1ULL << m) - 1; bits < last; bits = (m ? next_combination(bits) : last))
    {
        m1 = bits & 1;
        m2 = (bits >> 1) & 1;
        cb = bits >> 2;
        mix_columns<DT>(mix, d0, d1, l, cb);
        det = det_cof<DT>(cof, mix, l);
        if(m1 == 0 && m2 == 0)
        {
            out->s += det;
        }
        if(m2 == 0)
        {
            h = sp->scal[WICK_SF0H + m1] * det;
            h -= column_sum<DT>(SLAB(sp, WICK_FH + m1 * 2), SLAB(sp, WICK_FH + m1 * 2 + 1), cof,
                                rows, cols, l, n, cb);
            out->h1 += h;
        }
        h = sp->scal[WICK_SV0 + m1 + m2] * det;
        h -=
            2.0 * column_sum<DT>(SLAB(sp, WICK_V + m1 * 4 + m2 * 2),
                                 SLAB(sp, WICK_V + m1 * 4 + m2 * 2 + 1), cof, rows, cols, l, n, cb);
        if(l >= 2)
        {
            h += jterm<DT>(sp, mix, rows, cols, l, bits);
        }
        out->h2 += h;
    }
    free(d0);
}

/*
 * Different-spin pass.  The alpha and beta distributions
 *   m_{a0} + sum_z m_{az} = m_a,      m_{b0} + sum_y m_{by} = m_b
 * are enumerated independently; bit 0 holds the operator contraction.
 */
template<typename DT>
static DT diff_terms(const wick_env *env, const int64_t *info, const DT *scal, const spin_t<DT> *sa,
                     const int *rows_a, const int *cols_a, int la, const spin_t<DT> *sb,
                     const int *rows_b, const int *cols_b, int lb)
{
    const DT *slab = (const DT *) env->slab;
    const int64_t *off = info + WICK_AB;
    const size_t n = sa->nmo;
    const int ma = sa->m;
    const int mb = sb->m;
    int64_t nca = ncombinations(la + 1, ma);
    int64_t ncb = ncombinations(lb + 1, mb);
    int64_t ia, ib;
    int sza = la * la + 1;
    int szb = lb * lb + 1;
    DT *buf, *deta, *detb, *cofa, *cofb, *mix, *d0, *d1;
    uint64_t *bitsa, *bitsb, bits, last;
    DT acc = 0;
    DT contrib, da, db, inner;
    int ma0, mb0, i, j, k, r;
    size_t ra, ca;

    if(nca == 0 || ncb == 0)
    {
        return 0;
    }
    buf = (DT *) malloc(sizeof(DT) * ((nca + ncb) * 2 + nca * sza + ncb * szb + 3 * (sza + szb)));
    bitsa = (uint64_t *) malloc(sizeof(uint64_t) * (nca + ncb));
    bitsb = bitsa + nca;
    deta = buf;
    detb = deta + nca;
    cofa = detb + ncb;
    cofb = cofa + nca * sza;
    mix = cofb + ncb * szb;
    d0 = mix + sza + szb;
    d1 = d0 + sza + szb;

    build_d<DT>(d0, SLAB(sa, WICK_X), SLAB(sa, WICK_Y), rows_a, cols_a, la, n);
    build_d<DT>(d1, SLAB(sa, WICK_X + 1), SLAB(sa, WICK_Y + 1), rows_a, cols_a, la, n);
    last = 1ULL << (la + 1);
    for(ia = 0, bits = (1ULL << ma) - 1; ia < nca;
        ia++, bits = (ia < nca) ? next_combination(bits) : 0)
    {
        bitsa[ia] = bits;
        mix_columns<DT>(mix, d0, d1, la, bits >> 1);
        deta[ia] = det_cof<DT>(cofa + ia * sza, mix, la);
    }
    build_d<DT>(d0, SLAB(sb, WICK_X), SLAB(sb, WICK_Y), rows_b, cols_b, lb, n);
    build_d<DT>(d1, SLAB(sb, WICK_X + 1), SLAB(sb, WICK_Y + 1), rows_b, cols_b, lb, n);
    last = 1ULL << (lb + 1);
    for(ib = 0, bits = (1ULL << mb) - 1; ib < ncb;
        ib++, bits = (ib < ncb) ? next_combination(bits) : 0)
    {
        bitsb[ib] = bits;
        mix_columns<DT>(mix, d0, d1, lb, bits >> 1);
        detb[ib] = det_cof<DT>(cofb + ib * szb, mix, lb);
    }
    (void) last;

    for(ia = 0; ia < nca; ia++)
    {
        for(ib = 0; ib < ncb; ib++)
        {
            const DT *ca_ = cofa + ia * sza;
            const DT *cb_ = cofb + ib * szb;
            uint64_t ba = bitsa[ia];
            uint64_t bb = bitsb[ib];
            ma0 = ba & 1;
            mb0 = bb & 1;
            da = deta[ia];
            db = detb[ib];
            contrib = scal[WICK_SVAB0 + ma0 * 2 + mb0] * da * db;
            contrib -= db * column_sum<DT>(slab + off[WICK_VAB + ma0 * 4 + mb0 * 2],
                                           slab + off[WICK_VAB + ma0 * 4 + mb0 * 2 + 1], ca_,
                                           rows_a, cols_a, la, n, ba >> 1);
            contrib -= da * column_sum<DT>(slab + off[WICK_VBA + mb0 * 4 + ma0 * 2],
                                           slab + off[WICK_VBA + mb0 * 4 + ma0 * 2 + 1], cb_,
                                           rows_b, cols_b, lb, n, bb >> 1);
            for(i = 0; i < la; i++)
            {
                for(j = 0; j < la; j++)
                {
                    int maj = (ba >> (j + 1)) & 1;
                    ra = rows_a[i];
                    ca = cols_a[j];
                    inner = 0;
                    for(k = 0; k < lb; k++)
                    {
                        int mbk = (bb >> (k + 1)) & 1;
                        const DT *t = slab + off[WICK_IIAB + ma0 * 8 + maj * 4 + mb0 * 2 + mbk] +
                                      ((ra * n + ca) * n) * n + cols_b[k];
                        for(r = 0; r < lb; r++)
                        {
                            inner += t[rows_b[r] * n] * cb_[r * lb + k];
                        }
                    }
                    acc += ca_[i * la + j] * inner;
                }
            }
            acc += contrib;
        }
    }
    free(bitsa);
    free(buf);
    return acc;
}

/*
 * Fixed-rank m_a = m_b = 0 Hamiltonian and overlap (no prefactors).  The
 * one-column classes use the precombined hcol0 = F + V + V^{ab}.
 */
template<typename DT>
ALWAYS_INLINE void hs_m0_fixed(DT *hout, DT *sout, const wick_env *env, const int64_t *info,
                               const DT *scal, const spin_t<DT> *sa, const int *rows_a,
                               const int *cols_a, const int la, const spin_t<DT> *sb,
                               const int *rows_b, const int *cols_b, const int lb)
{
    const size_t n = sa->nmo;
    const size_t n2 = n * n;
    const size_t n3 = n2 * n;
    DT da[WICK_MAXL * WICK_MAXL + 1], cofa[WICK_MAXL * WICK_MAXL + 1];
    DT db[WICK_MAXL * WICK_MAXL + 1], cofb[WICK_MAXL * WICK_MAXL + 1];
    DT mnr[WICK_MAXL * WICK_MAXL + 1];
    DT deta, detb, ja, jb, repa, repb, ii, inner, m2, g0, core;
    const DT *jsl, *hcol, *iisl;
    int eta, xi, z, y;

    build_d<DT>(da, SLAB(sa, WICK_X), SLAB(sa, WICK_Y), rows_a, cols_a, la, n);
    build_d<DT>(db, SLAB(sb, WICK_X), SLAB(sb, WICK_Y), rows_b, cols_b, lb, n);
    deta = fl_det_cof<DT>(cofa, da, la, 1);
    detb = fl_det_cof<DT>(cofb, db, lb, 1);

    ja = 0;
    jsl = SLAB(sa, WICK_J);
    for(eta = 0; eta < la; eta++)
    {
        for(xi = eta + 1; xi < la; xi++)
        {
            for(z = 0; z < la; z++)
            {
                for(y = z + 1; y < la; y++)
                {
                    m2 = second_minor_det<DT>(mnr, da, la, eta, xi, z, y);
                    m2 *= jsl[rows_a[eta] * n3 + cols_a[z] * n2 + rows_a[xi] * n + cols_a[y]] -
                          jsl[rows_a[eta] * n3 + cols_a[y] * n2 + rows_a[xi] * n + cols_a[z]];
                    if((eta + xi + z + y) & 1)
                    {
                        ja -= m2;
                    }
                    else
                    {
                        ja += m2;
                    }
                }
            }
        }
    }
    jb = 0;
    jsl = SLAB(sb, WICK_J);
    for(eta = 0; eta < lb; eta++)
    {
        for(xi = eta + 1; xi < lb; xi++)
        {
            for(z = 0; z < lb; z++)
            {
                for(y = z + 1; y < lb; y++)
                {
                    m2 = second_minor_det<DT>(mnr, db, lb, eta, xi, z, y);
                    m2 *= jsl[rows_b[eta] * n3 + cols_b[z] * n2 + rows_b[xi] * n + cols_b[y]] -
                          jsl[rows_b[eta] * n3 + cols_b[y] * n2 + rows_b[xi] * n + cols_b[z]];
                    if((eta + xi + z + y) & 1)
                    {
                        jb -= m2;
                    }
                    else
                    {
                        jb += m2;
                    }
                }
            }
        }
    }

    repa = 0;
    hcol = SLAB(sa, WICK_HCOL0);
    for(z = 0; z < la; z++)
    {
        const DT *t = hcol + cols_a[z] * n;
        for(eta = 0; eta < la; eta++)
        {
            repa += cofa[eta * la + z] * t[rows_a[eta]];
        }
    }
    repb = 0;
    hcol = SLAB(sb, WICK_HCOL0);
    for(z = 0; z < lb; z++)
    {
        const DT *t = hcol + cols_b[z] * n;
        for(eta = 0; eta < lb; eta++)
        {
            repb += cofb[eta * lb + z] * t[rows_b[eta]];
        }
    }

    ii = 0;
    iisl = (const DT *) env->slab + info[WICK_AB + WICK_IIAB];
    for(z = 0; z < la; z++)
    {
        for(eta = 0; eta < la; eta++)
        {
            const DT *t = iisl + rows_a[eta] * n3 + cols_a[z] * n2;
            inner = 0;
            for(y = 0; y < lb; y++)
            {
                for(xi = 0; xi < lb; xi++)
                {
                    inner += cofb[xi * lb + y] * t[rows_b[xi] * n + cols_b[y]];
                }
            }
            ii += cofa[eta * la + z] * inner;
        }
    }

    g0 = env->enuc + sa->scal[WICK_SF0H] + .5 * sa->scal[WICK_SV0] + sb->scal[WICK_SF0H] +
         .5 * sb->scal[WICK_SV0] + scal[WICK_SVAB0];
    core = g0 * deta * detb - detb * repa - deta * repb + ja * detb + jb * deta + ii;
    *hout = core;
    *sout = deta * detb;
}

/* General Hamiltonian and overlap (no prefactors) for any m_a, m_b, L */
template<typename DT>
static void hs_general(DT *hout, DT *sout, const wick_env *env, const int64_t *info, const DT *scal,
                       const spin_t<DT> *sa, const int *rows_a, const int *cols_a, int la,
                       const spin_t<DT> *sb, const int *rows_b, const int *cols_b, int lb)
{
    spinterms_t<DT> ta, tb;
    DT hab;
    spin_terms<DT>(&ta, sa, rows_a, cols_a, la);
    spin_terms<DT>(&tb, sb, rows_b, cols_b, lb);
    hab = diff_terms<DT>(env, info, scal, sa, rows_a, cols_a, la, sb, rows_b, cols_b, lb);
    *sout = ta.s * tb.s;
    *hout = ta.h1 * tb.s + tb.h1 * ta.s + .5 * (ta.h2 * tb.s + tb.h2 * ta.s) + hab +
            env->enuc * ta.s * tb.s;
}

#define HS_M0_CASE(LA, LB) \
    case LA * (WICK_MAXL + 1) + LB: \
        hs_m0_fixed<DT>(&h, &s, env, info, scal, &sa, rows_a, cols_a, LA, \
                        &sb, rows_b, cols_b, LB); \
        break;

/*
 * Same-parent Slater-Condon rules on occupation strings, adapted from
 * FCIpspace_h0tril_uhf of pyscf/lib/mcscf/fci_contract.c to complex orbitals,
 * two-word strings of up to WICK_MAXMO MOs and diagonal elements.  The
 * integrals are in the parent's orthonormal MO basis, with the same-spin
 * integrals antisymmetrised.
 */
/* popcount without the POPCNT instruction, as FCIpopcount_1 */
static inline int popcount64(uint64_t x)
{
    x = x - ((x >> 1) & 0x5555555555555555ULL);
    x = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x = (x + (x >> 4)) & 0x0f0f0f0f0f0f0f0fULL;
    return (int) ((x * 0x0101010101010101ULL) >> 56);
}

static inline int nbits2(uint64_t w0, uint64_t w1)
{
    return popcount64(w0) + popcount64(w1);
}

static inline int first1_2(uint64_t w0, uint64_t w1)
{
    return w0 ? __builtin_ctzll(w0) : 64 + __builtin_ctzll(w1);
}

/* lowest set bit other than bit k */
static inline int second1_2(uint64_t w0, uint64_t w1, int k)
{
    uint64_t w[2] = { w0, w1 };
    w[k >> 6] &= ~(1ULL << (k & 63));
    return first1_2(w[0], w[1]);
}

/* number of occupied orbitals of string w below orbital k */
static inline int below2(const uint64_t *w, int k)
{
    if(k <= 64)
    {
        return popcount64(k == 64 ? w[0] : w[0] & ((1ULL << k) - 1));
    }
    return popcount64(w[0]) + popcount64(k == 128 ? w[1] : w[1] & ((1ULL << (k - 64)) - 1));
}

/* sign of a^+_p a_q acting on string w, as FCIcre_des_sign */
static inline int cre_des_sign2(int p, int q, const uint64_t *w)
{
    int lo = p < q ? p : q;
    int hi = p < q ? q : p;
    return ((below2(w, hi) - below2(w, lo + 1)) & 1) ? -1 : 1;
}

/* occupied orbitals of string w, ascending */
static inline int occ_list2(int *occ, const uint64_t *w)
{
    uint64_t x;
    int k, n = 0;
    for(k = 0; k < 2; k++)
    {
        for(x = w[k]; x; x &= x - 1)
        {
            occ[n++] = 64 * k + __builtin_ctzll(x);
        }
    }
    return n;
}

/*
 * <x|H|w> and <x|w> between the strings sx and sw of one parent.  sign
 * converts the strings in the occupied-first order to the determinants.
 */
template<typename DT>
ALWAYS_INLINE void hs_orth(DT *hout, DT *sout, const wick_env *env, const int64_t *info,
                           const uint64_t *sx, const uint64_t *sw, double sign)
{
    const DT *slab = (const DT *) env->slab;
    const size_t n = info[WICK_NMO];
    const size_t n2 = n * n;
    const size_t n3 = n2 * n;
    const DT *ha = slab + info[WICK_OH_A];
    const DT *hb = slab + info[WICK_OH_B];
    const DT *aa = slab + info[WICK_OAA];
    const DT *bb = slab + info[WICK_OBB];
    const DT *ab = slab + info[WICK_OAB];
    const uint64_t da[2] = { sx[0] ^ sw[0], sx[1] ^ sw[1] };
    const uint64_t db[2] = { sx[2] ^ sw[2], sx[3] ^ sw[3] };
    const int n1da = nbits2(da[0], da[1]);
    const int n1db = nbits2(db[0], db[1]);
    int oa[WICK_MAXMO], ob[WICK_MAXMO];
    int na, nb, i, j;
    size_t p, q, r, s;
    uint64_t str1[2];
    DT h = 0;

    *hout = 0;
    *sout = 0;
    if(n1da + n1db > 4)
    {
        return;
    }

    if(n1da == 0 && n1db == 0)
    {
        na = occ_list2(oa, sw);
        nb = occ_list2(ob, sw + 2);
        h = env->enuc;
        for(i = 0; i < na; i++)
        {
            p = oa[i];
            h += ha[p * n + p];
            for(j = 0; j < na; j++)
            {
                q = oa[j];
                h += .5 * aa[p * n3 + p * n2 + q * n + q];
            }
            for(j = 0; j < nb; j++)
            {
                q = ob[j];
                h += ab[p * n3 + p * n2 + q * n + q];
            }
        }
        for(i = 0; i < nb; i++)
        {
            p = ob[i];
            h += hb[p * n + p];
            for(j = 0; j < nb; j++)
            {
                q = ob[j];
                h += .5 * bb[p * n3 + p * n2 + q * n + q];
            }
        }
        *sout = sign;
    }
    else if(n1da == 2 && n1db == 0)
    {
        /* (p <- q): h_pq + sum_{r in w_a} <pr||qr> + sum_{r in w_b} (pq|rr) */
        p = first1_2(da[0] & sx[0], da[1] & sx[1]);
        q = first1_2(da[0] & sw[0], da[1] & sw[1]);
        na = occ_list2(oa, sw);
        nb = occ_list2(ob, sw + 2);
        h = ha[p * n + q];
        for(j = 0; j < na; j++)
        {
            r = oa[j];
            h += aa[p * n3 + q * n2 + r * n + r];
        }
        for(j = 0; j < nb; j++)
        {
            r = ob[j];
            h += ab[p * n3 + q * n2 + r * n + r];
        }
        sign *= cre_des_sign2(p, q, sw);
    }
    else if(n1da == 0 && n1db == 2)
    {
        p = first1_2(db[0] & sx[2], db[1] & sx[3]);
        q = first1_2(db[0] & sw[2], db[1] & sw[3]);
        na = occ_list2(oa, sw);
        nb = occ_list2(ob, sw + 2);
        h = hb[p * n + q];
        for(j = 0; j < nb; j++)
        {
            r = ob[j];
            h += bb[p * n3 + q * n2 + r * n + r];
        }
        for(j = 0; j < na; j++)
        {
            r = oa[j];
            h += ab[r * n3 + r * n2 + p * n + q];
        }
        sign *= cre_des_sign2(p, q, sw + 2);
    }
    else if(n1da == 4)
    {
        p = first1_2(da[0] & sx[0], da[1] & sx[1]);
        q = first1_2(da[0] & sw[0], da[1] & sw[1]);
        r = second1_2(da[0] & sx[0], da[1] & sx[1], p);
        s = second1_2(da[0] & sw[0], da[1] & sw[1], q);
        str1[0] = sw[0];
        str1[1] = sw[1];
        str1[p >> 6] ^= 1ULL << (p & 63);
        str1[q >> 6] ^= 1ULL << (q & 63);
        h = aa[p * n3 + q * n2 + r * n + s];
        sign *= cre_des_sign2(p, q, sw) * cre_des_sign2(r, s, str1);
    }
    else if(n1db == 4)
    {
        p = first1_2(db[0] & sx[2], db[1] & sx[3]);
        q = first1_2(db[0] & sw[2], db[1] & sw[3]);
        r = second1_2(db[0] & sx[2], db[1] & sx[3], p);
        s = second1_2(db[0] & sw[2], db[1] & sw[3], q);
        str1[0] = sw[2];
        str1[1] = sw[3];
        str1[p >> 6] ^= 1ULL << (p & 63);
        str1[q >> 6] ^= 1ULL << (q & 63);
        h = bb[p * n3 + q * n2 + r * n + s];
        sign *= cre_des_sign2(p, q, sw + 2) * cre_des_sign2(r, s, str1);
    }
    else
    {
        /* alpha (p <- q), beta (r <- s): (pq|rs) */
        p = first1_2(da[0] & sx[0], da[1] & sx[1]);
        q = first1_2(da[0] & sw[0], da[1] & sw[1]);
        r = first1_2(db[0] & sx[2], db[1] & sx[3]);
        s = first1_2(db[0] & sw[2], db[1] & sw[3]);
        h = ab[p * n3 + q * n2 + r * n + s];
        sign *= cre_des_sign2(p, q, sw) * cre_des_sign2(r, s, sw + 2);
    }
    *hout = sign * h;
}

/*
 * One row of a same-parent block: the bra dx against the nw kets dws of the
 * same parent, with the occupation strings sx and sws of wick_det_strings.
 */
template<typename DT>
void hs_orth_row(DT *hrow, DT *srow, const wick_env *env, const int *dx, const uint64_t *sx,
                 const int *dws, const uint64_t *sws, int nw)
{
    const int64_t *info =
        env->pairinfo + ((size_t) dx[WICK_DPARENT] * env->nref + dx[WICK_DPARENT]) * WICK_PAIRLEN;
    const int *dw;
    int j;
    for(j = 0; j < nw; j++)
    {
        dw = dws + (size_t) j * env->dstride;
        hs_orth<DT>(hrow + j, srow + j, env, info, sx, sws + (size_t) j * WICK_NSTR,
                    (double) (dx[WICK_DOSIGN] * dw[WICK_DOSIGN]));
    }
}

/* Hamiltonian and overlap between records of different parents at one
 * corner combination */
template<typename DT>
void hamiltonian_corner(DT *hout, DT *sout, const wick_env *env, const int64_t *info,
                        const DT *scal, const int *dx, const int *dw)
{
    const int kmax = env->kmax;
    int rxa = dx[WICK_DRA], rxb = dx[WICK_DRB];
    int rwa = dw[WICK_DRA], rwb = dw[WICK_DRB];
    int la = rxa + rwa;
    int lb = rxb + rwb;
    int rows_a[WICK_MAXREC], cols_a[WICK_MAXREC];
    int rows_b[WICK_MAXREC], cols_b[WICK_MAXREC];
    spin_t<DT> sa, sb;
    DT h, s, pref;
    /* a zero-overlap pair needs a contraction in the operator or in D */
    if(info[WICK_MA] > la + 2 || info[WICK_MB] > lb + 2 ||
       info[WICK_MA] + info[WICK_MB] > la + lb + 2)
    {
        *hout = 0;
        *sout = 0;
        return;
    }
    spin_view<DT>(&sa, env, info, scal, 1);
    spin_view<DT>(&sb, env, info, scal, 0);
    labels<DT>(rows_a, cols_a, dx + WICK_DHEAD, dx + WICK_DHEAD + kmax, rxa, dw + WICK_DHEAD,
               dw + WICK_DHEAD + kmax, rwa, sa.nocc, sa.nmo);
    labels<DT>(rows_b, cols_b, dx + WICK_DHEAD + 2 * kmax, dx + WICK_DHEAD + 3 * kmax, rxb,
               dw + WICK_DHEAD + 2 * kmax, dw + WICK_DHEAD + 3 * kmax, rwb, sb.nocc, sb.nmo);
    if(sa.m == 0 && sb.m == 0 && la + lb <= WICK_MAXL)
    {
        switch(la * (WICK_MAXL + 1) + lb)
        {
            WICK_LALB_CASES(HS_M0_CASE)
        }
    }
    else
    {
        hs_general<DT>(&h, &s, env, info, scal, &sa, rows_a, cols_a, la, &sb, rows_b, cols_b, lb);
    }
    pref = (double) (dx[WICK_DSIGN] * dw[WICK_DSIGN]) * spin_pref<DT>(&sa) * spin_pref<DT>(&sb);
    *hout = pref * h;
    *sout = pref * s;
}

/* Hamiltonian and overlap between two determinant records */
template<typename DT>
void hamiltonian(DT *hout, DT *sout, const wick_env *env, const int *dx, const int *dw)
{
    const size_t k = (size_t) dx[WICK_DPARENT] * env->nref + dw[WICK_DPARENT];
    const int64_t *info = PAIRINFO(env, k);
    DT h, s;
    int64_t c;
    size_t kc;
    uint64_t sx[WICK_NSTR], sw[WICK_NSTR];
    if(dx[WICK_DPARENT] == dw[WICK_DPARENT])
    {
        wick_det_strings(sx, dx, env->kmax, info[WICK_NOCCA], info[WICK_NOCCB]);
        wick_det_strings(sw, dw, env->kmax, info[WICK_NOCCA], info[WICK_NOCCB]);
        hs_orth<DT>(hout, sout, env, info, sx, sw, (double) (dx[WICK_DOSIGN] * dw[WICK_DOSIGN]));
        return;
    }
    *hout = 0;
    *sout = 0;
    for(c = 0; c < info[WICK_NCORNER]; c++)
    {
        kc = corner_row(info, k, c);
        hamiltonian_corner<DT>(&h, &s, env, PAIRINFO(env, kc), PAIRSCAL(env, kc), dx, dw);
        *hout += h;
        *sout += s;
    }
}

/* Hamiltonian between two records at one corner combination with the
 * general evaluator only */
template<typename DT>
static void general_corner(DT *hout, DT *sout, const wick_env *env, const int64_t *info,
                           const DT *scal, const int *dx, const int *dw)
{
    const int kmax = env->kmax;
    int rxa = dx[WICK_DRA], rxb = dx[WICK_DRB];
    int rwa = dw[WICK_DRA], rwb = dw[WICK_DRB];
    int rows_a[WICK_MAXREC], cols_a[WICK_MAXREC];
    int rows_b[WICK_MAXREC], cols_b[WICK_MAXREC];
    spin_t<DT> sa, sb;
    DT h, s, pref;
    spin_view<DT>(&sa, env, info, scal, 1);
    spin_view<DT>(&sb, env, info, scal, 0);
    labels<DT>(rows_a, cols_a, dx + WICK_DHEAD, dx + WICK_DHEAD + kmax, rxa, dw + WICK_DHEAD,
               dw + WICK_DHEAD + kmax, rwa, sa.nocc, sa.nmo);
    labels<DT>(rows_b, cols_b, dx + WICK_DHEAD + 2 * kmax, dx + WICK_DHEAD + 3 * kmax, rxb,
               dw + WICK_DHEAD + 2 * kmax, dw + WICK_DHEAD + 3 * kmax, rwb, sb.nocc, sb.nmo);
    hs_general<DT>(&h, &s, env, info, scal, &sa, rows_a, cols_a, rxa + rwa, &sb, rows_b, cols_b,
                   rxb + rwb);
    pref = (double) (dx[WICK_DSIGN] * dw[WICK_DSIGN]) * spin_pref<DT>(&sa) * spin_pref<DT>(&sb);
    *hout = pref * h;
    *sout = pref * s;
}

/* Hamiltonian between two records with the general evaluator only */
template<typename DT>
void hamiltonian_general(DT *hout, DT *sout, const wick_env *env, const int *dx, const int *dw)
{
    const size_t k = (size_t) dx[WICK_DPARENT] * env->nref + dw[WICK_DPARENT];
    const int64_t *info = PAIRINFO(env, k);
    DT h, s;
    int64_t c;
    size_t kc;
    *hout = 0;
    *sout = 0;
    for(c = 0; c < info[WICK_NCORNER]; c++)
    {
        kc = corner_row(info, k, c);
        general_corner<DT>(&h, &s, env, PAIRINFO(env, kc), PAIRSCAL(env, kc), dx, dw);
        *hout += h;
        *sout += s;
    }
}

#undef PAIRINFO
#undef PAIRSCAL

} // namespace wick_kernels

} // namespace libgnme

#endif // LIBGNME_WICK_KERNELS_H
