/* Packed m_a = m_b = 0 Hamiltonian and overlap kernels for wick_matrix. WICK_NLANE determinant 
 * pairs sharing (L_a, L_b) are evaluated together: determinants, cofactors and second minors 
 * are lane-parallel, and intermediates are gathered per lane. 
 *
 * This file is included inside one namespace per instruction set by wick_kernels_avx2.C and 
 * wick_kernels_avx512.C with WICK_NLANE defined. Real packets are GCC vector extensions, and 
 * complex packets hold separate real and imaginary vectors.
 */

typedef double vd __attribute__((vector_size(8 * WICK_NLANE)));

struct vz
{
    vd re;
    vd im;
};

inline vd padd(vd a, vd b)
{
    return a + b;
}
inline vd psub(vd a, vd b)
{
    return a - b;
}
inline vd pmul(vd a, vd b)
{
    return a * b;
}
inline vd pscale(vd a, double s)
{
    return a * s;
}

inline vz padd(vz a, vz b)
{
    vz c = { a.re + b.re, a.im + b.im };
    return c;
}
inline vz psub(vz a, vz b)
{
    vz c = { a.re - b.re, a.im - b.im };
    return c;
}
inline vz pmul(vz a, vz b)
{
    vz c = { a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
    return c;
}
inline vz pscale(vz a, double s)
{
    vz c = { a.re * s, a.im * s };
    return c;
}

template<typename P>
inline P pfma(P acc, P a, P b)
{
    return padd(acc, pmul(a, b));
}
template<typename P>
inline P pfms(P acc, P a, P b)
{
    return psub(acc, pmul(a, b));
}

/** \brief Packet type and its construction, loads and stores for each scalar type **/
template<typename DT>
struct packet;

template<>
struct packet<double>
{
    typedef vd type;
    static vd zero()
    {
        vd v;
        std::memset(&v, 0, sizeof(v));
        return v;
    }
    static vd one()
    {
        return zero() + 1.0;
    }
    static vd bcast(double x)
    {
        return zero() + x;
    }
    static vd load(const double *p)
    {
        vd v;
        std::memcpy(&v, p, sizeof(v));
        return v;
    }
    static void store(double *p, vd v)
    {
        std::memcpy(p, &v, sizeof(v));
    }
};

template<>
struct packet<std::complex<double>>
{
    typedef vz type;
    static vz zero()
    {
        vz v = { packet<double>::zero(), packet<double>::zero() };
        return v;
    }
    static vz one()
    {
        vz v = { packet<double>::one(), packet<double>::zero() };
        return v;
    }
    static vz bcast(std::complex<double> x)
    {
        vz v = { packet<double>::bcast(x.real()), packet<double>::bcast(x.imag()) };
        return v;
    }
    static vz load(const std::complex<double> *p)
    {
        double re[WICK_NLANE], im[WICK_NLANE];
        for(int k = 0; k < WICK_NLANE; k++)
        {
            re[k] = p[k].real();
            im[k] = p[k].imag();
        }
        vz v = { packet<double>::load(re), packet<double>::load(im) };
        return v;
    }
    static void store(std::complex<double> *p, vz v)
    {
        for(int k = 0; k < WICK_NLANE; k++) p[k] = std::complex<double>(v.re[k], v.im[k]);
    }
};

template<typename DT>
ALWAYS_INLINE typename packet<DT>::type pdet_cof(typename packet<DT>::type *cof,
                                                 const typename packet<DT>::type *a, const int n,
                                                 const int want_cof)
{
    typename packet<DT>::type b[WICK_MAXL * WICK_MAXL];
    typename packet<DT>::type p[WICK_MAXL * WICK_MAXL];
    typename packet<DT>::type c, v;
    int i, j, k, l;
    if(n == 0)
    {
        return packet<DT>::one();
    }
    if(n == 1)
    {
        if(want_cof)
        {
            cof[0] = packet<DT>::one();
        }
        return a[0];
    }
    if(n == 2)
    {
        if(want_cof)
        {
            cof[0] = a[3];
            cof[1] = psub(packet<DT>::zero(), a[2]);
            cof[2] = psub(packet<DT>::zero(), a[1]);
            cof[3] = a[0];
        }
        return psub(pmul(a[0], a[3]), pmul(a[1], a[2]));
    }
    c = packet<DT>::zero();
    for(i = 0; i < n; i++)
    {
        c = psub(c, a[i * n + i]);
    }
    for(i = 0; i < n * n; i++)
    {
        b[i] = a[i];
    }
    for(i = 0; i < n; i++)
    {
        b[i * n + i] = padd(b[i * n + i], c);
    }
    for(k = 2; k <= n; k++)
    {
        if(want_cof && k == n)
        {
            for(i = 0; i < n; i++)
            {
                for(j = 0; j < n; j++)
                {
                    cof[i * n + j] =
                        (n & 1) ? b[j * n + i] : psub(packet<DT>::zero(), b[j * n + i]);
                }
            }
        }
        if(k == n)
        {
            c = packet<DT>::zero();
            for(i = 0; i < n; i++)
            {
                for(l = 0; l < n; l++)
                {
                    c = pfma(c, a[i * n + l], b[l * n + i]);
                }
            }
            c = pscale(c, -1. / n);
            return (n & 1) ? psub(packet<DT>::zero(), c) : c;
        }
        for(i = 0; i < n; i++)
        {
            for(j = 0; j < n; j++)
            {
                v = packet<DT>::zero();
                for(l = 0; l < n; l++)
                {
                    v = pfma(v, a[i * n + l], b[l * n + j]);
                }
                p[i * n + j] = v;
            }
        }
        c = packet<DT>::zero();
        for(i = 0; i < n; i++)
        {
            c = padd(c, p[i * n + i]);
        }
        c = pscale(c, -1. / k);
        for(i = 0; i < n * n; i++)
        {
            b[i] = p[i];
        }
        for(i = 0; i < n; i++)
        {
            b[i * n + i] = padd(b[i * n + i], c);
        }
    }
    return packet<DT>::zero();
}

template<typename DT>
ALWAYS_INLINE typename packet<DT>::type
psecond_minor_det(const typename packet<DT>::type *a, const int n, int r0, int r1, int c0, int c1)
{
    typename packet<DT>::type buf[WICK_MAXL * WICK_MAXL];
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
    return pdet_cof<DT>(NULL, buf, n - 2, 0);
}

/* lane labels of one spin */
template<typename DT>
struct labels_t
{
    int rows[WICK_NLANE][WICK_MAXL];
    int cols[WICK_NLANE][WICK_MAXL];
};

template<typename DT>
ALWAYS_INLINE void pbuild_d(typename packet<DT>::type *d, const DT *x, const DT *y,
                            const labels_t<DT> *lb, const int l, size_t n)
{
    DT tmp[WICK_NLANE];
    int i, j, k;
    for(i = 0; i < l; i++)
    {
        for(j = 0; j < l; j++)
        {
            const DT *t = (i >= j) ? x : y;
            for(k = 0; k < WICK_NLANE; k++)
            {
                tmp[k] = t[lb->rows[k][i] * n + lb->cols[k][j]];
            }
            d[i * l + j] = packet<DT>::load(tmp);
        }
    }
}

/* same-spin second-minor J class, sum_{eta<xi,z<y} */
template<typename DT>
ALWAYS_INLINE typename packet<DT>::type pjterm(const typename packet<DT>::type *d, const DT *jsl,
                                               const labels_t<DT> *lb, const int l, size_t n)
{
    const size_t n2 = n * n;
    const size_t n3 = n2 * n;
    DT tmp[WICK_NLANE];
    typename packet<DT>::type acc = packet<DT>::zero();
    typename packet<DT>::type m2;
    int eta, xi, z, y, k;
    for(eta = 0; eta < l; eta++)
    {
        for(xi = eta + 1; xi < l; xi++)
        {
            for(z = 0; z < l; z++)
            {
                for(y = z + 1; y < l; y++)
                {
                    m2 = psecond_minor_det<DT>(d, l, eta, xi, z, y);
                    for(k = 0; k < WICK_NLANE; k++)
                    {
                        const int *r = lb->rows[k];
                        const int *c = lb->cols[k];
                        tmp[k] = jsl[r[eta] * n3 + c[z] * n2 + r[xi] * n + c[y]] -
                                 jsl[r[eta] * n3 + c[y] * n2 + r[xi] * n + c[z]];
                    }
                    if((eta + xi + z + y) & 1)
                    {
                        acc = pfms(acc, m2, packet<DT>::load(tmp));
                    }
                    else
                    {
                        acc = pfma(acc, m2, packet<DT>::load(tmp));
                    }
                }
            }
        }
    }
    return acc;
}

template<typename DT>
ALWAYS_INLINE typename packet<DT>::type prepl(const typename packet<DT>::type *cof, const DT *hcol,
                                              const labels_t<DT> *lb, const int l, size_t n)
{
    DT tmp[WICK_NLANE];
    typename packet<DT>::type acc = packet<DT>::zero();
    int eta, z, k;
    for(z = 0; z < l; z++)
    {
        for(eta = 0; eta < l; eta++)
        {
            for(k = 0; k < WICK_NLANE; k++)
            {
                tmp[k] = hcol[lb->cols[k][z] * n + lb->rows[k][eta]];
            }
            acc = pfma(acc, cof[eta * l + z], packet<DT>::load(tmp));
        }
    }
    return acc;
}

template<typename DT>
ALWAYS_INLINE void hs_m0_pkt(DT *hout, DT *sout, const wick_kernels::wick_env *env,
                             const int64_t *info, const DT *scal, size_t n, const labels_t<DT> *lba,
                             const int la, const labels_t<DT> *lbb, const int lb)
{
    const DT *slab = (const DT *) env->slab;
    const int64_t *offa = info + WICK_AA;
    const int64_t *offb = info + WICK_BB;
    const DT *sa = scal + WICK_SAA;
    const DT *sb = scal + WICK_SBB;
    const size_t n2 = n * n;
    const size_t n3 = n2 * n;
    typename packet<DT>::type da[WICK_MAXL * WICK_MAXL + 1], cofa[WICK_MAXL * WICK_MAXL + 1];
    typename packet<DT>::type db[WICK_MAXL * WICK_MAXL + 1], cofb[WICK_MAXL * WICK_MAXL + 1];
    typename packet<DT>::type deta, detb, ja, jb, repa, repb, ii, inner, core, g0;
    DT tmp[WICK_NLANE];
    const DT *iisl = slab + info[WICK_AB + WICK_IIAB];
    int eta, xi, z, y, k;

    pbuild_d<DT>(da, slab + offa[WICK_X], slab + offa[WICK_Y], lba, la, n);
    pbuild_d<DT>(db, slab + offb[WICK_X], slab + offb[WICK_Y], lbb, lb, n);
    deta = pdet_cof<DT>(cofa, da, la, 1);
    detb = pdet_cof<DT>(cofb, db, lb, 1);
    ja = pjterm<DT>(da, slab + offa[WICK_J], lba, la, n);
    jb = pjterm<DT>(db, slab + offb[WICK_J], lbb, lb, n);
    repa = prepl<DT>(cofa, slab + offa[WICK_HCOL0], lba, la, n);
    repb = prepl<DT>(cofb, slab + offb[WICK_HCOL0], lbb, lb, n);

    ii = packet<DT>::zero();
    for(z = 0; z < la; z++)
    {
        for(eta = 0; eta < la; eta++)
        {
            inner = packet<DT>::zero();
            for(y = 0; y < lb; y++)
            {
                for(xi = 0; xi < lb; xi++)
                {
                    for(k = 0; k < WICK_NLANE; k++)
                    {
                        tmp[k] = iisl[lba->rows[k][eta] * n3 + lba->cols[k][z] * n2 +
                                      lbb->rows[k][xi] * n + lbb->cols[k][y]];
                    }
                    inner = pfma(inner, cofb[xi * lb + y], packet<DT>::load(tmp));
                }
            }
            ii = pfma(ii, cofa[eta * la + z], inner);
        }
    }

    g0 = packet<DT>::bcast(env->enuc + sa[WICK_SF0H] + .5 * sa[WICK_SV0] + sb[WICK_SF0H] +
                           .5 * sb[WICK_SV0] + scal[WICK_SVAB0]);
    core = pmul(g0, pmul(deta, detb));
    core = pfms(core, detb, repa);
    core = pfms(core, deta, repb);
    core = pfma(core, ja, detb);
    core = pfma(core, jb, deta);
    core = padd(core, ii);
    packet<DT>::store(hout, core);
    packet<DT>::store(sout, pmul(deta, detb));
}

/* One kernel for each (L_a, L_b) */
template<typename DT, int LA, int LB>
__attribute__((noinline)) void hs_m0_fixed_pkt(DT *h, DT *s, const wick_kernels::wick_env *env,
                                               const int64_t *info, const DT *scal, size_t n,
                                               const labels_t<DT> *lba, const labels_t<DT> *lbb)
{
    hs_m0_pkt<DT>(h, s, env, info, scal, n, lba, LA, lbb, LB);
}

/* (L_a, L_b) with L_a + L_b <= WICK_MAXL */
#define PKT_CASE(LA, LB)                                                                           \
    case LA * (WICK_MAXL + 1) + LB:                                                                 \
        hs_m0_fixed_pkt<DT, LA, LB>(h, s, env, info, scal, n, lba, lbb);                           \
        break;

template<typename DT>
static void hs_m0_dispatch(DT *h, DT *s, const wick_kernels::wick_env *env, const int64_t *info,
                           const DT *scal, size_t n, const labels_t<DT> *lba, int la,
                           const labels_t<DT> *lbb, int lb)
{
    switch(la * (WICK_MAXL + 1) + lb)
    {
        WICK_LALB_CASES(PKT_CASE)
    }
}
#undef PKT_CASE

/*
 * One row of the m_a = m_b = 0 Hamiltonian block at one corner combination
 * (info, scal): the bra dx against the nw kets starting at dws, all with the
 * same parent.  Kets with
 * L_a + L_b <= WICK_MAXL are binned by (L_a, L_b) into WICK_NLANE packets; a
 * partial packet is padded with copies of a valid lane.  Other kets fall
 * back to the scalar evaluator.
 */
template<typename DT>
void hs_row(DT *hrow, DT *srow, const wick_kernels::wick_env *env, const int64_t *info,
            const DT *scal, const int *dx, const int *dws, int nw)
{
    const int kmax = env->kmax;
    const int stride = env->dstride;
    const int nocca = info[WICK_NOCCA];
    const int noccb = info[WICK_NOCCB];
    const size_t n = info[WICK_NMO];
    const int rxa = dx[WICK_DRA];
    const int rxb = dx[WICK_DRB];
    const DT refpref = scal[WICK_SAA + WICK_SPHASE] * scal[WICK_SAA + WICK_STSP] *
                       scal[WICK_SBB + WICK_SPHASE] * scal[WICK_SBB + WICK_STSP];
    int bins[(WICK_MAXL + 1) * (WICK_MAXL + 1)][WICK_NLANE];
    int counts[(WICK_MAXL + 1) * (WICK_MAXL + 1)];
    labels_t<DT> lba, lbb;
    DT h[WICK_NLANE], s[WICK_NLANE];
    int j, k, key, la, lb, cnt;

    /* The bra labels are shared by every lane of the row */
    for(k = 0; k < WICK_NLANE; k++)
    {
        wick_kernels::labels<DT>(lba.rows[k], lba.cols[k], dx + WICK_DHEAD, dx + WICK_DHEAD + kmax,
                                 rxa, NULL, NULL, 0, nocca, n);
        wick_kernels::labels<DT>(lbb.rows[k], lbb.cols[k], dx + WICK_DHEAD + 2 * kmax,
                                 dx + WICK_DHEAD + 3 * kmax, rxb, NULL, NULL, 0, noccb, n);
    }

    memset(counts, 0, sizeof(counts));
    for(j = 0; j <= nw; j++)
    {
        const int *dw;
        if(j < nw)
        {
            dw = dws + (size_t) j * stride;
            la = rxa + dw[WICK_DRA];
            lb = rxb + dw[WICK_DRB];
            if(la + lb > WICK_MAXL)
            {
                wick_kernels::hamiltonian_corner<DT>(hrow + j, srow + j, env, info, scal, dx, dw);
                continue;
            }
            key = la * (WICK_MAXL + 1) + lb;
            bins[key][counts[key]++] = j;
            if(counts[key] < WICK_NLANE)
            {
                continue;
            }
            cnt = WICK_NLANE;
        }
        else
        {
            /* flush partial packets */
            for(key = 0; key < (WICK_MAXL + 1) * (WICK_MAXL + 1); key++)
            {
                if(counts[key] == 0) continue;
                for(k = counts[key]; k < WICK_NLANE; k++)
                {
                    bins[key][k] = bins[key][0];
                }
                break;
            }
            if(key == (WICK_MAXL + 1) * (WICK_MAXL + 1))
            {
                break;
            }
            cnt = counts[key];
            la = key / (WICK_MAXL + 1);
            lb = key % (WICK_MAXL + 1);
            j--; /* revisit the flush step */
        }
        for(k = 0; k < WICK_NLANE; k++)
        {
            dw = dws + (size_t) bins[key][k] * stride;
            /* Ket labels of the lane, after the bra labels */
            const int *ha = dw + WICK_DHEAD, *pa = ha + kmax, *hb = pa + kmax, *pb = hb + kmax;
            for(int i = 0; i < la - rxa; i++)
            {
                lba.rows[k][rxa + i] = (int) n - nocca + ha[i];
                lba.cols[k][rxa + i] = pa[i];
            }
            for(int i = 0; i < lb - rxb; i++)
            {
                lbb.rows[k][rxb + i] = (int) n - noccb + hb[i];
                lbb.cols[k][rxb + i] = pb[i];
            }
        }
        hs_m0_dispatch<DT>(h, s, env, info, scal, n, &lba, la, &lbb, lb);
        for(k = 0; k < cnt; k++)
        {
            int jj = bins[key][k];
            DT pref = (double) (dx[WICK_DSIGN] * dws[(size_t) jj * stride + WICK_DSIGN]) * refpref;
            hrow[jj] = pref * h[k];
            srow[jj] = pref * s[k];
        }
        counts[key] = 0;
    }
}
