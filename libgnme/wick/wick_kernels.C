#include <vector>
#include "wick_kernels.h"

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

namespace wick_kernels {

WICK_ENTRY_POINTS(, double)
WICK_ENTRY_POINTS(, std::complex<double>)

int simd_level()
{
#if defined(LIBGNME_X86_SIMD)
    __builtin_cpu_init();
    if(__builtin_cpu_supports("avx512f")) return WICK_SIMD_AVX512;
    if(__builtin_cpu_supports("avx2") and __builtin_cpu_supports("fma")) return WICK_SIMD_AVX2;
#endif
    return WICK_SIMD_NONE;
}

template<typename DT>
void pair_row(DT *hrow, DT *srow, const wick_env *env, size_t k, const int *dx, const int *dws,
              int nw)
{
    const int64_t *info = env->pairinfo + k * WICK_PAIRLEN;
#if defined(LIBGNME_X86_SIMD)
    const bool packed = env->simd != WICK_SIMD_NONE and info[WICK_MA] == 0 and info[WICK_MB] == 0;
#endif
    std::vector<DT> buf(info[WICK_NCORNER] > 1 ? 2 * nw : 0);
    for(int64_t c=0; c < info[WICK_NCORNER]; c++)
    {
        const size_t kc = corner_row(info, k, c);
        const int64_t *infoc = env->pairinfo + kc * WICK_PAIRLEN;
        const DT *scal = (const DT *) env->scal + kc * WICK_NSCAL;
        DT *h = (c == 0) ? hrow : buf.data();
        DT *s = (c == 0) ? srow : buf.data() + nw;
#if defined(LIBGNME_X86_SIMD)
        if(packed and env->simd == WICK_SIMD_AVX512)
            wick_avx512::hs_row<DT>(h, s, env, infoc, scal, dx, dws, nw);
        else if(packed)
            wick_avx2::hs_row<DT>(h, s, env, infoc, scal, dx, dws, nw);
        else
#endif
        for(int j=0; j < nw; j++)
            hamiltonian_corner<DT>(h+j, s+j, env, infoc, scal, dx, dws + (size_t) j * env->dstride);
        if(c == 0) continue;
        for(int j=0; j < nw; j++) { hrow[j] += h[j]; srow[j] += s[j]; }
    }
}
template void pair_row<double>(
    double *hrow, double *srow, const wick_env *env, size_t k, const int *dx, const int *dws,
    int nw);
template void pair_row<std::complex<double> >(
    std::complex<double> *hrow, std::complex<double> *srow, const wick_env *env, size_t k,
    const int *dx, const int *dws, int nw);

} // namespace wick_kernels

} // namespace libgnme
