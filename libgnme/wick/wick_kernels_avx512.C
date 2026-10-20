#include <cstring>
#include <complex>
#include "wick_kernels.h"

/* Packed kernels for AVX-512, with 8 lanes. This file is compiled with the
   corresponding instruction set and only called when the CPU supports it. */

namespace libgnme {

namespace wick_avx512 {

#define WICK_NLANE 8
#include "wick_packets.h"
#undef WICK_NLANE

template void hs_row<double>(
    double *hrow, double *srow, const wick_kernels::wick_env *env, const int64_t *info, 
    const double *scal, const int *dx, const int *dws, int nw);
template void hs_row<std::complex<double> >(
    std::complex<double> *hrow, std::complex<double> *srow, const wick_kernels::wick_env *env,
    const int64_t *info, const std::complex<double> *scal, const int *dx, const int *dws, int nw);

} // namespace wick_avx512

} // namespace libgnme
