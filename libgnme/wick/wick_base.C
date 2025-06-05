#include "wick_base.h"
#include "wick_base_one_body.h"
#include "wick_base_overlap.h"
#include "wick_base_rdm.h"
#include "wick_base_two_body.h"


namespace libgnme {

template class wick_base<double, double, double>;
template class wick_base<std::complex<double>, double, double>;
template class wick_base<std::complex<double>, std::complex<double>, double>;
template class wick_base<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
