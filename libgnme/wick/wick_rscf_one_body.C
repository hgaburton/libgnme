#include <cassert>
#include <algorithm>
#include <libgnme/utils/linalg.h>
#include "wick_rscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::add_one_body(arma::Mat<Tf> &F) 
{
    // Check input
    assert(F.n_rows == m_nbsf);
    assert(F.n_cols == m_nbsf);

    // Setup control variable to indicate one-body initialised
    m_one_body = true;

    // Get dimensions needed for temporary arrays
    size_t d = (m_orb.m_nz > 0) ? 2 : 1;

    // Construct 'F0' terms
    m_F0.resize(d); 
    for(size_t i=0; i<d; i++)
        m_F0(i) = arma::dot(F, m_orb.m_M(i).st());

    // We only have to worry about
    //    xx[YFX]    xw[YFY]
    //    wx[XFX]    ww[XFY]
    // Construct the XFX super matrices
    m_XFX.set_size(d,d); 
    for(size_t i=0; i<d; i++)
    for(size_t j=0; j<d; j++)
        m_XFX(i,j) = m_orb.m_CX(i).t() * F * m_orb.m_XC(j);
}


template class wick_rscf<double, double, double>;
template class wick_rscf<std::complex<double>, double, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
