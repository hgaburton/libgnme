#include <cassert>
#include "one_body_rscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void one_body_rscf<Tc,Tf,Tb>::initialise(
    wick_orbitals<Tc,Tb> &orb, 
    arma::Mat<Tf> &F)
{
    // Check input
    assert(F.n_rows == orb.m_nbsf);
    assert(F.n_cols == orb.m_nbsf);

    // Get dimensions needed for temporary arrays
    size_t d = (orb.m_nz > 0) ? 2 : 1;

    // Construct 'F0' terms
    m_F0.resize(d); 
    for(size_t i=0; i<d; i++)
        m_F0(i) = arma::dot(F, orb.m_M(i).st());

    // We only have to worry about
    //    xx[YFX]    xw[YFY]
    //    wx[XFX]    ww[XFY]
    // Construct the XFX super matrices
    m_XFX.set_size(d,d); 
    for(size_t i=0; i<d; i++)
    for(size_t j=0; j<d; j++)
        m_XFX(i,j) = orb.m_CX(i).t() * F * orb.m_XC(j);
}

template class one_body_rscf<double, double, double>;
template class one_body_rscf<std::complex<double>, double, double>;
template class one_body_rscf<std::complex<double>, std::complex<double>, double>;
template class one_body_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
