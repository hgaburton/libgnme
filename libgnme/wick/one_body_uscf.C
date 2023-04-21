#include <cassert>
#include "one_body_uscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void one_body_uscf<Tc,Tf,Tb>::initialise(
    wick_orbitals<Tc,Tb> &orba, 
    wick_orbitals<Tc,Tb> &orbb,  
    arma::Mat<Tf> &Fa,
    arma::Mat<Tf> &Fb)
{
    // Check input
    assert(Fa.n_rows == orba.m_nbsf);
    assert(Fa.n_cols == orba.m_nbsf);
    assert(Fb.n_rows == orbb.m_nbsf);
    assert(Fb.n_cols == orbb.m_nbsf);

    // Get dimensions needed for temporary arrays
    size_t da = (orba.m_nz > 0) ? 2 : 1;
    size_t db = (orbb.m_nz > 0) ? 2 : 1;

    // Construct 'F0' terms
    m_F0a.resize(da); 
    for(size_t i=0; i<da; i++)
        m_F0a(i) = arma::dot(Fa, orba.m_M(i).st());
    m_F0b.resize(db);
    for(size_t i=0; i<db; i++)
        m_F0b(i) = arma::dot(Fb, orbb.m_M(i).st());

    // We only have to worry about
    //    xx[YFX]    xw[YFY]
    //    wx[XFX]    ww[XFY]
    // Construct the XFX super matrices
    m_XFXa.set_size(da,da); 
    for(size_t i=0; i<da; i++)
    for(size_t j=0; j<da; j++)
        m_XFXa(i,j) = orba.m_CX(i).t() * Fa * orba.m_XC(j);

    m_XFXb.set_size(db,db);
    for(size_t i=0; i<db; i++)
    for(size_t j=0; j<db; j++)
        m_XFXb(i,j) = orbb.m_CX(i).t() * Fb * orbb.m_XC(j);
}


template class one_body_uscf<double, double, double>;
template class one_body_uscf<std::complex<double>, double, double>;
template class one_body_uscf<std::complex<double>, std::complex<double>, double>;
template class one_body_uscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
