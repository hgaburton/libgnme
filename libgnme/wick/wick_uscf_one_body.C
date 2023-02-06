#include <cassert>
#include <algorithm>
#include <libgnme/utils/linalg.h>
#include "wick_uscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::add_one_body(arma::Mat<Tf> &F) 
{
    add_one_body(F,F);
}

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::add_one_body(arma::Mat<Tf> &Fa, arma::Mat<Tf> &Fb) 
{
    // Check input
    assert(Fa.n_rows == m_nbsf);
    assert(Fa.n_cols == m_nbsf);
    assert(Fb.n_rows == m_nbsf);
    assert(Fb.n_cols == m_nbsf);

    // Setup control variable to indicate one-body initialised
    m_one_body = true;

    // Get dimensions needed for temporary arrays
    size_t da = (m_orb_a.m_nz > 0) ? 2 : 1;
    size_t db = (m_orb_b.m_nz > 0) ? 2 : 1;

    // Construct 'F0' terms
    m_F0a.resize(da); 
    for(size_t i=0; i<da; i++)
        m_F0a(i) = arma::dot(Fa, m_orb_a.m_M(i).st());
    m_F0b.resize(db);
    for(size_t i=0; i<db; i++)
        m_F0b(i) = arma::dot(Fb, m_orb_b.m_M(i).st());

    // We only have to worry about
    //    xx[YFX]    xw[YFY]
    //    wx[XFX]    ww[XFY]
    // Construct the XFX super matrices
    m_XFXa.set_size(da,da); 
    for(size_t i=0; i<da; i++)
    for(size_t j=0; j<da; j++)
        m_XFXa(i,j) = m_orb_a.m_CX(i).t() * Fa * m_orb_a.m_XC(j);

    m_XFXb.set_size(da,da);
    for(size_t i=0; i<db; i++)
    for(size_t j=0; j<db; j++)
        m_XFXb(i,j) = m_orb_b.m_CX(i).t() * Fb * m_orb_b.m_XC(j);
}


template class wick_uscf<double, double, double>;
template class wick_uscf<std::complex<double>, double, double>;
template class wick_uscf<std::complex<double>, std::complex<double>, double>;
template class wick_uscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
