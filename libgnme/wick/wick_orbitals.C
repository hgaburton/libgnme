#include <cassert>
#include <libgnme/utils/lowdin_pair.h>
#include "wick_orbitals.h"

namespace libgnme {

template<typename Tc, typename Tb>
void wick_orbitals<Tc,Tb>::init(arma::Mat<Tc> Cx, arma::Mat<Tc> Cw)
{
    // Check all the dimensions
    assert(Cx.n_rows == m_nbsf);
    assert(Cx.n_cols == m_nmo);
    assert(Cw.n_rows == m_nbsf);
    assert(Cw.n_cols == m_nmo);

    // Take a safe copy of active orbitals
    m_Cx = Cx.cols(m_ncore,m_ncore+m_nact-1);
    m_Cw = Cw.cols(m_ncore,m_ncore+m_nact-1);

    // Make copy of orbitals for Lowdin pairing
    arma::Mat<Tc> Cw_tmp(Cw.memptr(), m_nbsf, m_nelec, true, true);
    arma::Mat<Tc> Cx_tmp(Cx.memptr(), m_nbsf, m_nelec, true, true);

    // Lowdin Pair occupied orbitals
    m_nz = 0; m_redS = 1.0;
    arma::uvec zeros(m_nelec, arma::fill::zeros);
    arma::Col<Tc> Sxx(m_nelec, arma::fill::zeros);
    arma::Col<Tc> inv_Sxx(m_nelec, arma::fill::zeros); 
    lowdin_pair(Cx_tmp, Cw_tmp, Sxx, m_metric, 1e-20);
    reduced_overlap(Sxx, inv_Sxx, m_redS, m_nz, zeros, 1e-8);

    // Construct co-density
    m_M.set_size(2);
    m_M(0).set_size(m_nbsf,m_nbsf); m_M(0).zeros();
    m_M(1).set_size(m_nbsf,m_nbsf); m_M(1).zeros();

    // Construct M matrices
    m_M(0) = Cw_tmp * arma::diagmat(inv_Sxx) * Cx_tmp.t();
    for(size_t i=0; i < m_nz; i++)
        m_M(0) += Cx_tmp.col(zeros(i)) * Cx_tmp.col(zeros(i)).t();
    // Construct P matrices
    for(size_t i=0; i < m_nz; i++)
        m_M(1) += Cw_tmp.col(zeros(i)) * Cx_tmp.col(zeros(i)).t();

    // Initialise X contraction arrays
    m_X.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        // Initialise the memory
        m_X(i).set_size(2*m_nact,2*m_nact); m_X(i).zeros();

        // Compute
        m_X(i).submat(0,0, m_nact-1,m_nact-1)               
            = m_Cx.t() * m_metric * m_M(i) * m_metric * m_Cx; // xx
        m_X(i).submat(0,m_nact, m_nact-1,2*m_nact-1)        
            = m_Cx.t() * m_metric * m_M(i) * m_metric * m_Cw; // xw
        m_X(i).submat(m_nact,0, 2*m_nact-1,m_nact-1)        
            = m_Cw.t() * m_metric * m_M(i) * m_metric * m_Cx; // wx
        m_X(i).submat(m_nact,m_nact, 2*m_nact-1,2*m_nact-1) 
            = m_Cw.t() * m_metric * m_M(i) * m_metric * m_Cw; // ww
    }

    // Intialise Y contraction arrays
    m_Y.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        // Initialise the memory
        m_Y(i).set_size(2*m_nact,2*m_nact); m_Y(i).zeros();

        // Compute
        m_Y(i).submat(0,0, m_nact-1,m_nact-1)             
            = m_Cx.t() * (m_metric * m_M(i) * m_metric - double(1-i) * m_metric) * m_Cx; // xx
        m_Y(i).submat(0,m_nact, m_nact-1,2*m_nact-1)       
            = m_Cx.t() * (m_metric * m_M(i) * m_metric - double(1-i) * m_metric) * m_Cw; // xw
        m_Y(i).submat(m_nact,0, 2*m_nact-1,m_nact-1)       
            = m_Cw.t() * (m_metric * m_M(i) * m_metric - double(1-i) * m_metric) * m_Cx; // wx
        m_Y(i).submat(m_nact,m_nact, 2*m_nact-1,2*m_nact-1) 
            = m_Cw.t() * (m_metric * m_M(i) * m_metric - double(1-i) * m_metric) * m_Cw; // ww
    }

    // 1-RDM variables
    m_wxP.set_size(2);
    for(size_t i=0; i<2; i++)
        m_wxP(i) = Cw.t() * m_metric * m_M(i) * m_metric * Cx; // wx
    m_R.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        m_R(i).resize(2*m_nact, m_nmo);
        m_R(i).rows(0,m_nact-1)        = m_Cx.t() * (m_metric * m_M(i) * m_metric - double(1-i) * m_metric) * Cx; 
        m_R(i).rows(m_nact,2*m_nact-1) = m_Cw.t() * m_metric * m_M(i) * m_metric * Cx;
    }
    m_Q.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        m_Q(i).resize(m_nmo, 2*m_nact);
        m_Q(i).cols(0,m_nact-1)        = Cw.t() * m_metric * m_M(i) * m_metric * m_Cx;
        m_Q(i).cols(m_nact,2*m_nact-1) = Cw.t() * (m_metric * m_M(i) * m_metric - double(1-i) * m_metric) * m_Cw; 
    }

    // Construct transformed coefficients
    m_CX.set_size(2);
    m_XC.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        m_CX(i).resize(m_nbsf,2*m_nact); m_CX(i).zeros();
        m_CX(i).cols(0,m_nact-1)        = m_M(i).t() * m_metric * m_Cx - (1-i) * m_Cx; // x[CY]
        m_CX(i).cols(m_nact,2*m_nact-1) = m_M(i).t() * m_metric * m_Cw;                // w[CX]

        m_XC(i).resize(m_nbsf,2*m_nact); m_XC(i).zeros();
        m_XC(i).cols(0,m_nact-1)        = m_M(i) * m_metric * m_Cx;                    // x[XC]
        m_XC(i).cols(m_nact,2*m_nact-1) = m_M(i) * m_metric * m_Cw - (1-i) * m_Cw;     // w[YC]
    }
}

template class wick_orbitals<double,double>;
template class wick_orbitals<std::complex<double>,double>;
template class wick_orbitals<std::complex<double>,std::complex<double> >;

} // namespace libgnme
