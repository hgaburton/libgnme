#include <cassert>
#include <libgnme/utils/lowdin_pair.h>
#include "wick_orbitals.h"

namespace libgnme {

template<typename Tc, typename Tb>
void wick_orbitals<Tc,Tb>::init()
{
    // Check our input is meaningful
    assert(m_refx.m_nbsf  == m_refw.m_nbsf);
    assert(m_refx.m_nmo   == m_refw.m_nmo);
    assert(m_refx.m_nelec == m_refw.m_nelec);

    // Get number of core/active orbitals for two reference states
    size_t ncorex = m_refx.m_ncore, nactx = m_refx.m_nact;
    size_t ncorew = m_refw.m_ncore, nactw = m_refw.m_nact;
    size_t nact   = nactx+nactw; // Total number of active orbitals

    // Get access to coefficients
    const arma::Mat<Tc> &Cx = m_refx.m_C;
    const arma::Mat<Tc> &Cw = m_refw.m_C;
    const arma::Mat<Tc> &Cx_act = m_refx.m_C.cols(ncorex,ncorex+nactx-1);
    const arma::Mat<Tc> &Cw_act = m_refw.m_C.cols(ncorew,ncorew+nactw-1);

    // Make copy of orbitals for Lowdin pairing
    arma::Mat<Tc> Cx_tmp = Cx.cols(0,m_nelec-1);
    arma::Mat<Tc> Cw_tmp = Cw.cols(0,m_nelec-1);
    // Lowdin Pair occupied orbitals
    m_nz = 0; m_redS = 1.0;
    arma::uvec      zeros(m_nelec, arma::fill::zeros);
    arma::Col<Tc>     Sxx(m_nelec, arma::fill::zeros);
    arma::Col<Tc> inv_Sxx(m_nelec, arma::fill::zeros); 
    lowdin_pair(Cx_tmp, Cw_tmp, Sxx, m_metric, 1e-20);
    reduced_overlap(Sxx, inv_Sxx, m_redS, m_nz, zeros, 1e-8);

    // Construct co-density
    m_M.set_size(2);
    m_M(0).set_size(m_nbsf, m_nbsf); m_M(0).zeros();
    m_M(1).set_size(m_nbsf, m_nbsf); m_M(1).zeros();
    // Construct M matrices
    m_M(0) = Cw_tmp * arma::diagmat(inv_Sxx) * Cx_tmp.t();
    for(size_t i=0; i < m_nz; i++)
        m_M(0) += Cx_tmp.col(zeros(i)) * Cx_tmp.col(zeros(i)).t();
    // Construct P matrices
    for(size_t i=0; i < m_nz; i++)
        m_M(1) += Cw_tmp.col(zeros(i)) * Cx_tmp.col(zeros(i)).t();


    // Initialise full X contraction arrays for RDM1
    m_fX.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        // Initialise the memory
        m_fX(i).set_size(2*m_nmo,2*m_nmo); m_fX(i).zeros();
        // Compute
        m_fX(i).submat(0,0, m_nmo-1,m_nmo-1)             = Cx.t() * m_metric * m_M(i) * m_metric * Cx; // xx
        m_fX(i).submat(0,m_nmo, m_nmo-1,2*m_nmo-1)       = Cx.t() * m_metric * m_M(i) * m_metric * Cw; // xw
        m_fX(i).submat(m_nmo,0, 2*m_nmo-1,m_nmo-1)       = Cw.t() * m_metric * m_M(i) * m_metric * Cx; // wx
        m_fX(i).submat(m_nmo,m_nmo, 2*m_nmo-1,2*m_nmo-1) = Cw.t() * m_metric * m_M(i) * m_metric * Cw; // ww
    }

 
    // Initialise X/Y contractions in active orbital spaces
    m_X.set_size(2);
    m_Y.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        // Initialise the memory
        m_X(i).set_size(nact, nact); m_X(i).zeros();
        m_Y(i).set_size(nact, nact); m_Y(i).zeros();

        // Compute X
        m_X(i).submat(0,0, nactx-1,nactx-1)       = Cx_act.t() * m_metric * m_M(i) * m_metric * Cx_act; // xx
        m_X(i).submat(0,nactx, nactx-1,nact-1)    = Cx_act.t() * m_metric * m_M(i) * m_metric * Cw_act; // xw
        m_X(i).submat(nactx,0, nact-1,nactx-1)    = Cw_act.t() * m_metric * m_M(i) * m_metric * Cx_act; // wx
        m_X(i).submat(nactx,nactx, nact-1,nact-1) = Cw_act.t() * m_metric * m_M(i) * m_metric * Cw_act; // ww 

        // Compute Y
        m_Y(i).submat(0,0, nactx-1,nactx-1)  
            = Cx_act.t() * (m_metric * m_M(i) * m_metric - double(1-i) * m_metric) * Cx_act; // xx
        m_Y(i).submat(0,nactx, nactx-1,nact-1)       
            = Cx_act.t() * (m_metric * m_M(i) * m_metric - double(1-i) * m_metric) * Cw_act; // xw
        m_Y(i).submat(nactx,0, nact-1,nactx-1)       
            = Cw_act.t() * (m_metric * m_M(i) * m_metric - double(1-i) * m_metric) * Cx_act; // wx
        m_Y(i).submat(nactx,nactx, nact-1,nact-1) 
            = Cw_act.t() * (m_metric * m_M(i) * m_metric - double(1-i) * m_metric) * Cw_act; // ww
    }

    // Construct transformed coefficients for computing integral intermediates
    m_CX.set_size(2);
    m_XC.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        m_CX(i).resize(m_nbsf,nact); m_CX(i).zeros();
        m_CX(i).cols(0,nactx-1)    = m_M(i).t() * m_metric * Cx_act - (1-i) * Cx_act; // x[CY]
        m_CX(i).cols(nactx,nact-1) = m_M(i).t() * m_metric * Cw_act;                // w[CX]

        m_XC(i).resize(m_nbsf,nact); m_XC(i).zeros();
        m_XC(i).cols(0,nactx-1)    = m_M(i) * m_metric * Cx_act;                    // x[XC]
        m_XC(i).cols(nactx,nact-1) = m_M(i) * m_metric * Cw_act - (1-i) * Cw_act;     // w[YC]
    }


    // Intermediates for computing density matrices
    m_wxP.set_size(2);
    for(size_t i=0; i<2; i++) 
        m_wxP(i) = Cw.t() * m_metric * m_M(i) * m_metric * Cx; // wx

    m_R.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        m_R(i).resize(nact, m_nmo);
        m_R(i).rows(0,nactx-1)    = Cx_act.t() * (m_metric * m_M(i) * m_metric - double(1-i) * m_metric) * Cx; // x 
        m_R(i).rows(nactx,nact-1) = Cw_act.t() * m_metric * m_M(i) * m_metric * Cx; // w
    }
    m_Q.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        m_Q(i).resize(m_nmo, nact);
        m_Q(i).cols(0,nactx-1)    = Cw.t() * m_metric * m_M(i) * m_metric * Cx_act;
        m_Q(i).cols(nactx,nact-1) = Cw.t() * (m_metric * m_M(i) * m_metric - double(1-i) * m_metric) * Cw_act; 
    }
}

template class wick_orbitals<double,double>;
template class wick_orbitals<std::complex<double>,double>;
template class wick_orbitals<std::complex<double>,std::complex<double> >;

} // namespace libgnme
