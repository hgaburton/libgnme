#include <cassert>
#include <algorithm>
#include "lowdin_pair.h"
#include "wick.h"

namespace libnome {

template<typename Tc, typename Tf, typename Tb>
void wick<Tc,Tf,Tb>::setup_orbitals(arma::Mat<Tc> Cx, arma::Mat<Tc> Cw) 
{
    // Take a safe copy
    m_Cxa = Cx.cols(0,m_nmo-1);
    m_Cwa = Cw.cols(0,m_nmo-1);
    m_Cxb = Cx.cols(m_nmo,2*m_nmo-1);
    m_Cwb = Cw.cols(m_nmo,2*m_nmo-1);
    
    // Initialise reduced overlap and number of zero overlaps
    m_redSa = 1.0, m_redSb = 1.0;
    m_nza = 0; m_nzb = 0;

    // Lowdin Pair occupied orbitals
    arma::Mat<Tc> Cw_a(Cw.memptr(), m_nbsf, m_nalpha, true, true);
    arma::Mat<Tc> Cw_b(Cw.colptr(m_nmo), m_nbsf, m_nbeta, true, true);
    arma::Mat<Tc> Cx_a(Cx.memptr(), m_nbsf, m_nalpha, true, true);
    arma::Mat<Tc> Cx_b(Cx.colptr(m_nmo), m_nbsf, m_nbeta, true, true);

    arma::uvec zeros_a(m_nalpha), zeros_b(m_nbeta);
    arma::Col<Tc> Sxx_a(m_nalpha, arma::fill::zeros);
    arma::Col<Tc> Sxx_b(m_nbeta, arma::fill::zeros);
    arma::Col<Tc> inv_Sxx_a(m_nalpha, arma::fill::zeros); 
    arma::Col<Tc> inv_Sxx_b(m_nbeta, arma::fill::zeros); 
    lowdin_pair(Cx_a, Cw_a, Sxx_a, m_metric);
    lowdin_pair(Cx_b, Cw_b, Sxx_b, m_metric);
    reduced_overlap(Sxx_a, inv_Sxx_a, m_redSa, m_nza, zeros_a);
    reduced_overlap(Sxx_b, inv_Sxx_b, m_redSb, m_nzb, zeros_b);

    // Construct co-density
    m_wxMa.set_size(2);
    m_wxMb.set_size(2);
    m_wxMa(0).set_size(m_nbsf,m_nbsf); m_wxMa(0).zeros();
    m_wxMa(1).set_size(m_nbsf,m_nbsf); m_wxMa(1).zeros();
    m_wxMb(0).set_size(m_nbsf,m_nbsf); m_wxMb(0).zeros();
    m_wxMb(1).set_size(m_nbsf,m_nbsf); m_wxMb(1).zeros();

    // Construct M matrices
    m_wxMa(0) = Cw_a * arma::diagmat(inv_Sxx_a) * Cx_a.t();
    m_wxMb(0) = Cw_b * arma::diagmat(inv_Sxx_b) * Cx_b.t();
    for(size_t i=0; i < m_nza; i++)
        m_wxMa(0) += Cx_a.col(zeros_a(i)) * Cx_a.col(zeros_a(i)).t();
    for(size_t i=0; i < m_nzb; i++)
        m_wxMb(0) += Cx_b.col(zeros_b(i)) * Cx_b.col(zeros_b(i)).t();

    // Construct P matrices
    for(size_t i=0; i < m_nza; i++)
        m_wxMa(1) += Cw_a.col(zeros_a(i)) * Cx_a.col(zeros_a(i)).t();
    for(size_t i=0; i < m_nzb; i++)
        m_wxMb(1) += Cw_b.col(zeros_b(i)) * Cx_b.col(zeros_b(i)).t();

    // Initialise contraction arrays 
    m_wwXa.set_size(4); m_xwXa.set_size(4);
    m_wxXa.set_size(4); m_xxXa.set_size(4);
    m_wwXb.set_size(4); m_xwXb.set_size(4);
    m_wxXb.set_size(4); m_xxXb.set_size(4);

    for(size_t i=0; i<2; i++)
    {
        // Construct X contractions
        m_wwXa(i) = m_Cwa.t() * m_metric * m_wxMa(i) * m_metric * m_Cwa; 
        m_wxXa(i) = m_Cwa.t() * m_metric * m_wxMa(i) * m_metric * m_Cxa; 
        m_xwXa(i) = m_Cxa.t() * m_metric * m_wxMa(i) * m_metric * m_Cwa; 
        m_xxXa(i) = m_Cxa.t() * m_metric * m_wxMa(i) * m_metric * m_Cxa; 
        m_wwXb(i) = m_Cwb.t() * m_metric * m_wxMb(i) * m_metric * m_Cwb; 
        m_wxXb(i) = m_Cwb.t() * m_metric * m_wxMb(i) * m_metric * m_Cxb; 
        m_xwXb(i) = m_Cxb.t() * m_metric * m_wxMb(i) * m_metric * m_Cwb; 
        m_xxXb(i) = m_Cxb.t() * m_metric * m_wxMb(i) * m_metric * m_Cxb; 
        // Construct Y contractions
        m_wwXa(i+2) = - m_Cwa.t() * (m_metric * m_wxMa(i) * m_metric - double(1-i) * m_metric) * m_Cwa; 
        m_wxXa(i+2) = - m_Cwa.t() * (m_metric * m_wxMa(i) * m_metric - double(1-i) * m_metric) * m_Cxa; 
        m_xwXa(i+2) = - m_Cxa.t() * (m_metric * m_wxMa(i) * m_metric - double(1-i) * m_metric) * m_Cwa; 
        m_xxXa(i+2) = - m_Cxa.t() * (m_metric * m_wxMa(i) * m_metric - double(1-i) * m_metric) * m_Cxa; 
        m_wwXb(i+2) = - m_Cwb.t() * (m_metric * m_wxMb(i) * m_metric - double(1-i) * m_metric) * m_Cwb; 
        m_wxXb(i+2) = - m_Cwb.t() * (m_metric * m_wxMb(i) * m_metric - double(1-i) * m_metric) * m_Cxb; 
        m_xwXb(i+2) = - m_Cxb.t() * (m_metric * m_wxMb(i) * m_metric - double(1-i) * m_metric) * m_Cwb; 
        m_xxXb(i+2) = - m_Cxb.t() * (m_metric * m_wxMb(i) * m_metric - double(1-i) * m_metric) * m_Cxb; 
    }
}

template<typename Tc, typename Tf, typename Tb>
void wick<Tc,Tf,Tb>::spin_overlap(
    arma::umat &xhp, arma::umat &whp,
    Tc &S, bool alpha)
{
    // Ensure output is zero'd
    S = 0.0;

    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations

    // Inform if we can't handle that excitation
    if(nx > 2 || nw > 2 || (nx+nw) > 4)
    {
        std::cout << "wick::spin_overlap: Bra excitations = " << nx << std::endl;
        std::cout << "wick::spin_overlap: Ket excitations = " << nw << std::endl;
        throw std::runtime_error(
           "wick::spin_overlap: Requested excitation level not yet implemented");
    } 

    // Get reference to relevant X matrices for this spin
    const arma::field<arma::Mat<Tc> > &wwX = alpha ? m_wwXa : m_wwXb;
    const arma::field<arma::Mat<Tc> > &wxX = alpha ? m_wxXa : m_wxXb;
    const arma::field<arma::Mat<Tc> > &xwX = alpha ? m_xwXa : m_xwXb;
    const arma::field<arma::Mat<Tc> > &xxX = alpha ? m_xxXa : m_xxXb;

    // Get reference to number of zeros for this spin
    const size_t &nz = alpha ? m_nza : m_nzb; 

    // Check we don't have a non-zero element
    if(nz > nw + nx) return;
    
    /* Evaluate the corresponding element */
    // < X | W >
    if(nx == 0 and nw == 0)
    {
        S = nz == 0 ? 1.0 : 0.0;
    }
    // < X_i^a | W >
    else if(nx == 1 and nw == 0)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        S = xxX(nz)(a,i);
    }
    // < X | W_i^a >
    else if(nx == 0 and nw == 1)
    {
        size_t i = whp(0,0), a = whp(0,1);
        S = wwX(nz)(i,a);
    }
    // < X_{ij}^{ab} | W > 
    else if(nx == 2 and nw == 0)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        size_t j = xhp(1,0), b = xhp(1,1);
        // Distribute the NZ zeros among 2 contractions
        std::vector<size_t> m(nz, 1); m.resize(2, 0); 
        do {
            S += xxX(m[0])(b,j) * xxX(m[1])(a,i) - xxX(m[0])(b,i) * xxX(m[1])(a,j); 
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X | W_{ij}^{ab} > 
    else if(nx == 0 and nw == 2)
    {   
        size_t i = whp(0,0), a = whp(0,1);
        size_t j = whp(1,0), b = whp(1,1);
        // Distribute the NZ zeros among 2 contractions
        std::vector<size_t> m(nz, 1); m.resize(2, 0); 
        do {
            S += wwX(m[0])(j,b) * wwX(m[1])(i,a) - wwX(m[0])(j,a) * wwX(m[1])(i,b); 
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X_i^a | W_j^b > 
    else if(nx == 1 and nw == 1)
    {   
        size_t i = xhp(0,0), a = xhp(0,1);
        size_t j = whp(0,0), b = whp(0,1);
        // Distribute the NZ zeros among 2 contractions
        std::vector<size_t> m(nz, 1); m.resize(2, 0); 
        do {
            S += xxX(m[0])(a,i) * wwX(m[1])(j,b) + wxX(m[0])(j,i) * xwX(2+m[1])(a,b); 
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X_{ij}^{ab} | W_k^c >
    else if(nx == 2 and nw == 1)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        size_t j = xhp(1,0), b = xhp(1,1);
        size_t k = whp(0,0), c = whp(0,1);
        // Distribute the NZ zeros among 3 contractions
        std::vector<size_t> m(nz, 1); m.resize(3, 0); 
        do {
            S += wwX(0+m[0])(k,c) * (xxX(m[1])(a,i) * xxX(m[2])(b,j) - xxX(m[1])(a,j) * xxX(m[2])(b,i))
               + xwX(2+m[0])(a,c) * (wxX(m[1])(k,i) * xxX(m[2])(b,j) - wxX(m[1])(k,j) * xxX(m[2])(b,i))
               + xwX(2+m[0])(b,c) * (wxX(m[1])(k,j) * xxX(m[2])(a,i) - wxX(m[1])(k,i) * xxX(m[2])(a,j));
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X_k^c | W_{ij}^{ab} >
    else if(nx == 1 and nw == 2)
    {
        size_t i = whp(0,0), a = whp(0,1);
        size_t j = whp(1,0), b = whp(1,1);
        size_t k = xhp(0,0), c = xhp(0,1);
        // Distribute the NZ zeros among 3 contractions
        std::vector<size_t> m(nz, 1); m.resize(3, 0); 
        do {
            S += xxX(0+m[0])(c,k) * (wwX(m[1])(i,a) * wwX(m[2])(j,b) - wwX(m[1])(j,a) * wwX(m[2])(i,b))
               + xwX(2+m[0])(c,a) * (wxX(m[1])(i,k) * wwX(m[2])(j,b) - wxX(m[1])(j,k) * wwX(m[2])(i,b))
               + xwX(2+m[0])(c,b) * (wxX(m[1])(j,k) * wwX(m[2])(i,a) - wxX(m[1])(i,k) * wwX(m[2])(j,a));
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X_{ij}^{ab} | W_{kl}^{cd} >
    else if(nx == 2 and nw == 2)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        size_t j = xhp(1,0), b = xhp(1,1);
        size_t k = whp(0,0), c = whp(0,1);
        size_t l = whp(1,0), d = whp(1,1);
        // Distribute the NZ zeros among 4 contractions
        std::vector<size_t> m(nz, 1); m.resize(4, 0); 
        do {
            S += (xxX(0+m[0])(a,i) * xxX(0+m[1])(b,j) - xxX(0+m[0])(a,j) * xxX(0+m[1])(b,i)) * (wwX(m[2])(k,c) * wwX(m[3])(l,d) - wwX(m[2])(k,d) * wwX(m[3])(l,c))
               + (xwX(2+m[0])(a,c) * xxX(0+m[1])(b,j) - xwX(2+m[0])(b,c) * xxX(0+m[1])(a,j)) * (wxX(m[2])(k,i) * wwX(m[3])(l,d) - wxX(m[2])(l,i) * wwX(m[3])(k,d))
               + (xwX(2+m[0])(a,d) * xxX(0+m[1])(b,j) - xwX(2+m[0])(b,d) * xxX(0+m[1])(a,j)) * (wxX(m[2])(l,i) * wwX(m[3])(k,c) - wxX(m[2])(k,i) * wwX(m[3])(l,c))
               + (xwX(2+m[0])(a,c) * xxX(0+m[1])(b,i) - xwX(2+m[0])(b,c) * xxX(0+m[1])(a,i)) * (wxX(m[2])(l,j) * wwX(m[3])(k,d) - wxX(m[2])(k,j) * wwX(m[3])(l,d))
               + (xwX(2+m[0])(a,d) * xxX(0+m[1])(b,i) - xwX(2+m[0])(b,d) * xxX(0+m[1])(a,i)) * (wxX(m[2])(k,j) * wwX(m[3])(l,c) - wxX(m[2])(l,j) * wwX(m[3])(k,c))
               + (xwX(2+m[0])(a,c) * xwX(2+m[1])(b,d) - xwX(2+m[0])(b,c) * xwX(2+m[1])(a,d)) * (wxX(m[2])(k,i) * wxX(m[3])(l,j) - wxX(m[2])(l,i) * wxX(m[3])(k,j));
        } while(std::prev_permutation(m.begin(), m.end()));
    }
}

template class wick<double, double, double>;
template class wick<std::complex<double>, double, double>;
template class wick<std::complex<double>, std::complex<double>, double>;
template class wick<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libnome
