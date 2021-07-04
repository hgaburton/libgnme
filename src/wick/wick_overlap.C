#include <cassert>
#include <algorithm>
#include "lowdin_pair.h"
#include "wick.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick<Tc,Tf,Tb>::setup_orbitals(arma::Mat<Tc> Cx, arma::Mat<Tc> Cw)
{
    setup_orbitals(Cx, Cw, 0, m_nmo);
}

template<typename Tc, typename Tf, typename Tb>
void wick<Tc,Tf,Tb>::setup_orbitals(
    arma::Mat<Tc> Cx, arma::Mat<Tc> Cw, 
    size_t ncore, size_t nactive) 
{
    // Store number of core and active orbitals
    m_nact = nactive;

    // Save reference coefficients for integrals 
    // TODO: Can we replace this?
    m_Cref = Cx;

    // Take a safe copy of active orbitals
    m_Cxa = Cx.cols(ncore,ncore+m_nact-1);
    m_Cwa = Cw.cols(ncore,ncore+m_nact-1);
    m_Cxb = Cx.cols(m_nmo+ncore,m_nmo+ncore+m_nact-1);
    m_Cwb = Cw.cols(m_nmo+ncore,m_nmo+ncore+m_nact-1);
    
    // Initialise reduced overlap and number of zero overlaps
    m_redSa = 1.0, m_redSb = 1.0;
    m_nza = 0; m_nzb = 0;

    // Take copy of orbitals for Lowdin pairing
    arma::Mat<Tc> Cw_a(Cw.memptr(), m_nbsf, m_nalpha, true, true);
    arma::Mat<Tc> Cw_b(Cw.colptr(m_nmo), m_nbsf, m_nbeta, true, true);
    arma::Mat<Tc> Cx_a(Cx.memptr(), m_nbsf, m_nalpha, true, true);
    arma::Mat<Tc> Cx_b(Cx.colptr(m_nmo), m_nbsf, m_nbeta, true, true);

    // Lowdin Pair occupied orbitals
    arma::uvec zeros_a(m_nalpha), zeros_b(m_nbeta);
    arma::Col<Tc> Sxx_a(m_nalpha, arma::fill::zeros);
    arma::Col<Tc> Sxx_b(m_nbeta, arma::fill::zeros);
    arma::Col<Tc> inv_Sxx_a(m_nalpha, arma::fill::zeros); 
    arma::Col<Tc> inv_Sxx_b(m_nbeta, arma::fill::zeros); 
    lowdin_pair(Cx_a, Cw_a, Sxx_a, m_metric,1e-20);
    lowdin_pair(Cx_b, Cw_b, Sxx_b, m_metric,1e-20);
    reduced_overlap(Sxx_a, inv_Sxx_a, m_redSa, m_nza, zeros_a,1e-8);
    reduced_overlap(Sxx_b, inv_Sxx_b, m_redSb, m_nzb, zeros_b,1e-8);

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

    // Initialise X contraction arrays
    m_Xa.set_size(2); m_Xb.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        // Initialise the memory
        m_Xa(i).set_size(2*m_nact,2*m_nact); m_Xa(i).zeros();
        m_Xb(i).set_size(2*m_nact,2*m_nact); m_Xb(i).zeros();

        // Compute alpha terms
        m_Xa(i).submat(0,0, m_nact-1,m_nact-1)               
            = m_Cxa.t() * m_metric * m_wxMa(i) * m_metric * m_Cxa; // xx
        m_Xa(i).submat(0,m_nact, m_nact-1,2*m_nact-1)        
            = m_Cxa.t() * m_metric * m_wxMa(i) * m_metric * m_Cwa; // xw
        m_Xa(i).submat(m_nact,0, 2*m_nact-1,m_nact-1)        
            = m_Cwa.t() * m_metric * m_wxMa(i) * m_metric * m_Cxa; // wx
        m_Xa(i).submat(m_nact,m_nact, 2*m_nact-1,2*m_nact-1) 
            = m_Cwa.t() * m_metric * m_wxMa(i) * m_metric * m_Cwa; // ww

        // Compute beta terms
        m_Xb(i).submat(0,0, m_nact-1,m_nact-1)               
            = m_Cxb.t() * m_metric * m_wxMb(i) * m_metric * m_Cxb; // xx
        m_Xb(i).submat(0,m_nact, m_nact-1,2*m_nact-1)        
            = m_Cxb.t() * m_metric * m_wxMb(i) * m_metric * m_Cwb; // xw
        m_Xb(i).submat(m_nact,0, 2*m_nact-1,m_nact-1)        
            = m_Cwb.t() * m_metric * m_wxMb(i) * m_metric * m_Cxb; // wx
        m_Xb(i).submat(m_nact,m_nact, 2*m_nact-1,2*m_nact-1) 
            = m_Cwb.t() * m_metric * m_wxMb(i) * m_metric * m_Cwb; // ww
    }

    // Intialise Y contraction arrays
    m_Ya.set_size(2); m_Yb.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        // Initialise the memory
        m_Ya(i).set_size(2*m_nact,2*m_nact); m_Ya(i).zeros();
        m_Yb(i).set_size(2*m_nact,2*m_nact); m_Yb(i).zeros();

        // Compute alpha terms
        m_Ya(i).submat(0,0, m_nact-1,m_nact-1)             
            = m_Cxa.t() * (m_metric * m_wxMa(i) * m_metric - double(1-i) * m_metric) * m_Cxa; // xx
        m_Ya(i).submat(0,m_nact, m_nact-1,2*m_nact-1)       
            = m_Cxa.t() * (m_metric * m_wxMa(i) * m_metric - double(1-i) * m_metric) * m_Cwa; // xw
        m_Ya(i).submat(m_nact,0, 2*m_nact-1,m_nact-1)       
            = m_Cwa.t() * (m_metric * m_wxMa(i) * m_metric - double(1-i) * m_metric) * m_Cxa; // wx
        m_Ya(i).submat(m_nact,m_nact, 2*m_nact-1,2*m_nact-1) 
            = m_Cwa.t() * (m_metric * m_wxMa(i) * m_metric - double(1-i) * m_metric) * m_Cwa; // ww

        // Compute beta terms
        m_Yb(i).submat(0,0, m_nact-1,m_nact-1)             
            = m_Cxb.t() * (m_metric * m_wxMb(i) * m_metric - double(1-i) * m_metric) * m_Cxb; // xx
        m_Yb(i).submat(0,m_nact, m_nact-1,2*m_nact-1)       
            = m_Cxb.t() * (m_metric * m_wxMb(i) * m_metric - double(1-i) * m_metric) * m_Cwb; // xw
        m_Yb(i).submat(m_nact,0, 2*m_nact-1,m_nact-1)       
            = m_Cwb.t() * (m_metric * m_wxMb(i) * m_metric - double(1-i) * m_metric) * m_Cxb; // wx
        m_Yb(i).submat(m_nact,m_nact, 2*m_nact-1,2*m_nact-1) 
            = m_Cwb.t() * (m_metric * m_wxMb(i) * m_metric - double(1-i) * m_metric) * m_Cwb; // ww
    }

    // Construct transformed coefficients
    m_CXa.set_size(2); m_CXb.set_size(2);
    m_XCa.set_size(2); m_XCb.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        // Alpha space
        m_CXa(i).resize(m_nbsf,2*m_nact); m_CXa(i).zeros();
        m_CXa(i).cols(0,m_nact-1)        = m_wxMa(i).t() * m_metric * m_Cxa - (1-i) * m_Cxa; // x[CY]
        m_CXa(i).cols(m_nact,2*m_nact-1) = m_wxMa(i).t() * m_metric * m_Cwa;                 // w[CX]

        m_XCa(i).resize(m_nbsf,2*m_nact); m_XCa(i).zeros();
        m_XCa(i).cols(0,m_nact-1)        = m_wxMa(i) * m_metric * m_Cxa;                     // x[XC]
        m_XCa(i).cols(m_nact,2*m_nact-1) = m_wxMa(i) * m_metric * m_Cwa - (1-i) * m_Cwa;     // w[YC]

        // Beta space
        m_CXb(i).resize(m_nbsf,2*m_nact); m_CXb(i).zeros();
        m_CXb(i).cols(0,m_nact-1)        = m_wxMb(i).t() * m_metric * m_Cxb - (1-i) * m_Cxb; // x[CY]
        m_CXb(i).cols(m_nact,2*m_nact-1) = m_wxMb(i).t() * m_metric * m_Cwb;                 // w[CX]

        m_XCb(i).resize(m_nbsf,2*m_nact); m_XCb(i).zeros();
        m_XCb(i).cols(0,m_nact-1)        = m_wxMb(i) * m_metric * m_Cxb;                     // x[XC]
        m_XCb(i).cols(m_nact,2*m_nact-1) = m_wxMb(i) * m_metric * m_Cwb - (1-i) * m_Cwb;     // w[YC]
    }

 
    // TODO: Refactor these into oblivion!
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

    // Initiate transformed coefficient matrices 
    m_xCXa.set_size(4); m_wCXa.set_size(4);
    m_xCXb.set_size(4); m_wCXb.set_size(4);
    m_xXCa.set_size(4); m_wXCa.set_size(4);
    m_xXCb.set_size(4); m_wXCb.set_size(4);
    
    // Construct transformed coefficient matrices
    for(size_t i=0; i<2; i++)
    {
        m_xCXa(i) = m_Cxa.t() * m_metric * m_wxMa(i);
        m_xCXb(i) = m_Cxb.t() * m_metric * m_wxMb(i);
        m_wCXa(i) = m_Cwa.t() * m_metric * m_wxMa(i);
        m_wCXb(i) = m_Cwb.t() * m_metric * m_wxMb(i);
        m_xXCa(i) = m_wxMa(i) * m_metric * m_Cxa;
        m_xXCb(i) = m_wxMb(i) * m_metric * m_Cxb;
        m_wXCa(i) = m_wxMa(i) * m_metric * m_Cwa;
        m_wXCb(i) = m_wxMb(i) * m_metric * m_Cwb;

        m_xCXa(2+i) = (1-i) * m_Cxa.t() - m_Cxa.t() * m_metric * m_wxMa(i);
        m_xCXb(2+i) = (1-i) * m_Cxb.t() - m_Cxb.t() * m_metric * m_wxMb(i);
        m_wCXa(2+i) = (1-i) * m_Cwa.t() - m_Cwa.t() * m_metric * m_wxMa(i);
        m_wCXb(2+i) = (1-i) * m_Cwb.t() - m_Cwb.t() * m_metric * m_wxMb(i);
        m_xXCa(2+i) = (1-i) * m_Cxa - m_wxMa(i) * m_metric * m_Cxa;
        m_xXCb(2+i) = (1-i) * m_Cxb - m_wxMb(i) * m_metric * m_Cxb;
        m_wXCa(2+i) = (1-i) * m_Cwa - m_wxMa(i) * m_metric * m_Cwa;
        m_wXCb(2+i) = (1-i) * m_Cwb - m_wxMb(i) * m_metric * m_Cwb;
    }

    // Setup relevant one- and two-body terms 
    if(m_one_body) setup_one_body();
    if(m_two_body) setup_two_body();
}

template<typename Tc, typename Tf, typename Tb>
void wick<Tc,Tf,Tb>::spin_overlap(
    arma::umat xhp, arma::umat whp,
    Tc &S, bool alpha)
{
    // Ensure output is zero'd
    S = 0.0;

    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations

    // Get reference to number of zeros for this spin
    const size_t &nz = alpha ? m_nza : m_nzb; 

    // Check we don't have a non-zero element
    if(nz > nw + nx) return;

    // Get reference to relevant X/Y matrices for this spin
    const arma::field<arma::Mat<Tc> > &X = alpha ? m_Xa : m_Xb;
    const arma::field<arma::Mat<Tc> > &Y = alpha ? m_Ya : m_Yb;

    // Shift w indices
    // TODO: Do we want to keep this?
    whp += m_nact;

    // Get particle-hole indices
    arma::uvec rows, cols;
    if(nx == 0 xor nw == 0)
    {
        rows = (nx > 0) ? xhp.col(1) : whp.col(0);
        cols = (nx > 0) ? xhp.col(0) : whp.col(1);
    }
    else if(nx > 0 and nw > 0) 
    {
        rows = arma::join_cols(xhp.col(1),whp.col(0));
        cols = arma::join_cols(xhp.col(0),whp.col(1));
    }

    // Test the determinantal version
    if(nx == 0 and nw == 0)
    {   // No excitations, so return simple overlap
        S = (nz == 0) ? 1.0 : 0.0;
    }
    else if(nx+nw == 1)
    {   // One excitation doesn't require determinant
        S = X(nz)(rows(0),cols(0));
    }
    else
    {   // General case does require determinant
        // Construct matrix for no zero overlaps
        arma::Mat<Tc> D    = arma::trimatl(X(0).submat(rows,cols))
                           + arma::trimatu(Y(0).submat(rows,cols),1);
        // Construct matrix with all zero overlaps
        arma::Mat<Tc> Dbar = arma::trimatl(X(1).submat(rows,cols)) 
                           + arma::trimatu(Y(1).submat(rows,cols),1);

        // Distribute nz zeros among columns of D 
        // This corresponds to inserting nz columns of Dbar into D for every
        // permutations of the nz zeros.
        std::vector<size_t> m(nz, 1); m.resize(nx+nw, 0); 
        arma::Col<size_t> ind(&m[0], m.size(), false, true);
        do {
            S += arma::det(D * arma::diagmat(1-ind) + Dbar * arma::diagmat(ind));
        } while(std::prev_permutation(m.begin(), m.end()));
    }

    // Shift w indices
    // TODO: Do we want to keep this?
    whp -= m_nact;

    return;
}

template class wick<double, double, double>;
template class wick<std::complex<double>, double, double>;
template class wick<std::complex<double>, std::complex<double>, double>;
template class wick<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
