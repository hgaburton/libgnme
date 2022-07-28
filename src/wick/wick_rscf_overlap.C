#include <cassert>
#include <algorithm>
#include "lowdin_pair.h"
#include "wick_rscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::setup_orbitals(arma::Mat<Tc> Cx, arma::Mat<Tc> Cw)
{
    setup_orbitals(Cx, Cw, 0, m_nmo);
}

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::setup_orbitals(
    arma::Mat<Tc> Cx, arma::Mat<Tc> Cw, 
    size_t ncore, size_t nactive) 
{
    // Store number of core and active orbitals
    m_nact = nactive;

    // Take a safe copy of active orbitals
    m_Cx = Cx.cols(ncore,ncore+m_nact-1);
    m_Cw = Cw.cols(ncore,ncore+m_nact-1);
    
    // Initialise reduced overlap and number of zero overlaps
    m_redS = 1.0;
    m_nz = 0;

    // Take copy of orbitals for Lowdin pairing
    arma::Mat<Tc> Cw_tmp(Cw.memptr(), m_nbsf, m_nelec, true, true);
    arma::Mat<Tc> Cx_tmp(Cx.memptr(), m_nbsf, m_nelec, true, true);

    // Lowdin Pair occupied orbitals
    arma::uvec zeros(m_nelec);
    arma::Col<Tc> Sxx(m_nelec, arma::fill::zeros);
    arma::Col<Tc> inv_Sxx(m_nelec, arma::fill::zeros); 
    lowdin_pair(Cx_tmp, Cw_tmp, Sxx, m_metric,1e-20);
    reduced_overlap(Sxx, inv_Sxx, m_redS, m_nz, zeros,1e-8);

    // Construct co-density
    m_wxM.set_size(2);
    m_wxM(0).set_size(m_nbsf,m_nbsf); m_wxM(0).zeros();
    m_wxM(1).set_size(m_nbsf,m_nbsf); m_wxM(1).zeros();

    // Construct M matrices
    m_wxM(0) = Cw_tmp * arma::diagmat(inv_Sxx) * Cx_tmp.t();
    for(size_t i=0; i < m_nz; i++)
        m_wxM(0) += Cx_tmp.col(zeros(i)) * Cx_tmp.col(zeros(i)).t();

    // Construct P matrices
    for(size_t i=0; i < m_nz; i++)
        m_wxM(1) += Cw_tmp.col(zeros(i)) * Cx_tmp.col(zeros(i)).t();

    // Initialise X contraction arrays
    m_X.set_size(2); m_X.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        // Initialise the memory
        m_X(i).set_size(2*m_nact,2*m_nact); m_X(i).zeros();

        // Compute terms
        m_X(i).submat(0,0, m_nact-1,m_nact-1)               
            = m_Cx.t() * m_metric * m_wxM(i) * m_metric * m_Cx; // xx
        m_X(i).submat(0,m_nact, m_nact-1,2*m_nact-1)        
            = m_Cx.t() * m_metric * m_wxM(i) * m_metric * m_Cw; // xw
        m_X(i).submat(m_nact,0, 2*m_nact-1,m_nact-1)        
            = m_Cw.t() * m_metric * m_wxM(i) * m_metric * m_Cx; // wx
        m_X(i).submat(m_nact,m_nact, 2*m_nact-1,2*m_nact-1) 
            = m_Cw.t() * m_metric * m_wxM(i) * m_metric * m_Cw; // ww
    }

    // Intialise Y contraction arrays
    m_Y.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        // Initialise the memory
        m_Y(i).set_size(2*m_nact,2*m_nact); m_Y(i).zeros();

        // Compute terms
        m_Y(i).submat(0,0, m_nact-1,m_nact-1)             
            = m_Cx.t() * (m_metric * m_wxM(i) * m_metric - double(1-i) * m_metric) * m_Cx; // xx
        m_Y(i).submat(0,m_nact, m_nact-1,2*m_nact-1)       
            = m_Cx.t() * (m_metric * m_wxM(i) * m_metric - double(1-i) * m_metric) * m_Cw; // xw
        m_Y(i).submat(m_nact,0, 2*m_nact-1,m_nact-1)       
            = m_Cw.t() * (m_metric * m_wxM(i) * m_metric - double(1-i) * m_metric) * m_Cx; // wx
        m_Y(i).submat(m_nact,m_nact, 2*m_nact-1,2*m_nact-1) 
            = m_Cw.t() * (m_metric * m_wxM(i) * m_metric - double(1-i) * m_metric) * m_Cw; // ww
    }

    // Construct transformed coefficients
    m_CX.set_size(2);
    m_XC.set_size(2);
    for(size_t i=0; i<2; i++)
    {
        // Alpha space
        m_CX(i).resize(m_nbsf,2*m_nact); m_CX(i).zeros();
        m_CX(i).cols(0,m_nact-1)        = m_wxM(i).t() * m_metric * m_Cx - (1-i) * m_Cx; // x[CY]
        m_CX(i).cols(m_nact,2*m_nact-1) = m_wxM(i).t() * m_metric * m_Cw;                // w[CX]

        m_XC(i).resize(m_nbsf,2*m_nact); m_XC(i).zeros();
        m_XC(i).cols(0,m_nact-1)        = m_wxM(i) * m_metric * m_Cx;                    // x[XC]
        m_XC(i).cols(m_nact,2*m_nact-1) = m_wxM(i) * m_metric * m_Cw - (1-i) * m_Cw;     // w[YC]
    }

    // Setup relevant one- and two-body terms 
    if(m_one_body) setup_one_body();
    if(m_two_body) setup_two_body();
}

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::spin_overlap(
    arma::umat xhp, arma::umat whp, Tc &S)
{
    // Ensure output is zero'd
    S = 0.0;

    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations

    // Check we don't have a non-zero element
    if(m_nz > nw + nx) return;

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
        S = (m_nz == 0) ? 1.0 : 0.0;
    }
    else if(nx+nw == 1)
    {   // One excitation doesn't require determinant
        S = m_X(m_nz)(rows(0),cols(0));
    }
    else
    {   // General case does require determinant
        // Construct matrix for no zero overlaps
        arma::Mat<Tc> D    = arma::trimatl(m_X(0).submat(rows,cols))
                           + arma::trimatu(m_Y(0).submat(rows,cols),1);
        // Construct matrix with all zero overlaps
        arma::Mat<Tc> Dbar = arma::trimatl(m_X(1).submat(rows,cols)) 
                           + arma::trimatu(m_Y(1).submat(rows,cols),1);

        // Distribute nz zeros among columns of D 
        // This corresponds to inserting nz columns of Dbar into D for every
        // permutations of the nz zeros.
        std::vector<size_t> m(m_nz, 1); m.resize(nx+nw, 0); 
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

template class wick_rscf<double, double, double>;
template class wick_rscf<std::complex<double>, double, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
