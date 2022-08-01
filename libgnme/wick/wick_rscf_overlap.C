#include <cassert>
#include <algorithm>
#include <libgnme/utils/lowdin_pair.h>
#include "wick_rscf.h"

namespace libgnme {

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
    if(m_orb.m_nz > nw + nx) return;

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
        S = (m_orb.m_nz == 0) ? 1.0 : 0.0;
    }
    else if(nx+nw == 1)
    {   // One excitation doesn't require determinant
        S = m_orb.m_X(m_orb.m_nz)(rows(0),cols(0));
    }
    else
    {   // General case does require determinant
        // Construct matrix for no zero overlaps
        arma::Mat<Tc> D    = arma::trimatl(m_orb.m_X(0).submat(rows,cols))
                           + arma::trimatu(m_orb.m_Y(0).submat(rows,cols),1);
        // Construct matrix with all zero overlaps
        arma::Mat<Tc> Dbar = arma::trimatl(m_orb.m_X(1).submat(rows,cols)) 
                           + arma::trimatu(m_orb.m_Y(1).submat(rows,cols),1);

        // Distribute nz zeros among columns of D 
        // This corresponds to inserting nz columns of Dbar into D for every
        // permutations of the nz zeros.
        std::vector<size_t> m(m_orb.m_nz, 1); m.resize(nx+nw, 0); 
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
