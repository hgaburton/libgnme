#include <cassert>
#include <algorithm>
#include "lowdin_pair.h"
#include "wick_rscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::add_one_body(arma::Mat<Tf> &F) 
{
    // Check input
    assert(F.n_rows == m_nbsf);
    assert(F.n_cols == m_nbsf);

    // Save a copy of matrices
    m_F = F;

    // Setup control variable to indicate one-body initialised
    m_one_body = true;
}


template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::setup_one_body()
{
    // Get dimensions needed for temporary arrays
    size_t d = (m_nz > 0) ? 2 : 1;

    // Construct 'F0' terms
    m_F0.resize(d); 
    for(size_t i=0; i<d; i++)
        m_F0(i) = arma::dot(m_F, m_wxM(i).st());

    // We only have to worry about
    //    xx[YFX]    xw[YFY]
    //    wx[XFX]    ww[XFY]
    // Construct the XFX super matrices
    m_XFX.set_size(d,d); 
    for(size_t i=0; i<d; i++)
    for(size_t j=0; j<d; j++)
        m_XFX(i,j) = m_CX(i).t() * m_F * m_XC(j);
}

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::spin_one_body(
    arma::umat xhp, arma::umat whp, Tc &F)
{
    // Ensure outputs are zero'd
    F = 0.0; 
    
    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations

    // Get dimensions of zero-contractions
    size_t dim = (m_nz > 0) ? 2 : 1; 

    // Check we don't have a non-zero element
    if(m_nz > nw + nx + 1) return;

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

    // Start with overlap contribution
    if(nx == 0 and nw == 0)
    {   // No excitations, so return simple overlap
        F = m_F0(m_nz);
    }
    else if((nx+nw) == 1)
    {   // One excitation doesn't require determinant
        // Distribute zeros over 2 contractions
        std::vector<size_t> m(m_nz, 1); m.resize(2, 0); 
        do {
            F += m_X(m[0])(rows(0),cols(0)) * m_F0(m[1]) - m_XFX(m[0],m[1])(rows(0),cols(0));
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    else
    {   // General case does require determinant
        // Construct matrix for no zero overlaps
        arma::Mat<Tc> D  = arma::trimatl(m_X(0).submat(rows,cols))
                         + arma::trimatu(m_Y(0).submat(rows,cols),1);
        // Construct matrix with all zero overlaps
        arma::Mat<Tc> Db = arma::trimatl(m_X(1).submat(rows,cols)) 
                         + arma::trimatu(m_Y(1).submat(rows,cols),1);

        // Matrix of F contractions
        arma::field<arma::Mat<Tc> > Ftmp(dim,dim); 
        for(size_t i=0; i<dim; i++)
        for(size_t j=0; j<dim; j++)
            Ftmp(i,j) = m_XFX(i,j).submat(rows,cols);

        // Compute contribution from the overlap and zeroth term
        std::vector<size_t> m(m_nz, 1); m.resize(nx+nw+1, 0); 
        arma::Col<size_t> ind(&m[1], nx+nw, false, true);
        // Loop over all possible contributions of zero overlaps
        do {
            // Evaluate overlap contribution
            arma::Mat<Tc> Dtmp = D * arma::diagmat(1-ind) + Db * arma::diagmat(ind);
            
            // Get the overlap contributions 
            F += m_F0(m[0]) * arma::det(Dtmp);
            
            // Loop over the column swaps for contracted terms
            for(size_t i=0; i < nx+nw; i++)
            {   
                // Take a safe copy of the column
                arma::Col<Tc> Dcol = Dtmp.col(i);
                // Make the swap
                Dtmp.col(i) = Ftmp(m[0],m[i+1]).col(i);
                // Add the one-body contribution
                F -= arma::det(Dtmp);
                // Restore the column
                Dtmp.col(i) = Dcol;
            }
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
