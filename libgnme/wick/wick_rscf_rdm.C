#include <cassert>
#include <algorithm>
#include <libgnme/utils/linalg.h>
#include "wick_rscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::spin_rdm1(
    arma::umat xhp, arma::umat whp, 
    arma::uvec xocc, arma::uvec wocc,
    arma::Mat<Tc> &P)
{
    // Temporary density matrix
    assert(P.n_rows == m_nmo);
    assert(P.n_cols == m_nmo);
    P.zeros();

    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations

    // Check we don't have a non-zero element
    if(m_orb.m_nz > nw + nx + 1) return;

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
        P.submat(wocc,xocc) = m_orb.m_wxP(m_orb.m_nz).submat(wocc,xocc);
    }
    else if((nx + nw) ==1)
    {   // One excitation is a special case
        std::vector<size_t> m(m_orb.m_nz, 1); m.resize(2, 0); 
        do {
            P.submat(wocc,xocc) += m_orb.m_wxP(m[0]).submat(wocc,xocc) * m_orb.m_X(m[1])(rows(0),cols(0))
                                 - m_orb.m_Q(m[0]).submat(wocc,cols) * m_orb.m_R(m[1]).submat(rows,xocc);
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    else
    {   // General case does require determinant
        // Construct matrix for no zero overlaps
        arma::Mat<Tc> D  = arma::trimatl(m_orb.m_X(0).submat(rows,cols))
                         + arma::trimatu(m_orb.m_Y(0).submat(rows,cols),1);
        // Construct matrix with all zero overlaps
        arma::Mat<Tc> Db = arma::trimatl(m_orb.m_X(1).submat(rows,cols)) 
                         + arma::trimatu(m_orb.m_Y(1).submat(rows,cols),1);

        // Temporary matrices
        arma::field<arma::Mat<Tc> > R(2);
        R(0) = m_orb.m_R(0).submat(rows,xocc);
        R(1) = m_orb.m_R(1).submat(rows,xocc);
        arma::field<arma::Mat<Tc> > Q(2);
        Q(0) = m_orb.m_Q(0).submat(wocc,cols);
        Q(1) = m_orb.m_Q(1).submat(wocc,cols);

        // Compute contribution from the overlap and zeroth term
        std::vector<size_t> m(m_orb.m_nz, 1); m.resize(nx+nw+1, 0); 
        arma::Col<size_t> ind(&m[1], nx+nw, false, true);
        // Loop over all possible contributions of zero overlaps
        do {
            // Evaluate overlap contribution
            arma::Mat<Tc> Dtmp = D * arma::diagmat(1-ind) + Db * arma::diagmat(ind);

            // Get matrix adjoint and determinant
            Tc detDtmp;
            size_t nzero;
            arma::Mat<Tc> adjDtmp; 
            adjoint_matrix(Dtmp, adjDtmp, detDtmp, nzero);
            adjDtmp = adjDtmp.t(); // Transpose makes row access easier later
            
            // Get the overlap contributions 
            P.submat(wocc,xocc) += m_orb.m_wxP(m[0]).submat(wocc,xocc) * detDtmp;
            
            // Loop over the column swaps for contracted terms
            for(size_t i=0; i < nx+nw; i++)
            {   
                // Get relevant column from transposed inverse matrix
                arma::Col<Tc> Qtmp(Q(m[0]).colptr(i), m_orb.m_nelec, false, true);
                arma::Col<Tc> a(adjDtmp.colptr(i), nx+nw, false, true); 
                // Additional constant
                Tc shift = arma::dot(a, Dtmp.col(i));
                P.submat(wocc,xocc) -= Qtmp * (detDtmp - shift + a.t() * R(m[i+1]));
            }
        } while(std::prev_permutation(m.begin(), m.end()));
    }

    whp -= m_nact;
}

template class wick_rscf<double, double, double>;
template class wick_rscf<std::complex<double>, double, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
