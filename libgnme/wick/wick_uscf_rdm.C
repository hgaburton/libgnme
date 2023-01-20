#include <cassert>
#include <algorithm>
#include <libgnme/utils/linalg.h>
#include "wick_uscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::spin_rdm1(
    arma::umat xhp, arma::umat whp, 
    arma::uvec xocc, arma::uvec wocc,
    arma::Mat<Tc> &P, bool alpha)
{
    // Temporary density matrix
    assert(P.n_rows == m_nmo);
    assert(P.n_cols == m_nmo);
    P.zeros();

    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations

    // Get reference to number of zeros for this spin
    const size_t &nz = alpha ? m_orb_a.m_nz : m_orb_b.m_nz; 
    const size_t &ne = alpha ? m_orb_a.m_nelec  : m_orb_b.m_nelec;

    // Check we don't have a non-zero element
    if(nz > nw + nx + 1) return;

    // Shift w indices
    // TODO: Do we want to keep this?
    whp += m_nact;
   
    // Get reference to relevant contractions
    const arma::field<arma::Mat<Tc> > &X = alpha ? m_orb_a.m_X : m_orb_b.m_X;
    const arma::field<arma::Mat<Tc> > &Y = alpha ? m_orb_a.m_Y : m_orb_b.m_Y;
    const arma::field<arma::Mat<Tc> > &Q = alpha ? m_orb_a.m_Q : m_orb_b.m_Q;
    const arma::field<arma::Mat<Tc> > &R = alpha ? m_orb_a.m_R : m_orb_b.m_R;
    const arma::field<arma::Mat<Tc> > &wxP = alpha ? m_orb_a.m_wxP : m_orb_b.m_wxP;

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
        P.submat(wocc,xocc) = wxP(nz).submat(wocc,xocc);
    }
    else if((nx + nw) ==1)
    {   // One excitation is a special case
        std::vector<size_t> m(nz, 1); m.resize(2, 0); 
        do {
            P.submat(wocc,xocc) += wxP(m[0]).submat(wocc,xocc) * X(m[1])(rows(0),cols(0))
                                   - Q(m[0]).submat(wocc,cols) * R(m[1]).submat(rows,xocc);
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    else
    {   // General case does require determinant
        // Construct matrix for no zero overlaps
        arma::Mat<Tc> D  = arma::trimatl(X(0).submat(rows,cols))
                         + arma::trimatu(Y(0).submat(rows,cols),1);
        // Construct matrix with all zero overlaps
        arma::Mat<Tc> Db = arma::trimatl(X(1).submat(rows,cols)) 
                         + arma::trimatu(Y(1).submat(rows,cols),1);

        // Temporary matrices
        arma::field<arma::Mat<Tc> > Rtmp(2);
        Rtmp(0) = R(0).submat(rows,xocc);
        Rtmp(1) = R(1).submat(rows,xocc);
        arma::field<arma::Mat<Tc> > Qtmp(2);
        Qtmp(0) = Q(0).submat(wocc,cols);
        Qtmp(1) = Q(1).submat(wocc,cols);

        // Compute contribution from the overlap and zeroth term
        std::vector<size_t> m(nz, 1); m.resize(nx+nw+1, 0); 
        arma::Col<size_t>   ind(&m[1], nx+nw, false, true);
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
            P.submat(wocc,xocc) += wxP(m[0]).submat(wocc,xocc) * detDtmp;
            
            // Loop over the column swaps for contracted terms
            for(size_t i=0; i < nx+nw; i++)
            {   
                // Get relevant column from transposed inverse matrix
                arma::Col<Tc> Qi(Qtmp(m[0]).colptr(i), ne, false, true);
                arma::Col<Tc> a(adjDtmp.colptr(i), nx+nw, false, true); 
                // Additional constant
                Tc shift = arma::dot(a, Dtmp.col(i));
                P.submat(wocc,xocc) -= Qi * (detDtmp - shift + a.t() * Rtmp(m[i+1]));
            }
        } while(std::prev_permutation(m.begin(), m.end()));
    }

    whp -= m_nact;
}

template class wick_uscf<double, double, double>;
template class wick_uscf<std::complex<double>, double, double>;
template class wick_uscf<std::complex<double>, std::complex<double>, double>;
template class wick_uscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
