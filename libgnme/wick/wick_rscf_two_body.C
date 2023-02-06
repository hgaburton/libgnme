#include <cassert>
#include <algorithm>
#include <libgnme/utils/eri_ao2mo.h>
#include <libgnme/utils/linalg.h>
#include "wick_rscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::add_two_body(arma::Mat<Tb> &V)
{
    // Check input
    assert(V.n_rows == m_nbsf * m_nbsf);
    assert(V.n_cols == m_nbsf * m_nbsf);

    // Setup control variable to indicate one-body initialised
    m_two_body = true;

    // Get dimensions of contractions
    size_t d = (m_orb.m_nz > 0) ? 2 : 1;

    // Initialise J/K matrices
    arma::field<arma::Mat<Tc> > J(d), K(d);
    for(size_t i=0; i<d; i++)
    {
        J(i).set_size(m_nbsf,m_nbsf); J(i).zeros();
        K(i).set_size(m_nbsf,m_nbsf); K(i).zeros();
    }

    // Construct J/K matrices in AO basis
    for(size_t m=0; m < m_nbsf; m++)
    for(size_t n=0; n < m_nbsf; n++)
    {
        # pragma omp parallel for schedule(static) collapse(2)
        for(size_t s=0; s < m_nbsf; s++)
        for(size_t t=0; t < m_nbsf; t++)
        {
            for(size_t i=0; i<d; i++)
            {
                // Coulomb matrices
                J(i)(s,t) += V(m*m_nbsf+n,s*m_nbsf+t) * m_orb.m_M(i)(n,m); 
                // Exchange matrices
                K(i)(s,t) += V(m*m_nbsf+t,s*m_nbsf+n) * m_orb.m_M(i)(n,m);
            }
        }
    }

    // Alpha-Alpha V terms
    m_Vsame.resize(3); m_Vsame.zeros();
    m_Vsame(0) = arma::dot(J(0).st() - K(0).st(), m_orb.m_M(0));
    if(m_orb.m_nz > 1)
    {
        m_Vsame(1) = 2.0 * arma::dot(J(0).st() - K(0).st(), m_orb.m_M(1));
        m_Vsame(2) = arma::dot(J(1).st() - K(1).st(), m_orb.m_M(1));
    }
    // Alpha-Beta V terms
    m_Vdiff.resize(d,d); m_Vdiff.zeros();
    for(size_t i=0; i<d; i++)
    for(size_t j=0; j<d; j++)
        m_Vdiff(i,j) = arma::dot(J(i).st(), m_orb.m_M(j));

    // Construct effective one-body terms
    // xx[Y(J-K)X]    xw[Y(J-K)Y]
    // wx[X(J-K)X]    ww[X(J-K)Y]
    m_XJX.set_size(d,d,d); 
    m_XKX.set_size(d,d,d); 
    for(size_t i=0; i<d; i++)
    for(size_t j=0; j<d; j++)
    for(size_t k=0; k<d; k++)
    {
        m_XJX(i,k,j) = m_orb.m_CX(i).t() * J(k) * m_orb.m_XC(j);
        m_XKX(i,k,j) = m_orb.m_CX(i).t() * K(k) * m_orb.m_XC(j);
    }

    // Build the two-electron integrals
    // Bra: xY    wX
    // Ket: xX    wY
    m_IIsame.set_size(d*d,d*d);
    m_IIdiff.set_size(d*d,d*d); 
    for(size_t i=0; i<d; i++)
    for(size_t j=0; j<d; j++)
    for(size_t k=0; k<d; k++)
    for(size_t l=0; l<d; l++)
    {
        // Initialise the memory
        m_IIsame(2*i+j, 2*k+l).resize(4*m_nact*m_nact, 4*m_nact*m_nact); 
        m_IIdiff(2*i+j, 2*k+l).resize(4*m_nact*m_nact, 4*m_nact*m_nact); 
        // Construct two-electron integrals
        eri_ao2mo_split(m_orb.m_CX(i), m_orb.m_XC(j), m_orb.m_CX(k), m_orb.m_XC(l), 
                        V, m_IIdiff(2*i+j, 2*k+l), m_IIsame(2*i+j, 2*k+l), 2*m_nact, true); 
        m_IIsame(2*i+j, 2*k+l) += m_IIdiff(2*i+j, 2*k+l);
    }
}

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::same_spin_two_body(
    arma::umat xhp, arma::umat whp, Tc &V)
{
    // Zero the output
    V = 0.0;

    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations

    // Dimensions of multiple contractions
    size_t d = (m_orb.m_nz > 0) ? 2 : 1;

    // Check we don't have a non-zero element
    if(m_orb.m_nz > nw + nx + 2) return;

    // Get reference to relevant zeroth-order term
    const arma::Col<Tc> &V0  = m_Vsame;

    // TODO Correct indexing for new code
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

    /* Generalised cases */
    // No excitations, so return simple overlap
    if(nx == 0 and nw == 0)
    {   
        V = V0(m_orb.m_nz);
    }
    // One excitation doesn't require one-body determinant
    else if((nx+nw) == 1)
    {   
        // Distribute zeros over 3 contractions
        std::vector<size_t> m(m_orb.m_nz, 1); m.resize(3,0);
        do {
            // Zeroth-order term
            V += V0(m[0] + m[1]) * m_orb.m_X(m[2])(rows(0),cols(0));
            // First-order J/K term
            V -= 2.0 * m_XJX(m[0],m[1],m[2])(rows(0),cols(0));
            V += 2.0 * m_XKX(m[0],m[1],m[2])(rows(0),cols(0));
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // Full generalisation!
    else
    {
        // Construct matrix for no zero overlaps
        arma::Mat<Tc> D  = arma::trimatl(m_orb.m_X(0).submat(rows,cols))
                         + arma::trimatu(m_orb.m_Y(0).submat(rows,cols),1);
        // Construct matrix with all zero overlaps
        arma::Mat<Tc> Db = arma::trimatl(m_orb.m_X(1).submat(rows,cols)) 
                         + arma::trimatu(m_orb.m_Y(1).submat(rows,cols),1);

        // Matrix of F contractions
        arma::field<arma::Mat<Tc> > JKtmp(d,d,d); 
        for(size_t i=0; i<d; i++)
        for(size_t j=0; j<d; j++)
        for(size_t k=0; k<d; k++)
            JKtmp(i,j,k) = m_XJX(i,j,k).submat(rows,cols) 
                         - m_XKX(i,j,k).submat(rows,cols);

        // Compute contribution from the overlap and zeroth term
        std::vector<size_t> m(m_orb.m_nz, 1); m.resize(nx+nw+2, 0); 
        arma::Col<size_t> ind1(&m[2], nx+nw, false, true);
        arma::Col<size_t> ind2(&m[3], nx+nw-1, false, true);
        // Loop over all possible contributions of zero overlaps
        do {
            // Evaluate overlap contribution
            arma::Mat<Tc> Dtmp = D * arma::diagmat(1-ind1) + Db * arma::diagmat(ind1);

            // Get matrix adjoint and determinant
            Tc detDtmp;
            size_t nzero;
            arma::Mat<Tc> adjDtmp; 
            adjoint_matrix(Dtmp, adjDtmp, detDtmp, nzero);
            adjDtmp = adjDtmp.t(); // Transpose makes row access easier later
            
            // Get the overlap contributions 
            V += V0(m[0]+m[1]) * detDtmp;
            
            // Get the effective one-body contribution
            // Loop over the column swaps for contracted terms
            for(size_t i=0; i < nx+nw; i++)
            {   
                // Get replace column vector
                arma::Col<Tc> v1(JKtmp(m[0],m[1],m[i+2]).colptr(i), nx+nw, false, true);
                arma::Col<Tc> v2(Dtmp.colptr(i), nx+nw, false, true);
                // Get relevant column from transposed inverse matrix
                arma::Col<Tc> a(adjDtmp.colptr(i), nx+nw, false, true); 
                // Perform Shermann-Morrison style update
                V -= 2.0 * (detDtmp + arma::dot(v1 - v2, a));
            }

            arma::field<arma::Mat<Tc> > IItmp(d);
            arma::Mat<Tc> D2, Db2, Dtmp2;
            // Loop over particle-hole pairs for two-body interaction
            for(size_t i=0; i < nx+nw; i++)
            for(size_t j=0; j < nx+nw; j++)
            {
                // Get temporary two-electron indices for this pair of electrons
                for(size_t x=0; x < d; x++)
                {
                    arma::Mat<Tc> vIItmp(
                        m_IIsame(2*m[2]+x, 2*m[0]+m[1]).colptr(2*m_nact*rows(i)+cols(j)), 
                        2*m_nact, 2*m_nact, false, true);
                    IItmp(x) = vIItmp.submat(cols,rows).st();
                    IItmp(x).shed_row(i); 
                    IItmp(x).shed_col(j);
                }

                // New submatrices
                D2  = D;   D2.shed_row(i);  D2.shed_col(j);
                Db2 = Db; Db2.shed_row(i); Db2.shed_col(j);
                Dtmp2 = D2 * arma::diagmat(1-ind2) + Db2 * arma::diagmat(ind2);

                // Get matrix adjoint and determinant
                Tc detDtmp2;
                size_t nzero2;
                arma::Mat<Tc> adjDtmp2; 
                adjoint_matrix(Dtmp2, adjDtmp2, detDtmp2, nzero2);
                adjDtmp2 = adjDtmp2.t(); // Transpose makes row access faster

                // Get the phase factor
                double phase = (i % 2) xor (j % 2) ? -1.0 : 1.0;

                // Loop over remaining columns
                for(size_t k=0; k < nx+nw-1; k++)
                {   
                    // Get replace column vector
                    arma::Col<Tc> v1(IItmp(m[k+3]).colptr(k), nx+nw-1, false, true);
                    arma::Col<Tc> v2(Dtmp2.colptr(k), nx+nw-1, false, true);
                    // Get relevant column from transposed inverse matrix
                    arma::Col<Tc> a(adjDtmp2.colptr(k), nx+nw-1, false, true); 
                    // Perform Shermann-Morrison style update
                    V += 0.5 * phase * (detDtmp2 + arma::dot(v1-v2, a));
                }
            }
        } while(std::prev_permutation(m.begin(), m.end()));
    }

    // TODO Correct indexing for old code
    whp -= m_nact;
}



template class wick_rscf<double, double, double>;
template class wick_rscf<std::complex<double>, double, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
