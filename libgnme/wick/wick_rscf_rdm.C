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


template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::same_spin_rdm2(
        arma::umat xhp, arma::umat whp, 
        arma::uvec xocc, arma::uvec wocc, 
        arma::Mat<Tc> &P)
{
    // Temporary RDM-2
    assert(P.n_rows == m_nmo * m_nmo);
    assert(P.n_cols == m_nmo * m_nmo);
    P.zeros();

    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations

    // Check we don't have a non-zero element
    if(m_orb.m_nz > nw + nx + 2) return;

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

    // No excitations, so only standard term
    if(nx == 0 and nw == 0)
    {
        // Compute contribution from the overlap and zeroth term
        std::vector<size_t> m(m_orb.m_nz, 1); m.resize(2, 0); 
        // Loop over all possible contributions of zero overlaps
        do {
            for(size_t ip=0; ip<xocc.n_elem; ip++)
            for(size_t ir=0; ir<ip; ir++)
            for(size_t iq=0; iq<wocc.n_elem; iq++)
            for(size_t is=0; is<iq; is++)
            {
                // Get occupied orbital indices
                size_t p = xocc(ip), r = xocc(ir), q = wocc(iq), s = wocc(is);

                // Increment contribution
                P(p*m_nmo+q, r*m_nmo+s) += m_orb.m_fX(m[0])(q+m_nmo,p) * m_orb.m_fX(m[1])(s+m_nmo,r)
                                         - m_orb.m_fX(m[0])(s+m_nmo,p) * m_orb.m_fX(m[1])(q+m_nmo,r);
            }
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    else if((nx+nw) == 1)
    {
        // Distribute zeros over 3 contractions
        std::vector<size_t> m(m_orb.m_nz, 1); m.resize(3,0);
        do {
            for(size_t ip=0; ip<xocc.n_elem; ip++)
            for(size_t ir=0; ir<ip; ir++)
            for(size_t iq=0; iq<wocc.n_elem; iq++)
            for(size_t is=0; is<iq; is++)
            {
                // Get occupied orbital indices
                size_t p = xocc(ip), r = xocc(ir), q = wocc(iq), s = wocc(is);

                // Zeroth-order term
                P(p*m_nmo+q, r*m_nmo+s) += m_orb.m_X(m[2])(rows(0), cols(0)) * 
                                         ( m_orb.m_fX(m[0])(q+m_nmo,p) * m_orb.m_fX(m[1])(s+m_nmo,r)
                                         - m_orb.m_fX(m[0])(s+m_nmo,p) * m_orb.m_fX(m[1])(q+m_nmo,r));

                // First-order J/K term
                P(p*m_nmo+q, r*m_nmo+s) -= m_orb.m_fX(m[0])(q+m_nmo,p) 
                                         * m_orb.m_Q(m[1])(s,cols(0)) * m_orb.m_R(m[2])(rows(0),r);
                P(p*m_nmo+q, r*m_nmo+s) -= m_orb.m_fX(m[0])(s+m_nmo,r) 
                                         * m_orb.m_Q(m[1])(q,cols(0)) * m_orb.m_R(m[2])(rows(0),p);
                P(p*m_nmo+q, r*m_nmo+s) += m_orb.m_fX(m[0])(s+m_nmo,p) 
                                         * m_orb.m_Q(m[1])(q,cols(0)) * m_orb.m_R(m[2])(rows(0),r);
                P(p*m_nmo+q, r*m_nmo+s) += m_orb.m_fX(m[0])(q+m_nmo,r) 
                                         * m_orb.m_Q(m[1])(s,cols(0)) * m_orb.m_R(m[2])(rows(0),p);
            }
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    else
    {   // Full generalisation!
        // Construct matrix for no zero overlaps
        arma::Mat<Tc> D  = arma::trimatl(m_orb.m_X(0).submat(rows,cols))
                         + arma::trimatu(m_orb.m_Y(0).submat(rows,cols),1);
        // Construct matrix with all zero overlaps
        arma::Mat<Tc> Db = arma::trimatl(m_orb.m_X(1).submat(rows,cols)) 
                         + arma::trimatu(m_orb.m_Y(1).submat(rows,cols),1);

        // Temporary matrices
        arma::field<arma::Mat<Tc> > R(2);
        R(0) = m_orb.m_R(0).rows(rows);
        R(1) = m_orb.m_R(1).rows(rows);
        arma::field<arma::Mat<Tc> > Q(2);
        Q(0) = m_orb.m_Q(0).cols(cols);
        Q(1) = m_orb.m_Q(1).cols(cols);

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

            // Get first-order contribution
            // Temporary 1RDM
            arma::Mat<Tc> P1tmp(m_nmo,m_nmo,arma::fill::zeros);

            // Loop over the column swaps for contracted terms
            for(size_t i=0; i < nx+nw; i++)
            {   
                // Get relevant column from transposed inverse matrix
                arma::Col<Tc> Qtmp(Q(m[1]).colptr(i), m_nmo, false, true);
                arma::Col<Tc> a(adjDtmp.colptr(i), nx+nw, false, true); 
                // Additional constant
                Tc shift = arma::dot(a, Dtmp.col(i));
                P1tmp -= Qtmp * (detDtmp - shift + a.t() * R(m[i+2]));
            }
            
            // Get the overlap contributions 
            for(size_t ip=0; ip<xocc.n_elem; ip++)
            for(size_t ir=0; ir<ip; ir++)
            for(size_t iq=0; iq<wocc.n_elem; iq++)
            for(size_t is=0; is<iq; is++)
            {
                // Occupied orbital indices
                size_t p = xocc(ip), r = xocc(ir), q = wocc(iq), s = wocc(is);

                // Zeroth-order term
                P(p*m_nmo+q, r*m_nmo+s) += detDtmp * 
                                        ( m_orb.m_fX(m[0])(q+m_nmo,p) * m_orb.m_fX(m[1])(s+m_nmo,r)
                                        - m_orb.m_fX(m[0])(s+m_nmo,p) * m_orb.m_fX(m[1])(q+m_nmo,r));

                // First-order term
                P(p*m_nmo+q, r*m_nmo+s) += m_orb.m_fX(m[0])(q+m_nmo,p) * P1tmp(s,r);
                P(p*m_nmo+q, r*m_nmo+s) += m_orb.m_fX(m[0])(s+m_nmo,r) * P1tmp(q,p);
                P(p*m_nmo+q, r*m_nmo+s) -= m_orb.m_fX(m[0])(q+m_nmo,r) * P1tmp(s,p);
                P(p*m_nmo+q, r*m_nmo+s) -= m_orb.m_fX(m[0])(s+m_nmo,p) * P1tmp(q,r);
            }

            arma::Mat<Tc> D2, Db2, Dtmp2;
            // Loop over particle-hole pairs for two-body interaction
            for(size_t i=0; i < nx+nw; i++)
            for(size_t a=0; a < nx+nw; a++)
            {
                // New submatrices
                D2  = D;   D2.shed_row(i);  D2.shed_col(a);
                Db2 = Db; Db2.shed_row(i); Db2.shed_col(a);
                Dtmp2 = D2 * arma::diagmat(1-ind2) + Db2 * arma::diagmat(ind2);

                arma::field<arma::Mat<Tc> > Rtmp(2), Qtmp(2);
                Rtmp(0) = R(0); Rtmp(0).shed_row(i);
                Rtmp(1) = R(1); Rtmp(1).shed_row(i);
                Qtmp(0) = Q(0); Qtmp(0).shed_col(a);
                Qtmp(1) = Q(1); Qtmp(1).shed_col(a);

                // Get matrix adjoint and determinant
                Tc detDtmp2;
                size_t nzero2;
                arma::Mat<Tc> adjDtmp2; 
                adjoint_matrix(Dtmp2, adjDtmp2, detDtmp2, nzero2);
                adjDtmp2 = adjDtmp2.t(); // Transpose makes row access faster

                // Get the phase factor
                double phase = (i % 2) xor (a % 2) ? -1.0 : 1.0;

                // Get first-order contribution
                // Temporary 1RDM
                arma::Mat<Tc> P2tmp(m_nmo,m_nmo,arma::fill::zeros);

                // Loop over the column swaps for contracted terms
                for(size_t j=0; j < nx+nw-1; j++)
                {   
                    // Get relevant column from transposed inverse matrix
                    arma::Col<Tc> Qvec(Qtmp(m[2]).colptr(j), m_nmo, false, true);
                    arma::Col<Tc> a(adjDtmp2.colptr(j), nx+nw-1, false, true); 
                    // Additional constant
                    Tc shift = arma::dot(a, Dtmp2.col(j));
                    P2tmp -= Qvec * (detDtmp2 - shift + a.t() * Rtmp(m[j+3]));
                }
            
                for(size_t ip=0; ip<xocc.n_elem; ip++)
                for(size_t ir=0; ir<ip; ir++)
                for(size_t iq=0; iq<wocc.n_elem; iq++)
                for(size_t is=0; is<iq; is++)
                {
                    // Occupied orbital indices
                    size_t p = xocc(ip), r = xocc(ir), q = wocc(iq), s = wocc(is);

                    // Zeroth-order term
                    P(p*m_nmo+q, r*m_nmo+s) -= 0.25 * phase * P2tmp(s,r)
                                         * m_orb.m_Q(m[0])(q,cols(a)) * m_orb.m_R(m[1])(rows(i),p);
                    P(p*m_nmo+q, r*m_nmo+s) -= 0.25 * phase * P2tmp(q,p) 
                                         * m_orb.m_Q(m[0])(s,cols(a)) * m_orb.m_R(m[1])(rows(i),r);
                    P(p*m_nmo+q, r*m_nmo+s) += 0.25 * phase * P2tmp(q,r) 
                                         * m_orb.m_Q(m[0])(s,cols(a)) * m_orb.m_R(m[1])(rows(i),p);
                    P(p*m_nmo+q, r*m_nmo+s) += 0.25 * phase * P2tmp(s,p)
                                         * m_orb.m_Q(m[0])(q,cols(a)) * m_orb.m_R(m[1])(rows(i),r);
                }
            }

        } while(std::prev_permutation(m.begin(), m.end()));

    }

    // Apply symmetries for the RDM2
    for(size_t ip=0; ip<xocc.n_elem; ip++)
    for(size_t ir=0; ir<ip; ir++)
    for(size_t iq=0; iq<wocc.n_elem; iq++)
    for(size_t is=0; is<iq; is++)
    {
        size_t p = xocc(ip), r = xocc(ir), q = wocc(iq), s = wocc(is);
        P(r*m_nmo+q, p*m_nmo+s) = - P(p*m_nmo+q, r*m_nmo+s);
        P(p*m_nmo+s, r*m_nmo+q) = - P(p*m_nmo+q, r*m_nmo+s);
        P(r*m_nmo+s, p*m_nmo+q) =   P(p*m_nmo+q, r*m_nmo+s);
    }
        
    whp -= m_nact;
}
    


template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::diff_spin_rdm2(
        arma::umat xahp, arma::umat xbhp, 
        arma::umat wahp, arma::umat wbhp, 
        arma::uvec xocca, arma::uvec xoccb, 
        arma::uvec wocca, arma::uvec woccb, 
        arma::Mat<Tc> &P1a, arma::Mat<Tc> &P1b, 
        arma::Mat<Tc> &P2)
{
    // Temporary RDM-2
    assert(P1a.n_rows == m_nmo);
    assert(P1a.n_cols == m_nmo);
    assert(P1b.n_rows == m_nmo);
    assert(P1b.n_cols == m_nmo);

    // Temporary RDM-2
    assert(P2.n_rows == m_nmo * m_nmo);
    assert(P2.n_cols == m_nmo * m_nmo);
    P2.zeros();

    // Use 1-RDM to compute different spin 2-RDM
    for(size_t ip=0; ip<xocca.n_elem; ip++)
    for(size_t ir=0; ir<xoccb.n_elem; ir++)
    for(size_t iq=0; iq<wocca.n_elem; iq++)
    for(size_t is=0; is<woccb.n_elem; is++)
    {
        size_t p = xocca(ip), r = xoccb(ir), q = wocca(iq), s = woccb(is);
        P2(p*m_nmo+q, r*m_nmo+s) += P1a(q,p) * P1b(s,r);
        P2(r*m_nmo+s, p*m_nmo+q) += P1a(q,p) * P1b(s,r);
    }
}

template class wick_rscf<double, double, double>;
template class wick_rscf<std::complex<double>, double, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
