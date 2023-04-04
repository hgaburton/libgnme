#include <cassert>
#include <algorithm>
#include <libgnme/utils/eri_ao2mo.h>
#include <libgnme/utils/linalg.h>
#include "wick_base.h"

namespace libgnme {


template<typename Tc, typename Tf, typename Tb>
void wick_base<Tc,Tf,Tb>::same_spin_two_body(
    arma::umat xhp, arma::umat whp,
    Tc &V, bool alpha)
{
    // Zero the output
    V = 0.0;

    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations

    // Get reference to number of zeros for this spin
    const size_t &nz = alpha ? m_orba.m_nz : m_orbb.m_nz; 

    // Dimensions of multiple contractions
    size_t d = (nz > 0) ? 2 : 1;

    // Check we don't have a non-zero element
    if(nz > nw + nx + 2) return;

    // Get reference to relevant contractions
    const arma::field<arma::Mat<Tc> > &X = alpha ? m_orba.m_X : m_orbb.m_X;
    const arma::field<arma::Mat<Tc> > &Y = alpha ? m_orba.m_Y : m_orbb.m_Y;

    // Get reference to relevant zeroth-order term
    const arma::Col<Tc> &V0  = alpha ? m_two_body_int->Vaa 
                                     : m_two_body_int->Vbb;
    // Get reference to relevant J/K term
    const arma::field<arma::Mat<Tc> > &XVX = alpha ? m_two_body_int->XVaXa 
                                                   : m_two_body_int->XVbXb;
    // Get reference to relevant two-electron integrals
    arma::field<arma::Mat<Tc> > &II = alpha ? m_two_body_int->IIaa
                                            : m_two_body_int->IIbb;

    // TODO Correct indexing for new code
    const size_t nactx = alpha ? m_orba.m_refx.m_nact : m_orbb.m_refx.m_nact;
    const size_t nactw = alpha ? m_orba.m_refw.m_nact : m_orbb.m_refw.m_nact;
    const size_t nact = nactx + nactw;
    whp += nactx;

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

    // Generalised cases 
    // No excitations, so return simple overlap
    if(nx == 0 and nw == 0)
    {   
        V = V0(nz);
    }
    // One excitation doesn't require one-body determinant
    else if((nx+nw) == 1)
    {   
        // Distribute zeros over 3 contractions
        std::vector<size_t> m(nz, 1); m.resize(3,0);
        do {
            // Zeroth-order term
            V += V0(m[0] + m[1]) * X(m[2])(rows(0),cols(0));
            // First-order J/K term
            V -= 2.0 * XVX(m[0],m[1],m[2])(rows(0),cols(0));
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // Full generalisation!
    else
    {
        // Construct matrix for no zero overlaps
        arma::Mat<Tc> D  = arma::trimatl(X(0).submat(rows,cols))
                         + arma::trimatu(Y(0).submat(rows,cols),1);
        // Construct matrix with all zero overlaps
        arma::Mat<Tc> Db = arma::trimatl(X(1).submat(rows,cols)) 
                         + arma::trimatu(Y(1).submat(rows,cols),1);

        // Matrix of F contractions
        arma::field<arma::Mat<Tc> > JKtmp(d,d,d); 
        for(size_t i=0; i<d; i++)
        for(size_t j=0; j<d; j++)
        for(size_t k=0; k<d; k++)
            JKtmp(i,j,k) = XVX(i,j,k).submat(rows,cols);

        // Compute contribution from the overlap and zeroth term
        std::vector<size_t> m(nz, 1); m.resize(nx+nw+2, 0); 
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
                        II(2*m[2]+x, 2*m[0]+m[1]).colptr(nact*rows(i)+cols(j)), nact, nact, false, true);
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
    whp -= nactx;
}


template<typename Tc, typename Tf, typename Tb>
void wick_base<Tc,Tf,Tb>::diff_spin_two_body(
    arma::umat xahp, arma::umat xbhp,
    arma::umat wahp, arma::umat wbhp,
    Tc &V)
{
    // Zero the output
    V = 0.0;

    // Establish number of bra/ket excitations
    size_t nxa = xahp.n_rows; // Bra alpha excitations
    size_t nwa = wahp.n_rows; // Ket alpha excitations
    size_t nxb = xbhp.n_rows; // Bra beta excitations
    size_t nwb = wbhp.n_rows; // Ket beta excitations

    // Get references
    const size_t &nza = m_orba.m_nz;
    const size_t &nzb = m_orbb.m_nz;
    const arma::field<arma::Mat<Tc> > &Xa = m_orba.m_X;
    const arma::field<arma::Mat<Tc> > &Xb = m_orbb.m_X;
    const arma::field<arma::Mat<Tc> > &Ya = m_orba.m_Y;
    const arma::field<arma::Mat<Tc> > &Yb = m_orbb.m_Y;

    const arma::field<arma::Mat<Tc> > &XVaXb = m_two_body_int->XVaXb;
    const arma::field<arma::Mat<Tc> > &XVbXa = m_two_body_int->XVbXa;
    const arma::Mat<Tc> &Vab = m_two_body_int->Vab;
    arma::field<arma::Mat<Tc> > &IIab = m_two_body_int->IIab;
    arma::field<arma::Mat<Tc> > &IIba = m_two_body_int->IIba;

    // Dimensions of multiple contractions
    size_t da = (nza > 0) ? 2 : 1;
    size_t db = (nzb > 0) ? 2 : 1;

    // Check we don't have a non-zero element
    if(nza > nwa+nxa+1 || nzb > nwb+nxb+1) return;

    // TODO Correct indexing for new code
    const size_t nactxa = m_orba.m_refx.m_nact;
    const size_t nactxb = m_orba.m_refx.m_nact;
    const size_t nactwa = m_orbb.m_refw.m_nact;
    const size_t nactwb = m_orbb.m_refw.m_nact;
    const size_t nacta = nactxa + nactwa;
    const size_t nactb = nactxb + nactwb;
    wahp += nactxa;
    wbhp += nactxb;

    // Get alpha particle-hole indices
    arma::uvec rowa, cola;
    if(nxa == 0 xor nwa == 0) 
    {
        rowa = (nxa > 0) ? xahp.col(1) : wahp.col(0);
        cola = (nxa > 0) ? xahp.col(0) : wahp.col(1);
    } 
    else if(nxa > 0 and nwa > 0) 
    {
        rowa = arma::join_cols(xahp.col(1),wahp.col(0));
        cola = arma::join_cols(xahp.col(0),wahp.col(1));
    }
    // Get beta particle-hole indices
    arma::uvec rowb, colb;
    if(nxb == 0 xor nwb == 0)
    {
        rowb = (nxb > 0) ? xbhp.col(1) : wbhp.col(0);
        colb = (nxb > 0) ? xbhp.col(0) : wbhp.col(1);
    }
    else if(nxb > 0 and nwb > 0) 
    {
        rowb = arma::join_cols(xbhp.col(1),wbhp.col(0));
        colb = arma::join_cols(xbhp.col(0),wbhp.col(1));
    }

    /* Super generalised case */
    arma::Mat<Tc> Da, DaB;
    if(nxa+nwa == 1)
    {
        Da  = Xa(0)(rowa(0),cola(0));
        DaB = Xa(1)(rowa(0),cola(0));
    }
    else if(nxa+nwa > 1)
    {
        // Construct matrix for no zero overlaps
        Da  = arma::trimatl(Xa(0).submat(rowa,cola))
            + arma::trimatu(Ya(0).submat(rowa,cola),1);
        // Construct matrix with all zero overlaps
        DaB = arma::trimatl(Xa(1).submat(rowa,cola)) 
            + arma::trimatu(Ya(1).submat(rowa,cola),1);
    }
    arma::Mat<Tc> Db, DbB;
    if(nxb+nwb == 1)
    {
        Db  = Xb(0)(rowb(0),colb(0));
        DbB = Xb(1)(rowb(0),colb(0));
    }
    else if(nxb+nwb > 1)
    {
        // Construct matrix for no zero overlaps
        Db  = arma::trimatl(Xb(0).submat(rowb,colb))
            + arma::trimatu(Yb(0).submat(rowb,colb),1);
        // Construct matrix with all zero overlaps
        DbB = arma::trimatl(Xb(1).submat(rowb,colb)) 
            + arma::trimatu(Yb(1).submat(rowb,colb),1);
    }

    // Matrix of JK contractions
    arma::field<arma::Mat<Tc> > Jab(db,da,db), Jba(da,db,da);
    for(size_t i=0; i<da; i++)
    for(size_t j=0; j<da; j++)
    for(size_t k=0; k<db; k++)
        Jba(i,k,j) = XVbXa(i,k,j).submat(rowa,cola);
    for(size_t i=0; i<db; i++)
    for(size_t j=0; j<db; j++)
    for(size_t k=0; k<da; k++)
        Jab(i,k,j) = XVaXb(i,k,j).submat(rowb,colb);

    // Compute contribution from the overlap and zeroth term
    std::vector<size_t> ma(nza, 1); ma.resize(nxa+nwa+1, 0); 
    std::vector<size_t> mb(nzb, 1); mb.resize(nxb+nwb+1, 0); 
    arma::Col<size_t> inda1(&ma[1], nxa+nwa,   false, true);
    arma::Col<size_t> inda2(&ma[2], nxa+nwa-1, false, true);
    arma::Col<size_t> indb1(&mb[1], nxb+nwb,   false, true);
    arma::Col<size_t> indb2(&mb[2], nxb+nwb-1, false, true);
    // Loop over all possible contributions of zero overlaps
    arma::Mat<Tc> tmpDa, tmpDa2;
    arma::Mat<Tc> tmpDb, tmpDb2;
    do {
    do {
        // Evaluate overlap contribution
        tmpDa = Da * arma::diagmat(1-inda1) + DaB * arma::diagmat(inda1);
        tmpDb = Db * arma::diagmat(1-indb1) + DbB * arma::diagmat(indb1);
        
        // Get matrix adjoint and determinant
        Tc detDa, detDb;
        size_t nzDa, nzDb;
        arma::Mat<Tc> adjDa, adjDb; 
        adjoint_matrix(tmpDa, adjDa, detDa, nzDa);
        adjoint_matrix(tmpDb, adjDb, detDb, nzDb);
        adjDa = adjDa.t(); // Transpose makes row access easier later
        adjDb = adjDb.t(); // Transpose makes row access easier later
        
        // Get the zeroth-order contributions 
        V += Vab(ma[0],mb[0]) * detDa * detDb;

        // Get the effective one-body contribution
        // Loop over the alpha column swaps for contracted terms
        for(size_t i=0; i < nxa+nwa; i++)
        {   
            // Get replace column vector
            arma::Col<Tc> v1(Jba(ma[0],mb[0],ma[i+1]).colptr(i), nxa+nwa, false, true);
            arma::Col<Tc> v2(tmpDa.colptr(i), nxa+nwa, false, true);
            // Get relevant column from transposed inverse matrix
            arma::Col<Tc> a(adjDa.colptr(i), nxa+nwa, false, true); 
            // Perform determinant update formula
            V -= (detDa + arma::dot(v1-v2,a)) * detDb;
        }
        // Loop over the beta column swaps for contracted terms
        for(size_t i=0; i < nxb+nwb; i++)
        {   
            // Get replace column vector
            arma::Col<Tc> v1(Jab(mb[0],ma[0],mb[i+1]).colptr(i), nxb+nwb, false, true);
            arma::Col<Tc> v2(tmpDb.colptr(i), nxb+nwb, false, true);
            //arma::Col<Tc> v = Jab(mb[0],ma[0],mb[i+1]).col(i) - tmpDb.col(i);
            // Get relevant column from transposed inverse matrix
            arma::Col<Tc> a(adjDb.colptr(i), nxb+nwb, false, true); 
            // Perform determinant update formula
            V -= (detDb + arma::dot(v1-v2,a)) * detDa;
        }

        arma::field<arma::Mat<Tc> > IItmp(std::max(da,db));
        arma::Mat<Tc> D2, DB2;
        // Loop over alpha particle-hole pairs for two-body interaction
        for(size_t i=0; i < nxa+nwa; i++)
        for(size_t j=0; j < nxa+nwa; j++)
        {
            // Get temporary two-electron indices for this pair of electrons
            for(size_t x=0; x<db; x++)
            {
                arma::Mat<Tc> vIItmp(
                   IIba(2*mb[0]+x, 2*ma[0]+ma[1]).colptr(nacta*rowa(i)+cola(j)), nactb, nactb, false, true);
                IItmp(x) = vIItmp.submat(colb,rowb).st();
            }

            // New submatrices
            D2  = Da;   D2.shed_row(i);  D2.shed_col(j);
            DB2 = DaB; DB2.shed_row(i); DB2.shed_col(j);
            tmpDa2 = D2 * arma::diagmat(1-inda2) + DB2 * arma::diagmat(inda2);

            // Get determinants and inverses
            Tc detDa2 = arma::det(tmpDa2);

            // Get the phase factor
            double phase = (i % 2) xor (j % 2) ? -1.0 : 1.0;

            // Loop over beta columns
            for(size_t k=0; k < nxb+nwb; k++)
            {   
                // Get replace column vector
                arma::Col<Tc> v1(IItmp(mb[k+1]).colptr(k), nxb+nwb, false, true);
                arma::Col<Tc> v2(tmpDb.colptr(k), nxb+nwb, false, true);
                // Get relevant column from transposed inverse matrix
                arma::Col<Tc> a(adjDb.colptr(k), nxb+nwb, false, true); 
                // Perform determinant update formula
                V += 0.5 * phase * (detDb + arma::dot(v1-v2,a)) * detDa2;
           }
        }
        // Loop over beta particle-hole pairs for two-body interaction
        for(size_t i=0; i < nxb+nwb; i++)
        for(size_t j=0; j < nxb+nwb; j++)
        {
            // Get temporary two-electron indices for this pair of electrons
            for(size_t x=0; x<da; x++)
            {
                arma::Mat<Tc> vIItmp(
                    IIab(2*ma[0]+x, 2*mb[0]+mb[1]).colptr(nactb*rowb(i)+colb(j)), nacta, nacta, false, true);
                IItmp(x) = vIItmp.submat(cola,rowa).st();
            }

            // New submatrices
            D2  = Db;   D2.shed_row(i);  D2.shed_col(j);
            DB2 = DbB; DB2.shed_row(i); DB2.shed_col(j);
            tmpDb2 = D2 * arma::diagmat(1-indb2) + DB2 * arma::diagmat(indb2);

            // Get determinants and inverses
            Tc detDb2 = arma::det(tmpDb2);

            // Get the phase factor
            double phase = (i % 2) xor (j % 2) ? -1.0 : 1.0;

            // Loop over alpha columns
            for(size_t k=0; k < nxa+nwa; k++)
            {   
                // Get replace column vector
                arma::Col<Tc> v1(IItmp(ma[k+1]).colptr(k), nxa+nwa, false, true);
                arma::Col<Tc> v2(tmpDa.colptr(k), nxa+nwa, false, true);
                // Get relevant column from transposed inverse matrix
                arma::Col<Tc> a(adjDa.colptr(k), nxa+nwa, false, true); 
                // Perform determinant update formula
                V += 0.5 * phase * (detDa + arma::dot(v1-v2,a)) * detDb2;
            }
        }
    } while(std::prev_permutation(ma.begin(), ma.end()));
    } while(std::prev_permutation(mb.begin(), mb.end()));
    
    // TODO Correct indexing for old code
    wahp -= nactxa;
    wbhp -= nactxb;
}

} // namespace libgnme
