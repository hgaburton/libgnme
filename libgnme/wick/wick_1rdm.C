#include <cassert>
#include <algorithm>
#include <libgnme/utils/lowdin_pair.h>
#include "wick.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick<Tc,Tf,Tb>::spin_1rdm(
    arma::umat &xhp, arma::umat &whp, arma::Mat<Tc> &P, bool alpha)
{
    // Resize and zero output
    P.resize(m_nbsf, m_nbsf); P.zeros();
    
    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations

    // Inform if we can't handle that excitation
    if(nx > 2 || nw > 2 || (nx+nw) > 4)
    {
        std::cout << "wick::spin_1rdm: Bra excitations = " << nx << std::endl;
        std::cout << "wick::spin_1rdm: Ket excitations = " << nw << std::endl;
        throw std::runtime_error("wick::spin_1rdm: Requested excitation level not yet implemented");
    }

    // Get reference to relevant X matrices for this spin
    const arma::field<arma::Mat<Tc> > &wwX = alpha ? m_wwXa : m_wwXb;
    const arma::field<arma::Mat<Tc> > &wxX = alpha ? m_wxXa : m_wxXb;
    const arma::field<arma::Mat<Tc> > &xwX = alpha ? m_xwXa : m_xwXb;
    const arma::field<arma::Mat<Tc> > &xxX = alpha ? m_xxXa : m_xxXb;

    // Get reference to transformed coefficients
    const arma::field<arma::Mat<Tc> > &xXC = alpha ? m_xXCa : m_xXCb;
    const arma::field<arma::Mat<Tc> > &xCX = alpha ? m_xCXa : m_xCXb;
    const arma::field<arma::Mat<Tc> > &wXC = alpha ? m_wXCa : m_wXCb;
    const arma::field<arma::Mat<Tc> > &wCX = alpha ? m_wCXa : m_wCXb;

    // Get reference to relevant co-density matrix
    const arma::field<arma::Mat<Tc> > &wxM = alpha ? m_wxMa : m_wxMb;

    // Get reference to number of zeros for this spin
    const size_t &nz = alpha ? m_nza : m_nzb; 

    // Check we don't have a non-zero element
    if(nz > nw + nx + 1) return;

    // < X | F | W >
    if(nx == 0 and nw == 0)
    {
        P = wxM(nz);
    }
    // < X_i^a | F | W >
    else if(nx == 1 and nw == 0)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        // Distribute the NZ zeros among 2 contractions
        std::vector<size_t> m(nz, 1); m.resize(2, 0); 
        do {
            P += xxX(m[0])(a,i) * wxM(m[1]) + xXC(m[0]).col(i) * xCX(2+m[1]).row(a);
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X | F | W_i^a >
    else if(nx == 0 and nw == 1)
    {
        size_t i = whp(0,0), a = whp(0,1);
        // Distribute the NZ zeros among 2 contractions
        std::vector<size_t> m(nz, 1); m.resize(2, 0); 
        do {
            P += wwX(m[0])(i,a) * wxM(m[1]) + wXC(2+m[0]).col(a) * wCX(m[1]).row(i);
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X_i^a | F | W_j^b>
    else if(nx == 1 and nw == 1)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        size_t j = whp(0,0), b = whp(0,1);
        // Distribute the NZ zeros among 3 contractions
        std::vector<size_t> m(nz, 1); m.resize(3, 0); 
        do {
            P += wxM(m[0]) * (xxX(m[1])(a,i) * wwX(m[2])(j,b) + wxX(m[1])(j,i) * xwX(2+m[2])(a,b))
               + xXC(0+m[0]).col(i) * xCX(2+m[1]).row(a) * wwX(m[2])(j,b) + wXC(2+m[0]).col(b) * xCX(2+m[1]).row(a) * wxX(0+m[2])(j,i)
               + wXC(2+m[0]).col(b) * wCX(0+m[1]).row(j) * xxX(m[2])(a,i) - xXC(0+m[0]).col(i) * wCX(0+m[1]).row(j) * xwX(2+m[2])(a,b);
        } while(std::prev_permutation(m.begin(), m.end()));
    } 
    // < X_ij^ab | F | W > 
    else if(nx == 2 and nw == 0)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        size_t j = xhp(1,0), b = xhp(1,1);
        // Distribute the NZ zeros among 3 contractions
        std::vector<size_t> m(nz, 1); m.resize(3, 0); 
        do {
            P += wxM(m[0]) * (xxX(m[1])(a,i) * xxX(m[2])(b,j) - xxX(m[1])(a,j) * xxX(m[2])(b,i))
               + (xXC(0+m[0]).col(i) * xCX(2+m[1]).row(a) * xxX(m[2])(b,j) - xXC(m[0]).col(j) * xCX(2+m[1]).row(a) * xxX(m[2])(b,i))
               + (xXC(0+m[0]).col(j) * xCX(2+m[1]).row(b) * xxX(m[2])(a,i) - xXC(m[0]).col(i) * xCX(2+m[1]).row(b) * xxX(m[2])(a,j));
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X | F | W_ij^ab > 
    else if(nx == 0 and nw == 2)
    {
        size_t i = whp(0,0), a = whp(0,1);
        size_t j = whp(1,0), b = whp(1,1);
        // Distribute the NZ zeros among 3 contractions
        std::vector<size_t> m(nz, 1); m.resize(3, 0); 
        do {
            P += wxM(m[0]) * (wwX(m[1])(i,a) * wwX(m[2])(j,b) - wwX(m[1])(i,b) * wwX(m[2])(j,a))
               + (wXC(2+m[0]).col(b) * wCX(m[1]).row(j) * wwX(m[2])(i,a) - wXC(2+m[0]).col(a) * wCX(m[1]).row(j) * wwX(m[2])(i,b))
               + (wXC(2+m[0]).col(a) * wCX(m[0]).row(i) * wwX(m[2])(j,b) - wXC(2+m[0]).col(b) * wCX(m[1]).row(i) * wwX(m[2])(j,a));
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X_ij^ab | F | W_k^c>
    else if(nx == 2 and nw == 1)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        size_t j = xhp(1,0), b = xhp(1,1);
        size_t k = whp(0,0), c = whp(0,1);
        // Distribute the NZ zeros among 4 contractions
        std::vector<size_t> m(nz, 1); m.resize(4, 0); 
        do {
            P += (wxM(m[0]) * wwX(0+m[1])(k,c) + wXC(2+m[0]).col(c) * wCX(0+m[1]).row(k)) * (xxX(m[2])(a,i) * xxX(m[3])(b,j) - xxX(m[2])(b,i) * xxX(m[3])(a,j))
               + (wxM(m[0]) * xwX(2+m[1])(a,c) + wXC(2+m[0]).col(c) * xCX(2+m[1]).row(a)) * (xxX(m[2])(b,j) * wxX(m[3])(k,i) - xxX(m[2])(b,i) * wxX(m[3])(k,j))
               + (wxM(m[0]) * xwX(2+m[1])(b,c) + wXC(2+m[0]).col(c) * xCX(2+m[1]).row(b)) * (xxX(m[2])(a,i) * wxX(m[3])(k,j) - xxX(m[2])(a,j) * wxX(m[3])(k,i))
               + xXC(m[0]).col(j) * xCX(2+m[1]).row(b) * (xxX(m[2])(a,i) * wwX(0+m[3])(k,c) + wxX(m[2])(k,i) * xwX(2+m[3])(a,c))
               - xXC(m[0]).col(j) * xCX(2+m[1]).row(a) * (xxX(m[2])(b,i) * wwX(0+m[3])(k,c) + wxX(m[2])(k,i) * xwX(2+m[3])(b,c))
               + xXC(m[0]).col(i) * xCX(2+m[1]).row(a) * (xxX(m[2])(b,j) * wwX(0+m[3])(k,c) + wxX(m[2])(k,j) * xwX(2+m[3])(b,c))
               - xXC(m[0]).col(i) * xCX(2+m[1]).row(b) * (xxX(m[2])(a,j) * wwX(0+m[3])(k,c) + wxX(m[2])(k,j) * xwX(2+m[3])(a,c))
               + xXC(m[0]).col(j) * wCX(0+m[1]).row(k) * (xxX(m[2])(b,i) * xwX(2+m[3])(a,c) - xxX(m[2])(a,i) * xwX(2+m[3])(b,c))
               + xXC(m[0]).col(i) * wCX(0+m[1]).row(k) * (xxX(m[2])(a,j) * xwX(2+m[3])(b,c) - xxX(m[2])(b,j) * xwX(2+m[3])(a,c));
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X_k^c | F | W_ij^ab>
    else if(nx == 1 and nw == 2)
    {
        size_t i = whp(0,0), a = whp(0,1);
        size_t j = whp(1,0), b = whp(1,1);
        size_t k = xhp(0,0), c = xhp(0,1);
        // Distribute the NZ zeros among 4 contractions
        std::vector<size_t> m(nz, 1); m.resize(4, 0); 
        do {
            P += (wxM(m[0]) * xxX(0+m[1])(c,k) + xXC(0+m[0]).col(k) * xCX(2+m[1]).row(c)) * (wwX(m[2])(i,a) * wwX(m[3])(j,b) - wwX(m[2])(i,b) * wwX(m[3])(j,a))
               + (wxM(m[0]) * xwX(2+m[1])(c,a) + wXC(2+m[0]).col(a) * xCX(2+m[1]).row(c)) * (wwX(m[2])(j,b) * wxX(m[3])(i,k) - wwX(m[2])(i,b) * wxX(m[3])(j,k))
               + (wxM(m[0]) * xwX(2+m[1])(c,b) + wXC(2+m[0]).col(b) * xCX(2+m[1]).row(c)) * (wwX(m[2])(i,a) * wxX(m[3])(j,k) - wwX(m[2])(j,a) * wxX(m[3])(i,k))
               + wXC(2+m[0]).col(b) * wCX(m[1]).row(j) * (wwX(m[2])(i,a) * xxX(0+m[3])(c,k) + wxX(m[2])(i,k) * xwX(2+m[3])(c,a))
               - wXC(2+m[0]).col(a) * wCX(m[1]).row(j) * (wwX(m[2])(i,b) * xxX(0+m[3])(c,k) + wxX(m[2])(i,k) * xwX(2+m[3])(c,b))
               + wXC(2+m[0]).col(a) * wCX(m[1]).row(i) * (wwX(m[2])(j,b) * xxX(0+m[3])(c,k) + wxX(m[2])(j,k) * xwX(2+m[3])(c,b))
               - wXC(2+m[0]).col(b) * wCX(m[1]).row(i) * (wwX(m[2])(j,a) * xxX(0+m[3])(c,k) + wxX(m[2])(j,k) * xwX(2+m[3])(c,a))
               + xXC(0+m[0]).col(k) * wCX(m[1]).row(j) * (wwX(m[2])(i,b) * xwX(2+m[3])(c,a) - wwX(m[2])(i,a) * xwX(2+m[3])(c,b))
               + xXC(0+m[0]).col(k) * wCX(m[1]).row(i) * (wwX(m[2])(j,a) * xwX(2+m[3])(c,b) - wwX(m[2])(j,b) * xwX(2+m[3])(c,a));
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X_ij^ab | F | W_kl^cd>
    else if(nx == 2 and nw == 2)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        size_t j = xhp(1,0), b = xhp(1,1);
        size_t k = whp(0,0), c = whp(0,1);
        size_t l = whp(1,0), d = whp(1,1);
        // Distribute the NZ zeros among 5 contractions
        std::vector<size_t> m(nz, 1); m.resize(5, 0); 
        do {
            // Normal-order overlap term
            P += wxM(m[4]) * ( 
                   (xxX(0+m[0])(a,i) * xxX(0+m[1])(b,j) - xxX(0+m[0])(a,j) * xxX(0+m[1])(b,i)) * (wwX(m[2])(k,c) * wwX(m[3])(l,d) - wwX(m[2])(k,d) * wwX(m[3])(l,c))
                 + (xwX(2+m[0])(a,c) * xxX(0+m[1])(b,j) - xwX(2+m[0])(b,c) * xxX(0+m[1])(a,j)) * (wxX(m[2])(k,i) * wwX(m[3])(l,d) - wxX(m[2])(l,i) * wwX(m[3])(k,d))
                 + (xwX(2+m[0])(a,d) * xxX(0+m[1])(b,j) - xwX(2+m[0])(b,d) * xxX(0+m[1])(a,j)) * (wxX(m[2])(l,i) * wwX(m[3])(k,c) - wxX(m[2])(k,i) * wwX(m[3])(l,c))
                 + (xwX(2+m[0])(a,c) * xxX(0+m[1])(b,i) - xwX(2+m[0])(b,c) * xxX(0+m[1])(a,i)) * (wxX(m[2])(l,j) * wwX(m[3])(k,d) - wxX(m[2])(k,j) * wwX(m[3])(l,d))
                 + (xwX(2+m[0])(a,d) * xxX(0+m[1])(b,i) - xwX(2+m[0])(b,d) * xxX(0+m[1])(a,i)) * (wxX(m[2])(k,j) * wwX(m[3])(l,c) - wxX(m[2])(l,j) * wwX(m[3])(k,c))
                 + (xwX(2+m[0])(a,c) * xwX(2+m[1])(b,d) - xwX(2+m[0])(b,c) * xwX(2+m[1])(a,d)) * (wxX(m[2])(k,i) * wxX(m[3])(l,j) - wxX(m[2])(l,i) * wxX(m[3])(k,j)) );
            // Remaining Terms
            P += xXC(0+m[0]).col(i) * xCX(2+m[1]).row(a) * ( xxX(m[2])(b,j) * ( wwX(m[3])(k,c) * wwX(0+m[4])(l,d) - wwX(m[3])(k,d) * wwX(0+m[4])(l,c) )
                                                           + wxX(m[2])(k,j) * ( wwX(m[3])(l,d) * xwX(2+m[4])(b,c) - wwX(m[3])(l,c) * xwX(2+m[4])(b,d) )
                                                           + wxX(m[2])(l,j) * ( wwX(m[3])(k,c) * xwX(2+m[4])(b,d) - wwX(m[3])(k,d) * xwX(2+m[4])(b,c) ) )
               + xXC(0+m[0]).col(j) * xCX(2+m[1]).row(a) * ( xxX(m[2])(b,i) * ( wwX(m[3])(k,d) * wwX(0+m[4])(l,c) - wwX(m[3])(k,c) * wwX(0+m[4])(l,d) )
                                                           + wxX(m[2])(k,i) * ( wwX(m[3])(l,c) * xwX(2+m[4])(b,d) - wwX(m[3])(l,d) * xwX(2+m[4])(b,c) )
                                                           + wxX(m[2])(l,i) * ( wwX(m[3])(k,d) * xwX(2+m[4])(b,c) - wwX(m[3])(k,c) * xwX(2+m[4])(b,d) ) )
               + xXC(0+m[0]).col(i) * xCX(2+m[1]).row(b) * ( xxX(m[2])(a,j) * ( wwX(m[3])(k,d) * wwX(0+m[4])(l,c) - wwX(m[3])(k,c) * wwX(0+m[4])(l,d) )
                                                           + wxX(m[2])(k,j) * ( wwX(m[3])(l,c) * xwX(2+m[4])(a,d) - wwX(m[3])(l,d) * xwX(2+m[4])(a,c) )
                                                           + wxX(m[2])(l,j) * ( wwX(m[3])(k,d) * xwX(2+m[4])(a,c) - wwX(m[3])(k,c) * xwX(2+m[4])(a,d) ) )
               + xXC(0+m[0]).col(j) * xCX(2+m[1]).row(b) * ( xxX(m[2])(a,i) * ( wwX(m[3])(k,c) * wwX(0+m[4])(l,d) - wwX(m[3])(k,d) * wwX(0+m[4])(l,c) )
                                                           + wxX(m[2])(k,i) * ( wwX(m[3])(l,d) * xwX(2+m[4])(a,c) - wwX(m[3])(l,c) * xwX(2+m[4])(a,d) )
                                                           + wxX(m[2])(l,i) * ( wwX(m[3])(k,c) * xwX(2+m[4])(a,d) - wwX(m[3])(k,d) * xwX(2+m[4])(a,c) ) );

            P += wXC(2+m[0]).col(c) * wCX(0+m[1]).row(l) * ( xxX(m[2])(b,i) * (xxX(m[3])(a,j) * wwX(0+m[4])(k,d) + wxX(m[3])(k,j) * xwX(2+m[4])(a,d))
                                                           - xxX(m[2])(a,i) * (xxX(m[3])(b,j) * wwX(0+m[4])(k,d) + wxX(m[3])(k,j) * xwX(2+m[4])(b,d))
                                                           + wxX(m[2])(k,i) * (xxX(m[3])(a,j) * xwX(2+m[4])(b,d) - xxX(m[3])(b,j) * xwX(2+m[4])(a,d)) )
               + wXC(2+m[0]).col(d) * wCX(0+m[1]).row(l) * ( xxX(m[2])(a,i) * (xxX(m[3])(b,j) * wwX(0+m[4])(k,c) + wxX(m[3])(k,j) * xwX(2+m[4])(b,c))
                                                           - xxX(m[2])(b,i) * (xxX(m[3])(a,j) * wwX(0+m[4])(k,c) + wxX(m[3])(k,j) * xwX(2+m[4])(a,c))
                                                           + wxX(m[2])(k,i) * (xxX(m[3])(b,j) * xwX(2+m[4])(a,c) - xxX(m[3])(a,j) * xwX(2+m[4])(b,c)) )
               + wXC(2+m[0]).col(c) * wCX(0+m[1]).row(k) * ( xxX(m[2])(a,i) * (xxX(m[3])(b,j) * wwX(0+m[4])(l,d) + wxX(m[3])(l,j) * xwX(2+m[4])(b,d))
                                                           - xxX(m[2])(b,i) * (xxX(m[3])(a,j) * wwX(0+m[4])(l,d) + wxX(m[3])(l,j) * xwX(2+m[4])(a,d))
                                                           + wxX(m[2])(l,i) * (xxX(m[3])(b,j) * xwX(2+m[4])(a,d) - xxX(m[3])(a,j) * xwX(2+m[4])(b,d)) )
               + wXC(2+m[0]).col(d) * wCX(0+m[1]).row(k) * ( xxX(m[2])(b,i) * (xxX(m[3])(a,j) * wwX(0+m[4])(l,c) + wxX(m[3])(l,j) * xwX(2+m[4])(a,c))
                                                           - xxX(m[2])(a,i) * (xxX(m[3])(b,j) * wwX(0+m[4])(l,c) + wxX(m[3])(l,j) * xwX(2+m[4])(b,c))
                                                           + wxX(m[2])(l,i) * (xxX(m[3])(a,j) * xwX(2+m[4])(b,c) - xxX(m[3])(b,j) * xwX(2+m[4])(a,c)) );

            P += wXC(2+m[0]).col(c) * xCX(2+m[1]).row(a) * ( xxX(m[2])(b,i) * (wxX(m[3])(l,j) * wwX(m[4])(k,d) - wxX(m[3])(k,j) * wwX(0+m[4])(l,d))
                                                           - wxX(m[2])(l,i) * (xxX(m[3])(b,j) * wwX(m[4])(k,d) + wxX(m[3])(k,j) * xwX(2+m[4])(b,d))
                                                           + wxX(m[2])(k,i) * (xxX(m[3])(b,j) * wwX(m[4])(l,d) + wxX(m[3])(l,j) * xwX(2+m[4])(b,d)) )
               + wXC(2+m[0]).col(d) * xCX(2+m[1]).row(a) * ( xxX(m[2])(b,i) * (wxX(m[3])(k,j) * wwX(m[4])(l,c) - wxX(m[3])(l,j) * wwX(0+m[4])(k,c))
                                                           - wxX(m[2])(k,i) * (xxX(m[3])(b,j) * wwX(m[4])(l,c) + wxX(m[3])(l,j) * xwX(2+m[4])(b,c))
                                                           + wxX(m[2])(l,i) * (xxX(m[3])(b,j) * wwX(m[4])(k,c) + wxX(m[3])(k,j) * xwX(2+m[4])(b,c)) ) 
               + wXC(2+m[0]).col(c) * xCX(2+m[1]).row(b) * ( xxX(m[2])(a,i) * (wxX(m[3])(k,j) * wwX(m[4])(l,d) - wxX(m[3])(l,j) * wwX(0+m[4])(k,d))
                                                           - wxX(m[2])(k,i) * (xxX(m[3])(a,j) * wwX(m[4])(l,d) + wxX(m[3])(l,j) * xwX(2+m[4])(a,d))
                                                           + wxX(m[2])(l,i) * (xxX(m[3])(a,j) * wwX(m[4])(k,d) + wxX(m[3])(k,j) * xwX(2+m[4])(a,d)) )
               + wXC(2+m[0]).col(d) * xCX(2+m[1]).row(b) * ( xxX(m[2])(a,i) * (wxX(m[3])(l,j) * wwX(m[4])(k,c) - wxX(m[3])(k,j) * wwX(0+m[4])(l,c))
                                                           - wxX(m[2])(l,i) * (xxX(m[3])(a,j) * wwX(m[4])(k,c) + wxX(m[3])(k,j) * xwX(2+m[4])(a,c))
                                                           + wxX(m[2])(k,i) * (xxX(m[3])(a,j) * wwX(m[4])(l,c) + wxX(m[3])(l,j) * xwX(2+m[4])(a,c)) ); 

            P += xXC(0+m[0]).col(i) * wCX(0+m[1]).row(l) * ( xxX(m[2])(b,j) * (wwX(0+m[3])(k,d) * xwX(2+m[4])(a,c) - wwX(0+m[3])(k,c) * xwX(2+m[4])(a,d))
                                                           + xxX(m[2])(a,j) * (wwX(0+m[3])(k,c) * xwX(2+m[4])(b,d) - wwX(0+m[3])(k,d) * xwX(2+m[4])(b,c))
                                                           + wxX(m[2])(k,j) * (xwX(2+m[3])(b,d) * xwX(2+m[4])(a,c) - xwX(2+m[3])(b,c) * xwX(2+m[4])(a,d)) )
               + xXC(0+m[0]).col(j) * wCX(0+m[1]).row(l) * ( xxX(m[2])(b,i) * (wwX(0+m[3])(k,c) * xwX(2+m[4])(a,d) - wwX(0+m[3])(k,d) * xwX(2+m[4])(a,c))
                                                           + xxX(m[2])(a,i) * (wwX(0+m[3])(k,d) * xwX(2+m[4])(b,c) - wwX(0+m[3])(k,c) * xwX(2+m[4])(b,d))
                                                           + wxX(m[2])(k,i) * (xwX(2+m[3])(b,c) * xwX(2+m[4])(a,d) - xwX(2+m[3])(a,c) * xwX(2+m[4])(b,d)) )
               + xXC(0+m[0]).col(i) * wCX(0+m[1]).row(k) * ( xxX(m[2])(b,j) * (wwX(0+m[3])(l,c) * xwX(2+m[4])(a,d) - wwX(0+m[3])(l,d) * xwX(2+m[4])(a,c))
                                                           + xxX(m[2])(a,j) * (wwX(0+m[3])(l,d) * xwX(2+m[4])(b,c) - wwX(0+m[3])(l,c) * xwX(2+m[4])(b,d))
                                                           + wxX(m[2])(l,j) * (xwX(2+m[3])(b,c) * xwX(2+m[4])(a,d) - xwX(2+m[3])(b,d) * xwX(2+m[4])(a,c)) )
               + xXC(0+m[0]).col(j) * wCX(0+m[1]).row(k) * ( xxX(m[2])(b,i) * (wwX(0+m[3])(l,d) * xwX(2+m[4])(a,c) - wwX(0+m[3])(l,c) * xwX(2+m[4])(a,d))
                                                           + xxX(m[2])(a,i) * (wwX(0+m[3])(l,c) * xwX(2+m[4])(b,d) - wwX(0+m[3])(l,d) * xwX(2+m[4])(b,c))
                                                           + wxX(m[2])(l,i) * (xwX(2+m[3])(b,d) * xwX(2+m[4])(a,c) - xwX(2+m[3])(a,d) * xwX(2+m[4])(b,c)) );
        } while(std::prev_permutation(m.begin(), m.end()));
    }
}

template class wick<double, double, double>;
template class wick<std::complex<double>, double, double>;
template class wick<std::complex<double>, std::complex<double>, double>;
template class wick<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
