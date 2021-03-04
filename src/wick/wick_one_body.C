#include <cassert>
#include <algorithm>
#include "lowdin_pair.h"
#include "wick.h"

namespace libnome {

template<typename Tc, typename Tf, typename Tb>
void wick<Tc,Tf,Tb>::setup_one_body(arma::Mat<Tf> Fa, arma::Mat<Tf> Fb) 
{
    // Check input
    assert(Fa.n_rows == m_nbsf);
    assert(Fa.n_cols == m_nbsf);
    assert(Fb.n_rows == m_nbsf);
    assert(Fb.n_cols == m_nbsf);

    // Transform one-body matrix to X MO basis
    arma::Mat<Tc> xxFa = m_Cxa.t() * Fa * m_Cxa;
    arma::Mat<Tc> xxFb = m_Cxb.t() * Fb * m_Cxb;

    // Construct 'F0' terms
    m_F0a.resize(2); 
    m_F0a(0) = arma::dot(xxFa, m_xxXa(0).st());
    m_F0a(1) = arma::dot(xxFa, m_xxXa(1).st());
    m_F0b.resize(2);
    m_F0b(0) = arma::dot(xxFb, m_xxXb(0).st());
    m_F0b(1) = arma::dot(xxFb, m_xxXb(1).st());

    // Construct XFX matrices
    m_wwXFXa.set_size(4,4); m_xwXFXa.set_size(4,4);
    m_wxXFXa.set_size(4,4); m_xxXFXa.set_size(4,4);
    m_wwXFXb.set_size(4,4); m_xwXFXb.set_size(4,4);
    m_wxXFXb.set_size(4,4); m_xxXFXb.set_size(4,4);

    // Construct intermediates
    #pragma omp parallel for schedule(static) collapse(2)
    for(size_t x=0; x<4; x++)
    for(size_t y=0; y<4; y++)
    {
        // Alpha
        m_wwXFXa(x,y) = m_wxXa(x) * xxFa * m_xwXa(y);
        m_wxXFXa(x,y) = m_wxXa(x) * xxFa * m_xxXa(y);
        m_xwXFXa(x,y) = m_xxXa(x) * xxFa * m_xwXa(y);
        m_xxXFXa(x,y) = m_xxXa(x) * xxFa * m_xxXa(y);
        // Beta
        m_wwXFXb(x,y) = m_wxXb(x) * xxFb * m_xwXb(y);
        m_wxXFXb(x,y) = m_wxXb(x) * xxFb * m_xxXb(y);
        m_xwXFXb(x,y) = m_xxXb(x) * xxFb * m_xwXb(y);
        m_xxXFXb(x,y) = m_xxXb(x) * xxFb * m_xxXb(y);
    }

    // Setup control variable to indicate one-body initialised
    m_one_body = true;
}

template<typename Tc, typename Tf, typename Tb>
void wick<Tc,Tf,Tb>::spin_one_body(
    arma::umat &xhp, arma::umat &whp,
    Tc &F, bool alpha)
{
    // Ensure output is zero'd
    F = 0.0;
    
    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations

    // Inform if we can't handle that excitation
    if(nx > 2 || nw > 2 || (nx+nw) > 4)
    {
        std::cout << "wick::spin_one_body: Bra excitations = " << nx << std::endl;
        std::cout << "wick::spin_one_body: Ket excitations = " << nw << std::endl;
        throw std::runtime_error("wick::spin_one_body: Requested excitation level not yet implemented");
    }

    // Get reference to relevant X matrices for this spin
    const arma::field<arma::Mat<Tc> > &wwX = alpha ? m_wwXa : m_wwXb;
    const arma::field<arma::Mat<Tc> > &wxX = alpha ? m_wxXa : m_wxXb;
    const arma::field<arma::Mat<Tc> > &xwX = alpha ? m_xwXa : m_xwXb;
    const arma::field<arma::Mat<Tc> > &xxX = alpha ? m_xxXa : m_xxXb;

    // Get reference to relevant one-body matrix
    const arma::Col<Tc> &F0  = alpha ? m_F0a : m_F0b;
    const arma::field<arma::Mat<Tc> > &wwXFX = alpha ? m_wwXFXa : m_wwXFXb;
    const arma::field<arma::Mat<Tc> > &wxXFX = alpha ? m_wxXFXa : m_wxXFXb;
    const arma::field<arma::Mat<Tc> > &xwXFX = alpha ? m_xwXFXa : m_xwXFXb;
    const arma::field<arma::Mat<Tc> > &xxXFX = alpha ? m_xxXFXa : m_xxXFXb;

    // Get reference to number of zeros for this spin
    const size_t &nz = alpha ? m_nza : m_nzb; 

    // Check we don't have a non-zero element
    if(nz > nw + nx + 1) return;

    // < X | F | W >
    if(nx == 0 and nw == 0)
    {
        F = F0(nz); 
    }
    // < X_i^a | F | W >
    else if(nx == 1 and nw == 0)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        // Distribute the NZ zeros among 2 contractions
        std::vector<size_t> m(nz, 1); m.resize(2, 0); 
        do {
            F += xxX(m[0])(a,i) * F0(m[1]) + xxXFX(2+m[0],m[1])(a,i);
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X | F | W_i^a >
    else if(nx == 0 and nw == 1)
    {
        size_t i = whp(0,0), a = whp(0,1);
        // Distribute the NZ zeros among 2 contractions
        std::vector<size_t> m(nz, 1); m.resize(2, 0); 
        do {
            F += wwX(m[0])(i,a) * F0(m[1]) + wwXFX(m[0],2+m[1])(i,a);
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
            F += F0(m[0]) * (xxX(m[1])(a,i) * wwX(m[2])(j,b) + wxX(m[1])(j,i) * xwX(2+m[2])(a,b))
               + xxXFX(2+m[0],0+m[1])(a,i) * wwX(m[2])(j,b) + xwXFX(2+m[0],2+m[1])(a,b) * wxX(0+m[2])(j,i)
               + wwXFX(0+m[0],2+m[1])(j,b) * xxX(m[2])(a,i) - wxXFX(0+m[0],0+m[1])(j,i) * xwX(2+m[2])(a,b);
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
            F += F0(m[0]) * (xxX(m[1])(a,i) * xxX(m[2])(b,j) - xxX(m[1])(a,j) * xxX(m[2])(b,i))
               + (xxXFX(2+m[0],m[1])(a,i) * xxX(m[2])(b,j) - xxXFX(2+m[0],m[1])(a,j) * xxX(m[2])(b,i))
               + (xxXFX(2+m[0],m[1])(b,j) * xxX(m[2])(a,i) - xxXFX(2+m[0],m[1])(b,i) * xxX(m[2])(a,j));
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
            F += F0(m[0]) * (wwX(m[1])(i,a) * wwX(m[2])(j,b) - wwX(m[1])(i,b) * wwX(m[2])(j,a))
               + (wwXFX(m[0],2+m[1])(j,b) * wwX(m[2])(i,a) - wwXFX(m[0],2+m[1])(j,a) * wwX(m[2])(i,b))
               + (wwXFX(m[0],2+m[1])(i,a) * wwX(m[2])(j,b) - wwXFX(m[0],2+m[1])(i,b) * wwX(m[2])(j,a));
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
            F += (F0(m[0]) * wwX(0+m[1])(k,c) + wwXFX(0+m[0],2+m[1])(k,c)) * (xxX(m[2])(a,i) * xxX(m[3])(b,j) - xxX(m[2])(b,i) * xxX(m[3])(a,j))
               + (F0(m[0]) * xwX(2+m[1])(a,c) + xwXFX(2+m[0],2+m[1])(a,c)) * (xxX(m[2])(b,j) * wxX(m[3])(k,i) - xxX(m[2])(b,i) * wxX(m[3])(k,j))
               + (F0(m[0]) * xwX(2+m[1])(b,c) + xwXFX(2+m[0],2+m[1])(b,c)) * (xxX(m[2])(a,i) * wxX(m[3])(k,j) - xxX(m[2])(a,j) * wxX(m[3])(k,i))
               + xxXFX(2+m[0],m[1])(b,j) * (xxX(m[2])(a,i) * wwX(0+m[3])(k,c) + wxX(m[2])(k,i) * xwX(2+m[3])(a,c))
               - xxXFX(2+m[0],m[1])(a,j) * (xxX(m[2])(b,i) * wwX(0+m[3])(k,c) + wxX(m[2])(k,i) * xwX(2+m[3])(b,c))
               + xxXFX(2+m[0],m[1])(a,i) * (xxX(m[2])(b,j) * wwX(0+m[3])(k,c) + wxX(m[2])(k,j) * xwX(2+m[3])(b,c))
               - xxXFX(2+m[0],m[1])(b,i) * (xxX(m[2])(a,j) * wwX(0+m[3])(k,c) + wxX(m[2])(k,j) * xwX(2+m[3])(a,c))
               + wxXFX(0+m[0],m[1])(k,j) * (xxX(m[2])(b,i) * xwX(2+m[3])(a,c) - xxX(m[2])(a,i) * xwX(2+m[3])(b,c))
               + wxXFX(0+m[0],m[1])(k,i) * (xxX(m[2])(a,j) * xwX(2+m[3])(b,c) - xxX(m[2])(b,j) * xwX(2+m[3])(a,c));
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
            F += (F0(m[0]) * xxX(0+m[1])(c,k) + xxXFX(2+m[0],0+m[1])(c,k)) * (wwX(m[2])(i,a) * wwX(m[3])(j,b) - wwX(m[2])(i,b) * wwX(m[3])(j,a))
               + (F0(m[0]) * xwX(2+m[1])(c,a) + xwXFX(2+m[0],2+m[1])(c,a)) * (wwX(m[2])(j,b) * wxX(m[3])(i,k) - wwX(m[2])(i,b) * wxX(m[3])(j,k))
               + (F0(m[0]) * xwX(2+m[1])(c,b) + xwXFX(2+m[0],2+m[1])(c,b)) * (wwX(m[2])(i,a) * wxX(m[3])(j,k) - wwX(m[2])(j,a) * wxX(m[3])(i,k))
               + wwXFX(m[0],2+m[1])(j,b) * (wwX(m[2])(i,a) * xxX(0+m[3])(c,k) + wxX(m[2])(i,k) * xwX(2+m[3])(c,a))
               - wwXFX(m[0],2+m[1])(j,a) * (wwX(m[2])(i,b) * xxX(0+m[3])(c,k) + wxX(m[2])(i,k) * xwX(2+m[3])(c,b))
               + wwXFX(m[0],2+m[1])(i,a) * (wwX(m[2])(j,b) * xxX(0+m[3])(c,k) + wxX(m[2])(j,k) * xwX(2+m[3])(c,b))
               - wwXFX(m[0],2+m[1])(i,b) * (wwX(m[2])(j,a) * xxX(0+m[3])(c,k) + wxX(m[2])(j,k) * xwX(2+m[3])(c,a))
               + wxXFX(m[0],0+m[1])(j,k) * (wwX(m[2])(i,b) * xwX(2+m[3])(c,a) - wwX(m[2])(i,a) * xwX(2+m[3])(c,b))
               + wxXFX(m[0],0+m[1])(i,k) * (wwX(m[2])(j,a) * xwX(2+m[3])(c,b) - wwX(m[2])(j,b) * xwX(2+m[3])(c,a));
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
            F += F0(m[4]) * ( 
                   (xxX(0+m[0])(a,i) * xxX(0+m[1])(b,j) - xxX(0+m[0])(a,j) * xxX(0+m[1])(b,i)) * (wwX(m[2])(k,c) * wwX(m[3])(l,d) - wwX(m[2])(k,d) * wwX(m[3])(l,c))
                 + (xwX(2+m[0])(a,c) * xxX(0+m[1])(b,j) - xwX(2+m[0])(b,c) * xxX(0+m[1])(a,j)) * (wxX(m[2])(k,i) * wwX(m[3])(l,d) - wxX(m[2])(l,i) * wwX(m[3])(k,d))
                 + (xwX(2+m[0])(a,d) * xxX(0+m[1])(b,j) - xwX(2+m[0])(b,d) * xxX(0+m[1])(a,j)) * (wxX(m[2])(l,i) * wwX(m[3])(k,c) - wxX(m[2])(k,i) * wwX(m[3])(l,c))
                 + (xwX(2+m[0])(a,c) * xxX(0+m[1])(b,i) - xwX(2+m[0])(b,c) * xxX(0+m[1])(a,i)) * (wxX(m[2])(l,j) * wwX(m[3])(k,d) - wxX(m[2])(k,j) * wwX(m[3])(l,d))
                 + (xwX(2+m[0])(a,d) * xxX(0+m[1])(b,i) - xwX(2+m[0])(b,d) * xxX(0+m[1])(a,i)) * (wxX(m[2])(k,j) * wwX(m[3])(l,c) - wxX(m[2])(l,j) * wwX(m[3])(k,c))
                 + (xwX(2+m[0])(a,c) * xwX(2+m[1])(b,d) - xwX(2+m[0])(b,c) * xwX(2+m[1])(a,d)) * (wxX(m[2])(k,i) * wxX(m[3])(l,j) - wxX(m[2])(l,i) * wxX(m[3])(k,j)) );
            // Remaining Terms
            F += xxXFX(2+m[0],0+m[1])(a,i) * ( xxX(m[2])(b,j) * ( wwX(m[3])(k,c) * wwX(0+m[4])(l,d) - wwX(m[3])(k,d) * wwX(0+m[4])(l,c) )
                                             + wxX(m[2])(k,j) * ( wwX(m[3])(l,d) * xwX(2+m[4])(b,c) - wwX(m[3])(l,c) * xwX(2+m[4])(b,d) )
                                             + wxX(m[2])(l,j) * ( wwX(m[3])(k,c) * xwX(2+m[4])(b,d) - wwX(m[3])(k,d) * xwX(2+m[4])(b,c) ) )
               + xxXFX(2+m[0],0+m[1])(a,j) * ( xxX(m[2])(b,i) * ( wwX(m[3])(k,d) * wwX(0+m[4])(l,c) - wwX(m[3])(k,c) * wwX(0+m[4])(l,d) )
                                             + wxX(m[2])(k,i) * ( wwX(m[3])(l,c) * xwX(2+m[4])(b,d) - wwX(m[3])(l,d) * xwX(2+m[4])(b,c) )
                                             + wxX(m[2])(l,i) * ( wwX(m[3])(k,d) * xwX(2+m[4])(b,c) - wwX(m[3])(k,c) * xwX(2+m[4])(b,d) ) )
               + xxXFX(2+m[0],0+m[1])(b,i) * ( xxX(m[2])(a,j) * ( wwX(m[3])(k,d) * wwX(0+m[4])(l,c) - wwX(m[3])(k,c) * wwX(0+m[4])(l,d) )
                                             + wxX(m[2])(k,j) * ( wwX(m[3])(l,c) * xwX(2+m[4])(a,d) - wwX(m[3])(l,d) * xwX(2+m[4])(a,c) )
                                             + wxX(m[2])(l,j) * ( wwX(m[3])(k,d) * xwX(2+m[4])(a,c) - wwX(m[3])(k,c) * xwX(2+m[4])(a,d) ) )
               + xxXFX(2+m[0],0+m[1])(b,j) * ( xxX(m[2])(a,i) * ( wwX(m[3])(k,c) * wwX(0+m[4])(l,d) - wwX(m[3])(k,d) * wwX(0+m[4])(l,c) )
                                             + wxX(m[2])(k,i) * ( wwX(m[3])(l,d) * xwX(2+m[4])(a,c) - wwX(m[3])(l,c) * xwX(2+m[4])(a,d) )
                                             + wxX(m[2])(l,i) * ( wwX(m[3])(k,c) * xwX(2+m[4])(a,d) - wwX(m[3])(k,d) * xwX(2+m[4])(a,c) ) );

            F += wwXFX(0+m[0],2+m[1])(l,c) * ( xxX(m[2])(b,i) * (xxX(m[3])(a,j) * wwX(0+m[4])(k,d) + wxX(m[3])(k,j) * xwX(2+m[4])(a,d))
                                             - xxX(m[2])(a,i) * (xxX(m[3])(b,j) * wwX(0+m[4])(k,d) + wxX(m[3])(k,j) * xwX(2+m[4])(b,d))
                                             + wxX(m[2])(k,i) * (xxX(m[3])(a,j) * xwX(2+m[4])(b,d) - xxX(m[3])(b,j) * xwX(2+m[4])(a,d)) )
               + wwXFX(0+m[0],2+m[1])(l,d) * ( xxX(m[2])(a,i) * (xxX(m[3])(b,j) * wwX(0+m[4])(k,c) + wxX(m[3])(k,j) * xwX(2+m[4])(b,c))
                                             - xxX(m[2])(b,i) * (xxX(m[3])(a,j) * wwX(0+m[4])(k,c) + wxX(m[3])(k,j) * xwX(2+m[4])(a,c))
                                             + wxX(m[2])(k,i) * (xxX(m[3])(b,j) * xwX(2+m[4])(a,c) - xxX(m[3])(a,j) * xwX(2+m[4])(b,c)) )
               + wwXFX(0+m[0],2+m[1])(k,c) * ( xxX(m[2])(a,i) * (xxX(m[3])(b,j) * wwX(0+m[4])(l,d) + wxX(m[3])(l,j) * xwX(2+m[4])(b,d))
                                             - xxX(m[2])(b,i) * (xxX(m[3])(a,j) * wwX(0+m[4])(l,d) + wxX(m[3])(l,j) * xwX(2+m[4])(a,d))
                                             + wxX(m[2])(l,i) * (xxX(m[3])(b,j) * xwX(2+m[4])(a,d) - xxX(m[3])(a,j) * xwX(2+m[4])(b,d)) )
               + wwXFX(0+m[0],2+m[1])(k,d) * ( xxX(m[2])(b,i) * (xxX(m[3])(a,j) * wwX(0+m[4])(l,c) + wxX(m[3])(l,j) * xwX(2+m[4])(a,c))
                                             - xxX(m[2])(a,i) * (xxX(m[3])(b,j) * wwX(0+m[4])(l,c) + wxX(m[3])(l,j) * xwX(2+m[4])(b,c))
                                             + wxX(m[2])(l,i) * (xxX(m[3])(a,j) * xwX(2+m[4])(b,c) - xxX(m[3])(b,j) * xwX(2+m[4])(a,c)) );

            F += xwXFX(2+m[0],2+m[1])(a,c) * ( xxX(m[2])(b,i) * (wxX(m[3])(l,j) * wwX(m[4])(k,d) - wxX(m[3])(k,j) * wwX(0+m[4])(l,d))
                                             - wxX(m[2])(l,i) * (xxX(m[3])(b,j) * wwX(m[4])(k,d) + wxX(m[3])(k,j) * xwX(2+m[4])(b,d))
                                             + wxX(m[2])(k,i) * (xxX(m[3])(b,j) * wwX(m[4])(l,d) + wxX(m[3])(l,j) * xwX(2+m[4])(b,d)) )
               + xwXFX(2+m[0],2+m[1])(a,d) * ( xxX(m[2])(b,i) * (wxX(m[3])(k,j) * wwX(m[4])(l,c) - wxX(m[3])(l,j) * wwX(0+m[4])(k,c))
                                             - wxX(m[2])(k,i) * (xxX(m[3])(b,j) * wwX(m[4])(l,c) + wxX(m[3])(l,j) * xwX(2+m[4])(b,c))
                                             + wxX(m[2])(l,i) * (xxX(m[3])(b,j) * wwX(m[4])(k,c) + wxX(m[3])(k,j) * xwX(2+m[4])(b,c)) ) 
               + xwXFX(2+m[0],2+m[1])(b,c) * ( xxX(m[2])(a,i) * (wxX(m[3])(k,j) * wwX(m[4])(l,d) - wxX(m[3])(l,j) * wwX(0+m[4])(k,d))
                                             - wxX(m[2])(k,i) * (xxX(m[3])(a,j) * wwX(m[4])(l,d) + wxX(m[3])(l,j) * xwX(2+m[4])(a,d))
                                             + wxX(m[2])(l,i) * (xxX(m[3])(a,j) * wwX(m[4])(k,d) + wxX(m[3])(k,j) * xwX(2+m[4])(a,d)) )
               + xwXFX(2+m[0],2+m[1])(b,d) * ( xxX(m[2])(a,i) * (wxX(m[3])(l,j) * wwX(m[4])(k,c) - wxX(m[3])(k,j) * wwX(0+m[4])(l,c))
                                             - wxX(m[2])(l,i) * (xxX(m[3])(a,j) * wwX(m[4])(k,c) + wxX(m[3])(k,j) * xwX(2+m[4])(a,c))
                                             + wxX(m[2])(k,i) * (xxX(m[3])(a,j) * wwX(m[4])(l,c) + wxX(m[3])(l,j) * xwX(2+m[4])(a,c)) ); 

            F += wxXFX(0+m[0],0+m[1])(l,i) * ( xxX(m[2])(b,j) * (wwX(0+m[3])(k,d) * xwX(2+m[4])(a,c) - wwX(0+m[3])(k,c) * xwX(2+m[4])(a,d))
                                             + xxX(m[2])(a,j) * (wwX(0+m[3])(k,c) * xwX(2+m[4])(b,d) - wwX(0+m[3])(k,d) * xwX(2+m[4])(b,c))
                                             + wxX(m[2])(k,j) * (xwX(2+m[3])(b,d) * xwX(2+m[4])(a,c) - xwX(2+m[3])(b,c) * xwX(2+m[4])(a,d)) )
               + wxXFX(0+m[0],0+m[1])(l,j) * ( xxX(m[2])(b,i) * (wwX(0+m[3])(k,c) * xwX(2+m[4])(a,d) - wwX(0+m[3])(k,d) * xwX(2+m[4])(a,c))
                                             + xxX(m[2])(a,i) * (wwX(0+m[3])(k,d) * xwX(2+m[4])(b,c) - wwX(0+m[3])(k,c) * xwX(2+m[4])(b,d))
                                             + wxX(m[2])(k,i) * (xwX(2+m[3])(b,c) * xwX(2+m[4])(a,d) - xwX(2+m[3])(a,c) * xwX(2+m[4])(b,d)) )
               + wxXFX(0+m[0],0+m[1])(k,i) * ( xxX(m[2])(b,j) * (wwX(0+m[3])(l,c) * xwX(2+m[4])(a,d) - wwX(0+m[3])(l,d) * xwX(2+m[4])(a,c))
                                             + xxX(m[2])(a,j) * (wwX(0+m[3])(l,d) * xwX(2+m[4])(b,c) - wwX(0+m[3])(l,c) * xwX(2+m[4])(b,d))
                                             + wxX(m[2])(l,j) * (xwX(2+m[3])(b,c) * xwX(2+m[4])(a,d) - xwX(2+m[3])(b,d) * xwX(2+m[4])(a,c)) )
               + wxXFX(0+m[0],0+m[1])(k,j) * ( xxX(m[2])(b,i) * (wwX(0+m[3])(l,d) * xwX(2+m[4])(a,c) - wwX(0+m[3])(l,c) * xwX(2+m[4])(a,d))
                                             + xxX(m[2])(a,i) * (wwX(0+m[3])(l,c) * xwX(2+m[4])(b,d) - wwX(0+m[3])(l,d) * xwX(2+m[4])(b,c))
                                             + wxX(m[2])(l,i) * (xwX(2+m[3])(b,d) * xwX(2+m[4])(a,c) - xwX(2+m[3])(a,d) * xwX(2+m[4])(b,c)) );
        } while(std::prev_permutation(m.begin(), m.end()));
    }
}

template class wick<double, double, double>;
template class wick<std::complex<double>, double, double>;
template class wick<std::complex<double>, std::complex<double>, double>;
template class wick<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libnome
