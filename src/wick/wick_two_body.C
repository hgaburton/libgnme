#include <cassert>
#include <algorithm>
#include "lowdin_pair.h"
#include "wick.h"

namespace {

// Build the anti-symmetrised two-electron integrals
// Take the form (C1 C2 || C3 C4) = (C1 C2 | C3 C4) - (C1 C3 | C2 C4) 
template<typename Tc, typename Tb>
void mo_eri(
    arma::Mat<Tc> &C1, arma::Mat<Tc> &C2, arma::Mat<Tc> &C3, arma::Mat<Tc> &C4, 
    arma::Mat<Tb> &IIao, arma::Mat<Tc> &IImo, size_t nmo, bool same_spin)
{
    // Get array of MO numbers
    assert(C1.n_cols == nmo);
    assert(C2.n_cols == nmo);
    assert(C3.n_cols == nmo);
    assert(C4.n_cols == nmo);

    // Setup temporary memory
    size_t nbsf = C1.n_rows;
    assert(IIao.n_rows == nbsf * nbsf);
    assert(IIao.n_cols == nbsf * nbsf);
    size_t dim = std::max(nmo,nbsf);
    arma::Mat<Tc> IItmp1(dim*dim, dim*dim, arma::fill::zeros);
    arma::Mat<Tc> IItmp2(dim*dim, dim*dim, arma::fill::zeros);

    // (pq|r4)
    IItmp1.zeros();
    for(size_t l=0; l<nmo; l++)
        for(size_t p=0; p<nbsf; p++)
        for(size_t q=0; q<nbsf; q++)
        for(size_t r=0; r<nbsf; r++)
        for(size_t s=0; s<nbsf; s++)
            IItmp1(p*nbsf+q, r*nmo+l) += IIao(p*nbsf+q, r*nbsf+s) * C4(s,l);

    // (pq|34)
    IItmp2.zeros();
    for(size_t k=0; k<nmo; k++)
        for(size_t p=0; p<nbsf; p++)
        for(size_t q=0; q<nbsf; q++)
        for(size_t r=0; r<nbsf; r++)
        for(size_t l=0; l<nmo; l++)
            IItmp2(p*nbsf+q, k*nmo+l) += IItmp1(p*nbsf+q, r*nmo+l) * std::conj(C3(r,k));
     
    // (p2|34)
    IItmp1.zeros();
    for(size_t j=0; j<nmo; j++)
        for(size_t p=0; p<nbsf; p++)
        for(size_t q=0; q<nbsf; q++)
        for(size_t k=0; k<nmo; k++)
        for(size_t l=0; l<nmo; l++)
            IItmp1(p*nmo+j, k*nmo+l) += IItmp2(p*nbsf+q, k*nmo+l) * C2(q,j);

    // (12|34)
    for(size_t i=0; i<nmo; i++)
        for(size_t p=0; p<nbsf; p++)
        for(size_t j=0; j<nmo; j++)
        for(size_t k=0; k<nmo; k++)
        for(size_t l=0; l<nmo; l++)
        {
            // Save Coulomb integrals
            IImo(i*nmo+j, k*nmo+l) += IItmp1(p*nmo+j, k*nmo+l) * std::conj(C1(p,i));
            // Add exchange integral if same spin
            if(same_spin)
                IImo(i*nmo+j, k*nmo+l) -= IItmp1(p*nmo+l, k*nmo+j) * std::conj(C1(p,i)); 
        }
}
template void mo_eri(
    arma::mat &C1, arma::mat &C2, arma::mat &C3, arma::mat &C4, 
    arma::mat &IIao, arma::mat &IImo, size_t nmo, bool same_spin);
template void mo_eri(
    arma::cx_mat &C1, arma::cx_mat &C2, arma::cx_mat &C3, arma::cx_mat &C4, 
    arma::mat &IIao, arma::cx_mat &IImo, size_t nmo, bool same_spin);
template void mo_eri(
    arma::cx_mat &C1, arma::cx_mat &C2, arma::cx_mat &C3, arma::cx_mat &C4, 
    arma::cx_mat &IIao, arma::cx_mat &IImo, size_t nmo, bool same_spin);

} // unnamed namespace

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick<Tc,Tf,Tb>::add_two_body(arma::Mat<Tb> &V)
{
    // Check input
    assert(V.n_rows == m_nbsf * m_nbsf);
    assert(V.n_cols == m_nbsf * m_nbsf);

    // Save two-body integrals
    m_II = V;

    // Setup control variable to indicate one-body initialised
    m_two_body = true;
}


template<typename Tc, typename Tf, typename Tb>
void wick<Tc,Tf,Tb>::setup_two_body()
{
    // Initialise J/K matrices
    arma::field<arma::Mat<Tc> > Ja(2), Jb(2), Ka(2), Kb(2);
    for(size_t i=0; i<2; i++)
    {
        Ja(i).set_size(m_nbsf,m_nbsf); Ja(i).zeros();
        Jb(i).set_size(m_nbsf,m_nbsf); Jb(i).zeros();
        Ka(i).set_size(m_nbsf,m_nbsf); Ka(i).zeros();
        Kb(i).set_size(m_nbsf,m_nbsf); Kb(i).zeros();
    }

    // Construct J/K matrices in AO basis
    for(size_t m=0; m < m_nbsf; m++)
    for(size_t n=0; n < m_nbsf; n++)
    {
        # pragma omp parallel for schedule(static) collapse(2)
        for(size_t s=0; s < m_nbsf; s++)
        for(size_t t=0; t < m_nbsf; t++)
        {
            // Coulomb matrices
            Ja(0)(s,t) += m_II(m*m_nbsf+n,s*m_nbsf+t) * m_wxMa(0)(n,m); 
            Ja(1)(s,t) += m_II(m*m_nbsf+n,s*m_nbsf+t) * m_wxMa(1)(n,m); 
            Jb(0)(s,t) += m_II(m*m_nbsf+n,s*m_nbsf+t) * m_wxMb(0)(n,m); 
            Jb(1)(s,t) += m_II(m*m_nbsf+n,s*m_nbsf+t) * m_wxMb(1)(n,m); 
            // Exchange matrices
            Ka(0)(s,t) += m_II(m*m_nbsf+t,s*m_nbsf+n) * m_wxMa(0)(n,m);
            Ka(1)(s,t) += m_II(m*m_nbsf+t,s*m_nbsf+n) * m_wxMa(1)(n,m);
            Kb(0)(s,t) += m_II(m*m_nbsf+t,s*m_nbsf+n) * m_wxMb(0)(n,m);
            Kb(1)(s,t) += m_II(m*m_nbsf+t,s*m_nbsf+n) * m_wxMb(1)(n,m);
        }
    }

    // Alpha-Alpha V terms
    m_Vaa.resize(3); m_Vaa.zeros();
    m_Vaa(0) = arma::dot(Ja(0).st() - Ka(0).st(), m_wxMa(0));
    m_Vaa(1) = 2.0 * arma::dot(Ja(0).st() - Ka(0).st(), m_wxMa(1));
    m_Vaa(2) = arma::dot(Ja(1).st() - Ka(1).st(), m_wxMa(1));
    // Beta-Beta V terms
    m_Vbb.resize(3); m_Vbb.zeros();
    m_Vbb(0) = arma::dot(Jb(0).st() - Kb(0).st(), m_wxMb(0));
    m_Vbb(1) = 2.0 * arma::dot(Jb(0).st() - Kb(0).st(), m_wxMb(1));
    m_Vbb(2) = arma::dot(Jb(1).st() - Kb(1).st(), m_wxMb(1));
    // Alpha-Beta V terms
    m_Vab.resize(2,2); m_Vab.zeros();
    m_Vab(0,0) = arma::dot(Ja(0).st(), m_wxMb(0));
    m_Vab(1,0) = arma::dot(Ja(1).st(), m_wxMb(0));
    m_Vab(0,1) = arma::dot(Ja(0).st(), m_wxMb(1));
    m_Vab(1,1) = arma::dot(Ja(1).st(), m_wxMb(1));

    // Construct effective one-body terms
    // xx[Y(J-K)X]    xw[Y(J-K)Y]
    // wx[X(J-K)X]    ww[X(J-K)Y]
    m_XVaXa.set_size(2,2,2); 
    m_XVaXb.set_size(2,2,2);
    m_XVbXa.set_size(2,2,2); 
    m_XVbXb.set_size(2,2,2);
    #pragma omp parallel for schedule(static) collapse(3)
    for(size_t i=0; i<2; i++)
    for(size_t j=0; j<2; j++)
    for(size_t k=0; k<2; k++)
    {
        // Go straight to the answer
        m_XVaXa(i,j,k) = m_CXa(i).t() * (Ja(j) - Ka(j)) * m_XCa(k); // aa
        m_XVbXb(i,j,k) = m_CXb(i).t() * (Jb(j) - Kb(j)) * m_XCb(k); // bb
        m_XVbXa(i,j,k) = m_CXa(i).t() * Jb(j) * m_XCa(k); // ab
        m_XVaXb(i,j,k) = m_CXb(i).t() * Ja(j) * m_XCb(k); // ba
    }

    // TODO: Build the two-electron integrals
    // Bra: xY    wX
    // Ket: xX    wY
    m_IIaa.set_size(4,4); // aa
    m_IIbb.set_size(4,4); // bb
    m_IIab.set_size(4,4); // ab
    for(size_t i=0; i<2; i++)
    for(size_t j=0; j<2; j++)
    for(size_t k=0; k<2; k++)
    for(size_t l=0; l<2; l++)
    {
        // Initialise the memory
        m_IIaa(2*i+j, 2*k+l).resize(4*m_nact*m_nact, 4*m_nact*m_nact); m_IIaa(2*i+j, 2*k+l).zeros();
        m_IIbb(2*i+j, 2*k+l).resize(4*m_nact*m_nact, 4*m_nact*m_nact); m_IIbb(2*i+j, 2*k+l).zeros();
        m_IIab(2*i+j, 2*k+l).resize(4*m_nact*m_nact, 4*m_nact*m_nact); m_IIab(2*i+j, 2*k+l).zeros();

        // Construct two-electron integrals
        mo_eri(m_CXa(i), m_XCa(j), m_CXa(k), m_XCa(l), m_II, m_IIaa(2*i+j, 2*k+l), 2*m_nact, true); 
        mo_eri(m_CXb(i), m_XCb(j), m_CXb(k), m_XCb(l), m_II, m_IIbb(2*i+j, 2*k+l), 2*m_nact, true); 
        mo_eri(m_CXa(i), m_XCa(j), m_CXb(k), m_XCb(l), m_II, m_IIab(2*i+j, 2*k+l), 2*m_nact, true); 
    }

    // TODO: Factor out this old code...
    // Construct XVX matrices
    m_wwXVaXa.set_size(4,2,4); m_xwXVaXa.set_size(4,2,4);
    m_wxXVaXa.set_size(4,2,4); m_xxXVaXa.set_size(4,2,4);
    m_wwXVaXb.set_size(4,2,4); m_xwXVaXb.set_size(4,2,4);
    m_wxXVaXb.set_size(4,2,4); m_xxXVaXb.set_size(4,2,4);
    m_wwXVbXa.set_size(4,2,4); m_xwXVbXa.set_size(4,2,4);
    m_wxXVbXa.set_size(4,2,4); m_xxXVbXa.set_size(4,2,4);
    m_wwXVbXb.set_size(4,2,4); m_xwXVbXb.set_size(4,2,4);
    m_wxXVbXb.set_size(4,2,4); m_xxXVbXb.set_size(4,2,4);

    // Construct intermediates
    #pragma omp parallel for schedule(static) collapse(3)
    for(size_t x=0; x<4; x++)
    for(size_t y=0; y<4; y++)
    for(size_t z=0; z<2; z++)
    {
        // Alpha-Alpha
        m_wwXVaXa(x,z,y) = m_wxXa(x) * m_Cxa.t() * (Ja(z) - Ka(z)) * m_Cxa * m_xwXa(y);
        m_wxXVaXa(x,z,y) = m_wxXa(x) * m_Cxa.t() * (Ja(z) - Ka(z)) * m_Cxa * m_xxXa(y);
        m_xwXVaXa(x,z,y) = m_xxXa(x) * m_Cxa.t() * (Ja(z) - Ka(z)) * m_Cxa * m_xwXa(y);
        m_xxXVaXa(x,z,y) = m_xxXa(x) * m_Cxa.t() * (Ja(z) - Ka(z)) * m_Cxa * m_xxXa(y);
        // Beta-Beta
        m_wwXVbXb(x,z,y) = m_wxXb(x) * m_Cxb.t() * (Jb(z) - Kb(z)) * m_Cxb * m_xwXb(y);
        m_wxXVbXb(x,z,y) = m_wxXb(x) * m_Cxb.t() * (Jb(z) - Kb(z)) * m_Cxb * m_xxXb(y);
        m_xwXVbXb(x,z,y) = m_xxXb(x) * m_Cxb.t() * (Jb(z) - Kb(z)) * m_Cxb * m_xwXb(y);
        m_xxXVbXb(x,z,y) = m_xxXb(x) * m_Cxb.t() * (Jb(z) - Kb(z)) * m_Cxb * m_xxXb(y);
        // Alpha-Beta
        m_wwXVaXb(x,z,y) = m_wxXb(x) * m_Cxb.t() * Ja(z) * m_Cxb * m_xwXb(y);
        m_wxXVaXb(x,z,y) = m_wxXb(x) * m_Cxb.t() * Ja(z) * m_Cxb * m_xxXb(y);
        m_xwXVaXb(x,z,y) = m_xxXb(x) * m_Cxb.t() * Ja(z) * m_Cxb * m_xwXb(y);
        m_xxXVaXb(x,z,y) = m_xxXb(x) * m_Cxb.t() * Ja(z) * m_Cxb * m_xxXb(y);
        // Beta-Alpha
        m_wwXVbXa(x,z,y) = m_wxXa(x) * m_Cxa.t() * Jb(z) * m_Cxa * m_xwXa(y);
        m_wxXVbXa(x,z,y) = m_wxXa(x) * m_Cxa.t() * Jb(z) * m_Cxa * m_xxXa(y);
        m_xwXVbXa(x,z,y) = m_xxXa(x) * m_Cxa.t() * Jb(z) * m_Cxa * m_xwXa(y);
        m_xxXVbXa(x,z,y) = m_xxXa(x) * m_Cxa.t() * Jb(z) * m_Cxa * m_xxXa(y);
    }
}

template<typename Tc, typename Tf, typename Tb>
void wick<Tc,Tf,Tb>::same_spin_two_body(
    arma::umat &xhp, arma::umat &whp,
    Tc &V, bool alpha)
{
    // Zero the output
    V = 0.0;

    // Establish number of bra/ket excitations
    size_t nx = xhp.n_rows; // Bra excitations
    size_t nw = whp.n_rows; // Ket excitations

    // Inform if we can't handle that excitation
    if(nx * nw > 2 || nx + nw > 2)
    {
        std::cout << "wick::same_spin_two_body: Bra excitations = " << nx << std::endl;
        std::cout << "wick::same_spin_two_body: Ket excitations = " << nw << std::endl;
        throw std::runtime_error("wick::same_spin_two_body: Requested excitation level not yet implemented");
    }

    // Get referemce to relevant X matrices for this spin
    const arma::field<arma::Mat<Tc> > &wwX = alpha ? m_wwXa : m_wwXb;
    const arma::field<arma::Mat<Tc> > &wxX = alpha ? m_wxXa : m_wxXb;
    const arma::field<arma::Mat<Tc> > &xwX = alpha ? m_xwXa : m_xwXb;
    const arma::field<arma::Mat<Tc> > &xxX = alpha ? m_xxXa : m_xxXb;

    // Get reference to relevant one-body matrix
    const arma::Col<Tc> &V0  = alpha ? m_Vaa : m_Vbb;
    const arma::field<arma::Mat<Tc> > &wwXVX = alpha ? m_wwXVaXa : m_wwXVbXb;
    const arma::field<arma::Mat<Tc> > &wxXVX = alpha ? m_wxXVaXa : m_wxXVbXb;
    const arma::field<arma::Mat<Tc> > &xwXVX = alpha ? m_xwXVaXa : m_xwXVbXb;
    const arma::field<arma::Mat<Tc> > &xxXVX = alpha ? m_xxXVaXa : m_xxXVbXb;

    // Get reference to coefficients
    const arma::field<arma::Mat<Tc> > &xXC = alpha ? m_xXCa : m_xXCb;
    const arma::field<arma::Mat<Tc> > &xCX = alpha ? m_xCXa : m_xCXb;
    const arma::field<arma::Mat<Tc> > &wXC = alpha ? m_wXCa : m_wXCb;
    const arma::field<arma::Mat<Tc> > &wCX = alpha ? m_wCXa : m_wCXb;

    // Get reference to number of zeros for this spin
    const size_t &nz = alpha ? m_nza : m_nzb; 

    // Check we don't have a non-zero element
    if(nz > nw + nx + 2) return;

    // < X | V | W > 
    if(nx == 0 and nw == 0)
    {
        V = V0(nz);
    }
    // < X_i^a | V | W >
    else if(nx == 1 and nw == 0)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        // Distribute the NZ zeros among 3 contractions
        std::vector<size_t> m(nz, 1); m.resize(3, 0); 
        do {
            V += xxX(m[0])(a,i) * V0(m[1] + m[2]) + 2.0 * xxXVX(2+m[0],m[1],m[2])(a,i);
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X | V | W_i^a >
    else if(nx == 0 and nw == 1)
    {
        size_t i = whp(0,0), a = whp(0,1);
        // Distribute the NZ zeros among 3 contractions
        std::vector<size_t> m(nz, 1); m.resize(3, 0); 
        do {
            V += wwX(m[0])(i,a) * V0(m[1] + m[2]) + 2.0 * wwXVX(m[0],m[1],2+m[2])(i,a);
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X_{ij}^{ab} | V | W >
    else if (nx == 2 and nw == 0)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        size_t j = xhp(1,0), b = xhp(1,1);
        // Distribute the NZ zerps ampng 4 contractions
        std::vector<size_t> m(nz, 1); m.resize(4, 0); 
        do {
            V += V0(m[0]+m[1]) * (xxX(m[2])(a,i) * xxX(m[3])(b,j) - xxX(m[2])(a,j) * xxX(m[3])(b,i));
            V += 2.0 * (xxX(m[0])(b,j) * xxXVX(2+m[1],m[2],m[3])(a,i) - xxX(m[0])(b,i) * xxXVX(2+m[1],m[2],m[3])(a,j));
            V += 2.0 * (xxX(m[0])(a,i) * xxXVX(2+m[1],m[2],m[3])(b,j) - xxX(m[0])(a,j) * xxXVX(2+m[1],m[2],m[3])(b,i));
            for(size_t p=0; p<m_nbsf; p++)
            for(size_t q=0; q<m_nbsf; q++)
            for(size_t r=0; r<m_nbsf; r++)
            for(size_t s=0; s<m_nbsf; s++)
            {
                V += m_II(p*m_nbsf+q,r*m_nbsf+s)  
                   * (xCX(2+m[0])(a,p) * xCX(2+m[1])(b,r) - xCX(2+m[0])(a,r) * xCX(2+m[1])(b,p))
                   * (xXC(0+m[2])(q,i) * xXC(0+m[3])(s,j) - xXC(0+m[2])(s,i) * xXC(0+m[3])(q,j));
            }
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X | V | W_{ij}^{ab} >
    else if (nx == 0 and nw == 2)
    {
        size_t i = whp(0,0), a = whp(0,1);
        size_t j = whp(1,0), b = whp(1,1);
        // Distribute the NZ zerps ampng 4 contractions
        std::vector<size_t> m(nz, 1); m.resize(4, 0); 
        do {
            V += V0(m[0]+m[1]) * (wwX(m[2])(i,a) * wwX(m[3])(j,b) - wwX(m[2])(j,a) * wwX(m[3])(i,b));
            V += 2.0 * (wwX(m[0])(j,b) * wwXVX(m[1],m[2],2+m[3])(i,a) - wwX(m[0])(i,b) * wwXVX(m[1],m[2],2+m[3])(j,a));
            V += 2.0 * (wwX(m[0])(i,a) * wwXVX(m[1],m[2],2+m[3])(j,b) - wwX(m[0])(j,a) * wwXVX(m[1],m[2],2+m[3])(i,b));
            for(size_t p=0; p<m_nbsf; p++)
            for(size_t q=0; q<m_nbsf; q++)
            for(size_t r=0; r<m_nbsf; r++)
            for(size_t s=0; s<m_nbsf; s++)
            {
                V += m_II(p*m_nbsf+q,r*m_nbsf+s)  
                   * (wCX(0+m[0])(i,p) * wCX(0+m[1])(j,r) - wCX(0+m[0])(i,r) * wCX(0+m[1])(j,p))
                   * (wXC(2+m[2])(q,a) * wXC(2+m[3])(s,b) - wXC(2+m[2])(s,a) * wXC(2+m[3])(q,b));
            }
        } while(std::prev_permutation(m.begin(), m.end()));
    }
    // < X_i^a| V | W_j^b >
    else if (nx == 1 and nw == 1)
    {
        size_t i = xhp(0,0), a = xhp(0,1);
        size_t j = whp(0,0), b = whp(0,1);
        std::vector<size_t> m(nz,1); m.resize(4,0);
        do {
            V += V0(m[0]+m[1]) * (xxX(m[2])(a,i) * wwX(m[3])(j,b) + xwX(2+m[2])(a,b) * wxX(m[3])(j,i));
            V += 2.0 * (wwX(m[0])(j,b) * xxXVX(2+m[1],m[2],0+m[3])(a,i) + wxX(0+m[0])(j,i) * xwXVX(2+m[1],m[2],2+m[3])(a,b));
            V += 2.0 * (xxX(m[0])(a,i) * wwXVX(0+m[1],m[2],2+m[3])(j,b) - xwX(2+m[0])(a,b) * wxXVX(0+m[1],m[2],0+m[3])(j,i));
            // Direct term
            arma::Col<Tc> LHS = arma::vectorise(xXC(0+m[2]).col(i) * xCX(2+m[0]).row(a));
            arma::Col<Tc> RHS = arma::vectorise(wXC(2+m[3]).col(b) * wCX(0+m[1]).row(j));
            V += 2.0 * arma::dot(LHS, m_II * RHS);
            // Exchange term
            LHS = arma::vectorise(xXC(0+m[2]).col(i) * wCX(0+m[0]).row(j));
            RHS = arma::vectorise(wXC(2+m[3]).col(b) * xCX(2+m[1]).row(a));
            V -= 2.0 * arma::dot(LHS, m_II * RHS);
        } while(std::prev_permutation(m.begin(), m.end()));
    }
}

template<typename Tc, typename Tf, typename Tb>
void wick<Tc,Tf,Tb>::diff_spin_two_body(
    arma::umat &xahp, arma::umat &xbhp,
    arma::umat &wahp, arma::umat &wbhp,
    Tc &V)
{
    // Zero the output
    V = 0.0;

    // Establish number of bra/ket excitations
    size_t nxa = xahp.n_rows; // Bra alpha excitations
    size_t nwa = wahp.n_rows; // Ket alpha excitations
    size_t nxb = xbhp.n_rows; // Bra beta excitations
    size_t nwb = wbhp.n_rows; // Ket beta excitations
    size_t nx = nxa + nxb;
    size_t nw = nwa + nwb;

    // Inform if we can't handle that excitation
    if(nx * nw > 2 || nx + nw > 2)
    {
        std::cout << "wick::diff_spin_two_body: Bra excitations = " << nx << std::endl;
        std::cout << "wick::diff_spin_two_body: Ket excitations = " << nw << std::endl;
        throw std::runtime_error("wick::diff_spin_two_body: Requested excitation level not yet implemented");
    }
    
    // < X | V | W > 
    if(nx == 0 and nw == 0)
    {
        V = m_Vab(m_nza,m_nzb);
    }
    // < X_i^a | V | W >
    else if(nx == 1 and nw == 0)
    {
        if(nxa == 1) // Alpha excitation
        {
            size_t i = xahp(0,0), a = xahp(0,1);
            // Distribute the NZ alpha zeros among 2 contractions
            std::vector<size_t> ma(m_nza, 1); ma.resize(2, 0); 
            do {
                V += m_xxXa(ma[0])(a,i) * m_Vab(ma[1],m_nzb) + m_xxXVbXa(2+ma[0],m_nzb,ma[1])(a,i);
            } while(std::prev_permutation(ma.begin(), ma.end()));
        }
        else // Beta excitation
        {
            size_t i = xbhp(0,0), a = xbhp(0,1);
            // Distribute the NZ beta zeros among 2 contractions
            std::vector<size_t> mb(m_nzb, 1); mb.resize(2, 0); 
            do {
                V += m_xxXb(mb[0])(a,i) * m_Vab(m_nza,mb[1]) + m_xxXVaXb(2+mb[0],m_nza,mb[1])(a,i);
            } while(std::prev_permutation(mb.begin(), mb.end()));
        }
    }
    // < X | V | W_i^a >
    else if(nx == 0 and nw == 1)
    {
        if(nwa == 1) // Alpha excitation
        {
            size_t i = wahp(0,0), a = wahp(0,1);
            // Distribute the NZ alpha zeros among 2 contractions
            std::vector<size_t> ma(m_nza, 1); ma.resize(2, 0); 
            do {
                V += m_wwXa(ma[0])(i,a) * m_Vab(ma[1],m_nzb) + m_wwXVbXa(ma[0],m_nzb,2+ma[1])(i,a);
            } while(std::prev_permutation(ma.begin(), ma.end()));
        }
        else // Beta excitation
        {
            size_t i = wbhp(0,0), a = wbhp(0,1);
            // Distribute the NZ beta zeros among 2 contractions
            std::vector<size_t> mb(m_nzb, 1); mb.resize(2, 0); 
            do {
                V += m_wwXb(mb[0])(i,a) * m_Vab(m_nza,mb[1]) + m_wwXVaXb(mb[0],m_nza,2+mb[1])(i,a);
            } while(std::prev_permutation(mb.begin(), mb.end()));
        }
    }
    // < X_{ij}^{ab} | V | W >
    else if(nx == 2 and nw == 0)
    {
        if(nxa == 2) // Both alpha excitations
        {
            size_t i = xahp(0,0), a = xahp(0,1);
            size_t j = xahp(1,0), b = xahp(1,1);
            std::vector<size_t> m(m_nza, 1); m.resize(3, 0); 
            do {
                V += m_Vab(m[0],m_nzb) * (m_xxXa(m[1])(a,i) * m_xxXa(m[2])(b,j) - m_xxXa(m[1])(a,j) * m_xxXa(m[2])(b,i))
                   + m_xxXa(m[0])(b,j) * m_xxXVbXa(2+m[1],m_nzb,m[2])(a,i) - m_xxXa(m[0])(b,i) * m_xxXVbXa(2+m[1],m_nzb,m[2])(a,j)
                   + m_xxXa(m[0])(a,i) * m_xxXVbXa(2+m[1],m_nzb,m[2])(b,j) - m_xxXa(m[0])(a,j) * m_xxXVbXa(2+m[1],m_nzb,m[2])(b,i);
            } while(std::prev_permutation(m.begin(), m.end()));
        }
        else if(nxb == 2) // Both alpha excitations
        {
            size_t i = xbhp(0,0), a = xbhp(0,1);
            size_t j = xbhp(1,0), b = xbhp(1,1);
            std::vector<size_t> m(m_nzb, 1); m.resize(3, 0); 
            do {
                V += m_Vab(m_nza,m[0]) * (m_xxXb(m[1])(a,i) * m_xxXb(m[2])(b,j) - m_xxXb(m[1])(a,j) * m_xxXb(m[2])(b,i))
                   + m_xxXb(m[0])(b,j) * m_xxXVaXb(2+m[1],m_nza,m[2])(a,i) - m_xxXb(m[0])(b,i) * m_xxXVaXb(2+m[1],m_nza,m[2])(a,j)
                   + m_xxXb(m[0])(a,i) * m_xxXVaXb(2+m[1],m_nza,m[2])(b,j) - m_xxXb(m[0])(a,j) * m_xxXVaXb(2+m[1],m_nza,m[2])(b,i);
            } while(std::prev_permutation(m.begin(), m.end()));
        }
        else // One alpha and one beta excitation
        {
            size_t i = xahp(0,0), a = xahp(0,1);
            size_t j = xbhp(0,0), b = xbhp(0,1);
            std::vector<size_t> ma(m_nza, 1); ma.resize(2, 0); 
            std::vector<size_t> mb(m_nzb, 1); mb.resize(2, 0); 
            do {
            do {
                V += m_Vab(ma[0],mb[0]) * m_xxXa(ma[1])(a,i) * m_xxXb(mb[1])(b,j)
                   + m_xxXa(ma[0])(a,i) * m_xxXVaXb(2+mb[0],ma[1],mb[1])(b,j) 
                   + m_xxXb(mb[0])(b,j) * m_xxXVbXa(2+ma[0],mb[1],ma[1])(a,i);
                arma::Col<Tc> LHS = arma::vectorise(m_xXCa(ma[1]).col(i) * m_xCXa(2+ma[0]).row(a));
                arma::Col<Tc> RHS = arma::vectorise(m_xXCb(mb[1]).col(j) * m_xCXb(2+mb[0]).row(b));
                V += arma::as_scalar(LHS.st() * m_II * RHS);
            } while(std::prev_permutation(ma.begin(), ma.end()));
            } while(std::prev_permutation(mb.begin(), mb.end()));
        }
    }
    // < X | V | W_{ij}^{ab} >
    else if (nx == 0 and nw == 2)
    {
        if(nwa == 2) // Both alpha excitations
        {
            size_t i = wahp(0,0), a = wahp(0,1);
            size_t j = wahp(1,0), b = wahp(1,1);
            std::vector<size_t> m(m_nza, 1); m.resize(3, 0); 
            do {
                V += m_Vab(m[0],m_nzb) * (m_wwXa(m[1])(i,a) * m_wwXa(m[2])(j,b) - m_wwXa(m[1])(j,a) * m_wwXa(m[2])(i,b))
                   + m_wwXa(m[0])(j,b) * m_wwXVbXa(m[1],m_nzb,2+m[2])(i,a) - m_wwXa(m[0])(i,b) * m_wwXVbXa(m[1],m_nzb,2+m[2])(j,a)
                   + m_wwXa(m[0])(i,a) * m_wwXVbXa(m[1],m_nzb,2+m[2])(j,b) - m_wwXa(m[0])(j,a) * m_wwXVbXa(m[1],m_nzb,2+m[2])(i,b);
            } while(std::prev_permutation(m.begin(), m.end()));
        }
        else if(nwb == 2) // Both alpha excitations
        {
            size_t i = wbhp(0,0), a = wbhp(0,1);
            size_t j = wbhp(1,0), b = wbhp(1,1);
            std::vector<size_t> m(m_nzb, 1); m.resize(3, 0); 
            do {
                V += m_Vab(m_nza,m[0]) * (m_wwXb(m[1])(i,a) * m_wwXb(m[2])(j,b) - m_wwXb(m[1])(j,a) * m_wwXb(m[2])(i,b))
                   + m_wwXb(m[0])(j,b) * m_wwXVaXb(m[1],m_nza,2+m[2])(i,a) - m_wwXb(m[0])(i,b) * m_wwXVaXb(m[1],m_nza,2+m[2])(j,a)
                   + m_wwXb(m[0])(i,a) * m_wwXVaXb(m[1],m_nza,2+m[2])(j,b) - m_wwXb(m[0])(j,a) * m_wwXVaXb(m[1],m_nza,2+m[2])(i,b);
            } while(std::prev_permutation(m.begin(), m.end()));
        }
        else // One alpha and one beta excitation
        {
            size_t i = wahp(0,0), a = wahp(0,1);
            size_t j = wbhp(0,0), b = wbhp(0,1);
            std::vector<size_t> ma(m_nza, 1); ma.resize(2, 0); 
            std::vector<size_t> mb(m_nzb, 1); mb.resize(2, 0); 
            do {
            do {
                V += m_Vab(ma[0],mb[0]) * m_wwXa(ma[1])(i,a) * m_wwXb(mb[1])(j,b)
                   + m_wwXa(ma[0])(i,a) * m_wwXVaXb(mb[0],ma[1],2+mb[1])(j,b) 
                   + m_wwXb(mb[0])(j,b) * m_wwXVbXa(ma[0],mb[1],2+ma[1])(i,a);
                for(size_t p=0; p<m_nbsf; p++)
                for(size_t q=0; q<m_nbsf; q++)
                for(size_t r=0; r<m_nbsf; r++)
                for(size_t s=0; s<m_nbsf; s++)
                {
                    V += m_II(p*m_nbsf+q,r*m_nbsf+s) * m_wCXa(ma[0])(i,p) * m_wXCa(2+ma[1])(q,a) 
                                                     * m_wCXb(mb[0])(j,r) * m_wXCb(2+mb[1])(s,b);
                }
            } while(std::prev_permutation(ma.begin(), ma.end()));
            } while(std::prev_permutation(mb.begin(), mb.end()));
        }
    }
    // < X_i^a | V | W_j^b >
    else if (nx == 1 and nw == 1) 
    {
        if(nxa == 1 and nwa == 1) // Both alpha excitations 
        {
            size_t i = xahp(0,0), a = xahp(0,1);
            size_t j = wahp(0,0), b = wahp(0,1);
            std::vector<size_t> m(m_nza,1); m.resize(3,0);
            do {
                V += m_Vab(m[0],m_nzb) * (m_xxXa(0+m[1])(a,i) * m_wwXa(m[2])(j,b) 
                                        + m_xwXa(2+m[1])(a,b) * m_wxXa(m[2])(j,i));
                V += m_wwXa(0+m[0])(j,b) * m_xxXVbXa(2+m[1],m_nzb,0+m[2])(a,i) 
                   + m_wxXa(0+m[0])(j,i) * m_xwXVbXa(2+m[1],m_nzb,2+m[2])(a,b);
                V += m_xxXa(0+m[0])(a,i) * m_wwXVbXa(0+m[1],m_nzb,2+m[2])(j,b) 
                   - m_xwXa(2+m[0])(a,b) * m_wxXVbXa(0+m[1],m_nzb,0+m[2])(j,i);
            } while(std::prev_permutation(m.begin(), m.end()));
        }
        else if(nxa == 0 and nwa == 1) // Bra beta, ket alpha
        {
            size_t i = xbhp(0,0), a = xbhp(0,1);
            size_t j = wahp(0,0), b = wahp(0,1);
            std::vector<size_t> ma(m_nza,1); ma.resize(2,0);
            std::vector<size_t> mb(m_nzb,1); mb.resize(2,0);
            do {
            do {
                V += m_Vab(ma[0],mb[0]) * m_xxXb(mb[1])(a,i) * m_wwXa(ma[1])(j,b)
                   + m_wwXa(ma[0])(j,b) * m_xxXVaXb(2+mb[0],ma[1],0+mb[1])(a,i)
                   + m_xxXb(mb[0])(a,i) * m_wwXVbXa(0+ma[0],mb[1],2+ma[1])(j,b);
                arma::Col<Tc> LHS = arma::vectorise(m_wXCa(2+ma[1]).col(b) * m_wCXa(0+ma[0]).row(j));
                arma::Col<Tc> RHS = arma::vectorise(m_xXCb(0+mb[1]).col(i) * m_xCXb(2+mb[0]).row(a));
                V += arma::dot(LHS, m_II * RHS);
            } while(std::prev_permutation(ma.begin(), ma.end()));
            } while(std::prev_permutation(mb.begin(), mb.end()));
        }
        else if(nxa == 1 and nwa == 0) // Bra alpha, ket beta
        {
            size_t i = xahp(0,0), a = xahp(0,1);
            size_t j = wbhp(0,0), b = wbhp(0,1);
            std::vector<size_t> ma(m_nza,1); ma.resize(2,0);
            std::vector<size_t> mb(m_nzb,1); mb.resize(2,0);
            do {
            do {
                V += m_Vab(ma[0],mb[0]) * m_xxXa(ma[1])(a,i) * m_wwXb(mb[1])(j,b)
                   + m_wwXb(0+mb[0])(j,b) * m_xxXVbXa(2+ma[0],mb[1],0+ma[1])(a,i)
                   + m_xxXa(0+ma[0])(a,i) * m_wwXVaXb(0+mb[0],ma[1],2+mb[1])(j,b);
                arma::Col<Tc> LHS = arma::vectorise(m_wXCb(2+mb[1]).col(b) * m_wCXb(0+mb[0]).row(j));
                arma::Col<Tc> RHS = arma::vectorise(m_xXCa(0+ma[1]).col(i) * m_xCXa(2+ma[0]).row(a));
                V += arma::dot(LHS, m_II * RHS);
            } while(std::prev_permutation(ma.begin(), ma.end()));
            } while(std::prev_permutation(mb.begin(), mb.end()));

        }
        else // Both beta excitations
        {
            size_t i = xbhp(0,0), a = xbhp(0,1);
            size_t j = wbhp(0,0), b = wbhp(0,1);
            std::vector<size_t> m(m_nzb,1); m.resize(3,0);
            do {
                V += m_Vab(m_nza,m[0]) * (m_xxXb(0+m[1])(a,i) * m_wwXb(m[2])(j,b) 
                                        + m_xwXb(2+m[1])(a,b) * m_wxXb(m[2])(j,i));
                V += m_wwXb(0+m[0])(j,b) * m_xxXVaXb(2+m[1],m_nza,0+m[2])(a,i) 
                   + m_wxXb(0+m[0])(j,i) * m_xwXVaXb(2+m[1],m_nza,2+m[2])(a,b);
                V += m_xxXb(0+m[0])(a,i) * m_xxXVaXb(0+m[1],m_nza,2+m[2])(j,b) 
                   - m_xwXb(2+m[0])(a,b) * m_wxXVaXb(0+m[1],m_nza,0+m[2])(j,i);
            } while(std::prev_permutation(m.begin(), m.end()));
        }
    }
}

template class wick<double, double, double>;
template class wick<std::complex<double>, double, double>;
template class wick<std::complex<double>, std::complex<double>, double>;
template class wick<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
