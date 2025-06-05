#include <cassert>
#include <libgnme/utils/eri_ao2mo.h>
#include "two_body_rscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void two_body_rscf<Tc,Tf,Tb>::initialise(
    wick_orbitals<Tc,Tb> &orb, 
    arma::Mat<Tb> &V)
{
    // Get temporary numbers for easy code
    size_t nbsf = orb.m_nbsf;
    size_t nactx = orb.m_refx.m_nact, nactw = orb.m_refw.m_nact;
    size_t nact = nactx + nactw;

    // Check input
    assert(V.n_rows == nbsf * nbsf);
    assert(V.n_cols == nbsf * nbsf);

    // Get dimensions of contractions
    size_t d = (orb.m_nz > 0) ? 2 : 1;

    // Initialise J/K matrices
    arma::field<arma::Mat<Tc> > J(d), K(d);
    for(size_t i=0; i<d; i++)
    {
        J(i).set_size(nbsf,nbsf); J(i).zeros();
        K(i).set_size(nbsf,nbsf); K(i).zeros();
    }

    // Construct J/K matrices in AO basis
    for(size_t m=0; m < nbsf; m++)
    for(size_t n=0; n < nbsf; n++)
    {
        # pragma omp parallel for schedule(static) collapse(2)
        for(size_t s=0; s < nbsf; s++)
        for(size_t t=0; t < nbsf; t++)
        {
            for(size_t i=0; i<d; i++)
            {
                // Coulomb matrices
                J(i)(s,t) += V(m*nbsf+n,s*nbsf+t) * orb.m_M(i)(n,m); 
                // Exchange matrices
                K(i)(s,t) += V(m*nbsf+t,s*nbsf+n) * orb.m_M(i)(n,m);
            }
        }
    }

    // Alpha-Alpha V terms
    m_Vss.resize(3); m_Vss.zeros();
    m_Vss(0) = arma::dot(J(0).st() - K(0).st(), orb.m_M(0));
    if(orb.m_nz > 1)
    {
        m_Vss(1) = 2.0 * arma::dot(J(0).st() - K(0).st(), orb.m_M(1));
        m_Vss(2) = arma::dot(J(1).st() - K(1).st(), orb.m_M(1));
    }
    // Alpha-Beta V terms
    m_Vst.resize(d,d); m_Vst.zeros();
    for(size_t i=0; i<d; i++)
    for(size_t j=0; j<d; j++)
        m_Vst(i,j) = arma::dot(J(i).st(), orb.m_M(j));

    // Construct effective one-body terms
    // xx[Y(J-K)X]    xw[Y(J-K)Y]
    // wx[X(J-K)X]    ww[X(J-K)Y]
    m_XVsXs.set_size(d,d,d); 
    m_XVsXt.set_size(d,d,d); 
    for(size_t i=0; i<d; i++)
    for(size_t j=0; j<d; j++)
    for(size_t k=0; k<d; k++)
    {
        m_XVsXs(i,k,j) = orb.m_CX(i).t() * (J(k) - K(k)) * orb.m_XC(j);
        m_XVsXt(i,k,j) = orb.m_CX(i).t() * J(k) * orb.m_XC(j);
    }

    // Build the two-electron integrals
    // Bra: xY    wX
    // Ket: xX    wY
    m_IIss.set_size(d*d,d*d);
    m_IIst.set_size(d*d,d*d); 
    for(size_t i=0; i<d; i++)
    for(size_t j=0; j<d; j++)
    for(size_t k=0; k<d; k++)
    for(size_t l=0; l<d; l++)
    {
        // Initialise the memory
        m_IIss(2*i+j, 2*k+l).resize(nact*nact, nact*nact); 
        m_IIst(2*i+j, 2*k+l).resize(nact*nact, nact*nact); 
        // Construct two-electron integrals
        eri_ao2mo_split(orb.m_CX(i), orb.m_XC(j), orb.m_CX(k), orb.m_XC(l), 
                        V, m_IIst(2*i+j, 2*k+l), m_IIss(2*i+j, 2*k+l), true); 
        m_IIss(2*i+j, 2*k+l) += m_IIst(2*i+j, 2*k+l);
    }
}

template class two_body_rscf<double, double, double>;
template class two_body_rscf<std::complex<double>, double, double>;
template class two_body_rscf<std::complex<double>, std::complex<double>, double>;
template class two_body_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
