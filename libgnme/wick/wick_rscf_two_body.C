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
    m_XJKX.set_size(d,d,d); 
    m_XJX.set_size(d,d,d); 
    for(size_t i=0; i<d; i++)
    for(size_t j=0; j<d; j++)
    for(size_t k=0; k<d; k++)
    {
        m_XJKX(i,k,j) = m_orb.m_CX(i).t() * (J(k) - K(k)) * m_orb.m_XC(j);
        m_XJX(i,k,j)  = m_orb.m_CX(i).t() * J(k) * m_orb.m_XC(j);
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


template class wick_rscf<double, double, double>;
template class wick_rscf<std::complex<double>, double, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
