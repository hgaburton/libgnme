#include <cassert>
#include <libgnme/utils/eri_ao2mo.h>
#include "two_body_uscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void two_body_uscf<Tc,Tf,Tb>::initialise(
    wick_orbitals<Tc,Tb> &orba, 
    wick_orbitals<Tc,Tb> &orbb,  
    arma::Mat<Tb> &V)
{
    // Check input
    assert(orba.m_nbsf == orbb.m_nbsf);
    assert(V.n_rows == orba.m_nbsf * orba.m_nbsf);
    assert(V.n_cols == orba.m_nbsf * orba.m_nbsf);
    size_t nbsf = orba.m_nbsf;
    size_t nact = orba.m_nact;

    // Get dimensions of contractions
    size_t da = (orba.m_nz > 0) ? 2 : 1;
    size_t db = (orbb.m_nz > 0) ? 2 : 1;

    // Get view of two-electron AO integrals
    arma::Mat<Tb> IIao(V.memptr(), nbsf*nbsf, nbsf*nbsf, false, true);

    // Initialise J/K matrices
    arma::field<arma::Mat<Tc> > Ja(da), Ka(da);
    for(size_t i=0; i<da; i++)
    {
        Ja(i).set_size(nbsf,nbsf); Ja(i).zeros();
        Ka(i).set_size(nbsf,nbsf); Ka(i).zeros();
    }
    arma::field<arma::Mat<Tc> > Jb(db), Kb(db);
    for(size_t i=0; i<db; i++)
    {
        Jb(i).set_size(nbsf,nbsf); Jb(i).zeros();
        Kb(i).set_size(nbsf,nbsf); Kb(i).zeros();
    }

    // Construct J/K matrices in AO basis
    for(size_t m=0; m < nbsf; m++)
    for(size_t n=0; n < nbsf; n++)
    {
        # pragma omp parallel for schedule(static) collapse(2)
        for(size_t s=0; s < nbsf; s++)
        for(size_t t=0; t < nbsf; t++)
        {
            for(size_t i=0; i<da; i++)
            {
                // Coulomb matrices
                Ja(i)(s,t) += IIao(m*nbsf+n,s*nbsf+t) * orba.m_M(i)(n,m); 
                // Exchange matrices
                Ka(i)(s,t) += IIao(m*nbsf+t,s*nbsf+n) * orba.m_M(i)(n,m);
            }
            for(size_t i=0; i<db; i++)
            {
                // Coulomb matrices
                Jb(i)(s,t) += IIao(m*nbsf+n,s*nbsf+t) * orbb.m_M(i)(n,m); 
                // Exchange matrices
                Kb(i)(s,t) += IIao(m*nbsf+t,s*nbsf+n) * orbb.m_M(i)(n,m);
            }
        }
    }

    // Alpha-Alpha V terms
    m_Vaa.resize(3); m_Vaa.zeros();
    m_Vaa(0) = arma::dot(Ja(0).st() - Ka(0).st(), orba.m_M(0));
    if(orba.m_nz > 1)
    {
        m_Vaa(1) = 2.0 * arma::dot(Ja(0).st() - Ka(0).st(), orba.m_M(1));
        m_Vaa(2) = arma::dot(Ja(1).st() - Ka(1).st(), orba.m_M(1));
    }
    // Beta-Beta V terms
    m_Vbb.resize(3); m_Vbb.zeros();
    m_Vbb(0) = arma::dot(Jb(0).st() - Kb(0).st(), orbb.m_M(0));
    if(orbb.m_nz > 1)
    {
        m_Vbb(1) = 2.0 * arma::dot(Jb(0).st() - Kb(0).st(), orbb.m_M(1));
        m_Vbb(2) = arma::dot(Jb(1).st() - Kb(1).st(), orbb.m_M(1));
    }
    // Alpha-Beta V terms
    m_Vab.resize(da,db); m_Vab.zeros();
    for(size_t i=0; i<da; i++)
    for(size_t j=0; j<db; j++)
    {
        m_Vab(i,j) = arma::dot(Ja(i).st(), orbb.m_M(j));
    }

    // Construct effective one-body terms
    // xx[Y(J-K)X]    xw[Y(J-K)Y]
    // wx[X(J-K)X]    ww[X(J-K)Y]
    m_XVaXa.set_size(da,da,da); 
    m_XVaXb.set_size(db,da,db);
    m_XVbXa.set_size(da,db,da); 
    m_XVbXb.set_size(db,db,db);
    for(size_t i=0; i<da; i++)
    for(size_t j=0; j<da; j++)
    for(size_t k=0; k<da; k++)
        m_XVaXa(i,k,j) = orba.m_CX(i).t() * (Ja(k) - Ka(k)) * orba.m_XC(j);
    for(size_t i=0; i<da; i++)
    for(size_t j=0; j<da; j++)
    for(size_t k=0; k<db; k++)
        m_XVbXa(i,k,j) = orba.m_CX(i).t() * Jb(k) * orba.m_XC(j);
    for(size_t i=0; i<db; i++)
    for(size_t j=0; j<db; j++)
    for(size_t k=0; k<da; k++)
        m_XVaXb(i,k,j) = orbb.m_CX(i).t() * Ja(k) * orbb.m_XC(j);
    for(size_t i=0; i<db; i++)
    for(size_t j=0; j<db; j++)
    for(size_t k=0; k<db; k++)
        m_XVbXb(i,k,j) = orbb.m_CX(i).t() * (Jb(k) - Kb(k)) * orbb.m_XC(j);

    // Build the two-electron integrals
    // Bra: xY    wX
    // Ket: xX    wY
    m_IIaa.set_size(da*da,da*da); // aa
    m_IIbb.set_size(db*db,db*db); // bb
    m_IIab.set_size(da*da,db*db); // ab
    m_IIba.set_size(db*db,da*da); // ba
    for(size_t i=0; i<da; i++)
    for(size_t j=0; j<da; j++)
    for(size_t k=0; k<da; k++)
    for(size_t l=0; l<da; l++)
    {
        // Initialise the memory
        m_IIaa(2*i+j, 2*k+l).resize(4*nact*nact, 4*nact*nact); 
        // Construct two-electron integrals
        eri_ao2mo(orba.m_CX(i), orba.m_XC(j), orba.m_CX(k), orba.m_XC(l), 
                  IIao, m_IIaa(2*i+j, 2*k+l), 2*nact, true); 
    }
    for(size_t i=0; i<db; i++)
    for(size_t j=0; j<db; j++)
    for(size_t k=0; k<db; k++)
    for(size_t l=0; l<db; l++)
    {
        // Initialise the memory
        m_IIbb(2*i+j, 2*k+l).resize(4*nact*nact, 4*nact*nact); 
        // Construct two-electron integrals
        eri_ao2mo(orbb.m_CX(i), orbb.m_XC(j), orbb.m_CX(k), orbb.m_XC(l), 
                  IIao, m_IIbb(2*i+j, 2*k+l), 2*nact, true); 
    }
    for(size_t i=0; i<da; i++)
    for(size_t j=0; j<da; j++)
    for(size_t k=0; k<db; k++)
    for(size_t l=0; l<db; l++)
    {
        // Initialise the memory
        m_IIab(2*i+j, 2*k+l).resize(4*nact*nact, 4*nact*nact); m_IIab(2*i+j, 2*k+l).zeros();
        // Construct two-electron integrals
        eri_ao2mo(orba.m_CX(i), orba.m_XC(j), orbb.m_CX(k), orbb.m_XC(l), 
                  IIao, m_IIab(2*i+j, 2*k+l), 2*nact, false); 
        // Also store the transpose for IIab as it will make access quicker later
        m_IIba(2*k+l, 2*i+j) = m_IIab(2*i+j, 2*k+l).st();
    }
}


template class two_body_uscf<double, double, double>;
template class two_body_uscf<std::complex<double>, double, double>;
template class two_body_uscf<std::complex<double>, std::complex<double>, double>;
template class two_body_uscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
