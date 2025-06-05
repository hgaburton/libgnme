#ifndef LIBGNME_ONE_BODY_RSCF_H
#define LIBGNME_ONE_BODY_RSCF_H

#include "one_body.h"
#include "wick_orbitals.h"

namespace libgnme {

/** \brief Container for storing one-body intermediate integrals with restricted orbitals
    \tparam Tc Type defining orbital coefficient
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class one_body_rscf : public one_body<Tc,Tf,Tb>
{
private:
    arma::Col<Tc> m_F0; //!< Store the zeroth-order Fock term [nz]
    arma::field<arma::Mat<Tc> > m_XFX; //!< Store the first-order Fock terms [nz,nz][2*nmo,2*nmo]

public:
    /** \brief Default constructor
        \param orb Container for the paired set of orbitals
        \param F Matrix containing one-body integrals in AO basis
     **/
    one_body_rscf(
        wick_orbitals<Tc,Tb> &orb, 
        arma::Mat<Tf> &F) : one_body<Tc,Tf,Tb>(m_F0, m_F0, m_XFX, m_XFX)
    { 
        initialise(orb, F);
    }
        
    /** Default destructor **/
    virtual ~one_body_rscf() { }

private:
    /** \brief Initialise intermediates with a given set of orbitals 
        \param orb Container for the paired set of orbitals
        \param F Matrix containing one-body integrals in AO basis
     **/
    void initialise(wick_orbitals<Tc,Tb> &orb, arma::Mat<Tf> &F);
};

extern template class one_body_rscf<double, double, double>;
extern template class one_body_rscf<std::complex<double>, double, double>;
extern template class one_body_rscf<std::complex<double>, std::complex<double>, double>;
extern template class one_body_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme

#endif // LIBGNME_ONE_BODY_RSCF_H
