#ifndef LIBGNME_ONE_BODY_USCF_H
#define LIBGNME_ONE_BODY_USCF_H

#include "one_body.h"
#include "wick_orbitals.h"

namespace libgnme {

/** \brief Container for storing one-body intermediate integrals with unrestricted orbitals
    \tparam Tc Type defining orbital coefficient
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class one_body_uscf : public one_body<Tc,Tf,Tb>
{
private:
    arma::Col<Tc> m_F0a; //!< Store the zeroth-order (alpha) Fock terms [nza]
    arma::Col<Tc> m_F0b; //!< Store the zeroth-order (beta)  Fock terms [nzb]
    arma::field<arma::Mat<Tc> > m_XFXa; //!< Store the first-order (alpha) Fock terms [nza,nza][2*nmo,2*nmo] 
    arma::field<arma::Mat<Tc> > m_XFXb; //!< Store the first-order (beta)  Fock terms [nzb,nzb][2*nmo,2*nmo]

public:
    /** \brief Constructor for spin-independent one-body integrals
        \param orba Container for the paired set of high-spin orbitals
        \param orba Container for the paired set of low-spin orbitals
        \param F Matrix containing one-body integrals in AO basis
     **/
    one_body_uscf( 
        wick_orbitals<Tc,Tb> &orba, 
        wick_orbitals<Tc,Tb> &orbb,
        arma::Mat<Tf> &F) : 
    one_body<Tc,Tf,Tb>(m_F0a, m_F0b, m_XFXa, m_XFXb)
    { 
        initialise(orba, orbb, F, F);
    }

    /** \brief Constructor for spin-dependent one-body integrals
        \param orba Container for the paired set of high-spin orbitals
        \param orba Container for the paired set of low-spin orbitals
        \param Fa Matrix containing one-body integrals for high-spin electrons in AO basis
        \param Fb Matrix containing one-body integrals for low-spin electrons in AO basis
     **/
    one_body_uscf( 
        wick_orbitals<Tc,Tb> &orba, 
        wick_orbitals<Tc,Tb> &orbb,
        arma::Mat<Tf> &Fa,
        arma::Mat<Tf> &Fb) : 
    one_body<Tc,Tf,Tb>(m_F0a, m_F0b, m_XFXa, m_XFXb)
    { 
        initialise(orba, orbb, Fa, Fb);
    }
        
    /** Default destructor **/
    virtual ~one_body_uscf() { }

private:
    /** \brief Initialise intermediates with a given set of orbitals 
        \param orba Container for the paired set of high-spin orbitals
        \param orba Container for the paired set of low-spin orbitals
        \param Fa Matrix containing one-body integrals for high-spin electrons in AO basis
        \param Fb Matrix containing one-body integrals for low-spin electrons in AO basis
     **/
    void initialise(
        wick_orbitals<Tc,Tb> &orba, 
        wick_orbitals<Tc,Tb> &orbb,
        arma::Mat<Tf> &Fa, 
        arma::Mat<Tf> &Fb);
};

extern template class one_body_uscf<double, double, double>;
extern template class one_body_uscf<std::complex<double>, double, double>;
extern template class one_body_uscf<std::complex<double>, std::complex<double>, double>;
extern template class one_body_uscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme

#endif // LIBGNME_ONE_BODY_USCF_H
