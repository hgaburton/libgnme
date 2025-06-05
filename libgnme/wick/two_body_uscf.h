#ifndef LIBGNME_WICK_TWO_BODY_USCF_H
#define LIBGNME_WICK_TWO_BODY_USCF_H

#include <armadillo>
#include "wick_orbitals.h"
#include "two_body.h"

namespace libgnme {

/** \brief Container for storing two-body intermediate integrals for unrestricted orbitals
    \tparam Tc Type defining orbital coefficient
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class two_body_uscf : public two_body<Tc,Tf,Tb>
{
private:
    arma::Col<Tc> m_Vaa; //!< Zeroth-order alpha/alpha term [nza]
    arma::Col<Tc> m_Vbb; //!< Zeroth-order beta/beta   term [nzb]
    arma::Mat<Tc> m_Vab; //!< Zeroth-order alpha/beta  term [nza,nzb]

    arma::field<arma::Mat<Tc> > m_XVaXa; //!< First-order coupling (alpha/alpha) [nza,nza][2*nmo,2*nmo]
    arma::field<arma::Mat<Tc> > m_XVbXb; //!< First-order coupling (beta/beta)   [nzb,nzb][2*nmo,2*nmo]
    arma::field<arma::Mat<Tc> > m_XVaXb; //!< First-order coupling (alpha/beta)  [nza,nzb][2*nmo,2*nmo]
    arma::field<arma::Mat<Tc> > m_XVbXa; //!< First-order coupling (beta/alpha)  [nzb,nza][2*nmo,2*nmo]

    arma::field<arma::Mat<Tc> > m_IIaa; //!< Two-electron integrals (alpha/alpha) [nza,nza][4*nmo^2,4*nmo^2]
    arma::field<arma::Mat<Tc> > m_IIbb; //!< Two-electron integrals (beta/beta)   [nzb,nzb][4*nmo^2,4*nmo^2]
    arma::field<arma::Mat<Tc> > m_IIab; //!< Two-electron integrals (alpha/beta)  [nza,nzb][4*nmo^2,4*nmo^2]
    arma::field<arma::Mat<Tc> > m_IIba; //!< Two-electron integrals (beta/alpha)  [nzb,nza][4*nmo^2,4*nmo^2]

public:
    /** \brief Constructor from unrestricted orbital pairs
        \param orba Container for Lowdin-paired alpha orbitals
        \param orbb Container for Lowdin-paired beta orbitals
        \param V Two-electron integrals in AO basis (pq|rs) = V(p*nmo+q,r*nmo+s)
     **/
    two_body_uscf(
        wick_orbitals<Tc,Tb> &orba, 
        wick_orbitals<Tc,Tb> &orbb,
        arma::Mat<Tb> &V) :
    two_body<Tc,Tf,Tb>(m_Vaa, m_Vbb, m_Vab, 
                       m_XVaXa, m_XVaXb, 
                       m_XVbXa, m_XVbXb, 
                       m_IIaa, m_IIab, m_IIba, m_IIbb)
    {
        initialise(orba, orbb, V);
    }

    /** \brief Default destructor **/
    virtual ~two_body_uscf() { }

private:
    /** \brief Initialise intermediate terms from unrestricted orbital pairs
        \param orba Container for Lowdin-paired alpha orbitals
        \param orbb Container for Lowdin-paired beta orbitals
        \param V Two-electron integrals in AO basis (pq|rs) = V(p*nmo+q,r*nmo+s)
     **/
    void initialise(
        wick_orbitals<Tc,Tb> &orba, 
        wick_orbitals<Tc,Tb> &orbb,
        arma::Mat<Tb> &V);
};

extern template class two_body_uscf<double, double, double>;
extern template class two_body_uscf<std::complex<double>, double, double>;
extern template class two_body_uscf<std::complex<double>, std::complex<double>, double>;
extern template class two_body_uscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme

#endif // LIBGNME_WICK_TWO_BODY_USCF_H
