#ifndef LIBGNME_WICK_TWO_BODY_RSCF_H
#define LIBGNME_WICK_TWO_BODY_RSCF_H

#include <armadillo>
#include "wick_orbitals.h"
#include "two_body.h"

namespace libgnme {

/** \brief Container for storing two-body intermediate integrals for restricted orbitals
    \tparam Tc Type defining orbital coefficient
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class two_body_rscf : public two_body<Tc,Tf,Tb>
{
private:
    arma::Col<Tc> m_Vss; //!< Zeroth-order coupling for same-spin electrons [nz]
    arma::Mat<Tc> m_Vst; //!< Zeroth-order coupling for different-spin electrons [nz,nz]

    arma::field<arma::Mat<Tc> > m_XVsXs; //!< First-order coupling for same-spin electrons [nz,nz][2*nmo,2*nmo]
    arma::field<arma::Mat<Tc> > m_XVsXt; //!< First-order coupling for different-spin electrons [nz,nz][2*nmo,2*nmo]

    arma::field<arma::Mat<Tc> > m_IIss; //!< Same-spin two-electron ERIs [nz,nz][4*nmo^2,4*nmo^2]
    arma::field<arma::Mat<Tc> > m_IIst; //!< Different-spin two-electron ERIs [nz,nz][4*nmo^2,4*nmo^2]

public:
    /** \brief Constructor from a single set of Lowdin-paired orbitals
        \param orb Container for Lowdin-paired orbitals
        \param V Two-electron integrals in AO basis (pq|rs) = V(p*nmo+q,r*nmo+s)
     **/
    two_body_rscf(
        wick_orbitals<Tc,Tb> &orb, 
        arma::Mat<Tb> &V) :
    two_body<Tc,Tf,Tb>(m_Vss, m_Vss, m_Vst, 
                       m_XVsXs, m_XVsXt, m_XVsXt, m_XVsXs, 
                       m_IIss, m_IIst, m_IIst, m_IIss)
    {
        initialise(orb, V);
    }

    /** \brief Default destructor **/
    virtual ~two_body_rscf() { }

private:
    /** \brief Initialise intermediate terms from restricted orbital pair
        \param orb Container for Lowdin-paired orbitals
        \param V Two-electron integrals (pq|rs) = V(p*nmo+q,r*nmo+s)
     **/
    void initialise(wick_orbitals<Tc,Tb> &orb, arma::Mat<Tb> &V);
};

extern template class two_body_rscf<double, double, double>;
extern template class two_body_rscf<std::complex<double>, double, double>;
extern template class two_body_rscf<std::complex<double>, std::complex<double>, double>;
extern template class two_body_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme

#endif // LIBGNME_WICK_TWO_BODY_RSCF_H
