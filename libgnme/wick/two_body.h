#ifndef LIBGNME_WICK_TWO_BODY_H
#define LIBGNME_WICK_TWO_BODY_H

#include <armadillo>

namespace libgnme {

/** \brief Container for storing two-body intermediate integrals
    \tparam Tc Type defining orbital coefficient
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class two_body
{
public:
    const arma::Col<Tc> &Vaa; //!< Reference to zeroth-order alpha/alpha term [nza]
    const arma::Col<Tc> &Vbb; //!< Reference to zeroth-order beta/beta   term [nzb]
    const arma::Mat<Tc> &Vab; //!< Reference to zeroth-order alpha/beta  term [nza,nzb]

    const arma::field<arma::Mat<Tc> > &XVaXa; //!< Reference to first-order coupling (alpha/alpha) [nza,nza][2*nmo,2*nmo] 
    const arma::field<arma::Mat<Tc> > &XVbXb; //!< Reference to first-order coupling (beta/beta)   [nzb,nzb][2*nmo,2*nmo]
    const arma::field<arma::Mat<Tc> > &XVaXb; //!< Reference to first-order coupling (alpha/beta)  [nza,nzb][2*nmo,2*nmo]
    const arma::field<arma::Mat<Tc> > &XVbXa; //!< Reference to first-order coupling (beta/alpha)  [nzb,nza][2*nmo,2*nmo]

    arma::field<arma::Mat<Tc> > &IIaa; //!< Reference to two-electron integrals (alpha/alpha) [nza,nza][4*nmo^2,4*nmo^2]
    arma::field<arma::Mat<Tc> > &IIbb; //!< Reference to two-electron integrals (beta/beta)   [nzb,nzb][4*nmo^2,4*nmo^2]
    arma::field<arma::Mat<Tc> > &IIab; //!< Reference to two-electron integrals (alpha/beta)  [nza,nzb][4*nmo^2,4*nmo^2]
    arma::field<arma::Mat<Tc> > &IIba; //!< Reference to two-electron integrals (beta/alpha)  [nzb,nza][4*nmo^2,4*nmo^2]

public:
    /** \brief Standard constructor 
        \param _Vaa Zeroth-order coupling (alpha/alpha)
        \param _Vbb Zeroth-order coupling (beta/beta)
        \param _Vab Zeroth-order coupling (alpha/beta)
        \param _XVaXa First-order coupling terms in alpha/alpha molecular orbital basis
        \param _XVaXb First-order coupling terms in alpha/beta molecular orbital basis
        \param _XVbXa First-order coupling terms in beta/alpha molecular orbital basis
        \param _XVbXb First-order coupling terms in beta/beta molecular orbital basis
        \param _IIaa ERI in alpha/alpha molecular orbital basis
        \param _IIab ERI in alpha/beta molecular orbital basis
        \param _IIba ERI in beta/alpha molecular orbital basis
        \param _IIbb ERI in beta/beta molecular orbital basis
     **/
    two_body(
        const arma::Col<Tc> &_Vaa, const arma::Col<Tc> &_Vbb, const arma::Mat<Tc> &_Vab,
        const arma::field<arma::Mat<Tc> > &_XVaXa, const arma::field<arma::Mat<Tc> > &_XVaXb,
        const arma::field<arma::Mat<Tc> > &_XVbXa, const arma::field<arma::Mat<Tc> > &_XVbXb,
        arma::field<arma::Mat<Tc> > &_IIaa, arma::field<arma::Mat<Tc> > &_IIab,
        arma::field<arma::Mat<Tc> > &_IIba, arma::field<arma::Mat<Tc> > &_IIbb) :
      Vaa(_Vaa), Vbb(_Vbb), Vab(_Vab), 
      XVaXa(_XVaXa), XVbXb(_XVbXb), XVaXb(_XVaXb), XVbXa(_XVbXa),
      IIaa(_IIaa), IIbb(_IIbb), IIab(_IIab), IIba(_IIba)
    { }

    /** \brief Default destructor **/
    virtual ~two_body() { }
};

} // namespace libgnme

#endif // LIBGNME_WICK_TWO_BODY_H
