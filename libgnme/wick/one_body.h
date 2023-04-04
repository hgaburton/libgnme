#ifndef LIBGNME_WICK_ONE_BODY_H
#define LIBGNME_WICK_ONE_BODY_H

#include <armadillo>

namespace libgnme {

/** \brief Container for storing one-body intermediate integrals
    \tparam Tc Type defining orbital coefficient
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class one_body
{
public:
    const arma::Col<Tc> &F0a; //!< Reference to the zeroth-order Fock terms (alpha) [nza]
    const arma::Col<Tc> &F0b; //!< Reference to the zeroth-order Fock terms (beta) [nzb]
    const arma::field<arma::Mat<Tc> > &XFXa; //!< Reference to the first-order Fock terms (alpha) [nza,nza][2*nmo,2*nmo]
    const arma::field<arma::Mat<Tc> > &XFXb; //!< Reference to the first-order Fock terms (beta)  [nzb,nzb][2*nmo,2*nmo]

public:
    /** \brief Standard constructor
        \param _F0a Vector containing zeroth-order Fock terms (alpha) for different mz values.
        \param _F0b Vector containing zeroth-order Fock terms (beta) for different mz values.
        \param _XFXa Field containing first-order Fock terms (alpha). Each entry in the field is the matrix
                     corresponding to a different value of mz for the bra/ket states.
        \param _XFXb Field containing first-order Fock terms (beta). Each entry in the field is the matrix
                     corresponding to a different value of mz for the bra/ket states.
     **/
    one_body(
        const arma::Col<Tc> &_F0a, 
        const arma::Col<Tc> &_F0b,
        const arma::field<arma::Mat<Tc> > &_XFXa, 
        const arma::field<arma::Mat<Tc> > &_XFXb) : 
        F0a(_F0a), F0b(_F0b), XFXa(_XFXa), XFXb(_XFXb)
    { }

    /** Default destructor **/
    virtual ~one_body() { }
};

} // namespace libgnme

#endif // LIBGNME_WICK_ONE_BODY_H
