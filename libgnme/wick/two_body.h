#ifndef LIBGNME_WICK_TWO_BODY_H
#define LIBGNME_WICK_TWO_BODY_H

#include <armadillo>

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
class two_body
{
public:
    // Reference to the 'V0' terms
    const arma::Col<Tc> &Vaa;
    const arma::Col<Tc> &Vbb;
    const arma::Mat<Tc> &Vab;

    // Reference to the '[X/Y](J-K)[X/Y]' super matrices (8 * nmo^2)
    const arma::field<arma::Mat<Tc> > &XVaXa;
    const arma::field<arma::Mat<Tc> > &XVbXb;
    const arma::field<arma::Mat<Tc> > &XVaXb;
    const arma::field<arma::Mat<Tc> > &XVbXa;

    // Reference to the two-electron repulsion integrals (16 * nmo^4)
    arma::field<arma::Mat<Tc> > &IIaa;
    arma::field<arma::Mat<Tc> > &IIbb;
    arma::field<arma::Mat<Tc> > &IIab;
    arma::field<arma::Mat<Tc> > &IIba;

public:
    // Default constructor
    two_body(
        const arma::Col<Tc> &_Vaa, const arma::Col<Tc> &_Vbb, const arma::Mat<Tc> &_Vab,
        const arma::field<arma::Mat<Tc> > &_XVaXa, const arma::field<arma::Mat<Tc> > &_XVaXb,
        const arma::field<arma::Mat<Tc> > &_XVbXa, const arma::field<arma::Mat<Tc> > &_XVbXb,
        arma::field<arma::Mat<Tc> > &_IIaa, arma::field<arma::Mat<Tc> > &_IIab,
        arma::field<arma::Mat<Tc> > &_IIba, arma::field<arma::Mat<Tc> > &_IIbb) :
      Vaa(_Vaa), Vbb(_Vbb), Vab(_Vab), 
      XVaXa(_XVaXa), XVbXb(_XVbXb), XVaXb(_XVaXb), XVbXa(_XVbXa),
      IIaa(_IIaa), IIab(_IIab), IIba(_IIba), IIbb(_IIbb)
    { }

    // Default destructor
    virtual ~two_body() { }
};

} // namespace libgnme

#endif // LIBGNME_WICK_TWO_BODY_H
