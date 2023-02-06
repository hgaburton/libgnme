#ifndef LIBGNME_WICK_ONE_BODY_H
#define LIBGNME_WICK_ONE_BODY_H

#include <armadillo>

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
class one_body
{
public:
    // Reference to the 'F0' terms (2)
    const arma::Col<Tc> &F0a;
    const arma::Col<Tc> &F0b;
    
    // Reference to the '(X/Y)F(X/Y)' super matrices (4 * nmo * nmo)
    const arma::field<arma::Mat<Tc> > &XFXa;
    const arma::field<arma::Mat<Tc> > &XFXb; 

public:
    // Default constructor
    one_body(
        const arma::Col<Tc> &_F0a, 
        const arma::Col<Tc> &_F0b,
        const arma::field<arma::Mat<Tc> > &_XFXa, 
        const arma::field<arma::Mat<Tc> > &_XFXb) : 
        F0a(_F0a), F0b(_F0b), XFXa(_XFXa), XFXb(_XFXb)
    { }

    // Default destructor
    virtual ~one_body() { }
};

} // namespace libgnme

#endif // LIBGNME_WICK_ONE_BODY_H
