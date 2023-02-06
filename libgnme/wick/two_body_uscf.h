#ifndef LIBGNME_WICK_TWO_BODY_USCF_H
#define LIBGNME_WICK_TWO_BODY_USCF_H

#include <armadillo>
#include "wick_orbitals.h"
#include "two_body.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
class two_body_uscf : public two_body<Tc,Tf,Tb>
{
private:
    // Reference to the 'V0' terms
    arma::Col<Tc> m_Vaa;
    arma::Col<Tc> m_Vbb;
    arma::Mat<Tc> m_Vab;

    // Reference to the '[X/Y](J-K)[X/Y]' super matrices (8 * nmo^2)
    arma::field<arma::Mat<Tc> > m_XVaXa;
    arma::field<arma::Mat<Tc> > m_XVbXb;
    arma::field<arma::Mat<Tc> > m_XVaXb;
    arma::field<arma::Mat<Tc> > m_XVbXa;

    // Reference to the two-electron repulsion integrals (16 * nmo^4)
    arma::field<arma::Mat<Tc> > m_IIaa;
    arma::field<arma::Mat<Tc> > m_IIbb;
    arma::field<arma::Mat<Tc> > m_IIab;
    arma::field<arma::Mat<Tc> > m_IIba;

public:
    // Constructor
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

    // Default destructor
    virtual ~two_body_uscf() { }

private:
    // Initialise 
    void initialise(
        wick_orbitals<Tc,Tb> &orba, 
        wick_orbitals<Tc,Tb> &orbb,
        arma::Mat<Tb> &V);
};

} // namespace libgnme

#endif // LIBGNME_WICK_TWO_BODY_USCF_H
