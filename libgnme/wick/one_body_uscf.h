#ifndef LIBGNME_ONE_BODY_USCF_H
#define LIBGNME_ONE_BODY_USCF_H

#include "one_body.h"
#include "wick_orbitals.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
class one_body_uscf : public one_body<Tc,Tf,Tb>
{
private:
    // Store the 'F0' term
    arma::Col<Tc> m_F0a;
    arma::Col<Tc> m_F0b;

    // Store the '(X/Y)F(X/Y)' super matrices (4 * nmo * nmo)
    arma::field<arma::Mat<Tc> > m_XFXa;
    arma::field<arma::Mat<Tc> > m_XFXb;

public:
    // Constructor
    one_body_uscf( 
        wick_orbitals<Tc,Tb> &orba, 
        wick_orbitals<Tc,Tb> &orbb,
        arma::Mat<Tf> &F) : 
    one_body<Tc,Tf,Tb>(m_F0a, m_F0b, m_XFXa, m_XFXb)
    { 
        initialise(orba, orbb, F, F);
    }

    one_body_uscf( 
        wick_orbitals<Tc,Tb> &orba, 
        wick_orbitals<Tc,Tb> &orbb,
        arma::Mat<Tf> &Fa,
        arma::Mat<Tf> &Fb) : 
    one_body<Tc,Tf,Tb>(m_F0a, m_F0b, m_XFXa, m_XFXb)
    { 
        initialise(orba, orbb, Fa, Fb);
    }
        
    // Default destructor
    virtual ~one_body_uscf() { }

private:
    // Initialise 
    void initialise(
        wick_orbitals<Tc,Tb> &orba, 
        wick_orbitals<Tc,Tb> &orbb,
        arma::Mat<Tf> &Fa, 
        arma::Mat<Tf> &Fb);
};

} // namespace libgnme

#endif // LIBGNME_ONE_BODY_USCF_H
