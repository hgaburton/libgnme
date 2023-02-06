#ifndef LIBGNME_ONE_BODY_RSCF_H
#define LIBGNME_ONE_BODY_RSCF_H

#include "one_body.h"
#include "wick_orbitals.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
class one_body_rscf : public one_body<Tc,Tf,Tb>
{
private:
    // Store the 'F0' term
    arma::Col<Tc> m_F0;
    // Store the '(X/Y)F(X/Y)' super matrices (4 * nmo * nmo)
    arma::field<arma::Mat<Tc> > m_XFX;

public:
    // Constructor
    one_body_rscf(
        wick_orbitals<Tc,Tb> &orb, 
        arma::Mat<Tf> &F) : one_body<Tc,Tf,Tb>(m_F0, m_F0, m_XFX, m_XFX)
    { 
        initialise(orb, F);
    }
        
    // Default destructor
    virtual ~one_body_rscf() { }

    // Initialise 
    void initialise(wick_orbitals<Tc,Tb> &orb, arma::Mat<Tf> &F);
};

} // namespace libgnme

#endif // LIBGNME_ONE_BODY_RSCF_H
