#ifndef LIBGNME_WICK_TWO_BODY_RSCF_H
#define LIBGNME_WICK_TWO_BODY_RSCF_H

#include <armadillo>
#include "wick_orbitals.h"
#include "two_body.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
class two_body_rscf : public two_body<Tc,Tf,Tb>
{
private:
    // Reference to the 'V0' terms
    arma::Col<Tc> m_Vss; //!< Same spin
    arma::Mat<Tc> m_Vst; //!< Different spin

    // Reference to the '[X/Y](J-K)[X/Y]' super matrices (8 * nmo^2)
    arma::field<arma::Mat<Tc> > m_XVsXs; //!< Same spin
    arma::field<arma::Mat<Tc> > m_XVsXt; //!< Different spin

    // Reference to the two-electron repulsion integrals (16 * nmo^4)
    arma::field<arma::Mat<Tc> > m_IIss; //!< Same spin
    arma::field<arma::Mat<Tc> > m_IIst; //!< Different spin

public:
    // Constructor
    two_body_rscf(
        wick_orbitals<Tc,Tb> &orb, 
        arma::Mat<Tb> &V) :
    two_body<Tc,Tf,Tb>(m_Vss, m_Vss, m_Vst, 
                       m_XVsXs, m_XVsXt, m_XVsXt, m_XVsXs, 
                       m_IIss, m_IIst, m_IIst, m_IIss)
    {
        initialise(orb, V);
    }

    // Default destructor
    virtual ~two_body_rscf() { }

private:
    // Initialise 
    void initialise(wick_orbitals<Tc,Tb> &orb, arma::Mat<Tb> &V);
};

template class two_body_rscf<double, double, double>;
template class two_body_rscf<std::complex<double>, double, double>;
template class two_body_rscf<std::complex<double>, std::complex<double>, double>;
template class two_body_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme

#endif // LIBGNME_WICK_TWO_BODY_RSCF_H
