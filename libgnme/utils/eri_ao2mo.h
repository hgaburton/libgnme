#ifndef LIBGNME_ERI_AO2MO_H
#define LIBGNME_ERI_AO2MO_H

#include <armadillo> 

namespace libgnme {

/** \brief Perform two-electron integral transform from AO to MO basis using chemists
           indexing (C1 C2 | C3 C4)
    \param C1 Coefficients of index 1 in AO basis
    \param C2 Coefficients of index 2 in AO basis
    \param C3 Coefficients of index 3 in AO basis
    \param C4 Coefficients of index 4 in AO basis
    \param IIao Matrix representation of two-electron integrals in AO basis
    \param IImo Output matrix representation of two-electron integrals in MO basis
    \param antisym Antisymmetrise the integrals if true
 **/
template<typename Tc, typename Tb>
void eri_ao2mo(
    arma::Mat<Tc> &C1, arma::Mat<Tc> &C2, arma::Mat<Tc> &C3, arma::Mat<Tc> &C4, 
    arma::Mat<Tb> &IIao, arma::Mat<Tc> &IImo, 
    size_t nmo, bool antisym);

/** \brief Perform two-electron integral transform from AO to MO basis using chemists
           indexing (C1 C2 | C3 C4)
           Compute separate integrals for Coulomb (ij|ab) and Exchange (ib|aj)
    \param C1 Coefficients of index 1 in AO basis
    \param C2 Coefficients of index 2 in AO basis
    \param C3 Coefficients of index 3 in AO basis
    \param C4 Coefficients of index 4 in AO basis
    \param IIao Matrix representation of two-electron integrals in AO basis
    \param IImo Output matrix representation of two-electron integrals in MO basis
    \param antisym Antisymmetrise the integrals if true
 **/
template<typename Tc, typename Tb>
void eri_ao2mo_split(
    arma::Mat<Tc> &C1, arma::Mat<Tc> &C2, arma::Mat<Tc> &C3, arma::Mat<Tc> &C4, 
    arma::Mat<Tb> &IIao, arma::Mat<Tc> &II_J, arma::Mat<Tc> &II_K,
    size_t nmo, bool antisym);

} // namespace libgnme

#endif // LIBGNME_ERI_AO2MO_H
