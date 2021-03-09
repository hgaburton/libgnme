#ifndef LIBGNME_LOWDIN_PAIR_H
#define LIBGNME_LOWDIN_PAIR_H

#include <armadillo>

namespace libgnme {

/** \brief Biorthogonalise two sets of orbitals using Lowdin Pairing.
    \param Cw Orbital coefficients in the bra.
    \param Cx Orbital ceofficients in the ket.
    \param[out] Sxx Paired overlap eigenvalues.
    \param metric Metric matrix corresponding to underlying basis.
    \param thresh Floating-point cutoff threshold for testing whether overlap is diagonal (default 1e-10)
    \ingroup gnme_utils
**/
template<typename Tc, typename Ti>
void lowdin_pair(arma::Mat<Tc>& Cw, arma::Mat<Tc>& Cx, arma::Col<Tc>& Sxx, const arma::Mat<Ti>& metric, double thresh=1e-10);

/** \brief Compute inverse overlap, reduced overlap and locate orbital pairs with zero overlap.
    \param Sxx Paired overlap eigenvalues.
    \param[out] invSxx Inverse paired overlap eigenvalues (0 where Sxx[i] = 0)
    \param[out] reduced_Ov Product of non-zero elements in Sxx
    \param[out] nZeros Number of zero elements in Sxx
    \param[out] zeros Vector containing indices of zero elements in Sxx
    \param thresh Floating-point cutoff threshold for zero biorthogonal orbital overlap (default 1e-8)
    \ingroup gnme_utils
**/
template<typename T>
void reduced_overlap(arma::Col<T> Sxx, arma::Col<T>& invSxx, T& reduced_Ov, size_t& nZeros, arma::uvec& zeros, double thresh=1e-8);

}

#endif // LIBGNME_LOWDIN_PAIR_H
