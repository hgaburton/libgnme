#ifndef LIBNOCI_NOCI_KERNELS_H
#define LIBNOCI_NOCI_KERNELS_H

#include <armadillo>

namespace libnoci {

/** \brief Biorthogonalise two sets of orbitals.
    \param Cw Orbital coefficients in the bra.
    \param Cx Orbital ceofficients in the ket.
    \param[out] Sxx Paired overlap eigenvalues.
    \param metric Metric matrix corresponding to underlying basis.
    \ingroup noci_matrix
**/
template<typename Tc, typename Ti>
void lowdin_pair(arma::Mat<Tc>& Cw, arma::Mat<Tc>& Cx, arma::Col<Tc>& Sxx, const arma::Mat<Ti>& metric);

/** \brief Compute inverse overlap, reduced overlap and locate orbital pairs with zero overlap.
    \param Sxx Paired overlap eigenvalues.
    \param[out] invSxx Inverse paired overlap eigenvalues (0 where Sxx[i] = 0)
    \param[out] reduced_Ov Product of non-zero elements in Sxx
    \param[out] nZeros Number of zero elements in Sxx
    \param[out] zeros Vector containing indices of zero elements in Sxx
    \ingroup noci_matrix
**/
template<typename T>
void reduced_overlap(arma::Col<T> Sxx, arma::Col<T>& invSxx, T& reduced_Ov, size_t& nZeros, arma::uvec& zeros);

/** \brief Compute NOCI density matrix with RSCF reference orbitals.
    \param C Cube containing RSCF reference determinant orbitals.
    \param Anoci Vector of NOCI coefficients.
    \param metric Atomic orbital metric (overlap) matrix.
    \param nmo Number of molecular orbitals
    \param nbsf Number of basis functions
    \param nelec Number of occupied orbitals
    \param nstates Number of reference determinants in NOCI wave function
    \param[out] P Matrix containing computed NOCI density matrix
 **/
template<typename Tc, typename Tb>
void rscf_noci_density(
    arma::Cube<Tc> C, const arma::Col<Tc> Anoci, const arma::Mat<Tb> metric, 
    const size_t nmo, const size_t nbsf, const size_t nelec, const size_t nstates,
    arma::Mat<Tc> &P); 

/** \brief Compute total NOCI density matrix with USCF reference orbitals.
    \param C Cube containing USCF reference determinant orbitals [Ca, Cb].
    \param Anoci Vector of NOCI coefficients.
    \param metric Atomic orbital metric (overlap) matrix.
    \param nmo Number of molecular orbitals
    \param nbsf Number of basis functions
    \param nelec Number of occupied orbitals
    \param nstates Number of reference determinants in NOCI wave function
    \param[out] P Matrix containing computed NOCI density matrix
 **/
template<typename Tc, typename Tb>
void uscf_noci_density(
    arma::Cube<Tc> C, const arma::Col<Tc> Anoci, const arma::Mat<Tb> metric, 
    const size_t nmo, const size_t nbsf, const size_t nalpha, const size_t nbeta, const size_t nstates,
    arma::Mat<Tc> &P); 

/** \brief Compute total NOCI spin density matrices with GSCF reference orbitals.
    \param C Cube containing USCF reference determinant orbitals [Ca, Cb].
    \param Anoci Vector of NOCI coefficients.
    \param metric Atomic orbital metric (overlap) matrix.
    \param nmo Number of molecular orbitals
    \param nbsf Number of basis functions
    \param nelec Number of occupied orbitals
    \param nstates Number of reference determinants in NOCI wave function
    \param[out] Pa Matrix containing computed alpha NOCI density matrix
    \param[out] Pb Matrix containing computed beta NOCI density matrix
 **/
template<typename Tc, typename Tb>
void uscf_noci_density(
    arma::Cube<Tc> C, const arma::Col<Tc> Anoci, const arma::Mat<Tb> metric, 
    const size_t nmo, const size_t nbsf, const size_t nalpha, const size_t nbeta, const size_t nstates,
    arma::Mat<Tc> &Pa, arma::Mat<Tc> &Pb); 

/** \brief Compute NOCI density matrix with GSCF reference orbitals.
    \param C Cube containing GSCF reference determinant orbitals.
    \param Anoci Vector of NOCI coefficients.
    \param metric Atomic orbital metric (overlap) matrix.
    \param nmo Number of molecular orbitals
    \param nbsf Number of basis functions
    \param nelec Number of occupied orbitals
    \param nstates Number of reference determinants in NOCI wave function
    \param[out] P Matrix containing computed NOCI density matrix
 **/
template<typename Tc, typename Tb>
void gscf_noci_density(
    arma::Cube<Tc> C, const arma::Col<Tc> Anoci, const arma::Mat<Tb> metric, 
    const size_t nmo, const size_t nbsf, const size_t nelec, const size_t nstates,
    arma::Mat<Tc> &P); 

/** \brief Solve the generalised eigenvalue problem
    \param dim Dimensions of the eigenvalue target matrices
    \param M Matrix to be diagonalised
    \param S Corresponding overlap matrix of generalised eigenvalue problem
    \param X Orthogonalisation transformation matrix to be identified
    \param eigval Vector of eigenvalues
    \param eigvec Matrix with eigenvectors as columns
    \param thresh Threshold for null space in overlap matrix
 **/
template<typename T>
void gen_eig_sym(
    const size_t dim, arma::Mat<T> &M, arma::Mat<T> &S, arma::Mat<T> &X, 
    arma::Col<double> &eigval, arma::Mat<T> &eigvec, double thresh);

} // namespace libnoci

#endif // LIBNOCI_NOCI_KERNELS_H
