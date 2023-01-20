#ifndef LIBGNME_LINALG_H_
#define LIBGNME_LINALG_H_

#include <cstddef>
#include <armadillo>

namespace libgnme {

/** \brief Compute the orthogonalisation transformation for a matrix
    \param dim Dimensions of input matrix
    \param M Input matrix
    \param thresh Threshold for zero eigenvalues
    \param[out] X Orthogonalisation matrix
    \ingroup gnme_utils
**/
template<typename T>
size_t orthogonalisation_matrix(size_t dim, const arma::Mat<T> &M, double thresh, arma::Mat<T> &X);

/** \brief Solve the generalised eigenvalue problem
    \param dim Dimensions of the eigenvalue target matrices
    \param M Matrix to be diagonalised
    \param S Corresponding overlap matrix of generalised eigenvalue problem
    \param X Orthogonalisation transformation matrix to be identified
    \param eigval Vector of eigenvalues
    \param eigvec Matrix with eigenvectors as columns
    \param thresh Threshold for null space in overlap matrix
    \ingroup gnme_utils
 **/
template<typename T>
void gen_eig_sym(
    const size_t dim, arma::Mat<T> &M, arma::Mat<T> &S, arma::Mat<T> &X, 
    arma::Col<double> &eigval, arma::Mat<T> &eigvec, double thresh=1e-8);


/** \brief Compute the adjoint of a matrix
    \param M Input matrix of interest
    \param A Adjoint matrix of M
    \param D Value of the determinant
 **/
template<typename T>
void adjoint_matrix(arma::Mat<T> &M, arma::Mat<T> &A, T &det, size_t &nzero, double thresh=1e-16);



} // namespace libgnme

#endif // LIBGNME_LINALG_H_
