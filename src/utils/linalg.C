#include "linalg.h"
#include <cassert>

namespace libnome {

template<typename T>
size_t orthogonalisation_matrix(size_t dim, const arma::Mat<T> &M, double thresh, arma::Mat<T> &X)
{
    // Check input
    assert(M.n_rows == dim); 
    assert(M.n_cols == dim);
    
    // Diagonalise
    arma::Mat<T> eigvec;
    arma::vec eigval;
    if(!arma::eig_sym(eigval, eigvec, M, "std"))
    {
        throw std::runtime_error("orthogonalisation_matrix: Unable to diagonalise M matrix");
    }
    
    // Remove null space
    size_t null_dim=0;
    while(eigval(null_dim) <= thresh) null_dim++;
    arma::vec eprime(dim - null_dim);
    for(size_t i=null_dim; i < dim; i++)
        eprime(i-null_dim) = 1.0 / std::sqrt(eigval(i));
    X = eigvec.cols(null_dim, dim-1) * arma::diagmat(eprime);

    return dim - null_dim;
}
template size_t orthogonalisation_matrix(size_t dim, const arma::mat &M, double thresh, arma::mat &X);
template size_t orthogonalisation_matrix(size_t dim, const arma::cx_mat &M, double thresh, arma::cx_mat &X);

template<typename T>
void gen_eig_sym(
    const size_t dim, arma::Mat<T> &M, arma::Mat<T> &S, arma::Mat<T> &X, 
    arma::Col<double> &eigval, arma::Mat<T> &eigvec, double thresh)
{
    // Check the input
    assert(M.n_rows == dim && M.n_cols == dim); 
    assert(S.n_rows == dim && S.n_cols == dim); 

    // Solve the generalised eigenvalue problem
    size_t n_span = orthogonalisation_matrix(dim, S, thresh, X);
    arma::Mat<T> ortho_M = X.t() * M * X;
    arma::eig_sym(eigval, eigvec, ortho_M, "dc");

    // Transform back to original space
    eigvec = X * eigvec;
}
template void gen_eig_sym(
    const size_t dim, arma::Mat<double> &M, arma::Mat<double> &S, arma::Mat<double> &X, 
    arma::Col<double> &eigval, arma::Mat<double> &eigvec, double thresh);
template void gen_eig_sym(
    const size_t dim, arma::Mat<std::complex<double> > &M, arma::Mat<std::complex<double> > &S, arma::Mat<std::complex<double> > &X, 
    arma::Col<double> &eigval, arma::Mat<std::complex<double> > &eigvec, double thresh);

} // namespace libnome
