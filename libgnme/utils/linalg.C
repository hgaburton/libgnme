#include "linalg.h"
#include <cassert>

namespace libgnme {

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

    // Sort eigenvalues small to high
    eigvec = eigvec.cols(arma::stable_sort_index(eigval));
    eigval = arma::sort(eigval);
}
template void gen_eig_sym(
    const size_t dim, arma::Mat<double> &M, arma::Mat<double> &S, arma::Mat<double> &X, 
    arma::Col<double> &eigval, arma::Mat<double> &eigvec, double thresh);
template void gen_eig_sym(
    const size_t dim, arma::Mat<std::complex<double> > &M, arma::Mat<std::complex<double> > &S, arma::Mat<std::complex<double> > &X, 
    arma::Col<double> &eigval, arma::Mat<std::complex<double> > &eigvec, double thresh);


template<typename T>
void adjoint_matrix(arma::Mat<T> &M, arma::Mat<T> &A, T &det, size_t &nzero, double thresh)
{
    // Compute matrix SVD
    arma::Mat<T> U, V;
    arma::vec S;
    arma::svd(U, S, V, M);

    // Compute determinant and reduced overlap
    arma::uvec  zeros(S.n_elem, arma::fill::zeros);
    arma::Col<T> invS(S.n_elem, arma::fill::zeros);
    nzero = 0;
    T red_det = arma::det(U) * arma::det(V.t()); 
    det = red_det;
    for(size_t i=0; i<S.n_elem; i++)
    {
        det *= S(i);
        if(std::abs(S(i)) > thresh) 
        {
            red_det *= S(i);
            invS(i) = 1.0 / S(i);
        }
        else zeros(nzero++) = i;
    }

    // Resize and zero output
    A.resize(M.n_rows, M.n_cols);
    A.zeros();

    // Compute adjoint depending on number of zeros
    if(nzero == 0)
    {
        for(size_t i=0; i<S.n_elem; i++)
            A += invS(i) * V.col(i) * U.col(i).t();
        A *= det; 
    }
    else if(nzero == 1)
    {
        A = red_det * V.col(zeros(0)) * U.col(zeros(0)).t();
    }
}
template void adjoint_matrix(
    arma::Mat<double> &M, arma::Mat<double> &A, 
    double &det, size_t &nzero, double thresh);
template void adjoint_matrix(
    arma::Mat<std::complex<double> > &M, arma::Mat<std::complex<double> > &A, 
    std::complex<double> &det, size_t &nzero, double thresh);

} // namespace libgnme
