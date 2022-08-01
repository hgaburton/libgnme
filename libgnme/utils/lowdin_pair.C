#include <cassert>
#include "lowdin_pair.h"

namespace libgnme {

template<typename Tc, typename Ti>
void lowdin_pair(
    arma::Mat<Tc> &Cw, arma::Mat<Tc> &Cx, 
    arma::Col<Tc> &Sxx, const arma::Mat<Ti>& metric, 
    double thresh) 
{
    // Check we have a meaningful threshold
    assert(thresh > 0);

    // Get initial overlap
    arma::Mat<Tc> Swx = Cw.t() * metric * Cx;

    // No pairing needed if off-diagonal is zero
    arma::Mat<Tc> diag_test = Swx - arma::diagmat(arma::diagvec(Swx));
    if(abs(diag_test).max() > thresh) 
    {   
        // Construct transformation matrices using SVD
        arma::Mat<Tc> U, V;
        arma::Col<double> D;
        arma::svd(U, D, V, Swx);

        // Transform orbital coefficients
        Cw = Cw * U;  
        Cx = Cx * V;
        Cw.col(0) = Cw.col(0) * arma::det(U.t());
        Cx.col(0) = Cx.col(0) * arma::det(V.t());
        
        // Recompute the overlap matrix and phase factor
        Swx = Cw.t() * metric * Cx;
    }

    // Get diagonal of overlap matrix
    Sxx = arma::diagvec(Swx); 
}
template void lowdin_pair<double, double>(
    arma::Mat<double>& Cw, arma::Mat<double>& Cx, 
    arma::Col<double>& Sxx, const arma::Mat<double>& metric, double thresh);
template void lowdin_pair<std::complex<double>, double>(
    arma::Mat<std::complex<double> >& Cw, arma::Mat<std::complex<double> >& Cx, 
    arma::Col<std::complex<double> >& Sxx, const arma::Mat<double>& metric, double thresh);
template void lowdin_pair<std::complex<double>, std::complex<double> >(
    arma::Mat<std::complex<double> >& Cw, arma::Mat<std::complex<double> >& Cx, 
    arma::Col<std::complex<double> >& Sxx, const arma::Mat<std::complex<double> >& metric, double thresh);

template<typename T>
void reduced_overlap(
    arma::Col<T> Sxx, arma::Col<T>& invSxx, 
    T& reduced_Ov, size_t& nZeros, arma::uvec& zeros, double thresh) 
{
    // Check we have a meaningful threshold
    assert(thresh > 0);

    // Initialise reduced overlap
    reduced_Ov = 1.0;

    // Resize and zero invSxx
    invSxx.resize(Sxx.n_elem); invSxx.zeros();

    for(int i=0; i < Sxx.n_elem; i++)
    {
        if(std::abs(Sxx(i)) > thresh)
        { // Non-zero
            reduced_Ov *= Sxx(i);
            invSxx(i) = 1.0 / Sxx(i);
        }
        else
        { // Zero
            invSxx(i) = 1.0;
            zeros(nZeros) = i; nZeros++;
        }
    }
}
template void reduced_overlap<double>(
    arma::Col<double> Sxx, arma::Col<double>& invSxx, 
    double& reduced_Ov, size_t& nZeros, arma::uvec& zeros, double thresh);
template void reduced_overlap<std::complex<double> >(
    arma::Col<std::complex<double> > Sxx, arma::Col<std::complex<double> >& invSxx, 
    std::complex<double>& reduced_Ov, size_t& nZeros, arma::uvec& zeros, double thresh);

} // namespace libgnme
