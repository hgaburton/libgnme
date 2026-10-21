#ifndef LIBGNME_UTILS_TESTING_H
#define LIBGNME_UTILS_TESTING_H

#include <iomanip>

template<typename T>
bool array_test(arma::Col<T> &arr, arma::Col<T> &arr_ref, int thresh=8)
{
    double tol = std::pow(0.1,thresh);

    assert(arr.n_elem == arr_ref.n_elem);

    bool retval = false;

    for(size_t i=0; i < arr.n_elem; i++)
        if(std::abs(arr(i) - arr_ref(i)) > tol)
        {
            std::cout << i << ": " << std::fixed << std::setprecision(thresh+2) 
                                   << std::scientific << arr(i) - arr_ref(i) 
                                   << std::endl;
            retval = true;
        }
     
    return retval;
};

template bool array_test(arma::Col<double> &arr, arma::Col<double> &arr_ref, int thresh);
template bool array_test(arma::Col<std::complex<double> > &arr, arma::Col<std::complex<double> > &arr_ref, int thresh);

inline arma::cx_mat random_unitary(size_t n, size_t nocc, double scale, bool mix, unsigned seed)
{
    // Unitary exp(A) for a random anti-Hermitian A. Unless mix is true, the 
    // transformation is restricted to the occupied and to the virtual orbitals.
    arma::arma_rng::set_seed(seed);
    arma::cx_mat A(arma::randn<arma::mat>(n,n), arma::randn<arma::mat>(n,n));
    A = scale * (A - A.t());
    if(!mix)
    {
        A.submat(0,nocc,nocc-1,n-1).zeros();
        A.submat(nocc,0,n-1,nocc-1).zeros();
    }
    return arma::expmat(A);
}

#endif // LIBGNME_UTILS_TESTING_H
