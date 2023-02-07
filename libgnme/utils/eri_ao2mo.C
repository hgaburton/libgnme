#include <cassert>
#include "eri_ao2mo.h"

namespace {

double conj2(double x) { return x; }
std::complex<double> conj2(std::complex<double> x) { return std::conj(x); }

} // unnamed namespace

namespace libgnme {

template<typename Tc, typename Tb>
void eri_ao2mo_split(
    arma::Mat<Tc> &C1, arma::Mat<Tc> &C2, arma::Mat<Tc> &C3, arma::Mat<Tc> &C4, 
    arma::Mat<Tb> &IIao, arma::Mat<Tc> &II_J, arma::Mat<Tc> &II_K, bool antisym)
{
    // Get dimensions of columns
    size_t d1 = C1.n_cols;
    size_t d2 = C2.n_cols;
    size_t d3 = C3.n_cols; 
    size_t d4 = C4.n_cols;
    if(antisym) assert(d2 == d4);

    // Check the dimensions of the output array
    size_t nbsf = C1.n_rows;
    assert(C2.n_rows == nbsf);
    assert(C3.n_rows == nbsf);
    assert(C4.n_rows == nbsf);
    assert(IIao.n_rows == nbsf * nbsf);
    assert(IIao.n_cols == nbsf * nbsf);

    // Create temporary memory for integral transform
    arma::Mat<Tc> IItmp1;
    arma::Mat<Tc> IItmp2;

    // (pq|r4)
    IItmp1.resize(nbsf*nbsf, nbsf*d4); IItmp1.zeros();
    #pragma omp parallel for schedule(static) collapse(4)
    for(size_t p=0; p<nbsf; p++)
    for(size_t q=0; q<nbsf; q++)
    for(size_t r=0; r<nbsf; r++)
    for(size_t l=0; l<d4; l++)
        for(size_t s=0; s<nbsf; s++)
            IItmp1(p*nbsf+q, r*d4+l) += IIao(p*nbsf+q, r*nbsf+s) * C4(s,l);

    // (pq|34)
    IItmp2.resize(nbsf*nbsf,d3*d4); IItmp2.zeros();
    #pragma omp parallel for schedule(static) collapse(4)
    for(size_t p=0; p<nbsf; p++)
    for(size_t q=0; q<nbsf; q++)
    for(size_t k=0; k<d3; k++)
    for(size_t l=0; l<d4; l++)
        for(size_t r=0; r<nbsf; r++)
            IItmp2(p*nbsf+q, k*d4+l) += IItmp1(p*nbsf+q, r*d4+l) * conj2(C3(r,k));
     
    // (p2|34)
    IItmp1.resize(nbsf*d2, d3*d4); IItmp1.zeros();
    #pragma omp parallel for schedule(static) collapse(4)
    for(size_t p=0; p<nbsf; p++)
    for(size_t j=0; j<d2; j++)
    for(size_t k=0; k<d3; k++)
    for(size_t l=0; l<d4; l++)
        for(size_t q=0; q<nbsf; q++)
            IItmp1(p*d2+j, k*d4+l) += IItmp2(p*nbsf+q, k*d4+l) * C2(q,j);

    // (12|34)
    II_J.reshape(d1*d2,d3*d4); II_J.zeros();
    II_K.reshape(d1*d2,d3*d4); II_K.zeros();
    #pragma omp parallel for schedule(static) collapse(4)
    for(size_t i=0; i<d1; i++)
    for(size_t j=0; j<d2; j++)
    for(size_t k=0; k<d3; k++)
    for(size_t l=0; l<d4; l++)
        for(size_t p=0; p<nbsf; p++)
        {
            // Save Coulomb integrals
            II_J(i*d2+j, k*d4+l) += IItmp1(p*d2+j, k*d4+l) * conj2(C1(p,i));
            // Add exchange integral if same spin
            if(antisym)
                II_K(i*d2+j, k*d4+l) -= IItmp1(p*d2+l, k*d4+j) * conj2(C1(p,i)); 
        }
}
template void eri_ao2mo_split(
    arma::mat &C1, arma::mat &C2, arma::mat &C3, arma::mat &C4, 
    arma::mat &IIao, arma::mat &II_J, arma::mat &II_K, bool antisym);
template void eri_ao2mo_split(
    arma::cx_mat &C1, arma::cx_mat &C2, arma::cx_mat &C3, arma::cx_mat &C4, 
    arma::mat &IIao, arma::cx_mat &II_J, arma::cx_mat &II_K, bool antisym);
template void eri_ao2mo_split(
    arma::cx_mat &C1, arma::cx_mat &C2, arma::cx_mat &C3, arma::cx_mat &C4, 
    arma::cx_mat &IIao, arma::cx_mat &II_J, arma::cx_mat &II_K, bool antisym);


template<typename Tc, typename Tb>
void eri_ao2mo(
    arma::Mat<Tc> &C1, arma::Mat<Tc> &C2, arma::Mat<Tc> &C3, arma::Mat<Tc> &C4, 
    arma::Mat<Tb> &IIao, arma::Mat<Tc> &IImo, bool antisym)
{
    eri_ao2mo_split(C1, C2, C3, C4, IIao, IImo, IImo, antisym);
}
template void eri_ao2mo(
    arma::mat &C1, arma::mat &C2, arma::mat &C3, arma::mat &C4, 
    arma::mat &IIao, arma::mat &IImo, bool antisym);
template void eri_ao2mo(
    arma::cx_mat &C1, arma::cx_mat &C2, arma::cx_mat &C3, arma::cx_mat &C4, 
    arma::mat &IIao, arma::cx_mat &IImo, bool antisym);
template void eri_ao2mo(
    arma::cx_mat &C1, arma::cx_mat &C2, arma::cx_mat &C3, arma::cx_mat &C4, 
    arma::cx_mat &IIao, arma::cx_mat &IImo, bool antisym);

} // namespace libgnme
