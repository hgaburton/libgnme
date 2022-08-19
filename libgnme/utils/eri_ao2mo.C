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
    arma::Mat<Tb> &IIao, arma::Mat<Tc> &II_J, arma::Mat<Tc> &II_K, size_t nmo, bool antisym)
{
    // Check the dimensions of input coefficients
    assert(C1.n_cols == nmo);
    assert(C2.n_cols == nmo);
    assert(C3.n_cols == nmo);
    assert(C4.n_cols == nmo);

    // Check the dimensions of the output array
    size_t nbsf = C1.n_rows;
    assert(IIao.n_rows == nbsf * nbsf);
    assert(IIao.n_cols == nbsf * nbsf);

    // Create temporary memory for integral transform
    size_t dim = std::max(nmo,nbsf);
    arma::Mat<Tc> IItmp1(dim*dim, dim*dim, arma::fill::zeros);
    arma::Mat<Tc> IItmp2(dim*dim, dim*dim, arma::fill::zeros);

    // (pq|r4)
    IItmp1.zeros();
    #pragma omp parallel for schedule(static) collapse(4)
    for(size_t p=0; p<nbsf; p++)
    for(size_t q=0; q<nbsf; q++)
    for(size_t r=0; r<nbsf; r++)
    for(size_t l=0; l<nmo; l++)
        for(size_t s=0; s<nbsf; s++)
            IItmp1(p*nbsf+q, r*nmo+l) += IIao(p*nbsf+q, r*nbsf+s) * C4(s,l);

    // (pq|34)
    IItmp2.zeros();
    #pragma omp parallel for schedule(static) collapse(4)
    for(size_t p=0; p<nbsf; p++)
    for(size_t q=0; q<nbsf; q++)
    for(size_t k=0; k<nmo; k++)
    for(size_t l=0; l<nmo; l++)
        for(size_t r=0; r<nbsf; r++)
            IItmp2(p*nbsf+q, k*nmo+l) += IItmp1(p*nbsf+q, r*nmo+l) * conj2(C3(r,k));
     
    // (p2|34)
    IItmp1.zeros();
    #pragma omp parallel for schedule(static) collapse(4)
    for(size_t p=0; p<nbsf; p++)
    for(size_t j=0; j<nmo; j++)
    for(size_t k=0; k<nmo; k++)
    for(size_t l=0; l<nmo; l++)
        for(size_t q=0; q<nbsf; q++)
            IItmp1(p*nmo+j, k*nmo+l) += IItmp2(p*nbsf+q, k*nmo+l) * C2(q,j);

    // (12|34)
    II_J.zeros();
    II_K.zeros();
    #pragma omp parallel for schedule(static) collapse(4)
    for(size_t i=0; i<nmo; i++)
    for(size_t j=0; j<nmo; j++)
    for(size_t k=0; k<nmo; k++)
    for(size_t l=0; l<nmo; l++)
        for(size_t p=0; p<nbsf; p++)
        {
            // Save Coulomb integrals
            II_J(i*nmo+j, k*nmo+l) += IItmp1(p*nmo+j, k*nmo+l) * conj2(C1(p,i));
            // Add exchange integral if same spin
            if(antisym)
                II_K(i*nmo+j, k*nmo+l) -= IItmp1(p*nmo+l, k*nmo+j) * conj2(C1(p,i)); 
        }
}
template void eri_ao2mo_split(
    arma::mat &C1, arma::mat &C2, arma::mat &C3, arma::mat &C4, 
    arma::mat &IIao, arma::mat &II_J, arma::mat &II_K, size_t nmo, bool antisym);
template void eri_ao2mo_split(
    arma::cx_mat &C1, arma::cx_mat &C2, arma::cx_mat &C3, arma::cx_mat &C4, 
    arma::mat &IIao, arma::cx_mat &II_J, arma::cx_mat &II_K, size_t nmo, bool antisym);
template void eri_ao2mo_split(
    arma::cx_mat &C1, arma::cx_mat &C2, arma::cx_mat &C3, arma::cx_mat &C4, 
    arma::cx_mat &IIao, arma::cx_mat &II_J, arma::cx_mat &II_K, size_t nmo, bool antisym);


template<typename Tc, typename Tb>
void eri_ao2mo(
    arma::Mat<Tc> &C1, arma::Mat<Tc> &C2, arma::Mat<Tc> &C3, arma::Mat<Tc> &C4, 
    arma::Mat<Tb> &IIao, arma::Mat<Tc> &IImo, size_t nmo, bool antisym)
{
    eri_ao2mo_split(C1, C2, C3, C4, IIao, IImo, IImo, nmo, antisym);
}
template void eri_ao2mo(
    arma::mat &C1, arma::mat &C2, arma::mat &C3, arma::mat &C4, 
    arma::mat &IIao, arma::mat &IImo, size_t nmo, bool antisym);
template void eri_ao2mo(
    arma::cx_mat &C1, arma::cx_mat &C2, arma::cx_mat &C3, arma::cx_mat &C4, 
    arma::mat &IIao, arma::cx_mat &IImo, size_t nmo, bool antisym);
template void eri_ao2mo(
    arma::cx_mat &C1, arma::cx_mat &C2, arma::cx_mat &C3, arma::cx_mat &C4, 
    arma::cx_mat &IIao, arma::cx_mat &IImo, size_t nmo, bool antisym);

} // namespace libgnme
