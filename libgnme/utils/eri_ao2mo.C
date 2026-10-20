#include <cassert>
#include "eri_ao2mo.h"

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

    // Each index is transformed by a matrix product. IIao(p*nbsf+q, r*nbsf+s) is stored
    // with q fastest, followed by p, s and r, and the intermediates keep this order.

    // (pq|r4) stored as [qp4, r], transforming s for each r
    arma::Mat<Tc> IItmp1(nbsf*nbsf*d4, nbsf);
    for(size_t r=0; r<nbsf; r++)
    {
        arma::Mat<Tc> IIr = arma::conv_to<arma::Mat<Tc> >::from(IIao.cols(r*nbsf, r*nbsf+nbsf-1));
        arma::Mat<Tc> IItmp(IItmp1.colptr(r), nbsf*nbsf, d4, false, true);
        IItmp = IIr * C4;
    }

    // (pq|34) stored as [qp4, 3]
    arma::Mat<Tc> IItmp2 = IItmp1 * arma::conj(C3);
    IItmp1.reset();

    // (p2|34) stored as [2, p43]
    IItmp2.reshape(nbsf, nbsf*d4*d3);
    arma::Mat<Tc> IItmp3 = C2.st() * IItmp2;
    IItmp2.reset();

    // (12|34) stored as [21, 43], which is the order of II_J
    arma::Mat<Tc> J(d1*d2, d3*d4);
    arma::Mat<Tc> C1c = arma::conj(C1);
    for(size_t kl=0; kl<d3*d4; kl++)
    {
        arma::Mat<Tc> IItmp(IItmp3.colptr(kl*nbsf), d2, nbsf, false, true);
        arma::Mat<Tc> Jkl(J.colptr(kl), d2, d1, false, true);
        Jkl = IItmp * C1c;
    }

    // Exchange integrals (12|34) -> -(14|32) if same spin
    arma::Mat<Tc> K(d1*d2, d3*d4, arma::fill::zeros);
    if(antisym)
    {
        for(size_t i=0; i<d1; i++)
        for(size_t j=0; j<d2; j++)
        for(size_t k=0; k<d3; k++)
        for(size_t l=0; l<d4; l++)
            K(i*d2+j, k*d4+l) = - J(i*d2+l, k*d4+j);
    }

    // Save Coulomb and exchange integrals, which are combined if II_J and II_K are the same
    if(&II_J == &II_K)
    {
        II_J = J + K;
    }
    else
    {
        II_J = J;
        II_K = K;
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
