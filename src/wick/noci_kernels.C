#include <cassert>
#include <libgscf/scf/scf_kernels.h>
#include <libgscf/util/arma_extra.h>
#include "noci_kernels.h"

namespace libnoci {

template<typename Tc, typename Ti>
void lowdin_pair(
    arma::Mat<Tc> &Cw, 
    arma::Mat<Tc> &Cx, 
    arma::Col<Tc> &Sxx, 
    const arma::Mat<Ti>& metric) 
{
    // Get initial overlap
    arma::Mat<Tc> Swx = Cw.t() * metric * Cx;

    // No pairing needed if off-diagonal is zero
    arma::Mat<Tc> diag_test = Swx - arma::diagmat(arma::diagvec(Swx));
    if(abs(diag_test).max() > 1e-10) 
    {   
        // Construct transformation matrices using SVD
        arma::Mat<Tc> U, V;
        arma::Col<double> D;
        arma::svd(U, D, V, Swx);

        // Transform orbital coefficients
        Cw = Cw*U;  
        Cx = Cx*V;
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
    arma::Col<double>& Sxx, const arma::Mat<double>& metric);
template void lowdin_pair<std::complex<double>, double>(
    arma::Mat<std::complex<double> >& Cw, arma::Mat<std::complex<double> >& Cx, 
    arma::Col<std::complex<double> >& Sxx, const arma::Mat<double>& metric);
template void lowdin_pair<std::complex<double>, std::complex<double> >(
    arma::Mat<std::complex<double> >& Cw, arma::Mat<std::complex<double> >& Cx, 
    arma::Col<std::complex<double> >& Sxx, const arma::Mat<std::complex<double> >& metric);

template<typename T>
void reduced_overlap(
    arma::Col<T> Sxx,
    arma::Col<T>& invSxx, 
    T& reduced_Ov,
    size_t& nZeros,
    arma::uvec& zeros) 
{
    // Resize and zero invSxx
    invSxx.resize(Sxx.n_elem); invSxx.zeros();

    for(int i=0; i < Sxx.n_elem; i++)
    {
        if(std::abs(Sxx(i)) > 1e-8)
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
    double& reduced_Ov, size_t& nZeros, arma::uvec& zeros);
template void reduced_overlap<std::complex<double> >(
    arma::Col<std::complex<double> > Sxx, arma::Col<std::complex<double> >& invSxx, 
    std::complex<double>& reduced_Ov, size_t& nZeros, arma::uvec& zeros);


template<typename Tc, typename Tb>
void rscf_noci_density(
    arma::Cube<Tc> C, const arma::Col<Tc> Anoci, const arma::Mat<Tb> metric, 
    const size_t nmo, const size_t nbsf, const size_t nelec, const size_t nstates, 
    arma::Mat<Tc> &P)
{
    // Verify input
    assert(C.n_cols == nmo);
    assert(C.n_rows == nbsf);
    assert(C.n_slices == Anoci.n_rows);
    assert(C.n_slices == nstates);

    // Resize outgoing NOCI density matrices
    P.resize(nbsf, nbsf); P.zeros();

    // Construct NOCI density matrix
    for(size_t iw=0; iw < nstates; iw++) 
    {
        // Extract bra coefficients
        arma::Mat<Tc> Cw = C.slice(iw).cols(0,nelec-1);

        for(size_t ix=iw; ix < nstates; ix++) 
        {        
            // Extract ket orbital coefficients
            arma::Mat<Tc> Cx = C.slice(ix).cols(0,nelec-1);

            // Biorthogonalise orbitals and compute reduced overlap
            size_t nZeros = 0;
            arma::uvec zeros(nelec);
            Tc redOv = 1.0;
            arma::Col<Tc> Sxx(nelec, arma::fill::zeros);
            arma::Col<Tc> inv_Sxx(nelec, arma::fill::zeros); 
            lowdin_pair(Cw, Cx, Sxx, metric);
            reduced_overlap(Sxx, inv_Sxx, redOv, nZeros, zeros);
            // Account for double occupancy
            redOv = 2.0 * redOv * redOv;

            // Store relevant matrices for matrix element computation
            if(nZeros == 0)
            {
                // Compute co-density matrices
                for(size_t icol=0; icol < nelec; icol++) 
                    Cx.col(icol) *= inv_Sxx(icol);
                arma::Mat<Tc> Pwx = Cx * Cw.t();

                // Compute contribution
                arma::Mat<Tc> P_tmp = redOv * conj2(Anoci(iw)) * Anoci(ix) * Pwx;

                // Accumulate density matrices
                P += P_tmp;
                if(iw != ix) P += P_tmp.t();
            }
        }
    }
} 
template void rscf_noci_density(
    arma::Cube<double> C, const arma::Col<double> Anoci, const arma::Mat<double> metric, 
    const size_t nmo, const size_t nbsf, const size_t nelec, const size_t nstates, arma::Mat<double> &P);
template void rscf_noci_density(
    arma::Cube<std::complex<double> > C, const arma::Col<std::complex<double> > Anoci, const arma::Mat<double> metric, 
    const size_t nmo, const size_t nbsf, const size_t nelec, const size_t nstates, arma::Mat<std::complex<double> > &P);


template<typename Tc, typename Tb>
void uscf_noci_density(
    arma::Cube<Tc> C, const arma::Col<Tc> Anoci, const arma::Mat<Tb> metric, 
    const size_t nmo, const size_t nbsf, const size_t nalpha, const size_t nbeta, const size_t nstates, 
    arma::Mat<Tc> &P)
{
    arma::Mat<Tc> Pa, Pb;
    uscf_noci_density(C, Anoci, metric, nmo, nbsf, nalpha, nbeta, nstates, Pa, Pb);
    P = Pa + Pb;

}
template void uscf_noci_density(
    arma::Cube<double> C, const arma::Col<double> Anoci, const arma::Mat<double> metric, 
    const size_t nmo, const size_t nbsf, const size_t nalpha, const size_t nbeta, const size_t nstates, 
    arma::Mat<double> &P);
template void uscf_noci_density(
    arma::Cube<std::complex<double> > C, const arma::Col<std::complex<double> > Anoci, const arma::Mat<double> metric, 
    const size_t nmo, const size_t nbsf, const size_t nalpha, const size_t nbeta, const size_t nstates, 
    arma::Mat<std::complex<double> > &P);


template<typename Tc, typename Tb>
void uscf_noci_density(
    arma::Cube<Tc> C, const arma::Col<Tc> Anoci, const arma::Mat<Tb> metric, 
    const size_t nmo, const size_t nbsf, const size_t nalpha, const size_t nbeta, const size_t nstates, 
    arma::Mat<Tc> &Pa, arma::Mat<Tc> &Pb)
{
    // Check input looks like
    assert(C.n_cols == 2*nmo);
    assert(C.n_rows == nbsf);
    assert(C.n_slices == Anoci.n_rows);
    assert(C.n_slices == nstates);

    // Resize outoing NOCI density matrices
    Pa.resize(nbsf, nbsf); Pa.zeros();
    Pb.resize(nbsf, nbsf); Pb.zeros();

    // Construct NOCI density matrix
    for(size_t iw=0; iw < nstates; iw++) 
    {
        // Extract bra coefficients
        arma::Mat<Tc> Cw_a(C.slice(iw).memptr(), nmo, nalpha, true, true);
        arma::Mat<Tc> Cw_b(C.slice(iw).colptr(nmo), nmo, nbeta, true, true);

        for(size_t ix=iw; ix < nstates; ix++) 
        {        
            // Extract ket orbital coefficients
            arma::Mat<Tc> Cx_a(C.slice(ix).memptr(), nmo, nalpha, true, true);
            arma::Mat<Tc> Cx_b(C.slice(ix).colptr(nmo), nmo, nbeta, true, true);

            // Biorthogonalise orbitals and compute reduced overlap
            size_t nZeros_a = 0, nZeros_b = 0;
            arma::uvec zeros_a(nalpha), zeros_b(nbeta);
            Tc redOv = 1.0;
            arma::Col<Tc> Sxx_a(nalpha, arma::fill::zeros);
            arma::Col<Tc> Sxx_b(nbeta, arma::fill::zeros);
            arma::Col<Tc> inv_Sxx_a(nalpha, arma::fill::zeros); 
            arma::Col<Tc> inv_Sxx_b(nbeta, arma::fill::zeros); 
            lowdin_pair(Cw_a, Cx_a, Sxx_a, metric);
            lowdin_pair(Cw_b, Cx_b, Sxx_b, metric);
            reduced_overlap(Sxx_a, inv_Sxx_a, redOv, nZeros_a, zeros_a);
            reduced_overlap(Sxx_b, inv_Sxx_b, redOv, nZeros_b, zeros_b);

            // Store relevant matrices for matrix element computation
            arma::Mat<Tc> Pwx_a(nbsf, nbsf, arma::fill::zeros);
            arma::Mat<Tc> Pwx_b(nbsf, nbsf, arma::fill::zeros);
            if((nZeros_a + nZeros_b) == 0)
            {
                // Compute co-density matrices
                for(size_t icol=0; icol < nalpha; icol++)
                    Cx_a.col(icol) *= inv_Sxx_a(icol);
                for(size_t icol=0; icol < nbeta; icol++)
                    Cx_b.col(icol) *= inv_Sxx_b(icol);
                Pwx_a = Cx_a * Cw_a.t();
                Pwx_b = Cx_b * Cw_b.t();
            }
            else if((nZeros_a + nZeros_b) == 1) 
            {   
                // Compute co-density matrices
                if(nZeros_a == 1)
                    Pwx_a = Cx_a.col(zeros_a(0)) * Cw_a.col(zeros_a(0)).t();
                else if(nZeros_b == 1)
                    Pwx_b = Cx_b.col(zeros_b(0)) * Cw_b.col(zeros_b(0)).t();
            } 

            // Account for overlap and NOCI coefficients
            Pwx_a = redOv * conj2(Anoci(iw)) * Anoci(ix) * Pwx_a;
            Pwx_b = redOv * conj2(Anoci(iw)) * Anoci(ix) * Pwx_b;
            
            // Add contribution
            Pa += Pwx_a;
            Pb += Pwx_b;
            // Add Hermitian conjugate if not diagonal contribution
            if(iw != ix)
            {   
                Pa += Pwx_a.t();
                Pb += Pwx_b.t();
            }
        }
    }
}
template void uscf_noci_density(
    arma::Cube<double> C, const arma::Col<double> Anoci, const arma::Mat<double> metric, 
    const size_t nmo, const size_t nbsf, const size_t nalpha, const size_t nbeta, const size_t nstates, 
    arma::Mat<double> &Pa, arma::Mat<double> &Pb);
template void uscf_noci_density(
    arma::Cube<std::complex<double> > C, const arma::Col<std::complex<double> > Anoci, const arma::Mat<double> metric, 
    const size_t nmo, const size_t nbsf, const size_t nalpha, const size_t nbeta, const size_t nstates, 
    arma::Mat<std::complex<double> > &Pa, arma::Mat<std::complex<double> > &Pb);

template<typename Tc, typename Tb>
void gscf_noci_density(
    arma::Cube<Tc> C, const arma::Col<Tc> Anoci, const arma::Mat<Tb> metric, 
    const size_t nmo, const size_t nbsf, const size_t nelec, const size_t nstates, 
    arma::Mat<Tc> &P)
{
    // Check input looks like
    assert(C.n_cols == nmo);
    assert(C.n_rows == 2*nbsf);
    assert(C.n_slices == Anoci.n_rows);
    assert(C.n_slices == nstates);

    // Construct GHF metric
    arma::Mat<Tb> metric_ghf(2*nbsf, 2*nbsf, arma::fill::zeros);
    metric_ghf.submat(0,0, nbsf-1,nbsf-1) = metric; 
    metric_ghf.submat(nbsf,nbsf, 2*nbsf-1,2*nbsf-1) = metric; 

    // Resize outoing NOCI density matrices
    P.resize(2*nbsf, 2*nbsf); P.zeros();

    // Construct NOCI density matrix
    for(size_t iw=0; iw < nstates; iw++) 
    {
        // Extract bra coefficients
        arma::Mat<Tc> Cw(C.slice(iw).memptr(), nmo, nelec, true, true);
        for(size_t ix=iw; ix < nstates; ix++) 
        {        
            // Extract ket orbital coefficients
            arma::Mat<Tc> Cx(C.slice(ix).memptr(), nmo, nelec, true, true);

            // Biorthogonalise orbitals and compute reduced overlap
            size_t nZeros = 0;
            arma::uvec zeros(nelec);
            Tc redOv = 1.0;
            arma::Col<Tc> Sxx(nelec, arma::fill::zeros);
            arma::Col<Tc> inv_Sxx(nelec, arma::fill::zeros); 
            lowdin_pair(Cw, Cx, Sxx, metric_ghf);
            reduced_overlap(Sxx, inv_Sxx, redOv, nZeros, zeros);

            // Store relevant matrices for matrix element computation
            arma::Mat<Tc> Pwx(2*nbsf, 2*nbsf, arma::fill::zeros);
            if(nZeros == 0)
            {
                // Compute co-density matrices
                for(size_t icol=0; icol < nelec; icol++)
                    Cx.col(icol) *= inv_Sxx(icol);
                Pwx = Cx * Cw.t();
            }
            else if(nZeros == 1)
            {
                // Compute co-density matrices
                Pwx = Cx.col(zeros(0)) * Cw.col(zeros(0)).t();
            }

            // Account for overlap and NOCI coefficients
            Pwx = redOv * conj2(Anoci(iw)) * Anoci(ix) * Pwx;
            
            // Add contribution
            P += Pwx;
            // Add Hermitian conjugate if not a diagonal contribution
            if(iw != ix) P += Pwx.t();
        }
    }
}
template void gscf_noci_density(
    arma::Cube<double> C, const arma::Col<double> Anoci, const arma::Mat<double> metric, 
    const size_t nmo, const size_t nbsf, const size_t nelec, const size_t nstates, arma::Mat<double> &P);
template void gscf_noci_density(
    arma::Cube<std::complex<double> > C, const arma::Col<std::complex<double> > Anoci, const arma::Mat<double> metric, 
    const size_t nmo, const size_t nbsf, const size_t nelec, const size_t nstates, arma::Mat<std::complex<double> > &P);

template<typename T>
void gen_eig_sym(
    const size_t dim, arma::Mat<T> &M, arma::Mat<T> &S, arma::Mat<T> &X, 
    arma::Col<double> &eigval, arma::Mat<T> &eigvec, double thresh)
{
    // Check the input
    assert(M.n_rows == dim && M.n_cols == dim); 
    assert(S.n_rows == dim && S.n_cols == dim); 

    // Solve the generalised eigenvalue problem
    size_t n_span = libgscf::compute_xmat(dim, S, thresh, X);
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

} // namespace libnoci
