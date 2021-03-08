#include <iostream>
#include <armadillo>
#include <libgnme/lowdin_pair.h>
#include <libgnme/wick.h>
#include <iomanip>
#include <libgnme/linalg.h>

typedef std::complex<double> cx_double;

using namespace libgnme;

namespace {

template<typename T>
void lowdin_one_body(
    arma::Mat<T> Cx_a, arma::Mat<T> Cx_b,
    arma::Mat<T> Cw_a, arma::Mat<T> Cw_b, 
    arma::Mat<T> Ha, arma::Mat<T> Hb,
    arma::Mat<double> &Smat, T &Ov, T &H)
{
    // Get number of occupied orbitals
    size_t nocca = Cw_a.n_cols; 
    size_t noccb = Cw_b.n_cols; 

    // Zero the output
    H = 0.0; Ov = 0.0;

    // Lowdin Pair
    arma::Col<T> Sxx_a(nocca); Sxx_a.zeros();
    arma::Col<T> Sxx_b(noccb); Sxx_b.zeros();
    arma::Col<T> inv_Sxx_a(nocca, arma::fill::zeros);
    arma::Col<T> inv_Sxx_b(noccb, arma::fill::zeros);
    size_t nZeros_a = 0, nZeros_b = 0;
    arma::uvec zeros_a(Sxx_a.n_elem), zeros_b(Sxx_b.n_elem);
    libgnme::lowdin_pair(Cx_a, Cw_a, Sxx_a, Smat);
    libgnme::lowdin_pair(Cx_b, Cw_b, Sxx_b, Smat);

    // Compute reduced overlap
    T reduced_Ov = 1.0;
    libgnme::reduced_overlap(Sxx_a, inv_Sxx_a, reduced_Ov, nZeros_a, zeros_a);
    libgnme::reduced_overlap(Sxx_b, inv_Sxx_b, reduced_Ov, nZeros_b, zeros_b);

    if((nZeros_a + nZeros_b) == 0)
    {   
        // Save non-zero overlap
        Ov = reduced_Ov;

        // Construct co-weighted matrices
        arma::Mat<T> xwPa = Cw_a * arma::diagmat(inv_Sxx_a) * Cx_a.t();  
        arma::Mat<T> xwPb = Cw_b * arma::diagmat(inv_Sxx_b) * Cx_b.t();  

        // Save Fock element
        H += arma::trace(Ha * xwPa) + arma::trace(Hb * xwPb);
    }
    else if((nZeros_a + nZeros_b) == 1)
    {   
        if(nZeros_a == 1)
        {
            arma::Mat<T> xwPa = Cw_a.col(zeros_a(0)) * Cx_a.col(zeros_a(0)).t();
            H += arma::trace(Ha * xwPa);
        }
        else 
        {
            arma::Mat<T> xwPb = Cw_b.col(zeros_b(0)) * Cx_b.col(zeros_b(0)).t();
            H += arma::trace(Hb * xwPb);
        }
    }

    // Account for reduced overlap 
    H *= reduced_Ov;
}
template void lowdin_one_body(
    arma::Mat<double> Cx_a, arma::Mat<double> Cx_b, 
    arma::Mat<double> Cw_a, arma::Mat<double> Cw_b, 
    arma::Mat<double> Ha, arma::Mat<double> Hb,
    arma::mat &Smat, double &Ov, double &H);
template void lowdin_one_body(
    arma::Mat<cx_double> Cx_a, arma::Mat<cx_double> Cx_b, 
    arma::Mat<cx_double> Cw_a, arma::Mat<cx_double> Cw_b, 
    arma::Mat<cx_double> Ha, arma::Mat<cx_double> Hb,
    arma::mat &Smat, cx_double &Ov, cx_double &H);

} // unnamed namespace

template<typename T>
int wick_one_body(double thresh)
{
    std::cout << "  libgnme::wick_one_body(" << thresh << ")" << std::endl;

    // Define dimensions
    size_t nbsf = 5, nmo = 5, ndets = 3, nocca = 2, noccb = 2;

    // Create random overlap matrix
    arma::mat S(nbsf, nbsf, arma::fill::randn);
    S = 0.5 * (S + S.t());
    S = S + nbsf * arma::eye(nbsf, nbsf);

    arma::Cube<T> C(nbsf, 2*nmo, ndets, arma::fill::randn);
    for(size_t idet=0; idet<ndets; idet++)
    {
        // Fill virtuals with random numbers in case they are empty
        arma::Mat<T> Ca(C.slice(idet).memptr(), nbsf, nmo, false, true);
        arma::Mat<T> Cb(C.slice(idet).colptr(nmo), nbsf, nmo, false, true);
        // Orbital overlap matrices
        arma::Mat<T> Saa = Ca.t() * S * Ca;
        arma::Mat<T> Sbb = Cb.t() * S * Cb;
        arma::Mat<T> Xa, Xb;
        orthogonalisation_matrix(nmo, Saa, 1e-10, Xa);
        orthogonalisation_matrix(nmo, Sbb, 1e-10, Xb);
        // Orthogonalize input orbitals
        Ca = Ca * Xa;
        Cb = Cb * Xb;
    }

    // Get a one-body Hamiltonian
    arma::Mat<T> ha(nbsf, nbsf, arma::fill::randn);
    ha += ha.t();
    arma::Mat<T> hb = ha;

    // Define "reference" occupation numbers
    arma::uvec ref_occa(nocca);
    arma::uvec ref_occb(noccb);
    for(size_t k=0; k<nocca; k++) ref_occa(k) = k;
    for(size_t k=0; k<noccb; k++) ref_occb(k) = k;

    // Loop over pairs to construct matrix elements
    for(size_t iw=0 ; iw < ndets ; iw++) 
    for(size_t ix=iw ; ix < ndets ; ix++) 
    {
        // Get access to coefficients
        arma::Mat<T> Cx_a(C.slice(ix).colptr(0), nbsf, nmo, true, true);
        arma::Mat<T> Cx_b(C.slice(ix).colptr(nmo), nbsf, nmo, true, true);
        arma::Mat<T> Cw_a(C.slice(iw).colptr(0), nbsf, nmo, true, true);
        arma::Mat<T> Cw_b(C.slice(iw).colptr(nmo), nbsf, nmo, true, true);

        arma::Mat<T> Sa = Cx_a.t() * S * Cw_a;
        arma::Mat<T> Sb = Cx_b.t() * S * Cw_b;

        // Setup matrix builder
        wick<T,T,double> mb(nbsf, nmo, nocca, noccb, S);
        mb.setup(C.slice(ix), C.slice(iw), ha, hb);

        // Reference coupling
        //std::cout << "< X       | W       > Ref   - Ref" << std::endl;
        {
            arma::umat xahp, xbhp;
            arma::umat wahp, wbhp;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "] | [" << iw << "] > " << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test one-body result
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "] | F | [" << iw << "] > " << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test 1RDM result
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "] | F | [" << iw << "] > " << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }

        // Alpha singles
        //std::cout << "< X_i^a   | W       > Alpha - Ref" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t a=nocca; a<nmo; a++)
        {
            arma::umat xahp(1,2), xbhp(0,2);
            arma::umat wahp(0,2), wbhp(0,2);
            xahp(0,0) = i; xahp(0,1) = a;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_{" << i << "}^{" << a << "} | [" << iw << "] > " << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test Wick's theorem result
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_{" << i << "}^{" << a << "} | F | [" << iw << "] > " << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test 1RDM result
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_{" << i << "}^{" << a << "} | F | [" << iw << "] > " << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }

        // Alpha singles
        //std::cout << "< X       | W_i^a   > Ref   - Alpha" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t a=nocca; a<nmo; a++)
        {
            arma::umat xahp(0,2), xbhp(0,2);
            arma::umat wahp(1,2), wbhp(0,2);
            wahp(0,0) = i; wahp(0,1) = a;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            wocca(i) = a;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "] | [" << iw << "]_{" << i << "}^{" << a << "} > " << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test two-body result
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "] | F | [" << iw << "]_{" << i << "}^{" << a << "} > " << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test 1RDM result
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "] | F | [" << iw << "]_{" << i << "}^{" << a << "} > " << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }

        // Alpha T-ref
        //std::cout << "< X       | W_ij^ab > Ref   - Alpha" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t j=0; j<i; j++)
        for(size_t a=nocca; a<nmo; a++)
        for(size_t b=nocca; b<a; b++)
        {
            arma::umat xahp(0,2), xbhp(0,2);
            arma::umat wahp(2,2), wbhp(0,2);
            wahp(0,0) = i; wahp(0,1) = a;
            wahp(1,0) = j; wahp(1,1) = b;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            wocca(i) = a;
            wocca(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "] | [" 
                          << iw << "]_{" << i << "," << j << "}^{" << a << "," << b << "} >" << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test two-body result
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "] | F | [" 
                          << iw << "]_{" << i <<"," << j << "}^{" << a << "," << b << "} >" << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test 1RDM result
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "] | F | [" 
                          << iw << "]_{" << i <<"," << j << "}^{" << a << "," << b << "} >" << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }

        // Alpha single-single
        //std::cout << "< X_i^a   | W_j^b   > Alpha - Alpha" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t j=0; j<nocca; j++)
        for(size_t a=nocca; a<nmo; a++)
        for(size_t b=nocca; b<nmo; b++)
        {
            arma::umat xahp(1,2), xbhp(0,2);
            arma::umat wahp(1,2), wbhp(0,2);
            xahp(0,0) = i; xahp(0,1) = a;
            wahp(0,0) = j; wahp(0,1) = b;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            wocca(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_" << i << "^" << a 
                          << " | [" << iw << "]_" << j << "^" << b << " >" << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test two-body result
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_" << i << "^" << a 
                          << " | F | [" << iw << "]_" << j << "^" << b << " >" << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test 1RDM result
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_" << i << "^" << a 
                          << " | F | [" << iw << "]_" << j << "^" << b << " >" << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }

        // Alpha T-ref
        //std::cout << "< X_ij^ab | W       > Alpha - Ref" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t j=0; j<i; j++)
        for(size_t a=nocca; a<nmo; a++)
        for(size_t b=nocca; b<a; b++)
        {
            arma::umat xahp(2,2), xbhp(0,2);
            arma::umat wahp(0,2), wbhp(0,2);
            xahp(0,0) = i; xahp(0,1) = a;
            xahp(1,0) = j; xahp(1,1) = b;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            xocca(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b 
                          << "} | [" << iw << "] >" << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test two-body result
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b 
                          << "} | F | [" << iw << "] >" << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }

        // Beta T-ref
        //std::cout << "< X_ij^ab | W       > Beta/Beta - Ref" << std::endl;
        for(size_t i=0; i<noccb; i++)
        for(size_t j=0; j<i; j++)
        for(size_t a=noccb; a<nmo; a++)
        for(size_t b=noccb; b<a; b++)
        {
            arma::umat xahp(0,2), xbhp(2,2);
            arma::umat wahp(0,2), wbhp(0,2);
            xbhp(0,0) = i; xbhp(0,1) = a;
            xbhp(1,0) = j; xbhp(1,1) = b;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp,xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xoccb(i) = a;
            xoccb(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b 
                          << "} | [" << iw << "] >" << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test one-body result
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b 
                          << "} | F | [" << iw << "] >" << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test 1RDM
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b 
                          << "} | F | [" << iw << "] >" << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }

        // Alpha/Beta T-ref
        //std::cout << "< X_ij^ab | W       > Alpha/Beta - Ref" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t j=0; j<noccb; j++)
        for(size_t a=nocca; a<nmo; a++)
        for(size_t b=noccb; b<nmo; b++)
        {
            arma::umat xahp(1,2), xbhp(1,2);
            arma::umat wahp(0,2), wbhp(0,2);
            xahp(0,0) = i; xahp(0,1) = a;
            xbhp(0,0) = j; xbhp(0,1) = b;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            xoccb(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b 
                          << "} | [" << iw << "] >" << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test one-body result
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b 
                          << "} | F | [" << iw << "] >" << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test 1RDM result
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b 
                          << "} | F | [" << iw << "] >" << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }

        // Alpha T-single
        //std::cout << "< X_ij^ab | W_k^c   > Alpha - Alpha" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t j=0; j<i; j++)
        for(size_t k=0; k<nocca; k++)
        for(size_t a=nocca; a<nmo; a++)
        for(size_t b=nocca; b<a; b++)
        for(size_t c=nocca; c<nmo; c++)
        {
            arma::umat xahp(2,2), xbhp(0,2);
            arma::umat wahp(1,2), wbhp(0,2);
            xahp(0,0) = i; xahp(0,1) = a;
            xahp(1,0) = j; xahp(1,1) = b;
            wahp(0,0) = k; wahp(0,1) = c;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            xocca(j) = b;
            wocca(k) = c;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b 
                          << "} | [" << iw << "]_{" << k << "}^{" << c << "} >" << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test overlap result
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b 
                          << "} | F | [" << iw << "]_{" << k << "}^{" << c << "} >" << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test 1RDM result
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b 
                          << "} | F | [" << iw << "]_{" << k << "}^{" << c << "} >" << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }
            
        // Alpha T-single
        //std::cout << "< X_k^c   | W_ij^ab > Alpha - Alpha" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t j=0; j<i; j++)
        for(size_t k=0; k<nocca; k++)
        for(size_t a=nocca; a<nmo; a++)
        for(size_t b=nocca; b<a; b++)
        for(size_t c=nocca; c<nmo; c++)
        {
            arma::umat xahp(1,2), xbhp(0,2);
            arma::umat wahp(2,2), wbhp(0,2);
            xahp(0,0) = k; xahp(0,1) = c;
            wahp(0,0) = i; wahp(0,1) = a;
            wahp(1,0) = j; wahp(1,1) = b;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(k) = c;
            wocca(i) = a;
            wocca(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << k << "}^{" << c
                          << "} | [" << iw << "]_{" << i <<"," << j << "}^{" << a << "," << b << "} >" << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test overlap result
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << k << "}^{" << c
                          << "} | F | [" << iw << "]_{" << i <<"," << j << "}^{" << a << "," << b << "} >" << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test 1RDM result
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << k << "}^{" << c
                          << "} | F | [" << iw << "]_{" << i <<"," << j << "}^{" << a << "," << b << "} >" << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }

        // Alpha T-T
        //std::cout << "< X_ij^ab | W_lk^cd > Alpha - Alpha" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t j=0; j<i; j++)
        for(size_t k=0; k<nocca; k++)
        for(size_t l=0; l<k; l++)
        for(size_t a=nocca; a<nmo; a++)
        for(size_t b=nocca; b<a; b++)
        for(size_t c=nocca; c<nmo; c++)
        for(size_t d=nocca; d<c; d++)
        {
            arma::umat xahp(2,2), xbhp(0,2);
            arma::umat wahp(2,2), wbhp(0,2);
            xahp(0,0) = i; xahp(0,1) = a;
            xahp(1,0) = j; xahp(1,1) = b;
            wahp(0,0) = k; wahp(0,1) = c;
            wahp(1,0) = l; wahp(1,1) = d;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            xocca(j) = b;
            wocca(k) = c;
            wocca(l) = d;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i << "," << j << "}^{" << a << "," << b 
                        << "} | [" << iw << "]_{" << k << "," << l << "}^{" << c << "," << d 
                        << "} >" << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test one-body result
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i << "," << j << "}^{" << a << "," << b 
                        << "} | F | [" << iw << "]_{" << k << "," << l << "}^{" << c << "," << d 
                        << "} >" << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test 1RDM result
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i << "," << j << "}^{" << a << "," << b 
                        << "} | F | [" << iw << "]_{" << k << "," << l << "}^{" << c << "," << d 
                        << "} >" << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }

        // Alpha single - beta single
        //std::cout << "< X_i^a   | W_j^b   > Alpha - Beta" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t j=0; j<noccb; j++)
        for(size_t a=nocca; a<nmo; a++)
        for(size_t b=noccb; b<nmo; b++)
        {
            arma::umat xahp(1,2), xbhp(0,2);
            arma::umat wahp(0,2), wbhp(1,2);
            xahp(0,0) = i; xahp(0,1) = a;
            wbhp(0,0) = j; wbhp(0,1) = b;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            woccb(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_" << i << "^" << a 
                          << " | [" << iw << "]_" << j << "^" << b << " >" << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test one-body result
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_" << i << "^" << a 
                          << " | F | [" << iw << "]_" << j << "^" << b << " >" << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test 1RDM approach
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_" << i << "^" << a 
                          << " | F | [" << iw << "]_" << j << "^" << b << " >" << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }

        // Alpha T - Beta single
        //std::cout << "< X_k^c   | W_ij^ab > Beta  - Alpha" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t j=0; j<i; j++)
        for(size_t k=0; k<noccb; k++)
        for(size_t a=nocca; a<nmo; a++)
        for(size_t b=nocca; b<a; b++)
        for(size_t c=noccb; c<nmo; c++)
        {
            arma::umat xahp(0,2), xbhp(1,2);
            arma::umat wahp(2,2), wbhp(0,2);
            xbhp(0,0) = k; xbhp(0,1) = c;
            wahp(0,0) = i; wahp(0,1) = a;
            wahp(1,0) = j; wahp(1,1) = b;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xoccb(k) = c;
            wocca(i) = a;
            wocca(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << k << "}^{" << c
                          << "} | [" << iw << "]_{" << i <<"," << j << "}^{" << a << "," << b << "} >" << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test wick's theorem
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << k << "}^{" << c
                          << "} | F | [" << iw << "]_{" << i <<"," << j << "}^{" << a << "," << b << "} >" << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test 1RDM
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << k << "}^{" << c
                          << "} | F | [" << iw << "]_{" << i <<"," << j << "}^{" << a << "," << b << "} >" << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }

        // Alpha T-T
        //std::cout << "< X_ij^ab | W_lk^cd > Alpha - Beta" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t j=0; j<i; j++)
        for(size_t k=0; k<noccb; k++)
        for(size_t l=0; l<k; l++)
        for(size_t a=nocca; a<nmo; a++)
        for(size_t b=nocca; b<a; b++)
        for(size_t c=noccb; c<nmo; c++)
        for(size_t d=noccb; d<c; d++)
        {
            arma::umat xahp(2,2), xbhp(0,2);
            arma::umat wahp(0,2), wbhp(2,2);
            xahp(0,0) = i; xahp(0,1) = a;
            xahp(1,0) = j; xahp(1,1) = b;
            wbhp(0,0) = k; wbhp(0,1) = c;
            wbhp(1,0) = l; wbhp(1,1) = d;

            // Wick test
            T swick = 0.0, fwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            xocca(j) = b;
            woccb(k) = c;
            woccb(l) = d;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

            // one-body RDM
            arma::Mat<T> RDM;
            T srdm = 0.0, frdm = 0.0;
            mb.evaluate_1rdm(xahp, xbhp, wahp, wbhp, srdm, RDM);
            frdm = arma::dot(ha, RDM.st());

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i << "," << j << "}^{" << a << "," << b 
                        << "} | [" << iw << "]_{" << k << "," << l << "}^{" << c << "," << d 
                        << "} >" << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test one-body result
            if(std::abs(fwick - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i << "," << j << "}^{" << a << "," << b 
                        << "} | F | [" << iw << "]_{" << k << "," << l << "}^{" << c << "," << d 
                        << "} >" << std::endl;
                std::cout << "F_wick   = " << std::setprecision(16) << fwick << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
            // Test one-body RDM
            if(std::abs(frdm - flowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i << "," << j << "}^{" << a << "," << b 
                        << "} | F | [" << iw << "]_{" << k << "," << l << "}^{" << c << "," << d 
                        << "} >" << std::endl;
                std::cout << "F_rdm    = " << std::setprecision(16) << frdm << std::endl;
                std::cout << "F_lowdin = " << std::setprecision(16) << flowdin << std::endl;
                return 1;
            }
        }
    }

    return 0;
}
template int wick_one_body<double>(double thresh);
template int wick_one_body<cx_double>(double thresh);

int main()
{
    return wick_one_body<double>(1e-7) || wick_one_body<cx_double>(1e-7);
}
