#include <iostream>
#include <armadillo>
#include <libgnme/lowdin_pair.h>
#include <libgnme/wick.h>
#include <iomanip>
#include <libgnme/linalg.h>
#include <libgnme/slater_uscf.h>

typedef std::complex<double> cx_double;

using namespace libgnme;

template<typename Tc>
void lowdin_two_body(
    arma::Mat<Tc> Cx_a, arma::Mat<Tc> Cx_b,
    arma::Mat<Tc> Cw_a, arma::Mat<Tc> Cw_b, 
    arma::Mat<double> &II, arma::Mat<double> &Smat, 
    Tc &Ov, Tc &V)
{
    // Get number of occupied orbitals
    size_t nocca = Cw_a.n_cols; 
    size_t noccb = Cw_b.n_cols; 
    size_t nbsf  = Cw_a.n_rows;
    
    // Zero the output
    V = 0.0; Ov = 0.0;
    Tc Va = 0.0, Vb = 0.0, Vab = 0.0;

    // Lowdin Pair
    arma::Col<Tc> Sxx_a(nocca); Sxx_a.zeros();
    arma::Col<Tc> Sxx_b(noccb); Sxx_b.zeros();
    arma::Col<Tc> inv_Sxx_a(nocca, arma::fill::zeros);
    arma::Col<Tc> inv_Sxx_b(noccb, arma::fill::zeros);
    size_t nZeros_a = 0, nZeros_b = 0;
    arma::uvec zeros_a(Sxx_a.n_elem), zeros_b(Sxx_b.n_elem);
    lowdin_pair(Cx_a, Cw_a, Sxx_a, Smat);
    lowdin_pair(Cx_b, Cw_b, Sxx_b, Smat);

    // Compute reduced overlap
    Tc reduced_Ov = 1.0;
    reduced_overlap(Sxx_a, inv_Sxx_a, reduced_Ov, nZeros_a, zeros_a);
    reduced_overlap(Sxx_b, inv_Sxx_b, reduced_Ov, nZeros_b, zeros_b);

    if((nZeros_a + nZeros_b) == 0)
    {   
        // Save non-zero overlap
        Ov = reduced_Ov;

        // Construct co-weighted matrices
        arma::Mat<Tc> xwWa = Cw_a * arma::diagmat(inv_Sxx_a) * Cx_a.t();  
        arma::Mat<Tc> xwWb = Cw_b * arma::diagmat(inv_Sxx_b) * Cx_b.t();  

        for(size_t p=0; p < nbsf; p++)
        for(size_t q=0; q < nbsf; q++)
        for(size_t r=0; r < nbsf; r++)
        for(size_t s=0; s < nbsf; s++)
        {
            Va += 0.5 * II(p*nbsf+q,r*nbsf+s) * xwWa(q,p) * xwWa(s,r);
            Vb += 0.5 * II(p*nbsf+q,r*nbsf+s) * xwWb(q,p) * xwWb(s,r);
            Vab+= 1.0 * II(p*nbsf+q,r*nbsf+s) * xwWa(q,p) * xwWb(s,r);
            Va -= 0.5 * II(p*nbsf+s,r*nbsf+q) * xwWa(q,p) * xwWa(s,r);
            Vb -= 0.5 * II(p*nbsf+s,r*nbsf+q) * xwWb(q,p) * xwWb(s,r);
        }
    }
    else if((nZeros_a + nZeros_b) == 1)
    {   
        if(nZeros_a == 1)
        {
            arma::Mat<Tc> xwPa = Cw_a.col(zeros_a(0)) * Cx_a.col(zeros_a(0)).t();
            arma::Mat<Tc> xwWa = Cw_a * arma::diagmat(inv_Sxx_a) * Cx_a.t();  
            arma::Mat<Tc> xwWb = Cw_b * arma::diagmat(inv_Sxx_b) * Cx_b.t();  
            for(size_t p=0; p < nbsf; p++)
            for(size_t q=0; q < nbsf; q++)
            for(size_t r=0; r < nbsf; r++)
            for(size_t s=0; s < nbsf; s++)
            {
                Va += II(p*nbsf+q,r*nbsf+s) * xwPa(q,p) * xwWa(s,r);
                Vab+= II(p*nbsf+q,r*nbsf+s) * xwPa(q,p) * xwWb(s,r);
                Va -= II(p*nbsf+s,r*nbsf+q) * xwPa(q,p) * xwWa(s,r);
            }
        }
        else 
        {
            arma::Mat<Tc> xwPb = Cw_b.col(zeros_b(0)) * Cx_b.col(zeros_b(0)).t();
            arma::Mat<Tc> xwWa = Cw_a * arma::diagmat(inv_Sxx_a) * Cx_a.t();  
            arma::Mat<Tc> xwWb = Cw_b * arma::diagmat(inv_Sxx_b) * Cx_b.t();  
            for(size_t p=0; p < nbsf; p++)
            for(size_t q=0; q < nbsf; q++)
            for(size_t r=0; r < nbsf; r++)
            for(size_t s=0; s < nbsf; s++)
            {
                Vb += II(p*nbsf+q,r*nbsf+s) * xwWb(q,p) * xwPb(s,r);
                Vab+= II(p*nbsf+q,r*nbsf+s) * xwWa(q,p) * xwPb(s,r);
                Vb -= II(p*nbsf+s,r*nbsf+q) * xwWb(q,p) * xwPb(s,r);
            }
        }
    }
    else if((nZeros_a + nZeros_b) == 2)
    {   
        if(nZeros_a == 2)
        {
            arma::Mat<Tc> xwP1a = Cw_a.col(zeros_a(0)) * Cx_a.col(zeros_a(0)).t();
            arma::Mat<Tc> xwP2a = Cw_a.col(zeros_a(1)) * Cx_a.col(zeros_a(1)).t();
            for(size_t p=0; p < nbsf; p++)
            for(size_t q=0; q < nbsf; q++)
            for(size_t r=0; r < nbsf; r++)
            for(size_t s=0; s < nbsf; s++)
            {
                Va += II(p*nbsf+q,r*nbsf+s) * xwP1a(q,p) * xwP2a(s,r);
                Va -= II(p*nbsf+s,r*nbsf+q) * xwP1a(q,p) * xwP2a(s,r);
            }
        }
        else if(nZeros_b == 2)
        {
            arma::Mat<Tc> xwP1b = Cw_b.col(zeros_b(0)) * Cx_b.col(zeros_b(0)).t();
            arma::Mat<Tc> xwP2b = Cw_b.col(zeros_b(1)) * Cx_b.col(zeros_b(1)).t();
            for(size_t p=0; p < nbsf; p++)
            for(size_t q=0; q < nbsf; q++)
            for(size_t r=0; r < nbsf; r++)
            for(size_t s=0; s < nbsf; s++)
            {
                Vb += II(p*nbsf+q,r*nbsf+s) * xwP1b(q,p) * xwP2b(s,r);
                Vb -= II(p*nbsf+s,r*nbsf+q) * xwP1b(q,p) * xwP2b(s,r);
            }
        }
        else 
        {
            arma::Mat<Tc> xwP1b = Cw_b.col(zeros_b(0)) * Cx_b.col(zeros_b(0)).t();
            arma::Mat<Tc> xwP1a = Cw_a.col(zeros_a(0)) * Cx_a.col(zeros_a(0)).t();
            for(size_t p=0; p < nbsf; p++)
            for(size_t q=0; q < nbsf; q++)
            for(size_t r=0; r < nbsf; r++)
            for(size_t s=0; s < nbsf; s++)
            {
                Vab += II(p*nbsf+q,r*nbsf+s) * xwP1a(q,p) * xwP1b(s,r);
            }
        }
    }

    // Account for reduced overlap 
    V = (Va + Vb + Vab) * reduced_Ov;
}

template<typename T>
int test_real_uhf(size_t thresh) 
{

    std::ostringstream tnss;
    tnss << "libgnme::wick_two_body(" << thresh << ")";
    std::cout << tnss.str() << std::endl;

    // Set random number seed
    arma::arma_rng::set_seed(7);

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

    // Get a set of two-body integrals
    arma::Mat<double> II(nbsf*nbsf, nbsf*nbsf, arma::fill::randn);
    for(size_t p=0; p < nbsf; p++)
    for(size_t q=0; q < nbsf; q++)
    for(size_t r=0; r < nbsf; r++)
    for(size_t s=0; s < nbsf; s++)
    {
        double tmp = II(p*nbsf+q,r*nbsf+s) + II(p*nbsf+q,s*nbsf+r) 
                   + II(q*nbsf+p,r*nbsf+s) + II(q*nbsf+p,s*nbsf+r)
                   + II(r*nbsf+s,p*nbsf+q) + II(s*nbsf+r,p*nbsf+q)
                   + II(r*nbsf+s,q+nbsf*p) + II(s*nbsf+r,q*nbsf+p);

        II(p*nbsf+q,r*nbsf+s) = tmp * 0.125;
        II(q*nbsf+p,r*nbsf+s) = tmp * 0.125;
        II(p*nbsf+q,s*nbsf+r) = tmp * 0.125;
        II(q*nbsf+p,s*nbsf+r) = tmp * 0.125;
        II(r*nbsf+s,p*nbsf+q) = tmp * 0.125;
        II(r*nbsf+s,q*nbsf+p) = tmp * 0.125;
        II(s*nbsf+r,p*nbsf+q) = tmp * 0.125;
        II(s*nbsf+r,q*nbsf+p) = tmp * 0.125;
    }
    for(size_t p=0; p < nbsf; p++)
    for(size_t q=0; q < nbsf; q++)
    for(size_t r=0; r < nbsf; r++)
    for(size_t s=0; s < nbsf; s++)
    {
        assert(II(p*nbsf+q,r*nbsf+s) == II(p*nbsf+q,s*nbsf+r));
        assert(II(p*nbsf+q,r*nbsf+s) == II(q*nbsf+p,r*nbsf+s));
        assert(II(p*nbsf+q,r*nbsf+s) == II(q*nbsf+p,s*nbsf+r));
        assert(II(p*nbsf+q,r*nbsf+s) == II(r*nbsf+s,p*nbsf+q));
        assert(II(p*nbsf+q,r*nbsf+s) == II(s*nbsf+r,p*nbsf+q));
        assert(II(p*nbsf+q,r*nbsf+s) == II(r*nbsf+s,q*nbsf+p));
        assert(II(p*nbsf+q,r*nbsf+s) == II(s*nbsf+r,q*nbsf+p));
    }

    // Setup matrix builder
    slater_uscf<T,T,double> slat(nbsf, nmo, nocca, noccb, S);
    wick<T,T,double> mb(nbsf, nmo, nocca, noccb, S);
    slat.add_two_body(II);
    mb.add_two_body(II);

    // Loop over pairs to construct matrix elements
    for(size_t iw=0 ; iw < ndets ; iw++) 
    for(size_t ix=iw ; ix < ndets ; ix++) 
    {
        // Get access to coefficients
        arma::Mat<T> Cx_a(C.slice(ix).colptr(0), nbsf, nmo, true, true);
        arma::Mat<T> Cx_b(C.slice(ix).colptr(nmo), nbsf, nmo, true, true);
        arma::Mat<T> Cw_a(C.slice(iw).colptr(0), nbsf, nmo, true, true);
        arma::Mat<T> Cw_b(C.slice(iw).colptr(nmo), nbsf, nmo, true, true);

        // Setup orbitals
        mb.setup_orbitals(C.slice(ix), C.slice(iw));

        // Reference coupling
        std::cout << "< X       | W       > Ref   - Ref" << std::endl;
        {
            arma::umat xahp, xbhp;
            arma::umat wahp, wbhp;

            // Wick test
            T swick = 0.0, vwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);

            // Standard overlap
            T slowdin = 0.0, vlowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
//            lowdin_two_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, II, S, slowdin, vlowdin);

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "] | [" << iw << "] > " << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test two-body result
            if(std::abs(vwick - vlowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "] | V | [" << iw << "] > " << std::endl;
                std::cout << "V_wick   = " << std::setprecision(16) << vwick << std::endl;
                std::cout << "V_lowdin = " << std::setprecision(16) << vlowdin << std::endl;
                return 1;
            }
        }

        // Alpha singles
        std::cout << "< X_i^a   | W       > Alpha - Ref" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t a=nocca; a<nmo; a++)
        {
            arma::umat xahp(1,2), xbhp(0,2);
            arma::umat wahp(0,2), wbhp(0,2);
            xahp(0,0) = i; xahp(0,1) = a;

            // Wick test
            T swick = 0.0, vwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);

            // Standard overlap
            T slowdin = 0.0, vlowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
            //lowdin_two_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, II, S, slowdin, vlowdin);

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_{" << i << "}^{" << a << "} | [" << iw << "] > " << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            if(std::abs(vwick - vlowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_{" << i << "}^{" << a << "} | V | [" << iw << "] > " << std::endl;
                std::cout << "V_wick   = " << std::setprecision(16) << vwick << std::endl;
                std::cout << "V_lowdin = " << std::setprecision(16) << vlowdin << std::endl;
                return 1;
            }
        }

        // Alpha singles
        std::cout << "< X       | W_i^a   > Ref   - Alpha" << std::endl;
        for(size_t i=0; i<nocca; i++)
        for(size_t a=nocca; a<nmo; a++)
        {
            arma::umat xahp(0,2), xbhp(0,2);
            arma::umat wahp(1,2), wbhp(0,2);
            wahp(0,0) = i; wahp(0,1) = a;

            // Wick test
            T swick = 0.0, vwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);

            // Standard overlap
            T slowdin = 0.0, vlowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            wocca(i) = a;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
            //lowdin_two_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, II, S, slowdin, vlowdin);

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "] | [" << iw << "]_{" << i << "}^{" << a << "} > " << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test two-body result
            if(std::abs(vwick - vlowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "] | V | [" << iw << "]_{" << i << "}^{" << a << "} > " << std::endl;
                std::cout << "V_wick   = " << std::setprecision(16) << vwick << std::endl;
                std::cout << "V_lowdin = " << std::setprecision(16) << vlowdin << std::endl;
                return 1;
            }
        }

        // Alpha T-ref
        std::cout << "< X       | W_ij^ab > Ref   - Alpha" << std::endl;
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
            T swick = 0.0, vwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);

            // Standard overlap
            T slowdin = 0.0, vlowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            wocca(i) = a;
            wocca(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
            //lowdin_two_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, II, S, slowdin, vlowdin);

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
            if(std::abs(vwick - vlowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "] | V | [" 
                          << iw << "]_{" << i <<"," << j << "}^{" << a << "," << b << "} >" << std::endl;
                std::cout << "V_wick   = " << std::setprecision(16) << vwick << std::endl;
                std::cout << "V_lowdin = " << std::setprecision(16) << vlowdin << std::endl;
                return 1;
            }
        }

        // Alpha single-single
        std::cout << "< X_i^a   | W_j^b   > Alpha - Alpha" << std::endl;
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
            T swick = 0.0, vwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);

            // Standard overlap
            T slowdin = 0.0, vlowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            wocca(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
            //lowdin_two_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, II, S, slowdin, vlowdin);

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
            if(std::abs(vwick - vlowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_" << i << "^" << a 
                          << " | V | [" << iw << "]_" << j << "^" << b << " >" << std::endl;
                std::cout << "V_wick   = " << std::setprecision(16) << vwick << std::endl;
                std::cout << "V_lowdin = " << std::setprecision(16) << vlowdin << std::endl;
                return 1;
            }
        }

        // Alpha T-ref
        std::cout << "< X_ij^ab | W       > Alpha - Ref" << std::endl;
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
            T swick = 0.0, vwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);

            // Standard overlap
            T slowdin = 0.0, vlowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            xocca(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
            //lowdin_two_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, II, S, slowdin, vlowdin);

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
            if(std::abs(vwick - vlowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b 
                          << "} | V | [" << iw << "] >" << std::endl;
                std::cout << "V_wick   = " << std::setprecision(16) << vwick << std::endl;
                std::cout << "V_lowdin = " << std::setprecision(16) << vlowdin << std::endl;
                return 1;
            }
        }

        // Beta T-ref
        std::cout << "< X_ij^ab | W       > Beta/Beta - Ref" << std::endl;
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
            T swick = 0.0, vwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);

            // Standard overlap
            T slowdin = 0.0, vlowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xoccb(i) = a;
            xoccb(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
            //lowdin_two_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, II, S, slowdin, vlowdin);

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
            if(std::abs(vwick - vlowdin) > std::pow(0.1, thresh))
            {
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b
                          << "} | V | [" << iw << "] >" << std::endl;
                std::cout << "V_wick   = " << std::setprecision(16) << vwick << std::endl;
                std::cout << "V_lowdin = " << std::setprecision(16) << vlowdin << std::endl;
                return 1;
            }
        }

        // Alpha/Beta T-ref
        std::cout << "< X_ij^ab | W       > Alpha/Beta - Ref" << std::endl;
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
            T swick = 0.0, vwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);

            // Standard overlap
            T slowdin = 0.0, flowdin = 0.0, vlowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            xoccb(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
            //lowdin_two_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, II, S, slowdin, vlowdin);

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
            if(std::abs(vwick - vlowdin) > std::pow(0.1, thresh))
            {
                std::cout << "< [" << ix << "]_{" << i <<"," << j << "}^{" << a << "," << b
                          << "} | V | [" << iw << "] >" << std::endl;
                std::cout << "V_wick   = " << std::setprecision(16) << vwick << std::endl;
                std::cout << "V_lowdin = " << std::setprecision(16) << vlowdin << std::endl;
                return 1;
            }
        }

        // Alpha single - beta single
        std::cout << "< X_i^a   | W_j^b   > Alpha - Beta" << std::endl;
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
            T swick = 0.0, vwick;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);

            // Standard overlap
            T slowdin = 0.0, vlowdin;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(i) = a;
            woccb(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
            //lowdin_two_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, II, S, slowdin, vlowdin);

            // Test overlap result
            if(std::abs(swick - slowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_" << i << "^" << a 
                          << " | [" << iw << "]_" << j << "^" << b << " >" << std::endl;
                std::cout << "S_wick   = " << std::setprecision(16) << swick << std::endl;
                std::cout << "S_lowdin = " << std::setprecision(16) << slowdin << std::endl;
                return 1;
            }
            // Test twp-body result
            if(std::abs(vwick - vlowdin) > std::pow(0.1, thresh))
            { 
                std::cout <<  "< [" << ix << "]_" << i << "^" << a 
                          << " | V | [" << iw << "]_" << j << "^" << b << " >" << std::endl;
                std::cout << "V_wick   = " << std::setprecision(16) << vwick << std::endl;
                std::cout << "V_lowdin = " << std::setprecision(16) << vlowdin << std::endl;
                return 1;
            }
        }

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
            T swick = 0.0, vwick = 0.0;
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);

            // Standard overlap
            T slowdin = 0.0, vlowdin = 0.0;
            arma::uvec xocca = ref_occa, xoccb = ref_occb;
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            xocca(k) = c;
            wocca(i) = a;
            wocca(j) = b;
            arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
            arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
            slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
            //lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, vlowdin);

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
            if(std::abs(vwick - vlowdin) > std::pow(0.1, thresh))
            { 
                std::cout << "< [" << ix << "]_{" << k << "}^{" << c
                          << "} | F | [" << iw << "]_{" << i <<"," << j << "}^{" << a << "," << b << "} >" << std::endl;
                std::cout << "V_wick   = " << std::setprecision(16) << vwick << std::endl;
                std::cout << "V_lowdin = " << std::setprecision(16) << vlowdin << std::endl;
                return 1;
            }
        }
    }

    return 0;
}

int main() {

    return

    test_real_uhf<double>(7) |
    test_real_uhf<cx_double>(7) |
    0;
}
