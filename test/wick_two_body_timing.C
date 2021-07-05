#include <iostream>
#include <armadillo>
#include <libgnme/lowdin_pair.h>
#include <libgnme/wick.h>
#include <iomanip>
#include <libgnme/linalg.h>
#include <libgnme/slater_uscf.h>
#include <chrono>
#include <ctime>

typedef std::complex<double> cx_double;

using namespace libgnme;

template<typename T>
int test_real_uhf(size_t thresh, size_t nbsf, size_t nelec) 
{
    std::ostringstream tnss;
    tnss << "libgnme::wick_two_body(" << thresh << ")";
    std::cout << tnss.str() << std::endl;

    // Set random number seed
    arma::arma_rng::set_seed(7);

    // Define dimensions
    size_t nmo = nbsf, ndets = 2, nocca = nelec, noccb = nelec;
    std::cout << " nbsf  = " << nbsf << std::endl;
    std::cout << " nelec = " << nelec << std::endl;

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
    std::cout << "Setup matrix builders...";
    slater_uscf<T,T,double> slat(nbsf, nmo, nocca, noccb, S);
    wick<T,T,double> mb(nbsf, nmo, nocca, noccb, S);
    slat.add_two_body(II);
    mb.add_two_body(II);
    std::cout << " done" << std::endl;
    size_t nza = mb.m_nza;
    std::cout << "Alpha zeros = " << nza << std::endl;

    // Get access to coefficients
    arma::Mat<T> Cx_a(C.slice(0).colptr(0), nbsf, nmo, true, true);
    arma::Mat<T> Cx_b(C.slice(0).colptr(nmo), nbsf, nmo, true, true);
    arma::Mat<T> Cw_a(C.slice(1).colptr(0), nbsf, nmo, true, true);
    arma::Mat<T> Cw_b(C.slice(1).colptr(nmo), nbsf, nmo, true, true);

    // Setup orbitals
    std::cout << "Setup orbitals..." << std::flush;
    mb.setup_orbitals(C.slice(0), C.slice(1));
    std::cout << " done" << std::endl;

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
//        lowdin_two_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, II, S, slowdin, vlowdin);

    }

    auto t_start = std::chrono::system_clock::now();
    auto t_end = std::chrono::system_clock::now();
    double ncount = 0.0;
    /* Single <=> Reference
    //std::cout << "Single <=> Reference..." << std::endl;

    // Nonorthogonal Wick's Theorem
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<nmo; a++)
    {
        T swick, vwick;
        arma::umat xahp(1,2), xbhp(0,2);
        arma::umat wahp(0,2), wbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);
        ncount += 1;
    }
    t_end = std::chrono::system_clock::now();
    */
    auto dtwick = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 

    /* Generalised Slater-Condon
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<nmo; a++)
    {
        T slowdin, vlowdin;
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        xocca(i) = a;
        arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
        arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
        slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
        ncount += 1;
    }
    t_end = std::chrono::system_clock::now();
    */
    auto dtslat = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    //std::cout << "          Wicks Theorem = " << std::setprecision(3) << std::fixed << dtwick / 1000.0 << " ms" << std::endl;
    //std::cout << "    Slater-Condon Rules = " << std::setprecision(3) << std::fixed << dtslat / 1000.0 << " ms" << std::endl;


    // Single <=> Single
    std::cout << "Single <=> Single..." << std::endl;

    // Nonorthogonal Wick's Theorem
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    for(size_t i=0; i<nocca; i++)
    for(size_t j=0; j<nocca; j++)
    for(size_t a=nocca; a<nmo; a++)
    for(size_t b=nocca; b<nmo; b++)
    {
        // Wick test
        arma::umat xahp(1,2), xbhp(0,2);
        arma::umat wahp(1,2), wbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        wahp(0,0) = j; wahp(0,1) = b;
        T swick = 0.0, vwick = 0.0;
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);
        ncount += 1;
    }
    t_end = std::chrono::system_clock::now();
    dtwick = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 

    // Generalised Slater-Condon
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    for(size_t i=0; i<nocca; i++)
    for(size_t j=0; j<nocca; j++)
    for(size_t a=nocca; a<nmo; a++)
    for(size_t b=nocca; b<nmo; b++)
    {
        // Standard overlap
        T slowdin = 0.0, vlowdin = 0.0;
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        xocca(i) = a;
        wocca(j) = b;
        arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
        arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
        slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
        ncount += 1;
    }
    t_end = std::chrono::system_clock::now();
    dtslat = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    std::cout << "         Basis Set Size = " << std::setprecision(0) << std::fixed << nbsf << std::endl;
    std::cout << "          Wicks Theorem = " << std::setprecision(6) << std::fixed << dtwick / 1000.0 << " ms" << std::endl;
    std::cout << "    Slater-Condon Rules = " << std::setprecision(6) << std::fixed << dtslat / 1000.0 << " ms" << std::endl;

    /* Alpha T-ref
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
                      << "} | V | [" << iw << "]_{" << i <<"," << j << "}^{" << a << "," << b << "} >" << std::endl;
            std::cout << "V_wick   = " << std::setprecision(16) << vwick << std::endl;
            std::cout << "V_lowdin = " << std::setprecision(16) << vlowdin << std::endl;
            return 1;
        }
    }

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
        T swick = 0.0, vwick = 0.0;
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);

        // Standard overlap
        T slowdin = 0.0, vlowdin = 0.0;
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        xocca(i) = a;
        xocca(j) = b;
        woccb(k) = c;
        woccb(l) = d;
        arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
        arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
        slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
        //lowdin_one_body(Cx_occa, Cx_occb, Cw_occa, Cw_occb, ha, hb, S, slowdin, flowdin);

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
        if(std::abs(vwick - vlowdin) > std::pow(0.1, thresh))
        { 
            std::cout << "< [" << ix << "]_{" << i << "," << j << "}^{" << a << "," << b 
                    << "} | V | [" << iw << "]_{" << k << "," << l << "}^{" << c << "," << d 
                    << "} >" << std::endl;
            std::cout << "V_wick   = " << std::setprecision(16) << vwick << std::endl;
            std::cout << "V_lowdin = " << std::setprecision(16) << vlowdin << std::endl;
            return 1;
        }
    }
    */

    return 0;
}

int main() {

    int ret = 0;
    for(size_t nbsf=2; nbsf<31; nbsf++)
        ret = ret | test_real_uhf<double>(7,nbsf,2);
    return ret;
}
