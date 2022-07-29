#include <iostream>
#include <armadillo>
#include <libgnme/lowdin_pair.h>
#include <libgnme/wick_rscf.h>
#include <iomanip>
#include <libgnme/linalg.h>
#include <libgnme/slater_uscf.h>
#include <chrono>
#include <ctime>

typedef std::complex<double> cx_double;

using namespace libgnme;

template<typename T>
int test_single_ref(size_t thresh, size_t nbsf, size_t nelec, double &dtwick, double &dtslat) 
{
    std::ostringstream tnss;
    tnss << "libgnme::wick_two_body(" << thresh << ")";
    //std::cout << tnss.str() << std::endl;

    // Set random number seed
    arma::arma_rng::set_seed(7);

    // Define dimensions
    size_t nmo = nbsf, ndets = 2, nocca = nelec, noccb = nelec;
    //std::cout << " nbsf  = " << nbsf << std::endl;
    //std::cout << " nelec = " << nelec << std::endl;

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
        arma::Mat<T> Xa;
        orthogonalisation_matrix(nmo, Saa, 1e-10, Xa);
        // Orthogonalize input orbitals
        Ca = Ca * Xa;
        Cb = Ca;
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
    //std::cout << "Setup matrix builders...";
    wick_rscf<T,T,double> mb(nbsf, nmo, nocca, S);
    mb.add_two_body(II);
    slater_uscf<T,T,double> slat(nbsf, nmo, nocca, noccb, S);
    slat.add_two_body(II);
    //std::cout << " done" << std::endl;
    size_t nza = mb.m_nz;
    //std::cout << "Alpha zeros = " << nza << std::endl;

    // Get access to coefficients
    arma::Mat<T> Cx_a(C.slice(0).colptr(0), nbsf, nmo, true, true);
    arma::Mat<T> Cx_b(C.slice(0).colptr(nmo), nbsf, nmo, true, true);
    arma::Mat<T> Cw_a(C.slice(1).colptr(0), nbsf, nmo, true, true);
    arma::Mat<T> Cw_b(C.slice(1).colptr(nmo), nbsf, nmo, true, true);

    // Setup orbitals
    //std::cout << "Setup orbitals..." << std::flush;
    mb.setup_orbitals(C.slice(0), C.slice(1));
    //std::cout << " done" << std::endl;

    // Reference coupling
    //std::cout << "< X       | W       > Ref   - Ref" << std::endl;
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
    }

    auto t_start = std::chrono::system_clock::now();
    auto t_end = std::chrono::system_clock::now();
    double ncount = 0.0;
    dtwick = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    dtslat = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 

    // Nonorthogonal Wick's Theorem
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    double wick_val = 0.0;
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<nmo; a++)
    {
        // Wick test
        arma::umat xahp(1,2), xbhp(0,2);
        arma::umat wahp(0,2), wbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        T swick = 0.0, vwick = 0.0;
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);
        ncount += 1;
        wick_val += vwick;
    }
    t_end = std::chrono::system_clock::now();
    dtwick = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 

    // Generalised Slater-Condon
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    double slat_val = 0.0;
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<nmo; a++)
    {
        // Standard overlap
        T slowdin = 0.0, vlowdin = 0.0;
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        xocca(i) = a;
        arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
        arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
        slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
        ncount += 1;
        slat_val += vlowdin;
    }
    t_end = std::chrono::system_clock::now();
    dtslat = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    //std::cout << "         Basis Set Size = " << std::setprecision(0) << std::fixed << nbsf << std::endl;
    //std::cout << "          Wicks Theorem = " << std::setprecision(6) << std::fixed << dtwick / 1000.0 << " ms" << std::endl;
    //std::cout << "    Slater-Condon Rules = " << std::setprecision(6) << std::fixed << dtslat / 1000.0 << " ms" << std::endl;
    std::cout << wick_val << " " << slat_val << std::endl;

    return 0;
}

template<typename T>
int test_single_single(size_t thresh, size_t nbsf, size_t nelec, double &dtwick, double &dtslat) 
{
    std::ostringstream tnss;
    tnss << "libgnme::wick_two_body(" << thresh << ")";
    //std::cout << tnss.str() << std::endl;

    // Set random number seed
    arma::arma_rng::set_seed(7);

    // Define dimensions
    size_t nmo = nbsf, ndets = 2, nocca = nelec, noccb = nelec;
    //std::cout << " nbsf  = " << nbsf << std::endl;
    //std::cout << " nelec = " << nelec << std::endl;

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
        arma::Mat<T> Xa;
        orthogonalisation_matrix(nmo, Saa, 1e-10, Xa);
        // Orthogonalize input orbitals
        Ca = Ca * Xa;
        Cb = Ca;
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
    slat.add_two_body(II);
    wick_rscf<T,T,double> mb(nbsf, nmo, nocca, S);
    mb.add_two_body(II);
    std::cout << " done" << std::endl;

    // Get access to coefficients
    arma::Mat<T> Cx_a(C.slice(0).colptr(0), nbsf, nmo, true, true);
    arma::Mat<T> Cx_b(C.slice(0).colptr(nmo), nbsf, nmo, true, true);
    arma::Mat<T> Cw_a(C.slice(1).colptr(0), nbsf, nmo, true, true);
    arma::Mat<T> Cw_b(C.slice(1).colptr(nmo), nbsf, nmo, true, true);

    // Setup orbitals
    //std::cout << "Setup orbitals..." << std::flush;
    mb.setup_orbitals(C.slice(0), C.slice(1));
    //std::cout << " done" << std::endl;

    auto t_start = std::chrono::system_clock::now();
    auto t_end = std::chrono::system_clock::now();
    double ncount = 0.0;
    dtwick = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    dtslat = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 

    // Nonorthogonal Wick's Theorem
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    double wick_val = 0.0;
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
        wick_val += vwick;
    }
    t_end = std::chrono::system_clock::now();
    dtwick = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 

    // Generalised Slater-Condon
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    double slat_val = 0.0;
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
        slat_val += vlowdin;
    }
    t_end = std::chrono::system_clock::now();
    dtslat = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    //std::cout << "         Basis Set Size = " << std::setprecision(0) << std::fixed << nbsf << std::endl;
    //std::cout << "          Wicks Theorem = " << std::setprecision(6) << std::fixed << dtwick / 1000.0 << " ms" << std::endl;
    //std::cout << "    Slater-Condon Rules = " << std::setprecision(6) << std::fixed << dtslat / 1000.0 << " ms" << std::endl;
    std::cout << wick_val << " " << slat_val << std::endl;

    return 0;
}

template<typename T>
int test_double_single(size_t thresh, size_t nbsf, size_t nelec, double &dtwick, double &dtslat) 
{
    std::ostringstream tnss;
    tnss << "libgnme::wick_two_body(" << thresh << ")";
    //std::cout << tnss.str() << std::endl;

    // Set random number seed
    arma::arma_rng::set_seed(7);

    // Define dimensions
    size_t nmo = nbsf, ndets = 2, nocca = nelec, noccb = nelec;
    //std::cout << " nbsf  = " << nbsf << std::endl;
    //std::cout << " nelec = " << nelec << std::endl;

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
        arma::Mat<T> Xa;
        orthogonalisation_matrix(nmo, Saa, 1e-10, Xa);
        // Orthogonalize input orbitals
        Ca = Ca * Xa;
        Cb = Ca;
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
    //std::cout << "Setup matrix builders...";
    slater_uscf<T,T,double> slat(nbsf, nmo, nocca, noccb, S);
    slat.add_two_body(II);
    wick_rscf<T,T,double> mb(nbsf, nmo, nocca, S);
    mb.add_two_body(II);
    //std::cout << " done" << std::endl;

    // Get access to coefficients
    arma::Mat<T> Cx_a(C.slice(0).colptr(0), nbsf, nmo, true, true);
    arma::Mat<T> Cx_b(C.slice(0).colptr(nmo), nbsf, nmo, true, true);
    arma::Mat<T> Cw_a(C.slice(1).colptr(0), nbsf, nmo, true, true);
    arma::Mat<T> Cw_b(C.slice(1).colptr(nmo), nbsf, nmo, true, true);

    // Setup orbitals
    //std::cout << "Setup orbitals..." << std::flush;
    mb.setup_orbitals(C.slice(0), C.slice(1));
    //std::cout << " done" << std::endl;

    // Reference coupling
    //std::cout << "< X       | W       > Ref   - Ref" << std::endl;
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
    }

    auto t_start = std::chrono::system_clock::now();
    auto t_end = std::chrono::system_clock::now();
    double ncount = 0.0;
    dtwick = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    dtslat = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 

    // Nonorthogonal Wick's Theorem
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    double wick_val = 0.0;
    for(size_t i=0; i<nocca; i++)
    for(size_t j=0; j<i; j++)
    for(size_t k=0; k<nocca; k++)
    for(size_t a=nocca; a<nmo; a++)
    for(size_t b=nocca; b<a; b++)
    for(size_t c=nocca; c<nmo; c++)
    {
        // Wick test
        arma::umat xahp(2,2), xbhp(0,2);
        arma::umat wahp(1,2), wbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        xahp(1,0) = j; xahp(1,1) = b;
        wahp(0,0) = k; wahp(0,1) = c;
        T swick = 0.0, vwick = 0.0;
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);
        ncount += 1;
        wick_val += vwick;
    }
    t_end = std::chrono::system_clock::now();
    dtwick = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    std::cout << dtwick << std::endl;

    // Generalised Slater-Condon
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    double slat_val = 0.0;
    for(size_t i=0; i<nocca; i++)
    for(size_t j=0; j<i; j++)
    for(size_t k=0; k<nocca; k++)
    for(size_t a=nocca; a<nmo; a++)
    for(size_t b=nocca; b<a; b++)
    for(size_t c=nocca; c<nmo; c++)
    {
        // Standard overlap
        T slowdin = 0.0, vlowdin = 0.0;
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        xocca(i) = a;
        xocca(j) = b;
        wocca(k) = c;
        arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
        arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
        slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
        ncount += 1;
        slat_val += vlowdin;
    }
    t_end = std::chrono::system_clock::now();
    dtslat = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    //std::cout << "         Basis Set Size = " << std::setprecision(0) << std::fixed << nbsf << std::endl;
    //std::cout << "          Wicks Theorem = " << std::setprecision(6) << std::fixed << dtwick / 1000.0 << " ms" << std::endl;
    //std::cout << "    Slater-Condon Rules = " << std::setprecision(6) << std::fixed << dtslat / 1000.0 << " ms" << std::endl;
    std::cout << wick_val << " " << slat_val << std::endl;

    return 0;
}

template<typename T>
int test_double_double(size_t thresh, size_t nbsf, size_t nelec, double &dtwick, double &dtslat) 
{
    std::ostringstream tnss;
    tnss << "libgnme::wick_two_body(" << thresh << ")";
    //std::cout << tnss.str() << std::endl;

    // Set random number seed
    arma::arma_rng::set_seed(7);

    // Define dimensions
    size_t nmo = nbsf, ndets = 2, nocca = nelec, noccb = nelec;
    //std::cout << " nbsf  = " << nbsf << std::endl;
    //std::cout << " nelec = " << nelec << std::endl;

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
        arma::Mat<T> Xa;
        orthogonalisation_matrix(nmo, Saa, 1e-10, Xa);
        // Orthogonalize input orbitals
        Ca = Ca * Xa;
        Cb = Ca;
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
    //std::cout << "Setup matrix builders...";
    slater_uscf<T,T,double> slat(nbsf, nmo, nocca, noccb, S);
    slat.add_two_body(II);
    wick_rscf<T,T,double> mb(nbsf, nmo, nocca, S);
    mb.add_two_body(II);
    //std::cout << " done" << std::endl;

    // Get access to coefficients
    arma::Mat<T> Cx_a(C.slice(0).colptr(0), nbsf, nmo, true, true);
    arma::Mat<T> Cx_b(C.slice(0).colptr(nmo), nbsf, nmo, true, true);
    arma::Mat<T> Cw_a(C.slice(1).colptr(0), nbsf, nmo, true, true);
    arma::Mat<T> Cw_b(C.slice(1).colptr(nmo), nbsf, nmo, true, true);

    // Setup orbitals
    //std::cout << "Setup orbitals..." << std::flush;
    mb.setup_orbitals(C.slice(0), C.slice(1));
    //std::cout << " done" << std::endl;

    // Reference coupling
    //std::cout << "< X       | W       > Ref   - Ref" << std::endl;
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
    }

    auto t_start = std::chrono::system_clock::now();
    auto t_end = std::chrono::system_clock::now();
    double ncount = 0.0;
    dtwick = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    dtslat = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 

    // Nonorthogonal Wick's Theorem
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    double wick_val = 0.0;
    for(size_t i=0; i<nocca; i++)
    for(size_t j=0; j<i; j++)
    for(size_t k=0; k<nocca; k++)
    for(size_t l=0; l<k; l++)
    for(size_t a=nocca; a<nmo; a++)
    for(size_t b=nocca; b<a; b++)
    for(size_t c=nocca; c<nmo; c++)
    for(size_t d=nocca; d<c; d++)
    {
        // Wick test
        arma::umat xahp(2,2), xbhp(0,2);
        arma::umat wahp(2,2), wbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        xahp(1,0) = j; xahp(1,1) = b;
        wahp(0,0) = k; wahp(0,1) = c;
        wahp(1,0) = l; wahp(1,1) = d;
        T swick = 0.0, vwick = 0.0;
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);
        ncount += 1;
        wick_val += vwick;
    }
    t_end = std::chrono::system_clock::now();
    dtwick = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    std::cout << dtwick << std::endl;

    // Generalised Slater-Condon
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    double slat_val = 0.0;
    for(size_t i=0; i<nocca; i++)
    for(size_t j=0; j<i; j++)
    for(size_t k=0; k<nocca; k++)
    for(size_t l=0; l<k; l++)
    for(size_t a=nocca; a<nmo; a++)
    for(size_t b=nocca; b<a; b++)
    for(size_t c=nocca; c<nmo; c++)
    for(size_t d=nocca; d<c; d++)
    {
        // Standard overlap
        T slowdin = 0.0, vlowdin = 0.0;
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        xocca(i) = a;
        xocca(j) = b;
        wocca(k) = c;
        wocca(l) = d;
        arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
        arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
        slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
        ncount += 1;
        slat_val += vlowdin;
    }
    t_end = std::chrono::system_clock::now();
    dtslat = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    //std::cout << "         Basis Set Size = " << std::setprecision(0) << std::fixed << nbsf << std::endl;
    //std::cout << "          Wicks Theorem = " << std::setprecision(6) << std::fixed << dtwick / 1000.0 << " ms" << std::endl;
    //std::cout << "    Slater-Condon Rules = " << std::setprecision(6) << std::fixed << dtslat / 1000.0 << " ms" << std::endl;
    std::cout << wick_val << " " << slat_val << std::endl;

    return 0;
}

template<typename T>
int test_double_triple(size_t thresh, size_t nbsf, size_t nelec, double &dtwick, double &dtslat) 
{
    std::ostringstream tnss;
    tnss << "libgnme::wick_two_body(" << thresh << ")";
    //std::cout << tnss.str() << std::endl;

    // Set random number seed
    arma::arma_rng::set_seed(7);

    // Define dimensions
    size_t nmo = nbsf, ndets = 2, nocca = nelec, noccb = nelec;
    //std::cout << " nbsf  = " << nbsf << std::endl;
    //std::cout << " nelec = " << nelec << std::endl;

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
        arma::Mat<T> Xa;
        orthogonalisation_matrix(nmo, Saa, 1e-10, Xa);
        // Orthogonalize input orbitals
        Ca = Ca * Xa;
        Cb = Ca;
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
    //std::cout << "Setup matrix builders...";
    slater_uscf<T,T,double> slat(nbsf, nmo, nocca, noccb, S);
    wick_rscf<T,T,double> mb(nbsf, nmo, nocca, S);
    slat.add_two_body(II);
    mb.add_two_body(II);
    //std::cout << " done" << std::endl;

    // Get access to coefficients
    arma::Mat<T> Cx_a(C.slice(0).colptr(0), nbsf, nmo, true, true);
    arma::Mat<T> Cx_b(C.slice(0).colptr(nmo), nbsf, nmo, true, true);
    arma::Mat<T> Cw_a(C.slice(1).colptr(0), nbsf, nmo, true, true);
    arma::Mat<T> Cw_b(C.slice(1).colptr(nmo), nbsf, nmo, true, true);

    // Setup orbitals
    //std::cout << "Setup orbitals..." << std::flush;
    mb.setup_orbitals(C.slice(0), C.slice(1));
    //std::cout << " done" << std::endl;

    // Reference coupling
    //std::cout << "< X       | W       > Ref   - Ref" << std::endl;
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
    }

    auto t_start = std::chrono::system_clock::now();
    auto t_end = std::chrono::system_clock::now();
    double ncount = 0.0;
    dtwick = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    dtslat = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 

    // Nonorthogonal Wick's Theorem
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    double wick_val = 0.0;
    for(size_t i=0; i<nocca; i++)
    for(size_t j=0; j<i; j++)
    for(size_t k=0; k<nocca; k++)
    for(size_t l=0; l<k; l++)
    for(size_t m=0; m<l; m++)
    for(size_t a=nocca; a<nmo; a++)
    for(size_t b=nocca; b<a; b++)
    for(size_t c=nocca; c<nmo; c++)
    for(size_t d=nocca; d<c; d++)
    for(size_t e=nocca; e<d; e++)
    {
        // Wick test
        arma::umat xahp(2,2), xbhp(0,2);
        arma::umat wahp(3,2), wbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        xahp(1,0) = j; xahp(1,1) = b;
        wahp(0,0) = k; wahp(0,1) = c;
        wahp(1,0) = l; wahp(1,1) = d;
        wahp(2,0) = m; wahp(2,1) = e;
        T swick = 0.0, vwick = 0.0;
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, vwick);
        ncount += 1;
        wick_val += vwick;
    }
    t_end = std::chrono::system_clock::now();
    dtwick = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    std::cout << dtwick << std::endl;

    // Generalised Slater-Condon
    t_start = std::chrono::system_clock::now();
    ncount = 0.0;
    double slat_val = 0.0;
    for(size_t i=0; i<nocca; i++)
    for(size_t j=0; j<i; j++)
    for(size_t k=0; k<nocca; k++)
    for(size_t l=0; l<k; l++)
    for(size_t m=0; m<l; m++)
    for(size_t a=nocca; a<nmo; a++)
    for(size_t b=nocca; b<a; b++)
    for(size_t c=nocca; c<nmo; c++)
    for(size_t d=nocca; d<c; d++)
    for(size_t e=nocca; e<d; e++)
    {
        // Standard overlap
        T slowdin = 0.0, vlowdin = 0.0;
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        xocca(i) = a;
        xocca(j) = b;
        wocca(k) = c;
        wocca(l) = d;
        wocca(m) = e;
        arma::Mat<T> Cx_occa = Cx_a.cols(xocca), Cx_occb = Cx_b.cols(xoccb);
        arma::Mat<T> Cw_occa = Cw_a.cols(wocca), Cw_occb = Cw_b.cols(woccb);
        slat.evaluate(Cx_occa, Cx_occb, Cw_occa, Cw_occb, slowdin, vlowdin);
        ncount += 1;
        slat_val += vlowdin;
    }
    t_end = std::chrono::system_clock::now();
    dtslat = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / ncount; 
    //std::cout << "         Basis Set Size = " << std::setprecision(0) << std::fixed << nbsf << std::endl;
    //std::cout << "          Wicks Theorem = " << std::setprecision(6) << std::fixed << dtwick / 1000.0 << " ms" << std::endl;
    //std::cout << "    Slater-Condon Rules = " << std::setprecision(6) << std::fixed << dtslat / 1000.0 << " ms" << std::endl;
    std::cout << wick_val << " " << slat_val << std::endl;

    return 0;
}

int main() {

    int ret = 0, nelec=5;
    double dtwick, dtslat;
    for(size_t nbsf=nelec+5; nbsf<nelec+10; nbsf++)
    {
        ret = ret | test_single_single<double>(7,nbsf,nelec,dtwick,dtslat);
        std::cout << std::setw(12) << std::setprecision(0) << nelec
                  << std::setw(12) << std::setprecision(0) << nbsf 
                  << std::setw(12) << std::setprecision(6) << std::fixed << dtwick / 1000.0 
                  << std::setw(12) << std::setprecision(6) << std::fixed << dtslat / 1000.0 
                  << std::endl;
    }
    for(size_t nbsf=nelec+5; nbsf<nelec+10; nbsf++)
    {
        ret = ret | test_double_double<double>(7,nbsf,nelec,dtwick,dtslat);
        std::cout << std::setw(12) << std::setprecision(0) << nelec
                  << std::setw(12) << std::setprecision(0) << nbsf 
                  << std::setw(12) << std::setprecision(6) << std::fixed << dtwick / 1000.0 
                  << std::setw(12) << std::setprecision(6) << std::fixed << dtslat / 1000.0 
                  << std::endl;
    }
    //size_t nelec = 10;
    //int ret = 0;
    //double dtwick, dtslat;
    //for(size_t nbsf=20; nbsf<31; nbsf++)
    //{
    //    ret = ret | test_real_uhf<double>(7,nbsf,nelec,dtwick,dtslat);
    //    std::cout << std::setw(12) << std::setprecision(0) << nbsf 
    //              << std::setw(12) << std::setprecision(6) << std::fixed << dtwick / 1000.0 
    //              << std::setw(12) << std::setprecision(6) << std::fixed << dtslat / 1000.0 
    //              << std::endl;
    //}
    return ret;
}
