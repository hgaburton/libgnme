#include <iostream>
#include <armadillo>
#include <cassert>
#include <libgnme/wick/wick_orbitals.h>
#include <libgnme/slater/slater_uscf.h>
#include <libgnme/wick/wick_uscf.h>
#include "testing.h"

using namespace libgnme;

int test_ref_ref(const char *testcase, unsigned thresh)
{
    // Report who we are
    std::ostringstream oss; 
    oss << "wick_uscf_two_body_test::test_ref_ref(" << testcase << ", " << thresh << ")";
    std::cout << oss.str() << std::endl;

    // Filenames
    std::string fname_nmo   = testcase + std::string("/nelec.txt");
    std::string fname_c     = testcase + std::string("/coeff.txt");
    std::string fname_ov    = testcase + std::string("/ovls.txt");
    std::string fname_II    = testcase + std::string("/teis.bin");

    // Read-in basis information
    size_t nbsf = 0, nocca = 0, noccb = 0, nmo = 0, nact = 0;
    std::ifstream nelec_file;
    nelec_file.open(fname_nmo);
    nelec_file >> nbsf >> nocca >> noccb >> nact;
    nelec_file.close();
    nmo = nbsf;

    // Check the input
    assert(nmo   > 0);
    assert(nbsf  > 0);
    assert(nocca > 0);
    assert(noccb > 0);
 
    // Read-in overlap matrix
    arma::mat S;
    if(!S.load(fname_ov, arma::raw_ascii)) return 1;
    assert(S.n_rows == nbsf);
    assert(S.n_cols == nbsf);

    // Read-in two-electron integrals
    arma::mat II;
    if(!II.load(fname_II, arma::raw_binary)) return 1;
    assert(II.n_elem == nbsf*nbsf*nbsf*nbsf);
    II.reshape(nbsf*nbsf, nbsf*nbsf);

    // Read-in orbital coefficients
    arma::mat Cread;
    if(!Cread.load(fname_c, arma::raw_ascii)) return 1;
    assert(Cread.n_rows == nbsf);
    assert(Cread.n_cols == 2*nmo);

    // Initialise memory for bra and ket orbitals with memory access views
    arma::mat Cx(nbsf, 2*nmo, arma::fill::zeros);
    arma::mat Cx_a(Cx.colptr(0), nbsf, nmo, false, true);
    arma::mat Cx_b(Cx.colptr(nmo), nbsf, nmo, false, true);
    arma::mat Cw(nbsf, 2*nmo, arma::fill::zeros);
    arma::mat Cw_a(Cw.colptr(0), nbsf, nmo, false, true);
    arma::mat Cw_b(Cw.colptr(nmo), nbsf, nmo, false, true);

    // Define bra and ket orbital coefficients as spin-flip pairs
    Cx_a = Cread.cols(0,nmo-1);     Cw_b = Cread.cols(0,nmo-1);
    Cx_b = Cread.cols(nmo,2*nmo-1); Cw_a = Cread.cols(nmo,2*nmo-1);

    // Construct the wick_orbitals object
    wick_orbitals<double,double> orbs_a(nbsf, nmo, nocca, Cx_a, Cw_a, S);
    wick_orbitals<double,double> orbs_b(nbsf, nmo, noccb, Cx_b, Cw_b, S);

    // Setup matrix builder
    slater_uscf<double,double,double> slat(nbsf, nmo, nocca, noccb, S);

    wick_uscf<double,double,double> mb(orbs_a, orbs_b, S);
    slat.add_two_body(II);
    mb.add_two_body(II);

    // Setup orbitals
    //mb.setup_orbitals(Cx, Cw);

    // Variables for testing
    double swick = 0.0, fwick = 0.0, sslat = 0.0, fslat = 0.0;

    // Define "reference" occupation numbers
    arma::uvec ref_occa(nocca);
    arma::uvec ref_occb(noccb);
    for(size_t k=0; k<nocca; k++) ref_occa(k) = k;
    for(size_t k=0; k<noccb; k++) ref_occb(k) = k;

    // Excitation indices
    arma::umat xahp, xbhp;
    arma::umat wahp, wbhp;
    
    // Occupied orbitals
    arma::uvec xocca = ref_occa, xoccb = ref_occb;
    arma::uvec wocca = ref_occa, woccb = ref_occb;

    // Compute matrix elements
    mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);
    slat.evaluate(Cx_a.cols(xocca), Cx_b.cols(xoccb), Cw_a.cols(wocca), Cw_b.cols(woccb), sslat, fslat);

    // Test overlap
    if(std::abs(swick - sslat) > std::pow(0.1,thresh))
    {
        std::cout << "Overlap error = " 
                  << std::fixed << std::setprecision(thresh+2) << std::scientific
                  << std::abs(swick - sslat) << std::endl;
        return 1;
    }

    // Test one-body matrix element
    if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
    {
        std::cout << "Two-body error = " 
                  << std::fixed << std::setprecision(thresh+2) << std::scientific
                  << std::abs(fwick - fslat) << std::endl;
        return 1;
    }

    return 0;
}


int test_ref_single(const char *testcase, unsigned thresh)
{
    // Report who we are
    std::ostringstream oss; 
    oss << "wick_uscf_two_body_test::test_ref_single(" << testcase << ", " << thresh << ")";
    std::cout << oss.str() << std::endl;

    // Filenames
    std::string fname_nmo   = testcase + std::string("/nelec.txt");
    std::string fname_c     = testcase + std::string("/coeff.txt");
    std::string fname_ov    = testcase + std::string("/ovls.txt");
    std::string fname_II    = testcase + std::string("/teis.bin");

    // Read-in basis information
    size_t nbsf = 0, nocca = 0, noccb = 0, nmo = 0, nact = 0;
    std::ifstream nelec_file;
    nelec_file.open(fname_nmo);
    nelec_file >> nbsf >> nocca >> noccb >> nact;
    nelec_file.close();
    nmo = nbsf;

    // Check the input
    assert(nmo   > 0);
    assert(nbsf  > 0);
    assert(nocca > 0);
    assert(noccb > 0);
 
    // Read-in overlap matrix
    arma::mat S;
    if(!S.load(fname_ov, arma::raw_ascii)) return 1;
    assert(S.n_rows == nbsf);
    assert(S.n_cols == nbsf);

    // Read-in two-electron integrals
    arma::mat II;
    if(!II.load(fname_II, arma::raw_binary)) return 1;
    assert(II.n_elem == nbsf*nbsf*nbsf*nbsf);
    II.reshape(nbsf*nbsf, nbsf*nbsf);

    // Read-in orbital coefficients
    arma::mat Cread;
    if(!Cread.load(fname_c, arma::raw_ascii)) return 1;
    assert(Cread.n_rows == nbsf);
    assert(Cread.n_cols == 2*nmo);

    // Initialise memory for bra and ket orbitals with memory access views
    arma::mat Cx(nbsf, 2*nmo, arma::fill::zeros);
    arma::mat Cx_a(Cx.colptr(0), nbsf, nmo, false, true);
    arma::mat Cx_b(Cx.colptr(nmo), nbsf, nmo, false, true);
    arma::mat Cw(nbsf, 2*nmo, arma::fill::zeros);
    arma::mat Cw_a(Cw.colptr(0), nbsf, nmo, false, true);
    arma::mat Cw_b(Cw.colptr(nmo), nbsf, nmo, false, true);

    // Define bra and ket orbital coefficients as spin-flip pairs
    Cx_a = Cread.cols(0,nmo-1);     Cw_b = Cread.cols(0,nmo-1);
    Cx_b = Cread.cols(nmo,2*nmo-1); Cw_a = Cread.cols(nmo,2*nmo-1);

    // Construct the wick_orbitals object
    wick_orbitals<double,double> orbs_a(nbsf, nmo, nocca, Cx_a, Cw_a, S);
    wick_orbitals<double,double> orbs_b(nbsf, nmo, noccb, Cx_b, Cw_b, S);

    // Setup matrix builder
    slater_uscf<double,double,double> slat(nbsf, nmo, nocca, noccb, S);
    wick_uscf<double,double,double> mb(orbs_a, orbs_b, S);
    slat.add_two_body(II);
    mb.add_two_body(II);

    // Setup orbitals
    //mb.setup_orbitals(Cx, Cw);

    // Variables for testing
    double swick = 0.0, fwick = 0.0, sslat = 0.0, fslat = 0.0;

    // Define "reference" occupation numbers
    arma::uvec ref_occa(nocca);
    arma::uvec ref_occb(noccb);
    for(size_t k=0; k<nocca; k++) ref_occa(k) = k;
    for(size_t k=0; k<noccb; k++) ref_occb(k) = k;

    // Alpha single excitation
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<nmo; a++)
    {
        // Excitation indices
        arma::umat xahp(1,2), xbhp(0,2);
        arma::umat wahp(0,2), wbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        
        // Occupied orbitals
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        xocca(i) = a;

        // Compute matrix elements
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);
        slat.evaluate(Cx_a.cols(xocca), Cx_b.cols(xoccb), Cw_a.cols(wocca), Cw_b.cols(woccb), sslat, fslat);

        // Test overlap
        if(std::abs(swick - sslat) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - sslat) << std::endl;
            return 1;
        }

        // Test one-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }

        // Compute matrix elements
        mb.evaluate(wahp, wbhp, xahp, xbhp, swick, fwick);
        slat.evaluate(Cx_a.cols(wocca), Cx_b.cols(woccb), Cw_a.cols(xocca), Cw_b.cols(xoccb), sslat, fslat);

        // Test overlap
        if(std::abs(swick - sslat) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - sslat) << std::endl;
            return 1;
        }

        // Test one-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    // Beta single excitation
    for(size_t i=0; i<noccb; i++)
    for(size_t a=noccb; a<nmo; a++)
    {
        // Excitation indices
        arma::umat xahp(0,2), xbhp(1,2);
        arma::umat wahp(0,2), wbhp(0,2);
        xbhp(0,0) = i; xbhp(0,1) = a;
        
        // Occupied orbitals
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        xoccb(i) = a;

        // Compute matrix elements
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);
        slat.evaluate(Cx_a.cols(xocca), Cx_b.cols(xoccb), Cw_a.cols(wocca), Cw_b.cols(woccb), sslat, fslat);

        // Test overlap
        if(std::abs(swick - sslat) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - sslat) << std::endl;
            return 1;
        }

        // Test one-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }

        // Compute matrix elements
        mb.evaluate(wahp, wbhp, xahp, xbhp, swick, fwick);
        slat.evaluate(Cx_a.cols(wocca), Cx_b.cols(woccb), Cw_a.cols(xocca), Cw_b.cols(xoccb), sslat, fslat);

        // Test overlap
        if(std::abs(swick - sslat) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - sslat) << std::endl;
            return 1;
        }

        // Test one-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    return 0;
}


int test_single_single(const char *testcase, unsigned thresh)
{
    // Report who we are
    std::ostringstream oss; 
    oss << "wick_uscf_two_body_test::test_single_single(" << testcase << ", " << thresh << ")";
    std::cout << oss.str() << std::endl;

    // Filenames
    std::string fname_nmo   = testcase + std::string("/nelec.txt");
    std::string fname_c     = testcase + std::string("/coeff.txt");
    std::string fname_ov    = testcase + std::string("/ovls.txt");
    std::string fname_II    = testcase + std::string("/teis.bin");

    // Read-in basis information
    size_t nbsf = 0, nocca = 0, noccb = 0, nmo = 0, nact = 0;
    std::ifstream nelec_file;
    nelec_file.open(fname_nmo);
    nelec_file >> nbsf >> nocca >> noccb >> nact;
    nelec_file.close();
    nmo = nbsf;

    // Check the input
    assert(nmo   > 0);
    assert(nbsf  > 0);
    assert(nocca > 0);
    assert(noccb > 0);
 
    // Read-in overlap matrix
    arma::mat S;
    if(!S.load(fname_ov, arma::raw_ascii)) return 1;
    assert(S.n_rows == nbsf);
    assert(S.n_cols == nbsf);

    // Read-in two-electron integrals
    arma::mat II;
    if(!II.load(fname_II, arma::raw_binary)) return 1;
    assert(II.n_elem == nbsf*nbsf*nbsf*nbsf);
    II.reshape(nbsf*nbsf, nbsf*nbsf);

    // Read-in orbital coefficients
    arma::mat Cread;
    if(!Cread.load(fname_c, arma::raw_ascii)) return 1;
    assert(Cread.n_rows == nbsf);
    assert(Cread.n_cols == 2*nmo);

    // Initialise memory for bra and ket orbitals with memory access views
    arma::mat Cx(nbsf, 2*nmo, arma::fill::zeros);
    arma::mat Cx_a(Cx.colptr(0), nbsf, nmo, false, true);
    arma::mat Cx_b(Cx.colptr(nmo), nbsf, nmo, false, true);
    arma::mat Cw(nbsf, 2*nmo, arma::fill::zeros);
    arma::mat Cw_a(Cw.colptr(0), nbsf, nmo, false, true);
    arma::mat Cw_b(Cw.colptr(nmo), nbsf, nmo, false, true);

    // Define bra and ket orbital coefficients as spin-flip pairs
    Cx_a = Cread.cols(0,nmo-1);     Cw_b = Cread.cols(0,nmo-1);
    Cx_b = Cread.cols(nmo,2*nmo-1); Cw_a = Cread.cols(nmo,2*nmo-1);

    // Construct the wick_orbitals object
    wick_orbitals<double,double> orbs_a(nbsf, nmo, nocca, Cx_a, Cw_a, S);
    wick_orbitals<double,double> orbs_b(nbsf, nmo, noccb, Cx_b, Cw_b, S);

    // Setup matrix builder
    slater_uscf<double,double,double> slat(nbsf, nmo, nocca, noccb, S);
    wick_uscf<double,double,double> mb(orbs_a, orbs_b, S);
    slat.add_two_body(II);
    mb.add_two_body(II);

    // Setup orbitals
    //mb.setup_orbitals(Cx, Cw);

    // Variables for testing
    double swick = 0.0, fwick = 0.0, sslat = 0.0, fslat = 0.0;

    // Define "reference" occupation numbers
    arma::uvec ref_occa(nocca);
    arma::uvec ref_occb(noccb);
    for(size_t k=0; k<nocca; k++) ref_occa(k) = k;
    for(size_t k=0; k<noccb; k++) ref_occb(k) = k;

    // Alpha/alpha single excitation
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<nmo; a++)
    for(size_t j=0; j<nocca; j++)
    for(size_t b=nocca; b<nmo; b++)
    {
        // Excitation indices
        arma::umat xahp(1,2), xbhp(0,2);
        arma::umat wahp(1,2), wbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        wahp(0,0) = j; wahp(0,1) = b;
        
        // Occupied orbitals
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        xocca(i) = a;
        wocca(j) = b;

        // Compute matrix elements
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);
        slat.evaluate(Cx_a.cols(xocca), Cx_b.cols(xoccb), Cw_a.cols(wocca), Cw_b.cols(woccb), sslat, fslat);

        // Test overlap
        if(std::abs(swick - sslat) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - sslat) << std::endl;
            return 1;
        }

        // Test one-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    // Beta/beta single excitation
    for(size_t i=0; i<noccb; i++)
    for(size_t a=noccb; a<nmo; a++)
    for(size_t j=0; j<noccb; j++)
    for(size_t b=noccb; b<nmo; b++)
    {
        // Excitation indices
        arma::umat xahp(0,2), xbhp(1,2);
        arma::umat wahp(0,2), wbhp(1,2);
        xbhp(0,0) = i; xbhp(0,1) = a;
        wbhp(0,0) = j; wbhp(0,1) = b;
        
        // Occupied orbitals
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        xoccb(i) = a;
        woccb(j) = b;

        // Compute matrix elements
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);
        slat.evaluate(Cx_a.cols(xocca), Cx_b.cols(xoccb), Cw_a.cols(wocca), Cw_b.cols(woccb), sslat, fslat);

        // Test overlap
        if(std::abs(swick - sslat) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - sslat) << std::endl;
            return 1;
        }

        // Test one-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    // Beta single excitation
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<nmo; a++)
    for(size_t j=0; j<noccb; j++)
    for(size_t b=noccb; b<nmo; b++)
    {
        // Excitation indices
        arma::umat xahp(1,2), xbhp(0,2);
        arma::umat wahp(0,2), wbhp(1,2);
        xahp(0,0) = i; xahp(0,1) = a;
        wbhp(0,0) = j; wbhp(0,1) = b;
        
        // Occupied orbitals
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        xocca(i) = a;
        woccb(j) = b;

        // Compute matrix elements
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);
        slat.evaluate(Cx_a.cols(xocca), Cx_b.cols(xoccb), Cw_a.cols(wocca), Cw_b.cols(woccb), sslat, fslat);

        // Test overlap
        if(std::abs(swick - sslat) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - sslat) << std::endl;
            return 1;
        }

        // Test one-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }

        // Compute matrix elements
        mb.evaluate(wahp, wbhp, xahp, xbhp, swick, fwick);
        slat.evaluate(Cx_a.cols(wocca), Cx_b.cols(woccb), Cw_a.cols(xocca), Cw_b.cols(xoccb), sslat, fslat);

        // Test overlap
        if(std::abs(swick - sslat) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - sslat) << std::endl;
            return 1;
        }

        // Test one-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    return 0;
}


int test_single_double(const char *testcase, unsigned thresh)
{
    // Report who we are
    std::ostringstream oss; 
    oss << "wick_uscf_two_body_test::test_single_double(" << testcase << ", " << thresh << ")";
    std::cout << oss.str() << std::endl;

    // Filenames
    std::string fname_nmo   = testcase + std::string("/nelec.txt");
    std::string fname_c     = testcase + std::string("/coeff.txt");
    std::string fname_ov    = testcase + std::string("/ovls.txt");
    std::string fname_II    = testcase + std::string("/teis.bin");

    // Read-in basis information
    size_t nbsf = 0, nocca = 0, noccb = 0, nmo = 0, nact = 0;
    std::ifstream nelec_file;
    nelec_file.open(fname_nmo);
    nelec_file >> nbsf >> nocca >> noccb >> nact;
    nelec_file.close();
    nmo = nbsf;

    // Check the input
    assert(nmo   > 0);
    assert(nbsf  > 0);
    assert(nocca > 0);
    assert(noccb > 0);
 
    // Read-in overlap matrix
    arma::mat S;
    if(!S.load(fname_ov, arma::raw_ascii)) return 1;
    assert(S.n_rows == nbsf);
    assert(S.n_cols == nbsf);

    // Read-in two-electron integrals
    arma::mat II;
    if(!II.load(fname_II, arma::raw_binary)) return 1;
    assert(II.n_elem == nbsf*nbsf*nbsf*nbsf);
    II.reshape(nbsf*nbsf, nbsf*nbsf);

    // Read-in orbital coefficients
    arma::mat Cread;
    if(!Cread.load(fname_c, arma::raw_ascii)) return 1;
    assert(Cread.n_rows == nbsf);
    assert(Cread.n_cols == 2*nmo);

    // Initialise memory for bra and ket orbitals with memory access views
    arma::mat Cx(nbsf, 2*nmo, arma::fill::zeros);
    arma::mat Cx_a(Cx.colptr(0), nbsf, nmo, false, true);
    arma::mat Cx_b(Cx.colptr(nmo), nbsf, nmo, false, true);
    arma::mat Cw(nbsf, 2*nmo, arma::fill::zeros);
    arma::mat Cw_a(Cw.colptr(0), nbsf, nmo, false, true);
    arma::mat Cw_b(Cw.colptr(nmo), nbsf, nmo, false, true);

    // Define bra and ket orbital coefficients as spin-flip pairs
    Cx_a = Cread.cols(0,nmo-1);     Cw_b = Cread.cols(0,nmo-1);
    Cx_b = Cread.cols(nmo,2*nmo-1); Cw_a = Cread.cols(nmo,2*nmo-1);

    // Construct the wick_orbitals object
    wick_orbitals<double,double> orbs_a(nbsf, nmo, nocca, Cx_a, Cw_a, S);
    wick_orbitals<double,double> orbs_b(nbsf, nmo, noccb, Cx_b, Cw_b, S);

    // Setup matrix builder
    slater_uscf<double,double,double> slat(nbsf, nmo, nocca, noccb, S);
    wick_uscf<double,double,double> mb(orbs_a, orbs_b, S);
    slat.add_two_body(II);
    mb.add_two_body(II);

    // Setup orbitals
    //mb.setup_orbitals(Cx, Cw);

    // Variables for testing
    double swick = 0.0, fwick = 0.0, sslat = 0.0, fslat = 0.0;

    // Define "reference" occupation numbers
    arma::uvec ref_occa(nocca);
    arma::uvec ref_occb(noccb);
    for(size_t k=0; k<nocca; k++) ref_occa(k) = k;
    for(size_t k=0; k<noccb; k++) ref_occb(k) = k;

    // Alpha single excitation
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<nmo; a++)
    {
        // The single excitation
        arma::umat xahp(1,2), xbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        xocca(i) = a; 

        // Alpha / Alpha double
        for(size_t j=0; j<nocca; j++)
        for(size_t k=0; k<j; k++)
        for(size_t b=nocca; b<nmo; b++)
        for(size_t c=nocca; c<b; c++)
        {
            // Excitation indices
            arma::umat wahp(2,2), wbhp(0,2);
            wahp(0,0) = j; wahp(0,1) = b;
            wahp(1,0) = k; wahp(1,1) = c;
            
            // Occupied orbitals
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            wocca(j) = b; wocca(k) = c;

            // Compute matrix elements
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);
            slat.evaluate(Cx_a.cols(xocca), Cx_b.cols(xoccb), Cw_a.cols(wocca), Cw_b.cols(woccb), sslat, fslat);

            // Test overlap
            if(std::abs(swick - sslat) > std::pow(0.1,thresh))
            {
                std::cout << "Overlap error = " 
                          << std::fixed << std::setprecision(thresh+2) << std::scientific
                          << std::abs(swick - sslat) << std::endl;
                return 1;
            }

            // Test one-body matrix element
            if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
            {
                std::cout << "Two-body error = " 
                          << std::fixed << std::setprecision(thresh+2) << std::scientific
                          << std::abs(fwick - fslat) << std::endl;
                return 1;
            }

            // Compute matrix elements while swapping bra and ket excitation
            mb.evaluate(wahp, wbhp, xahp, xbhp, swick, fwick);
            slat.evaluate(Cx_a.cols(wocca), Cx_b.cols(woccb), Cw_a.cols(xocca), Cw_b.cols(xoccb), sslat, fslat);

            // Test overlap
            if(std::abs(swick - sslat) > std::pow(0.1,thresh))
            {
                std::cout << "Overlap error = " 
                          << std::fixed << std::setprecision(thresh+2) << std::scientific
                          << std::abs(swick - sslat) << std::endl;
                return 1;
            }

            // Test one-body matrix element
            if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
            {
                std::cout << "Two-body error = " 
                          << std::fixed << std::setprecision(thresh+2) << std::scientific
                          << std::abs(fwick - fslat) << std::endl;
                return 1;
            }
        }

        // Alpha / Beta double
        for(size_t j=0; j<noccb; j++)
        for(size_t b=noccb; b<nmo; b++)
        for(size_t k=0; k<nocca; k++)
        for(size_t c=nocca; c<nmo; c++)
        {
            // Excitation indices
            arma::umat wahp(1,2), wbhp(1,2);
            wbhp(0,0) = j; wbhp(0,1) = b;
            wahp(0,0) = k; wahp(0,1) = c;
            
            // Occupied orbitals
            arma::uvec wocca = ref_occa, woccb = ref_occb;
            woccb(j) = b; wocca(k) = c;

            // Compute matrix elements
            mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);
            slat.evaluate(Cx_a.cols(xocca), Cx_b.cols(xoccb), Cw_a.cols(wocca), Cw_b.cols(woccb), sslat, fslat);

            // Test overlap
            if(std::abs(swick - sslat) > std::pow(0.1,thresh))
            {
                std::cout << "Overlap error = " 
                          << std::fixed << std::setprecision(thresh+2) << std::scientific
                          << std::abs(swick - sslat) << std::endl;
                return 1;
            }

            // Test one-body matrix element
            if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
            {
                std::cout << "Two-body error = " 
                          << std::fixed << std::setprecision(thresh+2) << std::scientific
                          << std::abs(fwick - fslat) << std::endl;
                return 1;
            }

            // Compute matrix elements while swapping bra and ket excitation
            mb.evaluate(wahp, wbhp, xahp, xbhp, swick, fwick);
            slat.evaluate(Cx_a.cols(wocca), Cx_b.cols(woccb), Cw_a.cols(xocca), Cw_b.cols(xoccb), sslat, fslat);

            // Test overlap
            if(std::abs(swick - sslat) > std::pow(0.1,thresh))
            {
                std::cout << "Overlap error = " 
                          << std::fixed << std::setprecision(thresh+2) << std::scientific
                          << std::abs(swick - sslat) << std::endl;
                return 1;
            }

            // Test one-body matrix element
            if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
            {
                std::cout << "Two-body error = " 
                          << std::fixed << std::setprecision(thresh+2) << std::scientific
                          << std::abs(fwick - fslat) << std::endl;
                return 1;
            }
        }
    }

    return 0;
}


int main () {
    int tol = 6;

    return 
    test_ref_ref("h2o_6-31g",tol)       | 
    test_ref_single("h2o_6-31g",tol)    | 
    test_single_single("h2o_6-31g",tol) | 
    test_single_double("h2o_6-31g",tol) |
    0;
}
