#include <iostream>
#include <armadillo>
#include <cassert>
#include <libgnme/wick/wick_orbitals.h>
#include <libgnme/slater/slater_uscf.h>
#include <libgnme/wick/wick_uscf.h>
#include <libgnme/utils/eri_ao2mo.h>
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

    wick_uscf<double,double,double> mb(orbs_a, orbs_b);
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
    wick_uscf<double,double,double> mb(orbs_a, orbs_b);
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
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
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
    for(size_t a=noccb; a<std::min(noccb+3,nmo); a++)
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
    wick_uscf<double,double,double> mb(orbs_a, orbs_b);
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
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    for(size_t j=0; j<nocca; j++)
    for(size_t b=nocca; b<std::min(nocca+3,nmo); b++)
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
    for(size_t a=noccb; a<std::min(noccb+3,nmo); a++)
    for(size_t j=0; j<noccb; j++)
    for(size_t b=noccb; b<std::min(noccb+3,nmo); b++)
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
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    for(size_t j=0; j<noccb; j++)
    for(size_t b=noccb; b<std::min(noccb+3,nmo); b++)
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
    wick_uscf<double,double,double> mb(orbs_a, orbs_b);
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
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    {
        // The single excitation
        arma::umat xahp(1,2), xbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        xocca(i) = a; 

        // Alpha / Alpha double
        for(size_t j=0; j<nocca; j++)
        for(size_t k=0; k<j; k++)
        for(size_t b=nocca; b<std::min(nocca+3,nmo); b++)
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
        for(size_t b=noccb; b<std::min(noccb+3,nmo); b++)
        for(size_t k=0; k<nocca; k++)
        for(size_t c=nocca; c<std::min(nocca+3,nmo); c++)
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


int test_zero_overlap(const char *testcase, unsigned thresh, size_t nza, size_t nzb)
{
    // Report who we are
    std::ostringstream oss; 
    oss << "wick_uscf_two_body_test::test_zero_overlap(" << testcase << ", " << thresh << ", " << nza << ", " << nzb << ")";
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

    // Initialise bra orbitals
    arma::mat Cx_a = Cread.cols(0,nmo-1);
    arma::mat Cx_b = Cread.cols(nmo,2*nmo-1);

    // Exchange occupied and virtual bra orbitals to give nza and nzb 
    // zero-overlap orbital pairs in the ket orbitals
    arma::mat Cw_a = Cx_a, Cw_b = Cx_b;
    for(size_t k=0; k<nza; k++) Cw_a.swap_cols(nocca-1-k, nocca+k);
    for(size_t k=0; k<nzb; k++) Cw_b.swap_cols(noccb-1-k, noccb+k);

    // Construct the wick_orbitals object
    wick_orbitals<double,double> orbs_a(nbsf, nmo, nocca, Cx_a, Cw_a, S);
    wick_orbitals<double,double> orbs_b(nbsf, nmo, noccb, Cx_b, Cw_b, S);

    // Setup matrix builder
    slater_uscf<double,double,double> slat(nbsf, nmo, nocca, noccb, S);
    wick_uscf<double,double,double> mb(orbs_a, orbs_b);
    slat.add_two_body(II);
    mb.add_two_body(II);

    // Variables for testing
    double swick = 0.0, fwick = 0.0, sslat = 0.0, fslat = 0.0;

    // Define "reference" occupation numbers
    arma::uvec ref_occa(nocca);
    arma::uvec ref_occb(noccb);
    for(size_t k=0; k<nocca; k++) ref_occa(k) = k;
    for(size_t k=0; k<noccb; k++) ref_occb(k) = k;

    // Reference / Reference
    {
        // Excitation indices
        arma::umat xahp(0,2), xbhp(0,2);
        arma::umat wahp(0,2), wbhp(0,2);

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

        // Test two-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    // Reference / Alpha single
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    {
        // Excitation indices
        arma::umat xahp(0,2), xbhp(0,2);
        arma::umat wahp(1,2), wbhp(0,2);
        wahp(0,0) = i; wahp(0,1) = a;

        // Occupied orbitals
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        wocca(i) = a;

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

        // Test two-body matrix element
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

        // Test two-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    // Reference / Beta single
    for(size_t i=0; i<noccb; i++)
    for(size_t a=noccb; a<std::min(noccb+3,nmo); a++)
    {
        // Excitation indices
        arma::umat xahp(0,2), xbhp(0,2);
        arma::umat wahp(0,2), wbhp(1,2);
        wbhp(0,0) = i; wbhp(0,1) = a;

        // Occupied orbitals
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        woccb(i) = a;

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

        // Test two-body matrix element
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

        // Test two-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    // Alpha single excitation
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    {
        // The single excitation
        arma::umat xahp(1,2), xbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        xocca(i) = a; 

        // Alpha / Alpha double
        for(size_t j=0; j<nocca; j++)
        for(size_t k=0; k<j; k++)
        for(size_t b=nocca; b<std::min(nocca+3,nmo); b++)
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

            // Test two-body matrix element
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

            // Test two-body matrix element
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
        for(size_t b=noccb; b<std::min(noccb+3,nmo); b++)
        for(size_t k=0; k<nocca; k++)
        for(size_t c=nocca; c<std::min(nocca+3,nmo); c++)
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

            // Test two-body matrix element
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

            // Test two-body matrix element
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


int test_complex(const char *testcase, unsigned thresh, size_t nza, size_t nzb)
{
    // Report who we are
    std::ostringstream oss; 
    oss << "wick_uscf_two_body_test::test_complex(" << testcase << ", " << thresh << ", " << nza << ", " << nzb << ")";
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

    // Complex bra orbitals from a unitary transformation of the real orbitals
    arma::cx_mat Cx_a = arma::cx_mat(Cread.cols(0,nmo-1), arma::zeros(nbsf,nmo)) 
                      * random_unitary(nmo, nocca, 0.1, true, 1);
    arma::cx_mat Cx_b = arma::cx_mat(Cread.cols(nmo,2*nmo-1), arma::zeros(nbsf,nmo)) 
                      * random_unitary(nmo, noccb, 0.1, true, 2);

    // Exchange occupied and virtual bra orbitals to give nza and nzb 
    // zero-overlap orbital pairs in the ket orbitals, followed by a unitary 
    // transformation that preserves them
    arma::cx_mat Cw_a = Cx_a, Cw_b = Cx_b;
    for(size_t k=0; k<nza; k++) Cw_a.swap_cols(nocca-1-k, nocca+k);
    for(size_t k=0; k<nzb; k++) Cw_b.swap_cols(noccb-1-k, noccb+k);
    Cw_a = Cw_a * random_unitary(nmo, nocca, 0.3, nza == 0, 3);
    Cw_b = Cw_b * random_unitary(nmo, noccb, 0.3, nzb == 0, 4);

    // Construct the wick_orbitals object
    wick_orbitals<std::complex<double>,double> orbs_a(nbsf, nmo, nocca, Cx_a, Cw_a, S);
    wick_orbitals<std::complex<double>,double> orbs_b(nbsf, nmo, noccb, Cx_b, Cw_b, S);

    // Setup matrix builder
    slater_uscf<std::complex<double>,double,double> slat(nbsf, nmo, nocca, noccb, S);
    wick_uscf<std::complex<double>,double,double> mb(orbs_a, orbs_b);
    slat.add_two_body(II);
    mb.add_two_body(II);

    // Variables for testing
    std::complex<double> swick = 0.0, fwick = 0.0, sslat = 0.0, fslat = 0.0;

    // Define "reference" occupation numbers
    arma::uvec ref_occa(nocca);
    arma::uvec ref_occb(noccb);
    for(size_t k=0; k<nocca; k++) ref_occa(k) = k;
    for(size_t k=0; k<noccb; k++) ref_occb(k) = k;

    // Reference / Reference
    {
        // Excitation indices
        arma::umat xahp(0,2), xbhp(0,2);
        arma::umat wahp(0,2), wbhp(0,2);

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

        // Test two-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    // Reference / Alpha single
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    {
        // Excitation indices
        arma::umat xahp(0,2), xbhp(0,2);
        arma::umat wahp(1,2), wbhp(0,2);
        wahp(0,0) = i; wahp(0,1) = a;

        // Occupied orbitals
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        wocca(i) = a;

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

        // Test two-body matrix element
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

        // Test two-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    // Reference / Beta single
    for(size_t i=0; i<noccb; i++)
    for(size_t a=noccb; a<std::min(noccb+3,nmo); a++)
    {
        // Excitation indices
        arma::umat xahp(0,2), xbhp(0,2);
        arma::umat wahp(0,2), wbhp(1,2);
        wbhp(0,0) = i; wbhp(0,1) = a;

        // Occupied orbitals
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        woccb(i) = a;

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

        // Test two-body matrix element
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

        // Test two-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    // Alpha single excitation
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    {
        // The single excitation
        arma::umat xahp(1,2), xbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        xocca(i) = a; 

        // Alpha / Alpha double
        for(size_t j=0; j<nocca; j++)
        for(size_t k=0; k<j; k++)
        for(size_t b=nocca; b<std::min(nocca+3,nmo); b++)
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

            // Test two-body matrix element
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

            // Test two-body matrix element
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
        for(size_t b=noccb; b<std::min(noccb+3,nmo); b++)
        for(size_t k=0; k<nocca; k++)
        for(size_t c=nocca; c<std::min(nocca+3,nmo); c++)
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

            // Test two-body matrix element
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

            // Test two-body matrix element
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


int test_complex_rdm(const char *testcase, unsigned thresh, size_t nza, size_t nzb)
{
    // Report who we are
    std::ostringstream oss; 
    oss << "wick_uscf_two_body_test::test_complex_rdm(" << testcase << ", " << thresh << ", " << nza << ", " << nzb << ")";
    std::cout << oss.str() << std::endl;

    // Filenames
    std::string fname_nmo   = testcase + std::string("/nelec.txt");
    std::string fname_c     = testcase + std::string("/coeff.txt");
    std::string fname_ov    = testcase + std::string("/ovls.txt");
    std::string fname_hcore = testcase + std::string("/oeis.txt");
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

    // Read-in one-electron integrals
    arma::mat hcore;
    if(!hcore.load(fname_hcore, arma::raw_ascii)) return 1;
    assert(hcore.n_rows == nbsf);
    assert(hcore.n_cols == nbsf);

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

    // Complex bra orbitals from a unitary transformation of the real orbitals
    arma::cx_mat Cx_a = arma::cx_mat(Cread.cols(0,nmo-1), arma::zeros(nbsf,nmo)) 
                      * random_unitary(nmo, nocca, 0.1, true, 1);
    arma::cx_mat Cx_b = arma::cx_mat(Cread.cols(nmo,2*nmo-1), arma::zeros(nbsf,nmo)) 
                      * random_unitary(nmo, noccb, 0.1, true, 2);

    // Exchange occupied and virtual bra orbitals to give nza and nzb 
    // zero-overlap orbital pairs in the ket orbitals, followed by a unitary 
    // transformation that preserves them
    arma::cx_mat Cw_a = Cx_a, Cw_b = Cx_b;
    for(size_t k=0; k<nza; k++) Cw_a.swap_cols(nocca-1-k, nocca+k);
    for(size_t k=0; k<nzb; k++) Cw_b.swap_cols(noccb-1-k, noccb+k);
    Cw_a = Cw_a * random_unitary(nmo, nocca, 0.3, nza == 0, 3);
    Cw_b = Cw_b * random_unitary(nmo, noccb, 0.3, nzb == 0, 4);

    // Construct the wick_orbitals object
    wick_orbitals<std::complex<double>,double> orbs_a(nbsf, nmo, nocca, Cx_a, Cw_a, S);
    wick_orbitals<std::complex<double>,double> orbs_b(nbsf, nmo, noccb, Cx_b, Cw_b, S);

    // Setup matrix builder
    wick_uscf<std::complex<double>,double,double> mb(orbs_a, orbs_b);
    mb.add_one_body(hcore);
    mb.add_two_body(II);

    // Two-electron integrals in the bra and ket orbital basis
    arma::cx_mat IIaa, IIbb, IIab;
    eri_ao2mo(Cx_a, Cw_a, Cx_a, Cw_a, II, IIaa, false);
    eri_ao2mo(Cx_b, Cw_b, Cx_b, Cw_b, II, IIbb, false);
    eri_ao2mo(Cx_a, Cw_a, Cx_b, Cw_b, II, IIab, false);

    // Variables for testing
    std::complex<double> swick = 0.0, fwick = 0.0, srdm = 0.0, Hrdm = 0.0;
    arma::cx_mat P1a(nmo,nmo), P1b(nmo,nmo), RDM1a, RDM1b;
    arma::cx_mat P2aa(nmo*nmo,nmo*nmo), P2bb(nmo*nmo,nmo*nmo), P2ab(nmo*nmo,nmo*nmo);

    // Define "reference" bitsets
    bitset ref_bsa((1 << nocca) - 1, nmo), ref_bsb((1 << noccb) - 1, nmo);

    // Reference / Reference
    {
        // Excitation bitsets
        bitset bxa(ref_bsa), bxb(ref_bsb);
        bitset bwa(ref_bsa), bwb(ref_bsb);

        // Compute matrix elements
        mb.evaluate(bxa, bxb, bwa, bwb, swick, fwick);
        mb.evaluate_rdm12(bxa, bxb, bwa, bwb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }
    }

    // Reference / Alpha single
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    {
        // Excitation bitsets
        bitset bxa(ref_bsa), bxb(ref_bsb);
        bitset bwa(ref_bsa), bwb(ref_bsb);
        bwa.flip(i); bwa.flip(a);

        // Compute matrix elements
        mb.evaluate(bxa, bxb, bwa, bwb, swick, fwick);
        mb.evaluate_rdm12(bxa, bxb, bwa, bwb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }

        // Compute matrix elements while swapping bra and ket excitation
        mb.evaluate(bwa, bwb, bxa, bxb, swick, fwick);
        mb.evaluate_rdm12(bwa, bwb, bxa, bxb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }
    }

    // Reference / Beta single
    for(size_t i=0; i<noccb; i++)
    for(size_t a=noccb; a<std::min(noccb+3,nmo); a++)
    {
        // Excitation bitsets
        bitset bxa(ref_bsa), bxb(ref_bsb);
        bitset bwa(ref_bsa), bwb(ref_bsb);
        bwb.flip(i); bwb.flip(a);

        // Compute matrix elements
        mb.evaluate(bxa, bxb, bwa, bwb, swick, fwick);
        mb.evaluate_rdm12(bxa, bxb, bwa, bwb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }

        // Compute matrix elements while swapping bra and ket excitation
        mb.evaluate(bwa, bwb, bxa, bxb, swick, fwick);
        mb.evaluate_rdm12(bwa, bwb, bxa, bxb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }
    }

    // Alpha single / Alpha single
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    for(size_t j=0; j<nocca; j++)
    for(size_t b=nocca; b<std::min(nocca+3,nmo); b++)
    {
        // Excitation bitsets
        bitset bxa(ref_bsa), bxb(ref_bsb);
        bitset bwa(ref_bsa), bwb(ref_bsb);
        bxa.flip(i); bxa.flip(a);
        bwa.flip(j); bwa.flip(b);

        // Compute matrix elements
        mb.evaluate(bxa, bxb, bwa, bwb, swick, fwick);
        mb.evaluate_rdm12(bxa, bxb, bwa, bwb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }
    }

    // Alpha single / Beta single
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    for(size_t j=0; j<noccb; j++)
    for(size_t b=noccb; b<std::min(noccb+3,nmo); b++)
    {
        // Excitation bitsets
        bitset bxa(ref_bsa), bxb(ref_bsb);
        bitset bwa(ref_bsa), bwb(ref_bsb);
        bxa.flip(i); bxa.flip(a);
        bwb.flip(j); bwb.flip(b);

        // Compute matrix elements
        mb.evaluate(bxa, bxb, bwa, bwb, swick, fwick);
        mb.evaluate_rdm12(bxa, bxb, bwa, bwb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }
    }

    return 0;
}


int test_near_zero(const char *testcase, unsigned thresh, double sa, double sb)
{
    // Report who we are
    std::ostringstream oss; 
    oss << "wick_uscf_two_body_test::test_near_zero(" << testcase << ", " << thresh << ", " << sa << ", " << sb << ")";
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

    // Initialise bra orbitals
    arma::mat Cx_a = Cread.cols(0,nmo-1);
    arma::mat Cx_b = Cread.cols(nmo,2*nmo-1);

    // Rotate the highest occupied and lowest virtual bra orbitals to give 
    // near-zero overlaps sa and sb with the ket orbitals
    arma::mat Cw_a = Cx_a, Cw_b = Cx_b;
    Cw_a.col(nocca-1) = sa * Cx_a.col(nocca-1) + std::sqrt(1-sa*sa) * Cx_a.col(nocca);
    Cw_a.col(nocca)   = sa * Cx_a.col(nocca) - std::sqrt(1-sa*sa) * Cx_a.col(nocca-1);
    Cw_b.col(noccb-1) = sb * Cx_b.col(noccb-1) + std::sqrt(1-sb*sb) * Cx_b.col(noccb);
    Cw_b.col(noccb)   = sb * Cx_b.col(noccb) - std::sqrt(1-sb*sb) * Cx_b.col(noccb-1);

    // Construct the wick_orbitals object
    wick_orbitals<double,double> orbs_a(nbsf, nmo, nocca, Cx_a, Cw_a, S);
    wick_orbitals<double,double> orbs_b(nbsf, nmo, noccb, Cx_b, Cw_b, S);

    // Setup matrix builder
    slater_uscf<double,double,double> slat(nbsf, nmo, nocca, noccb, S);
    wick_uscf<double,double,double> mb(orbs_a, orbs_b);
    slat.add_two_body(II);
    mb.add_two_body(II);

    // Variables for testing
    double swick = 0.0, fwick = 0.0, sslat = 0.0, fslat = 0.0;

    // Define "reference" occupation numbers
    arma::uvec ref_occa(nocca);
    arma::uvec ref_occb(noccb);
    for(size_t k=0; k<nocca; k++) ref_occa(k) = k;
    for(size_t k=0; k<noccb; k++) ref_occb(k) = k;

    // Reference / Reference
    {
        // Excitation indices
        arma::umat xahp(0,2), xbhp(0,2);
        arma::umat wahp(0,2), wbhp(0,2);

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

        // Test two-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    // Reference / Alpha single
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    {
        // Excitation indices
        arma::umat xahp(0,2), xbhp(0,2);
        arma::umat wahp(1,2), wbhp(0,2);
        wahp(0,0) = i; wahp(0,1) = a;

        // Occupied orbitals
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        wocca(i) = a;

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

        // Test two-body matrix element
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

        // Test two-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    // Reference / Beta single
    for(size_t i=0; i<noccb; i++)
    for(size_t a=noccb; a<std::min(noccb+3,nmo); a++)
    {
        // Excitation indices
        arma::umat xahp(0,2), xbhp(0,2);
        arma::umat wahp(0,2), wbhp(1,2);
        wbhp(0,0) = i; wbhp(0,1) = a;

        // Occupied orbitals
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        arma::uvec wocca = ref_occa, woccb = ref_occb;
        woccb(i) = a;

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

        // Test two-body matrix element
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

        // Test two-body matrix element
        if(std::abs(fwick - fslat) > std::pow(0.1,thresh))
        {
            std::cout << "Two-body error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - fslat) << std::endl;
            return 1;
        }
    }

    // Alpha single excitation
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    {
        // The single excitation
        arma::umat xahp(1,2), xbhp(0,2);
        xahp(0,0) = i; xahp(0,1) = a;
        arma::uvec xocca = ref_occa, xoccb = ref_occb;
        xocca(i) = a; 

        // Alpha / Alpha double
        for(size_t j=0; j<nocca; j++)
        for(size_t k=0; k<j; k++)
        for(size_t b=nocca; b<std::min(nocca+3,nmo); b++)
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

            // Test two-body matrix element
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

            // Test two-body matrix element
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
        for(size_t b=noccb; b<std::min(noccb+3,nmo); b++)
        for(size_t k=0; k<nocca; k++)
        for(size_t c=nocca; c<std::min(nocca+3,nmo); c++)
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

            // Test two-body matrix element
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

            // Test two-body matrix element
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


int test_near_zero_rdm(const char *testcase, unsigned thresh, double sa, double sb)
{
    // Report who we are
    std::ostringstream oss; 
    oss << "wick_uscf_two_body_test::test_near_zero_rdm(" << testcase << ", " << thresh << ", " << sa << ", " << sb << ")";
    std::cout << oss.str() << std::endl;

    // Filenames
    std::string fname_nmo   = testcase + std::string("/nelec.txt");
    std::string fname_c     = testcase + std::string("/coeff.txt");
    std::string fname_ov    = testcase + std::string("/ovls.txt");
    std::string fname_hcore = testcase + std::string("/oeis.txt");
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

    // Read-in one-electron integrals
    arma::mat hcore;
    if(!hcore.load(fname_hcore, arma::raw_ascii)) return 1;
    assert(hcore.n_rows == nbsf);
    assert(hcore.n_cols == nbsf);

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

    // Complex bra orbitals from a unitary transformation of the real orbitals
    arma::cx_mat Cx_a = arma::cx_mat(Cread.cols(0,nmo-1), arma::zeros(nbsf,nmo)) 
                      * random_unitary(nmo, nocca, 0.1, true, 1);
    arma::cx_mat Cx_b = arma::cx_mat(Cread.cols(nmo,2*nmo-1), arma::zeros(nbsf,nmo)) 
                      * random_unitary(nmo, noccb, 0.1, true, 2);

    // Rotate the highest occupied and lowest virtual bra orbitals to give 
    // near-zero overlaps sa and sb with the ket orbitals
    arma::cx_mat Cw_a = Cx_a, Cw_b = Cx_b;
    Cw_a.col(nocca-1) = sa * Cx_a.col(nocca-1) + std::sqrt(1-sa*sa) * Cx_a.col(nocca);
    Cw_a.col(nocca)   = sa * Cx_a.col(nocca) - std::sqrt(1-sa*sa) * Cx_a.col(nocca-1);
    Cw_b.col(noccb-1) = sb * Cx_b.col(noccb-1) + std::sqrt(1-sb*sb) * Cx_b.col(noccb);
    Cw_b.col(noccb)   = sb * Cx_b.col(noccb) - std::sqrt(1-sb*sb) * Cx_b.col(noccb-1);

    // Construct the wick_orbitals object
    wick_orbitals<std::complex<double>,double> orbs_a(nbsf, nmo, nocca, Cx_a, Cw_a, S);
    wick_orbitals<std::complex<double>,double> orbs_b(nbsf, nmo, noccb, Cx_b, Cw_b, S);

    // Setup matrix builder
    wick_uscf<std::complex<double>,double,double> mb(orbs_a, orbs_b);
    mb.add_one_body(hcore);
    mb.add_two_body(II);

    // Two-electron integrals in the bra and ket orbital basis
    arma::cx_mat IIaa, IIbb, IIab;
    eri_ao2mo(Cx_a, Cw_a, Cx_a, Cw_a, II, IIaa, false);
    eri_ao2mo(Cx_b, Cw_b, Cx_b, Cw_b, II, IIbb, false);
    eri_ao2mo(Cx_a, Cw_a, Cx_b, Cw_b, II, IIab, false);

    // Variables for testing
    std::complex<double> swick = 0.0, fwick = 0.0, srdm = 0.0, Hrdm = 0.0;
    arma::cx_mat P1a(nmo,nmo), P1b(nmo,nmo), RDM1a, RDM1b;
    arma::cx_mat P2aa(nmo*nmo,nmo*nmo), P2bb(nmo*nmo,nmo*nmo), P2ab(nmo*nmo,nmo*nmo);

    // Define "reference" bitsets
    bitset ref_bsa((1 << nocca) - 1, nmo), ref_bsb((1 << noccb) - 1, nmo);

    // Reference / Reference
    {
        // Excitation bitsets
        bitset bxa(ref_bsa), bxb(ref_bsb);
        bitset bwa(ref_bsa), bwb(ref_bsb);

        // Compute matrix elements
        mb.evaluate(bxa, bxb, bwa, bwb, swick, fwick);
        mb.evaluate_rdm12(bxa, bxb, bwa, bwb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }
    }

    // Reference / Alpha single
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    {
        // Excitation bitsets
        bitset bxa(ref_bsa), bxb(ref_bsb);
        bitset bwa(ref_bsa), bwb(ref_bsb);
        bwa.flip(i); bwa.flip(a);

        // Compute matrix elements
        mb.evaluate(bxa, bxb, bwa, bwb, swick, fwick);
        mb.evaluate_rdm12(bxa, bxb, bwa, bwb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }

        // Compute matrix elements while swapping bra and ket excitation
        mb.evaluate(bwa, bwb, bxa, bxb, swick, fwick);
        mb.evaluate_rdm12(bwa, bwb, bxa, bxb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }
    }

    // Reference / Beta single
    for(size_t i=0; i<noccb; i++)
    for(size_t a=noccb; a<std::min(noccb+3,nmo); a++)
    {
        // Excitation bitsets
        bitset bxa(ref_bsa), bxb(ref_bsb);
        bitset bwa(ref_bsa), bwb(ref_bsb);
        bwb.flip(i); bwb.flip(a);

        // Compute matrix elements
        mb.evaluate(bxa, bxb, bwa, bwb, swick, fwick);
        mb.evaluate_rdm12(bxa, bxb, bwa, bwb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }

        // Compute matrix elements while swapping bra and ket excitation
        mb.evaluate(bwa, bwb, bxa, bxb, swick, fwick);
        mb.evaluate_rdm12(bwa, bwb, bxa, bxb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }
    }

    // Alpha single / Alpha single
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    for(size_t j=0; j<nocca; j++)
    for(size_t b=nocca; b<std::min(nocca+3,nmo); b++)
    {
        // Excitation bitsets
        bitset bxa(ref_bsa), bxb(ref_bsb);
        bitset bwa(ref_bsa), bwb(ref_bsb);
        bxa.flip(i); bxa.flip(a);
        bwa.flip(j); bwa.flip(b);

        // Compute matrix elements
        mb.evaluate(bxa, bxb, bwa, bwb, swick, fwick);
        mb.evaluate_rdm12(bxa, bxb, bwa, bwb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
        }
    }

    // Alpha single / Beta single
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    for(size_t j=0; j<noccb; j++)
    for(size_t b=noccb; b<std::min(noccb+3,nmo); b++)
    {
        // Excitation bitsets
        bitset bxa(ref_bsa), bxb(ref_bsb);
        bitset bwa(ref_bsa), bwb(ref_bsb);
        bxa.flip(i); bxa.flip(a);
        bwb.flip(j); bwb.flip(b);

        // Compute matrix elements
        mb.evaluate(bxa, bxb, bwa, bwb, swick, fwick);
        mb.evaluate_rdm12(bxa, bxb, bwa, bwb, srdm, P1a, P1b, P2aa, P2bb, P2ab);

        // Energy from the RDMs, with the RDM1 in the AO basis
        RDM1a = Cw_a * P1a * Cx_a.t();
        RDM1b = Cw_b * P1b * Cx_b.t();
        Hrdm  = arma::dot(hcore, RDM1a) + arma::dot(hcore, RDM1b) 
              + 0.5 * (arma::dot(IIaa, P2aa) + arma::dot(IIbb, P2bb) + arma::dot(IIab, P2ab));

        // Test overlap
        if(std::abs(swick - srdm) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(swick - srdm) << std::endl;
            return 1;
        }

        // Test RDM energy
        if(std::abs(fwick - Hrdm) > std::pow(0.1,thresh))
        {
            std::cout << "RDM energy error = " 
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(fwick - Hrdm) << std::endl;
            return 1;
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
    test_zero_overlap("h2o_6-31g",tol,1,0) |
    test_zero_overlap("h2o_6-31g",tol,2,1) |
    test_complex("h2o_6-31g",tol,0,0) |
    test_complex("h2o_6-31g",tol,1,0) |
    test_complex("h2o_6-31g",tol,2,1) |
    test_complex_rdm("h2o_6-31g",tol,0,0) |
    test_complex_rdm("h2o_6-31g",tol,1,0) |
    test_complex_rdm("h2o_6-31g",tol,2,1) |
    test_near_zero("h2o_6-31g",tol,1e-6,1.0) |
    test_near_zero("h2o_6-31g",tol,1e-6,1e-5) |
    test_near_zero_rdm("h2o_6-31g",tol,1e-6,1.0) |
    test_near_zero_rdm("h2o_6-31g",tol,1e-6,1e-5) |
    0;
}
