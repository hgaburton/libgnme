#include <iostream>
#include <armadillo>
#include <cassert>
#include <libgnme/wick/wick_orbitals.h>
#include "testing.h"

using namespace libgnme;

int test_real_uhf(const char *testcase, unsigned thresh)
{
    // Report who we are
    std::ostringstream oss; 
    oss << "wick_orbitals_test::test_real_uhf(" << testcase << ", " << thresh << ")";
    std::cout << oss.str() << std::endl;

    // Filenames
    std::string fname_nmo = testcase + std::string("/nelec.txt");
    std::string fname_c   = testcase + std::string("/coeff.txt");
    std::string fname_ov  = testcase + std::string("/ovls.txt");

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

    // Read-in orbital coefficients
    arma::mat C;
    if(!C.load(fname_c, arma::raw_ascii)) return 1;
    assert(C.n_rows == nbsf);
    assert(C.n_cols == 2*nmo);

    // Access alpha and beta coefficients
    arma::mat Cx(C.colptr(0), nbsf, nmo, false, true);
    arma::mat Cw(C.colptr(nmo), nbsf, nmo, false, true);

    // Construct the wick_orbitals object
    wick_orbitals<double,double> orbs(nbsf, nmo, nocca, Cx, Cw, S);

    // Test transformed coefficients
    arma::vec CXref, XCref;
    arma::vec CX = arma::vectorise(arma::join_rows(orbs.m_CX(0), orbs.m_CX(1)));
    arma::vec XC = arma::vectorise(arma::join_rows(orbs.m_XC(0), orbs.m_XC(1)));
    if(!CXref.load(testcase + std::string("/CX.txt"), arma::raw_ascii)) return 1;
    if(array_test(CX, CXref, thresh)) return 1;
    if(!XCref.load(testcase + std::string("/XC.txt"), arma::raw_ascii)) return 1;
    if(array_test(XC, XCref, thresh)) return 1;

    // Test co-density matrices
    arma::vec Mref;
    if(!Mref.load(testcase + std::string("/M.txt"), arma::raw_ascii)) return 1;
    arma::vec M = arma::vectorise(arma::join_rows(orbs.m_M(0), orbs.m_M(1)));
    if(array_test(M, Mref, thresh)) return 1;

    // Test fundamental contractions
    arma::vec Xref, Yref;
    arma::vec X = arma::vectorise(arma::join_rows(orbs.m_X(0), orbs.m_X(1)));
    arma::vec Y = arma::vectorise(arma::join_rows(orbs.m_Y(0), orbs.m_Y(1)));
    if(!Xref.load(testcase + std::string("/X.txt"), arma::raw_ascii)) return 1;
    if(array_test(X, Xref, thresh)) return 1;
    if(!Yref.load(testcase + std::string("/Y.txt"), arma::raw_ascii)) return 1;
    if(array_test(Y, Yref, 8)) return 1;

    return 0;
}

int main () {
    return 

    test_real_uhf("h2o_6-31g",8) | 
    0;
}

