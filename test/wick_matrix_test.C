#include <iostream>
#include <armadillo>
#include <cassert>
#include <libgnme/wick/wick_orbitals.h>
#include <libgnme/wick/wick_uscf.h>
#include <libgnme/wick/wick_matrix.h>
#include "testing.h"

using namespace libgnme;

template<typename Tc>
int compare_with_wick_uscf(
    size_t nbsf, size_t nmo, size_t nocca, size_t noccb, arma::mat &S, arma::mat &hcore,
    arma::mat &II, std::vector<arma::Mat<Tc> > &Ca, std::vector<arma::Mat<Tc> > &Cb,
    unsigned thresh)
{
    const size_t nref = Ca.size();
    const double enuc = 1.5;

    // Define "reference" bitsets, with the rightmost element representing the first MO
    std::vector<uint8_t> refa(nmo, 0), refb(nmo, 0);
    for(size_t k=0; k<nocca; k++) refa[nmo-1-k] = 1;
    for(size_t k=0; k<noccb; k++) refb[nmo-1-k] = 1;

    // Determinants of every reference
    std::vector<size_t> ref;
    std::vector<bitset> ba, bb;
    for(size_t x=0; x<nref; x++)
    {
        // Reference
        ref.push_back(x); ba.push_back(bitset(refa)); bb.push_back(bitset(refb));

        // Alpha single excitation
        for(size_t i=0; i<nocca; i++)
        for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
        {
            std::vector<uint8_t> v = refa; v[nmo-1-i] = 0; v[nmo-1-a] = 1;
            ref.push_back(x); ba.push_back(bitset(v)); bb.push_back(bitset(refb));
        }

        // Beta single excitation
        for(size_t i=0; i<noccb; i++)
        for(size_t a=noccb; a<std::min(noccb+3,nmo); a++)
        {
            std::vector<uint8_t> v = refb; v[nmo-1-i] = 0; v[nmo-1-a] = 1;
            ref.push_back(x); ba.push_back(bitset(refa)); bb.push_back(bitset(v));
        }

        // Alpha / Alpha double excitation
        for(size_t j=0; j<nocca; j++)
        for(size_t k=0; k<j; k++)
        for(size_t b=nocca; b<std::min(nocca+3,nmo); b++)
        for(size_t c=nocca; c<b; c++)
        {
            std::vector<uint8_t> v = refa;
            v[nmo-1-j] = 0; v[nmo-1-k] = 0; v[nmo-1-b] = 1; v[nmo-1-c] = 1;
            ref.push_back(x); ba.push_back(bitset(v)); bb.push_back(bitset(refb));
        }
    }

    // Compute matrix elements
    wick_matrix<Tc,double,double> wm(nbsf, nmo, nocca, noccb, S, enuc);
    for(size_t x=0; x<nref; x++) wm.add_reference(Ca[x], Cb[x]);
    wm.add_one_body(hcore);
    wm.add_two_body(II);
    arma::Mat<Tc> Swm, Hwm;
    wm.evaluate(ref, ba, bb, Swm, Hwm);

    // Compare with each pair of references
    for(size_t x=0; x<nref; x++)
    for(size_t w=0; w<nref; w++)
    {
        // Construct the wick_orbitals object
        wick_orbitals<Tc,double> orbs_a(nbsf, nmo, nocca, Ca[x], Ca[w], S);
        wick_orbitals<Tc,double> orbs_b(nbsf, nmo, noccb, Cb[x], Cb[w], S);

        // Setup matrix builder
        wick_uscf<Tc,double,double> mb(orbs_a, orbs_b, enuc);
        mb.add_one_body(hcore);
        mb.add_two_body(II);

        for(size_t i=0; i<ref.size(); i++)
        for(size_t j=0; j<ref.size(); j++)
        {
            if(ref[i] != x or ref[j] != w) continue;

            // Compute matrix elements
            Tc swick = 0.0, fwick = 0.0;
            mb.evaluate(ba[i], bb[i], ba[j], bb[j], swick, fwick);

            // Test overlap
            if(std::abs(Swm(i,j) - swick) > std::pow(0.1,thresh))
            {
                std::cout << "Overlap error = "
                          << std::fixed << std::setprecision(thresh+2) << std::scientific
                          << std::abs(Swm(i,j) - swick) << std::endl;
                return 1;
            }

            // Test Hamiltonian matrix element
            if(std::abs(Hwm(i,j) - fwick) > std::pow(0.1,thresh))
            {
                std::cout << "Hamiltonian error = "
                          << std::fixed << std::setprecision(thresh+2) << std::scientific
                          << std::abs(Hwm(i,j) - fwick) << std::endl;
                return 1;
            }
        }
    }

    return 0;
}


int test_real(const char *testcase, unsigned thresh)
{
    // Report who we are
    std::ostringstream oss;
    oss << "wick_matrix_test::test_real(" << testcase << ", " << thresh << ")";
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
    arma::mat C_a = Cread.cols(0,nmo-1);
    arma::mat C_b = Cread.cols(nmo,2*nmo-1);

    // Reference determinants: the UHF orbitals, their spin-flip pair, a zero-overlap orbital
    // pair from exchanging occupied and virtual orbitals, and a near-zero overlap
    std::vector<arma::mat> Ca, Cb;
    Ca.push_back(C_a); Cb.push_back(C_b);
    Ca.push_back(C_b); Cb.push_back(C_a);
    arma::mat Cz = C_a;
    Cz.swap_cols(nocca-1, nocca);
    Ca.push_back(Cz); Cb.push_back(C_b);
    double s = 1e-5;
    arma::mat Cn = C_a;
    Cn.col(nocca-1) = s * C_a.col(nocca-1) + std::sqrt(1-s*s) * C_a.col(nocca);
    Cn.col(nocca)   = s * C_a.col(nocca) - std::sqrt(1-s*s) * C_a.col(nocca-1);
    Ca.push_back(Cn); Cb.push_back(C_b);

    return compare_with_wick_uscf<double>(nbsf, nmo, nocca, noccb, S, hcore, II, Ca, Cb, thresh);
}


int test_complex(const char *testcase, unsigned thresh)
{
    // Report who we are
    std::ostringstream oss;
    oss << "wick_matrix_test::test_complex(" << testcase << ", " << thresh << ")";
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

    // Complex orbitals from a unitary transformation of the real orbitals
    arma::mat zero(nbsf, nmo, arma::fill::zeros);
    arma::cx_mat C_a = arma::cx_mat(Cread.cols(0,nmo-1), zero)
                     * random_unitary(nmo, nocca, 0.1, true, 1);
    arma::cx_mat C_b = arma::cx_mat(Cread.cols(nmo,2*nmo-1), zero)
                     * random_unitary(nmo, noccb, 0.1, true, 2);

    // Reference determinants as in test_real
    std::vector<arma::cx_mat> Ca, Cb;
    Ca.push_back(C_a); Cb.push_back(C_b);
    Ca.push_back(C_b); Cb.push_back(C_a);
    arma::cx_mat Cz = C_a;
    Cz.swap_cols(nocca-1, nocca);
    Ca.push_back(Cz); Cb.push_back(C_b);
    double s = 1e-5;
    arma::cx_mat Cn = C_a;
    Cn.col(nocca-1) = s * C_a.col(nocca-1) + std::sqrt(1-s*s) * C_a.col(nocca);
    Cn.col(nocca)   = s * C_a.col(nocca) - std::sqrt(1-s*s) * C_a.col(nocca-1);
    Ca.push_back(Cn); Cb.push_back(C_b);

    return compare_with_wick_uscf<std::complex<double> >(
        nbsf, nmo, nocca, noccb, S, hcore, II, Ca, Cb, thresh);
}


int main () {
    int tol = 6;

    return
    test_real("h2o_6-31g",tol)    |
    test_complex("h2o_6-31g",tol) |
    0;
}
