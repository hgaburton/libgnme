#include <iostream>
#include <armadillo>
#include <cassert>
#include <libgnme/wick/wick_orbitals.h>
#include <libgnme/wick/wick_uscf.h>
#include <libgnme/wick/wick_rscf.h>
#include <libgnme/utils/bitset_tools.h>
#include "testing.h"

using namespace libgnme;

template<typename Tc, typename Tw>
int compare_with_evaluate(
    Tw &mb, wick_orbitals<Tc,double> &orbs_a, wick_orbitals<Tc,double> &orbs_b,
    std::vector<bitset> &xa, std::vector<bitset> &xb,
    std::vector<bitset> &wa, std::vector<bitset> &wb, unsigned thresh)
{
    // Compute matrix elements
    arma::Mat<Tc> Sd, Hd;
    mb.evaluate_dense(xa, xb, wa, wb, Sd, Hd);
    assert(Sd.n_rows == xa.size() and Sd.n_cols == wa.size());

    for(size_t i=0; i<xa.size(); i++)
    for(size_t j=0; j<wa.size(); j++)
    {
        // Get excitation indices
        arma::umat xahp, xbhp, wahp, wbhp;
        int pxa, pxb, pwa, pwb;
        orbs_a.m_refx.m_bs.excitation(xa[i], xahp, pxa);
        orbs_b.m_refx.m_bs.excitation(xb[i], xbhp, pxb);
        orbs_a.m_refw.m_bs.excitation(wa[j], wahp, pwa);
        orbs_b.m_refw.m_bs.excitation(wb[j], wbhp, pwb);

        // Compute matrix elements with the parity of the excitation
        Tc swick = 0.0, fwick = 0.0;
        mb.evaluate(xahp, xbhp, wahp, wbhp, swick, fwick);
        swick *= (double) (pxa * pxb * pwa * pwb);
        fwick *= (double) (pxa * pxb * pwa * pwb);

        // Test overlap
        if(std::abs(Sd(i,j) - swick) > std::pow(0.1,thresh))
        {
            std::cout << "Overlap error = "
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(Sd(i,j) - swick) << std::endl;
            return 1;
        }

        // Test Hamiltonian matrix element
        if(std::abs(Hd(i,j) - fwick) > std::pow(0.1,thresh))
        {
            std::cout << "Hamiltonian error = "
                      << std::fixed << std::setprecision(thresh+2) << std::scientific
                      << std::abs(Hd(i,j) - fwick) << std::endl;
            return 1;
        }
    }

    return 0;
}


template<typename Tc>
int compare_references(
    size_t nbsf, size_t nmo, size_t nocca, size_t noccb, arma::mat &S, arma::mat &hcore,
    arma::mat &II, std::vector<arma::Mat<Tc> > &Ca, std::vector<arma::Mat<Tc> > &Cb,
    unsigned thresh)
{
    const double enuc = 1.5;

    // Define "reference" bitsets, with the rightmost element representing the first MO
    std::vector<uint8_t> refa(nmo, 0), refb(nmo, 0);
    for(size_t k=0; k<nocca; k++) refa[nmo-1-k] = 1;
    for(size_t k=0; k<noccb; k++) refb[nmo-1-k] = 1;

    // Reference, alpha and beta single excitations, and alpha/alpha double excitations
    std::vector<bitset> ba, bb;
    ba.push_back(bitset(refa)); bb.push_back(bitset(refb));
    for(size_t i=0; i<nocca; i++)
    for(size_t a=nocca; a<std::min(nocca+3,nmo); a++)
    {
        std::vector<uint8_t> v = refa; v[nmo-1-i] = 0; v[nmo-1-a] = 1;
        ba.push_back(bitset(v)); bb.push_back(bitset(refb));
    }
    for(size_t i=0; i<noccb; i++)
    for(size_t a=noccb; a<std::min(noccb+3,nmo); a++)
    {
        std::vector<uint8_t> v = refb; v[nmo-1-i] = 0; v[nmo-1-a] = 1;
        ba.push_back(bitset(refa)); bb.push_back(bitset(v));
    }
    for(size_t j=0; j<nocca; j++)
    for(size_t k=0; k<j; k++)
    for(size_t b=nocca; b<std::min(nocca+3,nmo); b++)
    for(size_t c=nocca; c<b; c++)
    {
        std::vector<uint8_t> v = refa;
        v[nmo-1-j] = 0; v[nmo-1-k] = 0; v[nmo-1-b] = 1; v[nmo-1-c] = 1;
        ba.push_back(bitset(v)); bb.push_back(bitset(refb));
    }

    // Compare each pair of references
    for(size_t x=0; x<Ca.size(); x++)
    for(size_t w=0; w<Ca.size(); w++)
    {
        // Construct the wick_orbitals object
        wick_orbitals<Tc,double> orbs_a(nbsf, nmo, nocca, Ca[x], Ca[w], S);
        wick_orbitals<Tc,double> orbs_b(nbsf, nmo, noccb, Cb[x], Cb[w], S);

        // Spin-unrestricted matrix builder
        wick_uscf<Tc,double,double> mb(orbs_a, orbs_b, enuc);
        mb.add_one_body(hcore);
        mb.add_two_body(II);
        if(compare_with_evaluate(mb, orbs_a, orbs_b, ba, bb, ba, bb, thresh)) return 1;

        // Spin-restricted matrix builder with the high-spin orbitals
        if(nocca != noccb) continue;
        wick_rscf<Tc,double,double> mr(orbs_a, enuc);
        mr.add_one_body(hcore);
        mr.add_two_body(II);
        if(compare_with_evaluate(mr, orbs_a, orbs_a, ba, bb, ba, bb, thresh)) return 1;
    }

    return 0;
}


int test_real(const char *testcase, unsigned thresh)
{
    // Report who we are
    std::ostringstream oss;
    oss << "wick_dense_test::test_real(" << testcase << ", " << thresh << ")";
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

    return compare_references<double>(nbsf, nmo, nocca, noccb, S, hcore, II, Ca, Cb, thresh);
}


int test_complex(const char *testcase, unsigned thresh)
{
    // Report who we are
    std::ostringstream oss;
    oss << "wick_dense_test::test_complex(" << testcase << ", " << thresh << ")";
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

    return compare_references<std::complex<double> >(
        nbsf, nmo, nocca, noccb, S, hcore, II, Ca, Cb, thresh);
}


int test_active(const char *testx, const char *testw, unsigned thresh)
{
    // Report who we are
    std::ostringstream oss;
    oss << "wick_dense_test::test_active(" << testx << ", " << testw << ", " << thresh << ")";
    std::cout << oss.str() << std::endl;

    // Filenames
    std::string fname_ov    = testw + std::string("/ovls.txt");
    std::string fname_hcore = testw + std::string("/oeis.txt");
    std::string fname_II    = testw + std::string("/teis.bin");
    std::string fname_nmox  = testx + std::string("/nelec.txt");
    std::string fname_nmow  = testw + std::string("/nelec.txt");
    std::string fname_cx    = testx + std::string("/c1.txt");
    std::string fname_cw    = testw + std::string("/c1.txt");

    // Read-in basis information with the active space of each state
    size_t nbsf = 0, nocca = 0, noccb = 0, nact[2] = {0,0}, ncore[2] = {0,0};
    std::ifstream nelec_file;
    nelec_file.open(fname_nmox);
    nelec_file >> nbsf >> nocca >> noccb >> nact[0] >> ncore[0];
    nelec_file.close();
    nelec_file.open(fname_nmow);
    nelec_file >> nbsf >> nocca >> noccb >> nact[1] >> ncore[1];
    nelec_file.close();
    size_t nmo = nbsf;

    // Check the input
    assert(nbsf  > 0);
    assert(nocca == noccb);

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
    arma::mat Cx, Cw;
    if(!Cx.load(fname_cx, arma::raw_ascii)) return 1;
    if(!Cw.load(fname_cw, arma::raw_ascii)) return 1;
    assert(Cx.n_rows == nbsf and Cx.n_cols == nmo);
    assert(Cw.n_rows == nbsf and Cw.n_cols == nmo);

    // Construct the wick_orbitals object
    reference_state<double> refx(nbsf, nmo, nocca, nact[0], ncore[0], Cx);
    reference_state<double> refw(nbsf, nmo, nocca, nact[1], ncore[1], Cw);
    wick_orbitals<double,double> orbs(refx, refw, S);

    // Setup matrix builder
    wick_rscf<double,double,double> mb(orbs, 1.5);
    mb.add_one_body(hcore);
    mb.add_two_body(II);

    // Every determinant in each active space
    std::vector<bitset> xa = fci_bitset_list(nocca-ncore[0], nact[0]);
    std::vector<bitset> wa = fci_bitset_list(nocca-ncore[1], nact[1]);
    std::vector<bitset> xs, xb, ws, wb;
    for(size_t i=0; i<xa.size(); i++)
    for(size_t j=0; j<xa.size(); j++)
    {
        xs.push_back(xa[i]); xb.push_back(xa[j]);
    }
    for(size_t i=0; i<wa.size(); i++)
    for(size_t j=0; j<wa.size(); j++)
    {
        ws.push_back(wa[i]); wb.push_back(wa[j]);
    }

    return compare_with_evaluate(mb, orbs, orbs, xs, xb, ws, wb, thresh);
}


int main () {
    int tol = 8;

    return
    test_real("h2o_6-31g",tol)    |
    test_complex("h2o_6-31g",tol) |
    test_active("h6_sto-3g_4_4","h6_sto-3g_6_6",tol) |
    test_active("h6_sto-3g_6_6","h6_sto-3g_4_4",tol) |
    test_active("h6_sto-3g_4_4","h6_sto-3g_4_4",tol) |
    0;
}
