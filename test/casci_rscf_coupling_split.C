#include <iostream>
#define ARMA_ALLOW_FAKE_GCC
#include <armadillo>
#include <cassert>
#include <libgnme/wick/wick_orbitals.h>
#include <libgnme/wick/wick_rscf.h>
#include <libgnme/utils/bitset_tools.h>
#include <libgnme/utils/linalg.h>
#include <libgnme/utils/eri_ao2mo.h>

using namespace libgnme;

int main () {

    std::cout << " -------------------------------------------------" << std::endl;
    std::cout << "    LibGNME example: CAS-CI coupling terms        " << std::endl;
    std::cout << "    Hugh G. A. Burton (Jan 2023)                  " << std::endl;
    std::cout << " -------------------------------------------------" << std::endl;

    /* ----------------------------------------- */
    /*  Name and location of data files          */
    /* ----------------------------------------- */
    std::string fname_enuc  = std::string("h6_sto-3g_6_6/enuc.txt");
    std::string fname_h1e   = std::string("h6_sto-3g_6_6/oeis.txt");
    std::string fname_ov    = std::string("h6_sto-3g_6_6/ovls.txt");
    std::string fname_II    = std::string("h6_sto-3g_6_6/teis.bin");
    std::string fname_c1    = std::string("h6_sto-3g_4_4/c1.txt");
    std::string fname_nmo1  = std::string("h6_sto-3g_4_4/nelec.txt");
    std::string fname_v1    = std::string("h6_sto-3g_4_4/v1.txt");
    std::string fname_c2    = std::string("h6_sto-3g_6_6/c1.txt");
    std::string fname_nmo2  = std::string("h6_sto-3g_6_6/nelec.txt");
    std::string fname_v2    = std::string("h6_sto-3g_6_6/v1.txt");

    /* ----------------------------------------- */
    /*  Read in basis and integrals              */
    /* ----------------------------------------- */
    // Initialise matrices
    arma::mat S, h1e, II;
    size_t nbsfin[2] = {0,0}, nocca[2] = {0,0}, noccb[2] = {0,0}, nact[2] = {0,0}, ncore[2] = {0,0};
    std::ifstream nelec_file;
    nelec_file.open(fname_nmo1);
    nelec_file >> nbsfin[0] >> nocca[0] >> noccb[0] >> nact[0] >> ncore[0];
    nelec_file.close();
    nelec_file.open(fname_nmo2);
    nelec_file >> nbsfin[1] >> nocca[1] >> noccb[1] >> nact[1] >> ncore[1];
    nelec_file.close();
    // Check the input
    assert(nbsfin[0] == nbsfin[1]);
    assert(nocca[0] == nocca[1]);
    assert(noccb[0] == noccb[1]);
    size_t nbsf = nbsfin[0];
    size_t nmo  = nbsf;

    // Read-in nuclear repulsion energy
    double enuc;
    std::ifstream enuc_file;
    enuc_file.open(fname_enuc);
    enuc_file >> enuc;
    enuc_file.close();
    
    // Read-in overlap matrix and check dimensions
    if(!S.load(fname_ov, arma::raw_ascii)) return 1;
    assert(S.n_rows == nbsf); assert(S.n_cols == nbsf);

    // Read-in one-electron matrix
    if(!h1e.load(fname_h1e, arma::raw_ascii)) return 1;
    assert(h1e.n_rows == nbsf); assert(h1e.n_cols == nbsf);

    // Read-in two-electron integrals, structured as II_{ij,jk} = (ij | kl)
    if(!II.load(fname_II, arma::raw_binary)) return 1;
    assert(II.n_elem == nbsf*nbsf*nbsf*nbsf);
    II.reshape(nbsf*nbsf, nbsf*nbsf);
    /* ----------------------------------------- */

    /* ----------------------------------------- */
    /*  Read in orbital coefficients             */
    /* ----------------------------------------- */
    arma::field<arma::mat> C(2);
    if(!C(0).load(fname_c1, arma::raw_ascii)) return 1;
    assert(C(0).n_rows == nbsf); assert(C(0).n_cols == nmo);
    if(!C(1).load(fname_c2, arma::raw_ascii)) return 1;
    assert(C(1).n_rows == nbsf); assert(C(1).n_cols == nmo);
    /* ----------------------------------------- */

    /* ----------------------------------------- *
     *  Read in CI coefficients                  *
     *  Each V is 2d  with rows = alpha strings  *
     *                 and cols = beta  strings  *  
     * ----------------------------------------- */
    arma::field<arma::mat> V(2);
    if(!V(0).load(fname_v1, arma::raw_ascii)) return 1;
    if(!V(1).load(fname_v2, arma::raw_ascii)) return 1;
    /* ----------------------------------------- */

    /* Number of coupled CASCI states            */
    size_t nstate = C.n_rows; 

    /* Initialise H and Ov for CASCI coupling    */
    arma::mat H(nstate,nstate,arma::fill::zeros);
    arma::mat Ov(nstate,nstate,arma::fill::zeros);
    /* Store 1RDM for different couplings as arma::field */
    arma::field<arma::mat> RDM1(nstate,nstate);
    arma::field<arma::mat> RDM2(nstate,nstate);

    /* ----------------------------------------- *
     * Compute coupling terms                    *
     * ----------------------------------------- */
    for(size_t x=0; x<nstate; x++)
    for(size_t w=x; w<nstate; w++)
    {
        // Setup the biorthogonalized orbital pair
        reference_state<double> refx(nbsf, nmo, nocca[x], nact[x], ncore[x], C(x));
        reference_state<double> refw(nbsf, nmo, nocca[w], nact[w], ncore[w], C(w));
        wick_orbitals<double,double> orbs(refx, refw, S);
        // Setup the matrix builder object
        wick_rscf<double,double,double> mb(orbs, enuc);
        wick_rscf<double,double,double> mb2(orbs, enuc);
        // Add one- and two-body contributions
        mb.add_one_body(h1e);
        mb.add_two_body(II);
        // Initialise 1RDM
        RDM1(x,w).resize(nmo, nmo); RDM1(x,w).zeros();
        RDM2(x,w).resize(nmo*nmo, nmo*nmo); RDM2(x,w).zeros();

        // Transform 1e and 2e integrals into asymmetric basis
        arma::mat hmo = C(x).t() * h1e * C(w);
        arma::mat IImo(nmo*nmo,nmo*nmo,arma::fill::zeros);
        eri_ao2mo(C(x), C(w), C(x), C(w), II, IImo, false);

        /* List of bitsets for FCI configurations    */
        std::vector<bitset> v_xba, v_xbb, v_wba, v_wbb;
        v_wba = fci_bitset_list(nocca[w]-ncore[w], nact[w]);
        v_wbb = fci_bitset_list(noccb[w]-ncore[w], nact[w]);
        v_xba = fci_bitset_list(nocca[x]-ncore[x], nact[x]);
        v_xbb = fci_bitset_list(noccb[x]-ncore[x], nact[x]);

        // Loop over FCI occupation strings
        for(size_t iwa=0; iwa < v_wba.size(); iwa++)
        for(size_t iwb=0; iwb < v_wbb.size(); iwb++)
        for(size_t ixa=0; ixa < v_xba.size(); ixa++)
        for(size_t ixb=0; ixb < v_xbb.size(); ixb++)
        {
            /* ------------------------------------------------------------*/
            /*  Compute S and H contribution for this pair of determinants */
            /* ------------------------------------------------------------*/
            // Intialise temporary variables
            double Htmp = 0.0, Stmp = 0.0;
            // Evaluate matrix element
            mb.evaluate(v_xba[ixa], v_xbb[ixb], v_wba[iwa], v_wbb[iwb], Stmp, Htmp);
            // Increment total coupling
            H(x,w)  += Htmp * V(w)(iwa, iwb) * V(x)(ixa, ixb);
            Ov(x,w) += Stmp * V(w)(iwa, iwb) * V(x)(ixa, ixb);

            /* ------------------------------------------------------------*/
            /*  Compute 1RDM contribution for this pair of determinants    */
            /* ------------------------------------------------------------*/
            // Initialise temporary matrices
            arma::mat tmpP1(nmo, nmo, arma::fill::zeros);
            arma::mat tmpP2(nmo*nmo, nmo*nmo, arma::fill::zeros);
            // Evaluate contribution from Wick's theory
            mb2.evaluate_rdm12(v_xba[ixa], v_xbb[ixb], v_wba[iwa], v_wbb[iwb], Stmp, tmpP1, tmpP2);
            // Increment total 2RDM
            RDM1(x,w) += tmpP1 * V(w)(iwa, iwb) * V(x)(ixa, ixb);
            // Increment total 1RDM
            RDM2(x,w) += tmpP2 * V(w)(iwa, iwb) * V(x)(ixa, ixb);
        }

        // NOTE: The 1RDM and 2RDM are built in an asymmetric basis, with the x orbitals (bra)
        //       for the creation operator and w orbitals (ket) for the annihilation operator 

        // Make Hermitian
        H(w,x)    = H(x,w);
        Ov(w,x)   = Ov(x,w);
        RDM1(w,x) = RDM1(x,w).t();

        std::cout << enuc * Ov(x,w) + arma::dot(hmo.st(),RDM1(x,w)) + 0.5 * arma::dot(IImo, RDM2(x,w)) << " " << H(x,w) << std::endl;
    }

    // Print out the 1-RDM matrices
    for(size_t x=0; x<nstate; x++)
    for(size_t w=0; w<nstate; w++)
    {
        std::cout << "\n RDM-1 for <Psi_" << x << "| and |Psi_" << w << ">" << std::endl;
        RDM1(x,w).print();
    }

    // Print out the coupled matrices
    H.print("\n Hamiltonian");
    Ov.print("\n Overlap:");

    // Solve a generalised nonorthogonal CI
    arma::mat X, eigvec;
    arma::vec eigval;
    gen_eig_sym(nstate, H, Ov, X, eigval, eigvec);
    ((arma::rowvec) eigval.t()).print("\n Recoupled NO-CAS-CI eigenvalues");
    eigvec.print("\n Recoupled NO-CAS-CI eigenvectors");

    std::cout << std::endl;
    std::cout << " -------------------------------------------------" << std::endl;
    std::cout << std::endl;
    return 0;
}
