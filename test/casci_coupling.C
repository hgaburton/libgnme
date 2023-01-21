#include <iostream>
#include <armadillo>
#include <cassert>
#include <libgnme/wick/wick_orbitals.h>
#include <libgnme/slater/slater_uscf.h>
#include <libgnme/wick/wick_rscf.h>
#include <libgnme/wick/wick_uscf.h>
#include <libgnme/utils/bitset_tools.h>
#include <libgnme/utils/linalg.h>
#include "testing.h"

using namespace libgnme;


int main () {

    // Filenames
    std::string fname_nmo   = std::string("h6_sto-3g_4_4/nelec.txt");
    std::string fname_enuc  = std::string("h6_sto-3g_4_4/enuc.txt");
    std::string fname_c1    = std::string("h6_sto-3g_4_4/c1.txt");
    std::string fname_c2    = std::string("h6_sto-3g_4_4/c2.txt");
    std::string fname_v1    = std::string("h6_sto-3g_4_4/v1.txt");
    std::string fname_v2    = std::string("h6_sto-3g_4_4/v2.txt");
    std::string fname_ov    = std::string("h6_sto-3g_4_4/ovls.txt");
    std::string fname_h1e   = std::string("h6_sto-3g_4_4/oeis.txt");
    std::string fname_II    = std::string("h6_sto-3g_4_4/teis.bin");

    /* ----------------------------------------- */
    /*  Read in basis and integrals              */
    /* ----------------------------------------- */
    // Initialise matrices
    arma::mat S, h1e, II;
    size_t nbsf = 0, nocca = 0, noccb = 0, nmo = 0, nact = 0, ncore = 0;
    std::ifstream nelec_file;
    nelec_file.open(fname_nmo);
    nelec_file >> nbsf >> nocca >> noccb >> nact >> ncore;
    nelec_file.close();
    nmo = nbsf;

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
    size_t ndeta = V(0).n_rows, ndetb = V(0).n_cols;
    /* ----------------------------------------- */

    /* Number of coupled CASCI states            */
    size_t nstate = C.n_rows; 

    /* List of bitsets for FCI configurations    */
    std::vector<bitset> v_ba = fci_bitset_list(nocca-ncore, nact);
    std::vector<bitset> v_bb = fci_bitset_list(noccb-ncore, nact);

    /* Initialise H and Ov for CASCI coupling    */
    arma::mat H(nstate,nstate,arma::fill::zeros);
    arma::mat Ov(nstate,nstate,arma::fill::zeros);
    /* Store 1RDM for different couplings as arma::field */
    arma::field<arma::mat> RDM1(nstate,nstate);

    /* ----------------------------------------- *
     * Compute coupling terms                    *
     * ----------------------------------------- */
    for(size_t x=0; x<nstate; x++)
    for(size_t w=x; w<nstate; w++)
    {
        // Setup the biorthogonalized orbital pair
        wick_orbitals<double,double> orbs(nbsf, nmo, nocca, C(x), C(w), S, nact, ncore);
        // Setup the matrix builder object
        wick_rscf<double,double,double> mb(orbs, S, enuc);
        // Add one- and two-body contributions
        mb.add_one_body(h1e);
        mb.add_two_body(II);
        // Initialise 1RDM
        RDM1(x,w).resize(nmo, nmo); RDM1(x,w).zeros();

        // Loop over FCI occupation strings
        for(size_t iwa=0; iwa < v_ba.size(); iwa++)
        for(size_t iwb=0; iwb < v_bb.size(); iwb++)
        for(size_t ixa=0; ixa < v_ba.size(); ixa++)
        for(size_t ixb=0; ixb < v_bb.size(); ixb++)
        {
            /* ------------------------------------------------------------*/
            /*  Compute S and H contribution for this pair of determinants */
            /* ------------------------------------------------------------*/
            // Intialise temporary variables
            double Htmp = 0.0, Stmp = 0.0;
            // Evaluate matrix element
            mb.evaluate(v_ba[ixa], v_bb[ixb], v_ba[iwa], v_bb[iwb], Stmp, Htmp);
            // Increment total coupling
            H(x,w) += Htmp * V(w)(iwa, iwb) * V(x)(ixa, ixb);
            Ov(x,w) += Stmp * V(w)(iwa, iwb) * V(x)(ixa, ixb);

            /* ------------------------------------------------------------*/
            /*  Compute 1RDM contribution for this pair of determinants    */
            /* ------------------------------------------------------------*/
            // Initialise temporary matrices
            arma::mat tmpPa(orbs.m_nmo, orbs.m_nmo, arma::fill::zeros);
            arma::mat tmpPb(orbs.m_nmo, orbs.m_nmo, arma::fill::zeros);
            // Evaluate contribution from Wick's theory
            mb.evaluate_rdm1(v_ba[ixa], v_bb[ixb], v_ba[iwa], v_bb[iwb], Stmp, tmpPa, tmpPb);
            // Increment total 1RDM
            RDM1(x,w) += (tmpPa + tmpPb) * V(w)(iwa, iwb) * V(x)(ixa, ixb);
        }

        // NOTE: The 1RDM is built in an asymmetric basis, with the x orbitals (bra)
        //       for the creation operator and w orbitals (ket) for the annihilation operator 
        //       This can be illustrated by transforming to the AO basis.
        RDM1(x,w) = C(w) * RDM1(x,w) * C(x).t();

        // Make Hermitian
        H(w,x)    = H(x,w);
        Ov(w,x)   = Ov(x,w);
        RDM1(w,x) = RDM1(x,w).t();
    }


    // Print out the coupled matrices
    H.print("\nHamiltonian");
    Ov.print("\nOverlap:");


    // Solve a generalised nonorthogonal CI
    arma::mat X, eigvec;
    arma::vec eigval;
    gen_eig_sym(nstate, H, Ov, X, eigval, eigvec);
    ((arma::rowvec) eigval.t()).print("\nRecoupled NO-CAS-CI eigenvalues");
    eigvec.print("\nRecoupled NO-CAS-CI eigenvectors");
}

