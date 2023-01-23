#ifndef LIBGNME_WICK_RSCF_H 
#define LIBGNME_WICK_RSCF_H

#include <armadillo>
#include <libgnme/utils/bitset.h>
#include "wick_orbitals.h"

namespace libgnme {

/** \brief Implementation of the Extended Non-Orthogonal Wick's Theorem 
           for restricted orbitals
    \tparam Tc Type defining orbital coefficients
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class wick_rscf
{
private:
    /* Useful constants */
    const size_t m_nbsf; //!< Number of basis functions
    const size_t m_nmo; //!< Number of (linearly independent) MOs
    const size_t m_nelec; //!< Number of electrons
    size_t m_nact; //!< Number of active orbitals
    const arma::Mat<Tb> &m_metric; //!< Basis overlap metric

    double m_Vc; //!< constant component

    wick_orbitals<Tc,Tb> m_orb; //!< Orbital pair

    // One-body MO matrices
    bool m_one_body = false;
    bool m_two_body = false;

    // Occupied core
    arma::uvec m_core;

    /* Information about this pair */
public:
    size_t m_nz; //!< Number of zero-overlap orbitals

private:
    // Store the 'F0' terms (2)
    arma::Col<Tc> m_F0;

    // Store the '(X/Y)F(X/Y)' super matrices (4 * nmo * nmo)
    arma::field<arma::Mat<Tc> > m_XFX;

    // Store the 'V0' terms (3)
    arma::Col<Tc> m_Vsame;
    arma::Col<Tc> m_Vdiff;

    // Store the '[X/Y](J-K)[X/Y]' super matrices (8 * nmo^2)
    arma::field<arma::Mat<Tc> > m_XJX;
    arma::field<arma::Mat<Tc> > m_XKX;

    // Store two-electron repulsion integrals (16 * nmo^4)
    arma::field<arma::Mat<Tc> > m_IIsame;
    arma::field<arma::Mat<Tc> > m_IIdiff;

    // Reference bitset
    bitset m_bref; //!< Reference bitset for closed-shell determinant

public:
    /** \brief Constructor for the object
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nalpha Number of high-spin electrons
        \param nbeta Number of low-spin electrons
        \param metric Overlap matrix of the basis functions
        \param Vc Constant term in the corresponding operator
     **/
    wick_rscf(
        wick_orbitals<Tc,Tb> &orb,
        const arma::Mat<Tb> &metric, double Vc=0) :
        m_nbsf(orb.m_nbsf), m_nmo(orb.m_nmo), m_nelec(orb.m_nelec), 
        m_nact(orb.m_nact), m_metric(metric), m_Vc(Vc),
        m_orb(orb)
    { 
        // Set the reference bit strings
        size_t act_el = orb.m_nelec - orb.m_ncore; // Active alpha electrons
        std::vector<bool> ref(orb.m_nact-act_el, 0); ref.resize(orb.m_nact, 1);
        m_bref = bitset(ref);
        m_core.resize(orb.m_ncore);
        for(size_t i=0; i<orb.m_ncore; i++) m_core(i) = i;
    } 

    /** \brief Destructor **/
    virtual ~wick_rscf() { }

    /** \name Routines to add one- or two-body operators to the object **/
    ///@{
    
    /** \brief Add a one-body operator with spin-restricted integrals 
        \param F One-body integrals in AO basis
     **/
    virtual void add_one_body(arma::Mat<Tf> &F);

    /** \brief Add a two-body operator with spin-restricted integrals
        \param V Two-body integrals in AO basis. These are represented as matrices in chemists
                  notation, e.g. (ij|kl) = V(i*nbsf+j,k*nbsf+l)
     **/
    virtual void add_two_body(arma::Mat<Tb> &V);
    ///@}
    //
    
    virtual void evaluate(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S, Tc &V);
    
    virtual void evaluate_overlap(
        arma::umat &xa_hp, arma::umat &xb_hp,
        arma::umat &wa_hp, arma::umat &wb_hp,
        Tc &S);
    virtual void evaluate_one_body_spin(
        arma::umat &xhp, arma::umat &whp,
        Tc &S, Tc &V);
    virtual void evaluate(
        arma::umat &xa_hp, arma::umat &xb_hp,
        arma::umat &wa_hp, arma::umat &wb_hp,
        Tc &S, Tc &M);

    virtual void evaluate_rdm1(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S,
        arma::Mat<Tc> &Pa, arma::Mat<Tc> &Pb);

private:
    virtual void spin_rdm1(
        arma::umat xhp, arma::umat whp, 
        arma::uvec xocc, arma::uvec wocc, 
        arma::Mat<Tc> &P);
    virtual void spin_overlap(
        arma::umat xhp, arma::umat whp, Tc &S);
    virtual void spin_one_body(
        arma::umat xhp, arma::umat whp, Tc &F);
    virtual void same_spin_two_body(
        arma::umat xhp, arma::umat whp, Tc &V);
    virtual void diff_spin_two_body(
        arma::umat xa_hp, arma::umat xb_hp, 
        arma::umat wa_hp, arma::umat wb_hp, 
        Tc &V);
};

} // namespace libgnme

#endif // LIBGNME_WICK_RSCF_H
