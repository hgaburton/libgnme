#ifndef LIBGNME_WICK_RSCF_H 
#define LIBGNME_WICK_RSCF_H

#include <armadillo>
#include <libgnme/utils/bitset.h>
#include "wick_base.h"
#include "wick_orbitals.h"

namespace libgnme {

/** \brief Implementation of the Extended Non-Orthogonal Wick's Theorem for restricted orbitals
    \tparam Tc Type defining orbital coefficients
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class wick_rscf : public wick_base<Tc,Tf,Tb>
{
protected:
    using wick_base<Tc,Tf,Tb>::m_nmo;
    using wick_base<Tc,Tf,Tb>::m_one_body_int;
    using wick_base<Tc,Tf,Tb>::m_two_body_int;
    using wick_base<Tc,Tf,Tb>::m_orba;
    using wick_base<Tc,Tf,Tb>::m_orbb;

private:
    double m_Vc; //!< Constant component

    bool m_one_body = false; //!< Control variable if one-body elements are required
    bool m_two_body = false; //!< Control variable if two-body elements are required

public:
    /** \brief Constructor for the object from set of orbital pairs
        \param orb wick_orbitals containing Lowdin-paired set of orbitals
        \param Vc Constant contribution [default=0]
     **/
    wick_rscf(wick_orbitals<Tc,Tb> &orb, double Vc=0):
        wick_base<Tc,Tf,Tb>(orb, orb), m_Vc(Vc)
    { } 

    /** \brief Destructor **/
    virtual ~wick_rscf() 
    { 
        if(m_one_body) delete m_one_body_int;
        if(m_two_body) delete m_two_body_int;
    }

    /** \name Routines to add one- or two-body operators to the object **/
    ///@{
    
    /** \brief Add a one-body operator with spin-restricted integrals 
        \param F One-body integrals in AO basis
     **/
    void add_one_body(arma::Mat<Tf> &F);

    /** \brief Add a two-body operator with spin-restricted integrals
        \param V Two-body integrals in AO basis. These are represented as matrices in chemists
                  notation, e.g. (ij|kl) = V(i*nbsf+j,k*nbsf+l)
     **/
    void add_two_body(arma::Mat<Tb> &V);
    ///@}
    //
    
    /** \brief Evaluate integral using bitset representation
        \param bxa Bitset for bra alpha excitation
        \param bxb Bitset for bra beta  excitation
        \param bwa Bitset for ket alpha excitation
        \param bwb Bitset for ket beta  excitation
        \param S   Output overlap matrix element
        \param V   Output integral value
     **/
    void evaluate(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S, Tc &V);

    /** \brief Compute 1RDM using bitset representation
        \param bxa Bitset for bra alpha excitation
        \param bxb Bitset for bra beta  excitation
        \param bwa Bitset for ket alpha excitation
        \param bwb Bitset for ket beta  excitation
        \param S   Output overlap matrix element
        \param P1  Output 1RDM
     **/
    void evaluate_rdm1(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S, arma::Mat<Tc> &P1);

    /** \brief Compute 1RDM and 2RDM using bitset representation
        \param bxa Bitset for bra alpha excitation
        \param bxb Bitset for bra beta  excitation
        \param bwa Bitset for ket alpha excitation
        \param bwb Bitset for ket beta  excitation
        \param S   Output overlap matrix element
        \param P1  Output 1RDM
        \param P2  Output 2RDM
     **/
    void evaluate_rdm12(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S, 
        arma::Mat<Tc> &P1, arma::Mat<Tc> &P2);

    /** \brief Evaluate matrix element using particle-hole representation
        \param xa_hp Particle-hole indices for bra alpha excitation
        \param xb_hp Particle-hole indices for bra beta  excitation
        \param wa_hp Particle-hole indices for ket alpha excitation
        \param wb_hp Particle-hole indices for ket beta  excitation
        \param S   Output overlap matrix element
        \param V   Output one-body matrix element
     **/
    void evaluate(
        arma::umat &xa_hp, arma::umat &xb_hp,
        arma::umat &wa_hp, arma::umat &wb_hp,
        Tc &S, Tc &M);
    
    /** \brief Evaluate overlap value using particle-hole representation
        \param xa_hp Particle-hole indices for bra alpha excitation
        \param xb_hp Particle-hole indices for bra beta  excitation
        \param wa_hp Particle-hole indices for ket alpha excitation
        \param wb_hp Particle-hole indices for ket beta  excitation
        \param S   Output overlap matrix element
     **/
    void evaluate_overlap(
        arma::umat &xa_hp, arma::umat &xb_hp,
        arma::umat &wa_hp, arma::umat &wb_hp,
        Tc &S);

    /** \brief Evaluate one-body element for given spin sector using particle-hole representation
        \param xhp Particle-hole indices for bra excitation
        \param whp Particle-hole indices for ket excitation
        \param S   Output overlap matrix element
        \param V   Output one-body matrix element
     **/
    void evaluate_one_body_spin(
        arma::umat &xhp, arma::umat &whp,
        Tc &S, Tc &V);
};

} // namespace libgnme

#endif // LIBGNME_WICK_RSCF_H
