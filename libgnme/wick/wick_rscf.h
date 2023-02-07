#ifndef LIBGNME_WICK_RSCF_H 
#define LIBGNME_WICK_RSCF_H

#include <armadillo>
#include <libgnme/utils/bitset.h>
#include "wick_base.h"
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
class wick_rscf : public wick_base<Tc,Tf,Tb>
{
protected:
    using wick_base<Tc,Tf,Tb>::m_nmo;
    using wick_base<Tc,Tf,Tb>::m_one_body_int;
    using wick_base<Tc,Tf,Tb>::m_two_body_int;
    using wick_base<Tc,Tf,Tb>::m_orba;
    using wick_base<Tc,Tf,Tb>::m_orbb;

private:
    double m_Vc; //!< constant component

    // One-body MO matrices
    bool m_one_body = false;
    bool m_two_body = false;

public:
    /** \brief Constructor for the object
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nalpha Number of high-spin electrons
        \param nbeta Number of low-spin electrons
        \param metric Overlap matrix of the basis functions
        \param Vc Constant term in the corresponding operator
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
    
    void evaluate(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S, Tc &V);
    
    void evaluate_overlap(
        arma::umat &xa_hp, arma::umat &xb_hp,
        arma::umat &wa_hp, arma::umat &wb_hp,
        Tc &S);
    void evaluate_one_body_spin(
        arma::umat &xhp, arma::umat &whp,
        Tc &S, Tc &V);
    void evaluate(
        arma::umat &xa_hp, arma::umat &xb_hp,
        arma::umat &wa_hp, arma::umat &wb_hp,
        Tc &S, Tc &M);

    void evaluate_rdm1(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S, arma::Mat<Tc> &P1);

    void evaluate_rdm12(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S, 
        arma::Mat<Tc> &P1, arma::Mat<Tc> &P2);
};

template class wick_rscf<double, double, double>;
template class wick_rscf<std::complex<double>, double, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme

#endif // LIBGNME_WICK_RSCF_H
