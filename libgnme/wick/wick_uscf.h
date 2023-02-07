#ifndef LIBGNME_WICK_USCF_H 
#define LIBGNME_WICK_USCF_H

#include <armadillo>
#include <libgnme/utils/bitset.h>
#include "wick_base.h"
#include "wick_orbitals.h"

namespace libgnme {

/** \brief Implementation of the Extended Non-Orthogonal Wick's Theorem 
           for spin unrestricted orbitals
    \tparam Tc Type defining orbital coefficients
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class wick_uscf : public wick_base<Tc,Tf,Tb>
{
protected:
    using wick_base<Tc,Tf,Tb>::m_nbsf; 
    using wick_base<Tc,Tf,Tb>::m_nmo; 
    using wick_base<Tc,Tf,Tb>::m_nact; 
    using wick_base<Tc,Tf,Tb>::m_one_body_int;
    using wick_base<Tc,Tf,Tb>::m_two_body_int;

private:
    double m_Vc; //!< constant component

    wick_orbitals<Tc,Tb> m_orba; //!< Alpha orbital pair
    wick_orbitals<Tc,Tb> m_orbb; //!< Beta orbital pair

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
    wick_uscf(
        wick_orbitals<Tc,Tb> &orba, wick_orbitals<Tc,Tb> &orbb, double Vc=0) :
    wick_base<Tc,Tf,Tb>(orba.m_nbsf, orba.m_nmo, orba.m_nact, orba, orbb),
    m_Vc(Vc), m_orba(orba), m_orbb(orbb)
    { 
        assert(orba.m_nbsf == orbb.m_nbsf);
        assert(orba.m_nmo  == orbb.m_nmo);
        assert(orba.m_nact == orbb.m_nact);
    }
        
    /** \brief Destructor **/
    virtual ~wick_uscf() { }

    /** \name Routines to add one- or two-body operators to the object **/
    ///@{
    
    /** \brief Add a one-body operator with spin-restricted integrals 
        \param F One-body integrals in AO basis
     **/
    virtual void add_one_body(arma::Mat<Tf> &F);

    /** \brief Add a one-body operator with spin-unrestricted integrals 
        \param Fa One-body integrals for high-spin component in AO basis
        \param Fb One-body integrals for low-spin component in AO basis
     **/
    virtual void add_one_body(arma::Mat<Tf> &Fa, arma::Mat<Tf> &Fb);

    /** \brief Add a two-body operator with spin-restricted integrals
        \param V Two-body integrals in AO basis. These are represented as matrices in chemists
                  notation, e.g. (ij|kl) = V(i*nbsf+j,k*nbsf+l)
     **/
    virtual void add_two_body(arma::Mat<Tb> &V);
    ///@}
    
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
        Tc &S, Tc &V, bool alpha);
    virtual void evaluate(
        arma::umat &xa_hp, arma::umat &xb_hp,
        arma::umat &wa_hp, arma::umat &wb_hp,
        Tc &S, Tc &M);

    virtual void evaluate_rdm1(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S,
        arma::Mat<Tc> &Pa, arma::Mat<Tc> &Pb);

    virtual void evaluate_rdm12(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S, 
        arma::Mat<Tc> &P1a, arma::Mat<Tc> &P1b,
        arma::Mat<Tc> &P2aa, arma::Mat<Tc> &P2bb,
        arma::Mat<Tc> &P2ab);
};

template class wick_uscf<double, double, double>;
template class wick_uscf<std::complex<double>, double, double>;
template class wick_uscf<std::complex<double>, std::complex<double>, double>;
template class wick_uscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme

#endif // LIBGNME_WICK_USCF_H
