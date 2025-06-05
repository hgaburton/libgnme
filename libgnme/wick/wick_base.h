#ifndef LIBGNME_WICK_BASE_H 
#define LIBGNME_WICK_BASE_H

#include <armadillo>
#include <libgnme/utils/bitset.h>
#include "wick_orbitals.h"
#include "one_body.h"
#include "two_body.h"

namespace libgnme {

/** \brief Abstract base class for implementing nonorthogonal Wick's theorem
    \tparam Tc Type defining orbital coefficients
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class wick_base
{
protected:
    /* Useful constants */
    const size_t m_nmo;  //!< Number of (linearly independent) MOs

    wick_orbitals<Tc,Tb> &m_orba; //!< High-spin orbital pair
    wick_orbitals<Tc,Tb> &m_orbb; //!< Low-spin orbital pair

    one_body<Tc,Tf,Tb> *m_one_body_int; //!< Intermediates for one-body integrals
    two_body<Tc,Tf,Tb> *m_two_body_int; //!< Intermediates for two-body integrals

public:
    /** \brief Constructor using orbital pairs
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nalpha Number of high-spin electrons
        \param nbeta Number of low-spin electrons
        \param metric Overlap matrix of the basis functions
        \param Vc Constant term in the corresponding operator
     **/
    wick_base(wick_orbitals<Tc,Tb> &orba, wick_orbitals<Tc,Tb> &orbb) :
        m_nmo(orba.m_nmo), m_orba(orba), m_orbb(orbb)
    { 
        // Check for consistency
        assert(orba.m_nbsf == m_orbb.m_nbsf);
        assert(orba.m_nmo  == m_orbb.m_nmo);
    }

    /** \brief Destructor **/
    virtual ~wick_base() { }

protected:
    /** \brief Compute the overlap for a given spin sector
        \param xhp Particle-hole indices for bra state
        \param whp Particle-hole indices for ket state
        \param S Output overlap value
        \param alpha True for alpha; false for beta
     **/
    void spin_overlap(
        arma::umat xhp, arma::umat whp, 
        Tc &S, bool alpha);

    /** \brief Compute 1RDM for given spin sector
        \param xhp Particle-hole indices for bra state
        \param whp Particle-hole indices for ket state
        \param xocc Vector of occupied indices in bra state
        \param wocc Vector of occupied indices in ket state
        \param P Output 1RDM 
        \param alpha True for alpha; false for beta
     **/
    virtual void spin_rdm1(
        arma::umat xhp, arma::umat whp, 
        arma::uvec xocc, arma::uvec wocc, 
        arma::Mat<Tc> &P, bool alpha);

    /** \brief Compute same-spin 2RDM for given spin sector
        \param xhp Particle-hole indices for bra state
        \param whp Particle-hole indices for ket state
        \param xocc Vector of occupied indices in bra state
        \param wocc Vector of occupied indices in ket state
        \param P Output 2RDM in (pq,rs) format 
        \param alpha True for alpha; false for beta
     **/
    virtual void same_spin_rdm2(
        arma::umat xhp, arma::umat whp, 
        arma::uvec xocc, arma::uvec wocc, 
        arma::Mat<Tc> &P, bool alpha);

    /** \brief Compute different-spin 2RDM
        \param xahp Alpha particle-hole indices for bra state
        \param xbhp Beta  particle-hole indices for bra state
        \param wahp Alpha particle-hole indices for ket state
        \param wbhp Beta  particle-hole indices for ket state
        \param xocc Vector of alpha occupied indices in bra state
        \param xocc Vector of beta  occupied indices in bra state
        \param wocc Vector of alpha occupied indices in ket state
        \param wocc Vector of beta  occupied indices in ket state
        \param P1a Output 1RDM for alpha sector
        \param P1b Output 1RDM for beta sector
        \param P2ab Output different-spin 2RDM in (pq,rs) format 
     **/
    virtual void  diff_spin_rdm2(
        arma::umat xahp, arma::umat xbhp, 
        arma::umat wahp, arma::umat wbhp, 
        arma::uvec xocca, arma::uvec xoccb, 
        arma::uvec wocca, arma::uvec woccb, 
        arma::Mat<Tc> &P1a, arma::Mat<Tc> &P1b, 
        arma::Mat<Tc> &P2ab);

    /** \brief One-body coupling for a given spin sector
        \param xhp Particle-hole indices for bra state
        \param whp Particle-hole indices for ket state
        \param F Output one-body coupling 
        \param alpha True for alpha; false for beta
     **/
    virtual void spin_one_body(
        arma::umat xhp, arma::umat whp, Tc &F, bool alpha);

    /** \brief Same-spin two-body coupling 
        \param xhp Particle-hole indices for bra state
        \param whp Particle-hole indices for ket state
        \param V Output two-body coupling 
        \param alpha True for alpha; false for beta
     **/
    virtual void same_spin_two_body(
        arma::umat xhp, arma::umat whp, Tc &V, bool alpha);

    /** \brief Different-spin two-body coupling 
        \param xahp Alpha particle-hole indices for bra state
        \param xbhp Beta  particle-hole indices for bra state
        \param wahp Alpha particle-hole indices for ket state
        \param wbhp Beta  particle-hole indices for ket state
        \param V Output two-body coupling 
     **/
    virtual void diff_spin_two_body(
        arma::umat xa_hp, arma::umat xb_hp, 
        arma::umat wa_hp, arma::umat wb_hp, 
        Tc &V);
};

extern template class wick_base<double, double, double>;
extern template class wick_base<std::complex<double>, double, double>;
extern template class wick_base<std::complex<double>, std::complex<double>, double>;
extern template class wick_base<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme

#endif // LIBGNME_WICK_BASE_H
