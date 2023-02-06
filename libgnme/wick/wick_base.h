#ifndef LIBGNME_WICK_BASE_H 
#define LIBGNME_WICK_BASE_H

#include <armadillo>
#include <libgnme/utils/bitset.h>
#include "wick_orbitals.h"

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
    const size_t m_nbsf; //!< Number of basis functions
    const size_t m_nmo;  //!< Number of (linearly independent) MOs
    const size_t m_nact; //!< Number of active orbitals
   

public:
    /** \brief Constructor for the object
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nalpha Number of high-spin electrons
        \param nbeta Number of low-spin electrons
        \param metric Overlap matrix of the basis functions
        \param Vc Constant term in the corresponding operator
     **/
    wick_base(size_t nbsf, size_t nmo, size_t nact) : 
        m_nbsf(nbsf), m_nmo(nmo), m_nact(nact)
    { }

    /** \brief Destructor **/
    virtual ~wick_base() { }

protected:
    virtual void spin_overlap(
        arma::umat xhp, arma::umat whp, 
        Tc &S, bool alpha);

    virtual void spin_rdm1(
        arma::umat xhp, arma::umat whp, 
        arma::uvec xocc, arma::uvec wocc, 
        arma::Mat<Tc> &P, bool alpha);
    virtual void same_spin_rdm2(
        arma::umat xhp, arma::umat whp, 
        arma::uvec xocc, arma::uvec wocc, 
        arma::Mat<Tc> &P, bool alpha);
    virtual void  diff_spin_rdm2(
        arma::umat xahp, arma::umat xbhp, 
        arma::umat wahp, arma::umat wbhp, 
        arma::uvec xocca, arma::uvec xoccb, 
        arma::uvec wocca, arma::uvec woccb, 
        arma::Mat<Tc> &P1a, arma::Mat<Tc> &P1b, 
        arma::Mat<Tc> &P2ab);

    virtual void spin_one_body(
        arma::umat xhp, arma::umat whp, Tc &F, bool alpha);

    virtual void same_spin_two_body(
        arma::umat xhp, arma::umat whp, Tc &V, bool alpha);
    virtual void diff_spin_two_body(
        arma::umat xa_hp, arma::umat xb_hp, 
        arma::umat wa_hp, arma::umat wb_hp, 
        Tc &V);

private:
/*
    */

    /* Getters */
    virtual const size_t& get_nz(bool alpha) = 0;
    virtual const size_t& get_ne(bool alpha) = 0;
    virtual const arma::field<arma::Mat<Tc> >& get_fX(bool alpha) = 0;
    virtual const arma::field<arma::Mat<Tc> >& get_X(bool alpha) = 0;
    virtual const arma::field<arma::Mat<Tc> >& get_Y(bool alpha) = 0;
    virtual const arma::field<arma::Mat<Tc> >& get_Q(bool alpha) = 0;
    virtual const arma::field<arma::Mat<Tc> >& get_R(bool alpha) = 0;
    virtual const arma::field<arma::Mat<Tc> >& get_wxP(bool alpha) = 0;
    virtual const arma::Col<Tc>& get_F0(bool alpha) = 0; 
    virtual const arma::field<arma::Mat<Tc> >& get_XFX(bool alpha) = 0;
    virtual const arma::field<arma::Mat<Tc> >& get_XVX(bool a, bool b) = 0;
    virtual arma::field<arma::Mat<Tc> >& get_II(bool a, bool b) = 0;
    virtual const arma::Mat<Tc>& get_V0(bool a, bool b) = 0;
};

} // namespace libgnme

#endif // LIBGNME_WICK_BASE_H
