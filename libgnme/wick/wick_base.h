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
    const size_t m_nbsf; //!< Number of basis functions
    const size_t m_nmo;  //!< Number of (linearly independent) MOs
    const size_t m_nact; //!< Number of active orbitals

    wick_orbitals<Tc,Tb> &m_orba; //!< Orbital pair
    wick_orbitals<Tc,Tb> &m_orbb; //!< Orbital pair

    one_body<Tc,Tf,Tb> *m_one_body_int; //!< Intermediates for one-body integrals
    two_body<Tc,Tf,Tb> *m_two_body_int; //!< Intermediates for two-body integrals

public:
    /** \brief Constructor for the object
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nalpha Number of high-spin electrons
        \param nbeta Number of low-spin electrons
        \param metric Overlap matrix of the basis functions
        \param Vc Constant term in the corresponding operator
     **/
    wick_base(size_t nbsf, size_t nmo, size_t nact, 
              wick_orbitals<Tc,Tb> &orba, wick_orbitals<Tc,Tb> &orbb
    ) : m_nbsf(nbsf), m_nmo(nmo), m_nact(nact), m_orba(orba), m_orbb(orbb)
    { }

    /** \brief Destructor **/
    virtual ~wick_base() { }

protected:
    void spin_overlap(
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
    /* Getters */
    //virtual const arma::field<arma::Mat<Tc> >& get_XVX(bool a, bool b) = 0;
    //virtual arma::field<arma::Mat<Tc> >& get_II(bool a, bool b) = 0;
    //virtual const arma::Mat<Tc>& get_V0(bool a, bool b) = 0;
};

template class wick_base<double, double, double>;
template class wick_base<std::complex<double>, double, double>;
template class wick_base<std::complex<double>, std::complex<double>, double>;
template class wick_base<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme

#endif // LIBGNME_WICK_BASE_H
