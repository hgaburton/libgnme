#ifndef LIBGNME_WICK_ORBITALS_H 
#define LIBGNME_WICK_ORBITALS_H

#include <armadillo>
#include "reference_state.h"

namespace libgnme {

/** \brief Container to compute and store Lowdin-paired orbitals for two reference states
    \tparam Tc Type defining orbital coefficients
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tb>
class wick_orbitals
{
public:
    /* Useful constants */
    const size_t m_nbsf;  //!< Number of basis functions
    const size_t m_nmo;   //!< Number of (linearly independent) MOs
    const size_t m_nelec; //!< Number of electrons
    const arma::Mat<Tb> &m_metric; //!< Basis overlap metric

    /* Information about this pair */
    size_t m_nz; //!< Number of zero-overlap orbitals

    Tc m_redS; //!< Reduced overlap

    /* Near-zero overlaps 1e-8 < |S_k| < 1e-1 introduce large 1/S_k terms that cancel in the
       matrix elements. Each matrix element is linear in 1/S_k, so it is evaluated from the
       corners 1/S_k -> 0 and 1/S_k -> 1 as S_k F(1/S_k) = (S_k - 1) F(0) + F(1). */
    size_t m_nnear;   //!< Number of near-zero orbital overlaps
    size_t m_corner;  //!< Bit mask of near-zero overlaps entering with unit weight
    size_t m_ncorner; //!< Number of corners represented by this object

    arma::field<arma::Mat<Tc> > m_M;  //!< Co-density matrices [nz][nbsf,nbsf]
    arma::field<arma::Mat<Tc> > m_fX; //!< X fundamental contractions [nz][nbsf,nbsf]
    arma::field<arma::Mat<Tc> > m_X;  //!< X fundamental contractions [nz][nbsf,nbsf]
    arma::field<arma::Mat<Tc> > m_Y;  //!< Y fundamental contractions [nz][nbsf,nbsf]
    arma::field<arma::Mat<Tc> > m_CX; //!< CX transformed coefficients [nz][nbsf,nbsf]
    arma::field<arma::Mat<Tc> > m_XC; //!< XC transformed coefficients [nz][nbsf,nbsf]

    /* 1RDM variables */
    arma::field<arma::Mat<Tc> > m_wxP; //! Reference density [nz][nbsf,nbsf]
    arma::field<arma::Mat<Tc> > m_R;  //! (xxY_ap,...,wxX_lp)
    arma::field<arma::Mat<Tc> > m_Q;  //! (wxX_qi,...,wwY_qd)

    /* Reference configurations for bra and ket states */
    reference_state<Tc> m_refx; //!< Reference configuration for bra state 
    reference_state<Tc> m_refw; //!< Reference configuration for ket state

public:
    /** \brief Constructor for pair of wick orbitals from two reference state objects
        \param refx Reference configuration for bra state
        \param refw Reference configuration for ket state
        \param metric Overlap matrix in AO basis
     **/
    wick_orbitals(
        reference_state<Tc> &refx, 
        reference_state<Tc> &refw,
        const arma::Mat<Tb> &metric) :
    m_nbsf(refx.m_nbsf), m_nmo(refx.m_nmo), m_nelec(refx.m_nelec),
    m_metric(metric), m_corner(0), m_refx(refx), m_refw(refw)
    {
        init();
    }

    /** \brief Constructor for the wick orbitals from MO coefficients
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nelec Number of electrons
        \param Cx Matrix of orbital coefficients for bra state
        \param Cw Matrix of orbital coefficients for ket state
        \param metric Overlap matrix of the basis functions
     **/
    wick_orbitals(
        const size_t nbsf, const size_t nmo, const size_t nelec,
        arma::Mat<Tc> Cx, arma::Mat<Tc> Cw, const arma::Mat<Tb> &metric
    ) : m_nbsf(nbsf), m_nmo(nmo), m_nelec(nelec),
        m_metric(metric), m_corner(0),
        m_refx(nbsf, nmo, nelec, Cx),
        m_refw(nbsf, nmo, nelec, Cw)
    {
        assert(Cx.n_rows == nbsf);
        assert(Cx.n_cols == nmo);
        assert(Cw.n_rows == nbsf);
        assert(Cw.n_cols == nmo);

        init();
    }

    /** \brief Constructor for one near-zero corner of another pair of wick orbitals
        \param orb Wick orbitals for the same pair of reference states
        \param corner Bit mask of near-zero overlaps entering with unit weight
     **/
    wick_orbitals(const wick_orbitals<Tc,Tb> &orb, size_t corner) :
        m_nbsf(orb.m_nbsf), m_nmo(orb.m_nmo), m_nelec(orb.m_nelec),
        m_metric(orb.m_metric), m_corner(corner),
        m_refx(orb.m_refx), m_refw(orb.m_refw)
    {
        init();
        m_ncorner = 1;
    }

    /** \brief Destructor **/
    virtual ~wick_orbitals() { }

private:
    /** \brief Initialise all variables **/
    virtual void init();
};

} // namespace libgnme

#endif // LIBGNME_WICK_ORBITALS_H
