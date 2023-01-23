#ifndef LIBGNME_WICK_ORBITALS_H 
#define LIBGNME_WICK_ORBITALS_H

#include <armadillo>

namespace libgnme {

/** \brief Compute and store orbital pairs for Non-Orthogonal Wicks Theorem
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
    size_t m_nact;        //!< Number of active orbitals
    size_t m_ncore;       //!< Number of core orbitals

    /* Information about this pair */
    size_t m_nz; //!< Number of zero-overlap orbitals

    // Reference reduced overlaps
    Tc m_redS; //!< Reduced overlap

    arma::field<arma::Mat<Tc> > m_M; //! Co-density matrices (2 * nbsf * nbsf)
    arma::field<arma::Mat<Tc> > m_X; //! X fundamental contractions (2 * nbsf * nbsf)
    arma::field<arma::Mat<Tc> > m_Y; //! Y fundamental contractions (2 * nbsf * nbsf)
    arma::field<arma::Mat<Tc> > m_CX; //! CX transformed coefficients (2 * nbsf * nmo)
    arma::field<arma::Mat<Tc> > m_XC; //! XC transformed coefficients (2 * nbsf * nmo)

    // 1RDM variables
    arma::field<arma::Mat<Tc> > m_wxP; //! Reference density
    arma::field<arma::Mat<Tc> > m_R;  //! (xxY_ap,...,wxX_lp)
    arma::field<arma::Mat<Tc> > m_Q;  //! (wxX_qi,...,wwY_qd)

private:
    // Store the reference coefficients (nbsf * nmo)
    arma::Mat<Tc> m_Cx; // Bra coefficients
    arma::Mat<Tc> m_Cw; // Ket coefficients

public:
    /** \brief Constructor for the wick_orbitals object
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nelec Number of electrons
        \param Cx Matrix of orbital coefficients for bra state
        \param Cw Matrix of orbital coefficients for ket state
        \param metric Overlap matrix of the basis functions
     **/
    wick_orbitals(
        const size_t nbsf, const size_t nmo, const size_t nelec,
        arma::Mat<Tc> Cx, arma::Mat<Tc> Cw, 
        const arma::Mat<Tb> &metric) :
        m_nbsf(nbsf), m_nmo(nmo), m_nelec(nelec),
        m_metric(metric), m_nact(nmo), m_ncore(0) 
    { 
        init(Cx, Cw);
    }

    /** \brief Constructor for wick_orbitals defined with an active space
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nelec Number of electrons
        \param Cx Matrix of orbital coefficients for bra state
        \param Cw Matrix of orbital coefficients for ket state
        \param metric Overlap matrix of the basis functions
        \param nact Number of active orbitals
        \param ncore Number of core orbitals
     **/
    wick_orbitals(
        const size_t nbsf, const size_t nmo, const size_t nelec,
        arma::Mat<Tc> Cx, arma::Mat<Tc> Cw, 
        const arma::Mat<Tb> &metric,
        const size_t nact, const size_t ncore) :
        m_nbsf(nbsf), m_nmo(nmo), m_nelec(nelec),
        m_metric(metric), m_nact(nact), m_ncore(ncore)
    { 
        assert(ncore + nact <= nmo);
        init(Cx, Cw);
    }

    /** \brief Destructor **/
    virtual ~wick_orbitals() { }

    /** \brief Initialise all variables **/
    virtual void init(arma::Mat<Tc> Cx, arma::Mat<Tc> Cw);
};

} // namespace libgnme

#endif // LIBGNME_WICK_ORBITALS_H
