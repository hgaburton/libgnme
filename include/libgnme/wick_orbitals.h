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
private:
    /* Useful constants */
    const size_t m_nbsf;  //!< Number of basis functions
    const size_t m_nmo;   //!< Number of (linearly independent) MOs
    const size_t m_nelec; //!< Number of electrons
    const arma::Mat<Tb> &m_metric; //!< Basis overlap metric

    size_t m_nact;        //!< Number of active orbitals
    size_t m_ncore;       //!< Number of core orbitals

    /* Information about this pair */
public:
    size_t m_nz; //!< Number of zero-overlap orbitals

private:
    // Reference reduced overlaps
    Tc m_redS; //!< Reduced overlap

    // Store the reference coefficients (nbsf * nmo)
    arma::Mat<Tc> m_Cx; // Bra coefficients
    arma::Mat<Tc> m_Cw; // Ket coefficients

    // Store the co-density matrices (2 * nbsf * nbsf)
    arma::field<arma::Mat<Tc> > m_wxM;

    // Store the 'X' super matrices (2 * nmo * nmo)
    arma::field<arma::Mat<Tc> > m_X;

    // Store the 'Y' super matrices (2 * nmo * nmo)
    arma::field<arma::Mat<Tc> > m_Y;

    // Store the 'CX' and 'CY' matrices 
    arma::field<arma::Mat<Tc> > m_CX;
    arma::field<arma::Mat<Tc> > m_XC;

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
        arma::Mat<Tc> &Cx, arma::Mat<Tc> &Cw, 
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
        arma::Mat<Tc> &Cx, arma::Mat<Tc> &Cw, 
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
    virtual void init(arma::Mat<Tc> &Cx, arma::Mat<Tc> &Cw);

    //virtual void evaluate_overlap(
    //    arma::umat &xa_hp, arma::umat &xb_hp,
    //    arma::umat &wa_hp, arma::umat &wb_hp,
    //    Tc &S);
};

} // namespace libgnme

#endif // LIBGNME_WICK_ORBITALS_H
