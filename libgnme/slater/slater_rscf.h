#ifndef LIBGNME_SLATER_RSCF_H 
#define LIBGNME_SLATER_RSCF_H

#include <armadillo>
#include <cassert>

namespace libgnme {

/** \brief Implementation of the Generalised Slater-Condon Rules for RSCF orbitals
    \tparam Tc Type defining orbital coefficients
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_slater_uscf
 **/
template<typename Tc, typename Tf, typename Tb>
class slater_rscf
{
private:
    /* Useful constants */
    const size_t m_nbsf; //!< Number of basis functions
    const size_t m_nmo; //!< Number of (linearly independent) MOs
    const size_t m_nelec; //!< Number of alpha/beta electrons
    const arma::Mat<Tb> &m_metric; //!< Basis overlap metric

    // Integral constants in AO basis
    double m_Vc; //!< constant component
    arma::Mat<Tf> m_F; //!< Fock matrices
    arma::Mat<Tb> m_II; //!< Two-body integrals

    // Control variables for different components
    bool m_one_body = false;
    bool m_two_body = false;

public:
    /** \brief Constructor for the object
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nelec Number of electron pairs
        \param metric Overlap matrix of the basis functions
        \param Vc Constant term in the corresponding operator
     **/
    slater_rscf(
        const size_t nbsf, const size_t nmo, const size_t nelec,
        const arma::Mat<Tb> &metric, double Vc=0) :
        m_nbsf(nbsf), m_nmo(nmo), m_nelec(nelec), m_metric(metric), m_Vc(Vc)
    { }

    /** \brief Destructor **/
    virtual ~slater_rscf() { }

    /** \name Routines to add one- or two-body operators to the object **/
    ///@{
    
    /** \brief Add a one-body operator with spin-restricted integrals 
        \param F One-body integrals in AO basis
     **/
    virtual void add_one_body(arma::Mat<Tf> &F)
    {
        // Check input
        assert(F.n_rows == m_nbsf); 
        assert(F.n_cols == m_nbsf);

        // Save a copy of one-body matrices
        m_F = F;

        // Setup control variable
        m_one_body = true;
    }

    /** \brief Add a two-body operator with spin-restricted integrals
        \param V Two-body integrals in AO basis. These are represented as matrices in chemists
                  notation, e.g. (ij|kl) = V(i*nbsf+j,k*nbsf+l)
     **/
    virtual void add_two_body(arma::Mat<Tb> &II)
    {
        // Check input
        assert(II.n_rows == m_nbsf * m_nbsf);
        assert(II.n_cols == m_nbsf * m_nbsf);

        // Save two-body integrals
        m_II = II;

        // Setup control variable to indicate one-body initialised
        m_two_body = true;
    }
    ///@}

    virtual void evaluate_overlap(
        arma::Mat<Tc> Cx, arma::Mat<Tc> Cw, Tc &S);
    virtual void evaluate(
        arma::Mat<Tc> Cx, arma::Mat<Tc> Cw, Tc &Ov, Tc &H);
};

} // namespace libgnme

#endif // LIBGNME_SLATER_RSCF_H
