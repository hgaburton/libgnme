#ifndef LIBGNME_SLATER_USCF_H 
#define LIBGNME_SLATER_USCF_H

#include <armadillo>
#include <cassert>

namespace libgnme {

/** \brief Implementation of the Generalised Slater-Condon Rules for USCF orbitals
    \tparam Tc Type defining orbital coefficients
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_slater_uscf
 **/
template<typename Tc, typename Tf, typename Tb>
class slater_uscf
{
private:
    /* Useful constants */
    const size_t m_nbsf; //!< Number of basis functions
    const size_t m_nmo; //!< Number of (linearly independent) MOs
    const size_t m_nalpha; //!< Number of alpha electrons
    const size_t m_nbeta; //!< Number of beta electrons
    const arma::Mat<Tb> &m_metric; //!< Basis overlap metric

    // Integral constants in AO basis
    double m_Vc; //!< constant component
    arma::Mat<Tf> m_Fa; //!< Fock matrices
    arma::Mat<Tf> m_Fb; //!< Fock matrices
    arma::Mat<Tb> m_II; //!< Two-body integrals

    // Control variables for different components
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
    slater_uscf(
        const size_t nbsf, const size_t nmo, 
        const size_t nalpha, const size_t nbeta, 
        const arma::Mat<Tb> &metric, double Vc=0) :
        m_nbsf(nbsf), m_nmo(nmo), m_nalpha(nalpha), m_nbeta(nbeta), m_metric(metric), m_Vc(Vc)
    { }

    /** \brief Destructor **/
    virtual ~slater_uscf() { }

    /** \name Routines to add one- or two-body operators to the object **/
    ///@{
    
    /** \brief Add a one-body operator with spin-restricted integrals 
        \param F One-body integrals in AO basis
     **/
    virtual void add_one_body(arma::Mat<Tf> &F)
    {
        add_one_body(F,F);
    }

    /** \brief Add a one-body operator with spin-unrestricted integrals 
        \param Fa One-body integrals for high-spin component in AO basis
        \param Fb One-body integrals for low-spin component in AO basis
     **/
    virtual void add_one_body(arma::Mat<Tf> &Fa, arma::Mat<Tf> &Fb)
    {
        // Check input
        assert(Fa.n_rows == m_nbsf); 
        assert(Fa.n_cols == m_nbsf);
        assert(Fb.n_rows == m_nbsf); 
        assert(Fb.n_cols == m_nbsf);

        // Save a copy of one-body matrices
        m_Fa = Fa;
        m_Fb = Fb;

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
        arma::Mat<Tc> Cxa, arma::Mat<Tc> Cxb,
        arma::Mat<Tc> Cwa, arma::Mat<Tc> Cwb,
        Tc &S);
    virtual void evaluate(
        arma::Mat<Tc> Cxa, arma::Mat<Tc> Cxb,
        arma::Mat<Tc> Cwa, arma::Mat<Tc> Cwb,
        Tc &Ov, Tc &H);
};

} // namespace libgnme

#endif // LIBGNME_SLATER_USCF_H
