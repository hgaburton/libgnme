#ifndef LIBGNME_WICK_RSCF_H 
#define LIBGNME_WICK_RSCF_H

#include <armadillo>

namespace libgnme {

/** \brief Implementation of the Extended Non-Orthogonal Wick's Theorem 
           for restricted orbitals
    \tparam Tc Type defining orbital coefficients
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class wick_rscf
{
private:
    /* Useful constants */
    const size_t m_nbsf; //!< Number of basis functions
    const size_t m_nmo; //!< Number of (linearly independent) MOs
    const size_t m_nelec; //!< Number of electrons
    size_t m_nact; //!< Number of active orbitals
    const arma::Mat<Tb> &m_metric; //!< Basis overlap metric

    double m_Vc; //!< constant component
    arma::Mat<Tf> m_F; //!< Fock matrices
    Tb *m_IIao; //!< Pointer to two-body integral memory

    // One-body MO matrices
    bool m_one_body = false;
    bool m_two_body = false;

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

    // Store the 'F0' terms (2)
    arma::Col<Tc> m_F0;

    // Store the '(X/Y)F(X/Y)' super matrices (4 * nmo * nmo)
    arma::field<arma::Mat<Tc> > m_XFX;

    // Store the 'CX' and 'CY' matrices 
    arma::field<arma::Mat<Tc> > m_CX;
    arma::field<arma::Mat<Tc> > m_XC;

    // Store the 'V0' terms (3)
    arma::Col<Tc> m_Vsame;
    arma::Col<Tc> m_Vdiff;

    // Store the '[X/Y](J-K)[X/Y]' super matrices (8 * nmo^2)
    arma::field<arma::Mat<Tc> > m_XJX;
    arma::field<arma::Mat<Tc> > m_XKX;

    // Store two-electron repulsion integrals (16 * nmo^4)
    arma::field<arma::Mat<Tc> > m_II;

public:
    /** \brief Constructor for the object
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nalpha Number of high-spin electrons
        \param nbeta Number of low-spin electrons
        \param metric Overlap matrix of the basis functions
        \param Vc Constant term in the corresponding operator
     **/
    wick_rscf(
        const size_t nbsf, const size_t nmo, 
        const size_t nelec,
        const arma::Mat<Tb> &metric, double Vc=0) :
        m_nbsf(nbsf), m_nmo(nmo), m_nelec(nelec), m_metric(metric), m_Vc(Vc)
    { }

    /** \brief Destructor **/
    virtual ~wick_rscf() { }

    /** \brief Setup orbitals associated with the reference determinants
        \param Cx Molecular orbital coefficients for the bra state
        \param Cw Molecular orbital coefficients for the ket state
     **/
    virtual void setup_orbitals(arma::Mat<Tc> Cx, arma::Mat<Tc> Cw);
    virtual void setup_orbitals(arma::Mat<Tc> Cx, arma::Mat<Tc> Cw, size_t ncore, size_t nactive);

    /** \name Routines to add one- or two-body operators to the object **/
    ///@{
    
    /** \brief Add a one-body operator with spin-restricted integrals 
        \param F One-body integrals in AO basis
     **/
    virtual void add_one_body(arma::Mat<Tf> &F);

    /** \brief Add a two-body operator with spin-restricted integrals
        \param V Two-body integrals in AO basis. These are represented as matrices in chemists
                  notation, e.g. (ij|kl) = V(i*nbsf+j,k*nbsf+l)
     **/
    virtual void add_two_body(arma::Mat<Tb> &V);
    ///@}

    virtual void evaluate_overlap(
        arma::umat &xa_hp, arma::umat &xb_hp,
        arma::umat &wa_hp, arma::umat &wb_hp,
        Tc &S);
    virtual void evaluate_one_body_spin(
        arma::umat &xhp, arma::umat &whp,
        Tc &S, Tc &V);
    virtual void evaluate(
        arma::umat &xa_hp, arma::umat &xb_hp,
        arma::umat &wa_hp, arma::umat &wb_hp,
        Tc &S, Tc &M);

    virtual void evaluate_1rdm(
        arma::umat &xa_hp, arma::umat &xb_hp,
        arma::umat &wa_hp, arma::umat &wb_hp,
        Tc &S, arma::Mat<Tc> &P) {};


private:
    virtual void spin_overlap(
        arma::umat xhp, arma::umat whp, Tc &S);
    virtual void spin_one_body(
        arma::umat xhp, arma::umat whp, Tc &F);
    virtual void same_spin_two_body(
        arma::umat xhp, arma::umat whp, Tc &V);
    virtual void diff_spin_two_body(
        arma::umat xa_hp, arma::umat xb_hp, 
        arma::umat wa_hp, arma::umat wb_hp, 
        Tc &V);

    virtual void setup_one_body();
    virtual void setup_two_body();
};

} // namespace libgnme

#endif // LIBGNME_WICK_RSCF_H
