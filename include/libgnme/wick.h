#ifndef LIBGNME_WICK_H 
#define LIBGNME_WICK_H

#include <armadillo>

namespace libgnme {

/** \brief Implementation of the Extended Non-Orthogonal Wick's Theorem
    \tparam Tc Type defining orbital coefficients
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class wick
{
private:
    /* Useful constants */
    const size_t m_nbsf; //!< Number of basis functions
    const size_t m_nmo; //!< Number of (linearly independent) MOs
    const size_t m_nalpha; //!< Number of alpha electrons
    const size_t m_nbeta; //!< Number of beta electrons
    size_t m_nact; //!< Number of active orbitals
    const arma::Mat<Tb> &m_metric; //!< Basis overlap metric

    double m_Vc; //!< constant component
    arma::Mat<Tf> m_Fa; //!< Fock matrices
    arma::Mat<Tf> m_Fb; //!< Fock matrices
    Tb *m_II; //!< Pointer to two-body integral memory

    // One-body MO matrices
    bool m_one_body = false;
    bool m_two_body = false;

    /* Information about this pair */
public:
    size_t m_nza; //!< Number of alpha zero-overlap orbitals
private:
    size_t m_nzb; //!< Number of beta zero-overlap orbitals

    // Reference reduced overlaps
    Tc m_redSa; //!< Reduced overlap
    Tc m_redSb; //!< Reduced overlap

    // Store a set of reference orbitals
    arma::Mat<Tc> m_Cref; // Reference coefficients for integrals

    // Store the reference coefficients (nbsf * nmo)
    arma::Mat<Tc> m_Cxa; // Bra coefficients (alpha)
    arma::Mat<Tc> m_Cxb; // Bra coefficients (beta)
    arma::Mat<Tc> m_Cwa; // Ket coefficients (alpha)
    arma::Mat<Tc> m_Cwb; // Ket coefficients (beta)

    // Store the co-density matrices (2 * nbsf * nbsf)
    arma::field<arma::Mat<Tc> > m_wxMa;
    arma::field<arma::Mat<Tc> > m_wxMb;

    // Store the 'X' super matrices (2 * nmo * nmo)
    arma::field<arma::Mat<Tc> > m_Xa;
    arma::field<arma::Mat<Tc> > m_Xb;

    // Store the 'Y' super matrices (2 * nmo * nmo)
    arma::field<arma::Mat<Tc> > m_Ya;
    arma::field<arma::Mat<Tc> > m_Yb;

    // Store the 'F0' terms (2)
    arma::Col<Tc> m_F0a;
    arma::Col<Tc> m_F0b;

    // Store the '(X/Y)F(X/Y)' super matrices (4 * nmo * nmo)
    arma::field<arma::Mat<Tc> > m_XFXa;
    arma::field<arma::Mat<Tc> > m_XFXb;

    // Store the 'CX' and 'CY' matrices 
    arma::field<arma::Mat<Tc> > m_CXa;
    arma::field<arma::Mat<Tc> > m_CXb;
    arma::field<arma::Mat<Tc> > m_XCa;
    arma::field<arma::Mat<Tc> > m_XCb;

    // Store the 'V0' terms (3)
    arma::Col<Tc> m_Vaa;
    arma::Col<Tc> m_Vbb;
    arma::Mat<Tc> m_Vab;

    // Store the '[X/Y](J-K)[X/Y]' super matrices (8 * nmo^2)
    arma::field<arma::Mat<Tc> > m_XVaXa;
    arma::field<arma::Mat<Tc> > m_XVbXb;
    arma::field<arma::Mat<Tc> > m_XVaXb;
    arma::field<arma::Mat<Tc> > m_XVbXa;

    // Store two-electron repulsion integrals (16 * nmo^4)
    arma::field<arma::Mat<Tc> > m_IIaa;
    arma::field<arma::Mat<Tc> > m_IIbb;
    arma::field<arma::Mat<Tc> > m_IIab;
    arma::field<arma::Mat<Tc> > m_IIba;

    // Holds the `X' matrices
    // TODO: Remove
    arma::field<arma::Mat<Tc> > m_wwXa;
    arma::field<arma::Mat<Tc> > m_wxXa;
    arma::field<arma::Mat<Tc> > m_xwXa;
    arma::field<arma::Mat<Tc> > m_xxXa;
    arma::field<arma::Mat<Tc> > m_wwXb;
    arma::field<arma::Mat<Tc> > m_wxXb;
    arma::field<arma::Mat<Tc> > m_xwXb;
    arma::field<arma::Mat<Tc> > m_xxXb;

    // Hold the 'CX' and 'CY' matrices
    // TODO: Remove
    arma::field<arma::Mat<Tc> > m_xCXa;
    arma::field<arma::Mat<Tc> > m_xCXb;
    arma::field<arma::Mat<Tc> > m_wCXa;
    arma::field<arma::Mat<Tc> > m_wCXb;

    // Hold the 'XC' and 'YC' matrices
    // TODO: Remove
    arma::field<arma::Mat<Tc> > m_xXCa;
    arma::field<arma::Mat<Tc> > m_xXCb;
    arma::field<arma::Mat<Tc> > m_wXCa;
    arma::field<arma::Mat<Tc> > m_wXCb;

public:
    /** \brief Constructor for the object
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nalpha Number of high-spin electrons
        \param nbeta Number of low-spin electrons
        \param metric Overlap matrix of the basis functions
        \param Vc Constant term in the corresponding operator
     **/
    wick(
        const size_t nbsf, const size_t nmo, 
        const size_t nalpha, const size_t nbeta, 
        const arma::Mat<Tb> &metric, double Vc=0) :
        m_nbsf(nbsf), m_nmo(nmo), m_nalpha(nalpha), m_nbeta(nbeta), m_metric(metric), m_Vc(Vc)
    { }

    /** \brief Destructor **/
    virtual ~wick() { }

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

    virtual void evaluate_1rdm(
        arma::umat &xa_hp, arma::umat &xb_hp,
        arma::umat &wa_hp, arma::umat &wb_hp,
        Tc &S, arma::Mat<Tc> &P);


private:
    virtual void spin_1rdm(
        arma::umat &x_hp, arma::umat &w_hp, arma::Mat<Tc> &P, bool alpha);
    virtual void spin_overlap(
        arma::umat xhp, arma::umat whp,
        Tc &S, bool alpha);
    virtual void spin_one_body(
        arma::umat xhp, arma::umat whp,
        Tc &F, bool alpha);
    virtual void same_spin_two_body(
        arma::umat xhp, arma::umat whp,
        Tc &V, bool alpha);
    virtual void diff_spin_two_body(
        arma::umat &xa_hp, arma::umat &xb_hp, 
        arma::umat &wa_hp, arma::umat &wb_hp, 
        Tc &V);

    virtual void setup_one_body();
    virtual void setup_two_body();
};

} // namespace libgnme

#endif // LIBGNME_WICK_H
