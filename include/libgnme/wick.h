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
    const arma::Mat<Tb> &m_metric; //!< Basis overlap metric

    double m_Vc; //!< constant component
    arma::Mat<Tf> m_Fa; //!< Fock matrices
    arma::Mat<Tf> m_Fb; //!< Fock matrices
    arma::Mat<Tb> m_II; //!< Two-body integrals

    // One-body MO matrices
    bool m_one_body = false;
    bool m_two_body = false;

    /* Information about this pair */
    size_t m_nza; //!< Number of alpha zero-overlap orbitals
    size_t m_nzb; //!< Number of beta zero-overlap orbitals

    Tc m_redSa; //!< Reduced overlap
    Tc m_redSb; //!< Reduced overlap

    // Hold the reference coefficients
    arma::Mat<Tc> m_Cxa; // Bra coefficients (alpha)
    arma::Mat<Tc> m_Cxb; // Bra coefficients (beta)
    arma::Mat<Tc> m_Cwa; // Ket coefficients (alpha)
    arma::Mat<Tc> m_Cwb; // Ket coefficients (beta)

    // Hold the co-density matrices
    arma::field<arma::Mat<Tc> > m_wxMa;
    arma::field<arma::Mat<Tc> > m_wxMb;

    // Holds the `X' matrices
    arma::field<arma::Mat<Tc> > m_wwXa;
    arma::field<arma::Mat<Tc> > m_wxXa;
    arma::field<arma::Mat<Tc> > m_xwXa;
    arma::field<arma::Mat<Tc> > m_xxXa;
    arma::field<arma::Mat<Tc> > m_wwXb;
    arma::field<arma::Mat<Tc> > m_wxXb;
    arma::field<arma::Mat<Tc> > m_xwXb;
    arma::field<arma::Mat<Tc> > m_xxXb;

    // Hold the 'X' super matrices
    arma::field<arma::Mat<Tc> > m_Xa;
    arma::field<arma::Mat<Tc> > m_Xb;

    // Hold the 'Y' super matrices
    arma::field<arma::Mat<Tc> > m_Ya;
    arma::field<arma::Mat<Tc> > m_Yb;

    // Holds the 'F0' terms
    arma::Col<Tc> m_F0a;
    arma::Col<Tc> m_F0b;

    // Hold the '(X/Y)F(X/Y)' super matrices
    arma::field<arma::Mat<Tc> > m_XFXa;
    arma::field<arma::Mat<Tc> > m_XFXb;

    // Holds the `XFX' matrices
    arma::field<arma::Mat<Tc> > m_wwXFXa;
    arma::field<arma::Mat<Tc> > m_wxXFXa;
    arma::field<arma::Mat<Tc> > m_xwXFXa;
    arma::field<arma::Mat<Tc> > m_xxXFXa;
    arma::field<arma::Mat<Tc> > m_wwXFXb;
    arma::field<arma::Mat<Tc> > m_wxXFXb;
    arma::field<arma::Mat<Tc> > m_xwXFXb;
    arma::field<arma::Mat<Tc> > m_xxXFXb;

    // Holds the 'V0' terms
    arma::Col<Tc> m_Vaa;
    arma::Col<Tc> m_Vbb;
    arma::Mat<Tc> m_Vab;

    // Hold the 'XVX' matrices
    arma::field<arma::Mat<Tc> > m_wwXVaXa;
    arma::field<arma::Mat<Tc> > m_wxXVaXa;
    arma::field<arma::Mat<Tc> > m_xwXVaXa;
    arma::field<arma::Mat<Tc> > m_xxXVaXa;

    arma::field<arma::Mat<Tc> > m_wwXVaXb;
    arma::field<arma::Mat<Tc> > m_wxXVaXb;
    arma::field<arma::Mat<Tc> > m_xwXVaXb;
    arma::field<arma::Mat<Tc> > m_xxXVaXb;

    arma::field<arma::Mat<Tc> > m_wwXVbXa;
    arma::field<arma::Mat<Tc> > m_wxXVbXa;
    arma::field<arma::Mat<Tc> > m_xwXVbXa;
    arma::field<arma::Mat<Tc> > m_xxXVbXa;

    arma::field<arma::Mat<Tc> > m_wwXVbXb;
    arma::field<arma::Mat<Tc> > m_wxXVbXb;
    arma::field<arma::Mat<Tc> > m_xwXVbXb;
    arma::field<arma::Mat<Tc> > m_xxXVbXb;
    
    // Hold the 'CX' and 'CY' matrices
    arma::field<arma::Mat<Tc> > m_xCXa;
    arma::field<arma::Mat<Tc> > m_xCXb;
    arma::field<arma::Mat<Tc> > m_wCXa;
    arma::field<arma::Mat<Tc> > m_wCXb;

    // Hold the 'XC' and 'YC' matrices
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
        arma::umat &x_hp, arma::umat &w_hp,
        Tc &S, bool alpha);
    virtual void spin_one_body(
        arma::umat &x_hp, arma::umat &w_hp,
        Tc &F, bool alpha);
    virtual void same_spin_two_body(
        arma::umat &xhp, arma::umat &whp,
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
