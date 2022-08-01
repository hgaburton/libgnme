#ifndef LIBGNME_SLATER_CONDON_H 
#define LIBGNME_SLATER_CONDON_H

#include <armadillo>

namespace libgnme {

/** \brief Implementation of the Generalised Slater-Condon rules
    \tparam Tc Type defining orbital coefficients
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_slater_condon
 */
template<typename Tc, typename Tf, typename Tb>
class slater_condon
{
private:
    /* Useful constants */
    const size_t m_nbsf; //!< Number of basis functions
    const size_t m_nmo; //!< Number of (linearly independent) MOs
    const size_t m_nalpha; //!< Number of alpha electrons
    const size_t m_nbeta; //!< Number of beta electrons
    const arma::Mat<Tb> &m_metric; //!< Basis overlap metric

    double m_Vc = 0; //!< Constant component

    // One-body AO integrals
    arma::Mat<Tf> m_F; //!< Spin-restricted one-body integrals
    arma::Mat<Tf> m_Fa; //!< Spin-unrestricted one-body integrals (high-spin)
    arma::Mat<Tf> m_Fb; //!< Spin-unrestricted one-body integrals (low-spin)

    // Two-body AO integrals
    arma::Mat<Tb> m_II; //!< Two-body integrals

    // Control variables for different types of operator
    bool m_one_body = false;
    bool m_two_body = false;


public:
    /** \brief Constructor for the object
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nalpha Number of high-spin electrons
        \param nbeta Number of low-spin electrons
        \param metric Overlap matrix of the basis functions
     **/
    slater_condon(
        const size_t nbsf, const size_t nmo, 
        const size_t nalpha, const size_t nbeta, 
        const arma::Mat<Tb> &metric) :
        m_nbsf(nbsf), m_nmo(nmo), m_nalpha(nalpha), m_nbeta(nbeta), m_metric(metric)
    { }

    /** \brief Destructor **/
    virtual ~slater_condon() { }

    /** \name Routines to add constant, one-, or two-body operators to the object **/
    ///@{
    
    /** \brief Add a constant operator term 
        \param V Constant term in given operator
     **/ 
    virtual void add_constant(double V) 
    { 
        m_Vc = V; 
    };

    /** \brief Add a one-body operator with spin-restricted integrals 
        \param F One-body integrals in AO basis
     **/
    virtual void add_one_body(arma::Mat<Tf> &F) 
    { 
        m_F = F; 
    };

    /** \brief Add a one-body operator with spin-unrestricted integrals 
        \param Fa One-body integrals for high-spin component in AO basis
        \param Fb One-body integrals for low-spin component in AO basis
     **/
    virtual void add_one_body(arma::Mat<Tf> &Fa, arma::Mat<Tf> &Fb) 
    { 
        m_Fa = Fa; 
        m_Fb = Fb; 
    };

    /** \brief Add a two-body operator with spin-restricted integrals
        \param V Two-body integrals in AO basis. These are represented as matrices in chemists
                  notation, e.g. (ij|kl) = V(i*nbsf+j,k*nbsf+l)
     **/
    virtual void add_two_body(arma::Mat<Tb> &V)
    {
        m_II = V;
    };
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
};

} // namespace libgnme

#endif // LIBGNME_SLATER_CONDON_H
