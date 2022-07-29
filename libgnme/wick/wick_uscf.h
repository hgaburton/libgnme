#ifndef LIBGNME_WICK_USCF_H 
#define LIBGNME_WICK_USCF_H

#include <armadillo>
#include "wick_orbitals.h"

namespace libgnme {

/** \brief Implementation of the Extended Non-Orthogonal Wick's Theorem 
           for spin unrestricted orbitals
    \tparam Tc Type defining orbital coefficients
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class wick_uscf
{
private:
    /* Useful constants */
    const size_t m_nbsf; //!< Number of basis functions
    const size_t m_nmo; //!< Number of (linearly independent) MOs
    const size_t m_nalpha; //!< Number of alpha electrons
    const size_t m_nbeta; //!< Number of beta electrons
    const size_t m_nact; //!< Number of active orbitals
    const arma::Mat<Tb> &m_metric; //!< Basis overlap metric

    double m_Vc; //!< constant component

    wick_orbitals<Tc,Tb> m_orb_a; //!< Alpha orbital pair
    wick_orbitals<Tc,Tb> m_orb_b; //!< Beta orbital pair

    // One-body MO matrices
    bool m_one_body = false;
    bool m_two_body = false;

    /* Information about this pair */
public:
    size_t m_nza; //!< Number of alpha zero-overlap orbitals
    size_t m_nzb; //!< Number of beta zero-overlap orbitals

private:
    // Store the 'F0' terms (2)
    arma::Col<Tc> m_F0a;
    arma::Col<Tc> m_F0b;

    // Store the '(X/Y)F(X/Y)' super matrices (4 * nmo * nmo)
    arma::field<arma::Mat<Tc> > m_XFXa;
    arma::field<arma::Mat<Tc> > m_XFXb;

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

public:
    /** \brief Constructor for the object
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nalpha Number of high-spin electrons
        \param nbeta Number of low-spin electrons
        \param metric Overlap matrix of the basis functions
        \param Vc Constant term in the corresponding operator
     **/
    wick_uscf(
        wick_orbitals<Tc,Tb> &orba, wick_orbitals<Tc,Tb> &orbb,
        const arma::Mat<Tb> &metric, double Vc=0) :

        m_nbsf(orba.m_nbsf), m_nmo(orba.m_nmo), 
        m_nalpha(orba.m_nelec), m_nbeta(orbb.m_nelec),
        m_nact(orba.m_nact),
        m_metric(metric), m_Vc(Vc),
        m_orb_a(orba), m_orb_b(orbb)
    { 
        assert(orba.m_nbsf == orbb.m_nbsf);
        assert(orba.m_nmo  == orbb.m_nmo);
        assert(orba.m_nact == orbb.m_nact);
    }
        
    /** \brief Destructor **/
    virtual ~wick_uscf() { }

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

//    virtual void evaluate_1rdm(
//        arma::umat &xa_hp, arma::umat &xb_hp,
//        arma::umat &wa_hp, arma::umat &wb_hp,
//        Tc &S, arma::Mat<Tc> &P);


private:
//    virtual void spin_1rdm(
//        arma::umat &x_hp, arma::umat &w_hp, arma::Mat<Tc> &P, bool alpha);
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
        arma::umat xa_hp, arma::umat xb_hp, 
        arma::umat wa_hp, arma::umat wb_hp, 
        Tc &V);
};

} // namespace libgnme

#endif // LIBGNME_WICK_USCF_H
