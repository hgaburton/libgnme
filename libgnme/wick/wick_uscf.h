#ifndef LIBGNME_WICK_USCF_H 
#define LIBGNME_WICK_USCF_H

#include <armadillo>
#include <libgnme/utils/bitset.h>
#include "wick.h"
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
class wick_uscf : public wick<Tc,Tf,Tb>
{
protected:
    using wick<Tc,Tf,Tb>::m_nbsf; 
    using wick<Tc,Tf,Tb>::m_nmo; 
    using wick<Tc,Tf,Tb>::m_nact; 

private:
    /* Useful constants */
    const size_t m_nalpha; //!< Number of alpha electrons
    const size_t m_nbeta; //!< Number of beta electrons
    const arma::Mat<Tb> &m_metric; //!< Basis overlap metric

    double m_Vc; //!< constant component

    wick_orbitals<Tc,Tb> m_orb_a; //!< Alpha orbital pair
    wick_orbitals<Tc,Tb> m_orb_b; //!< Beta orbital pair

    // Reference bitsets
    bitset m_bref_a; //!< Reference alpha bitset
    bitset m_bref_b; //!< Reference beta bitset

    // One-body MO matrices
    bool m_one_body = false;
    bool m_two_body = false;

    // Occupied core
    arma::uvec m_corea;
    arma::uvec m_coreb;

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
        wick<Tc,Tf,Tb>(orba.m_nbsf, orba.m_nmo, orba.m_nact),
        m_nalpha(orba.m_nelec), m_nbeta(orbb.m_nelec),
        m_metric(metric), m_Vc(Vc),
        m_orb_a(orba), m_orb_b(orbb)
    { 
        assert(orba.m_nbsf == orbb.m_nbsf);
        assert(orba.m_nmo  == orbb.m_nmo);
        assert(orba.m_nact == orbb.m_nact);

        // Set the reference bit strings
        size_t act_el_a = orba.m_nelec - orba.m_ncore; // Active alpha electrons
        size_t act_el_b = orbb.m_nelec - orbb.m_ncore; // Active beta  electrons 
        std::vector<bool> refa(orba.m_nact-act_el_a, 0); refa.resize(orba.m_nact, 1);
        std::vector<bool> refb(orbb.m_nact-act_el_b, 0); refb.resize(orbb.m_nact, 1);
        m_bref_a = bitset(refa);
        m_bref_b = bitset(refb);  
        m_corea.resize(orba.m_ncore);
        m_coreb.resize(orbb.m_ncore);
        for(size_t i=0; i<orba.m_ncore; i++) m_corea(i) = i;
        for(size_t i=0; i<orbb.m_ncore; i++) m_coreb(i) = i;
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
    
    virtual void evaluate(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S, Tc &V);

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

    virtual void evaluate_rdm1(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S,
        arma::Mat<Tc> &Pa, arma::Mat<Tc> &Pb);

    virtual void evaluate_rdm12(
        bitset &bxa, bitset &bxb, 
        bitset &bwa, bitset &bwb,
        Tc &S, 
        arma::Mat<Tc> &P1a, arma::Mat<Tc> &P1b,
        arma::Mat<Tc> &P2aa, arma::Mat<Tc> &P2bb,
        arma::Mat<Tc> &P2ab);

private:
    //virtual void spin_rdm1(
    //    arma::umat xhp, arma::umat whp, 
    //    arma::uvec xocc, arma::uvec wocc, 
    //    arma::Mat<Tc> &P, bool alpha);
    //virtual void same_spin_rdm2(
    //    arma::umat xhp, arma::umat whp, 
    //    arma::uvec xocc, arma::uvec wocc, 
    //    arma::Mat<Tc> &P, bool alpha);
    //virtual void  diff_spin_rdm2(
    //    arma::umat xahp, arma::umat xbhp, 
    //    arma::umat wahp, arma::umat wbhp, 
    //    arma::uvec xocca, arma::uvec xoccb, 
    //    arma::uvec wocca, arma::uvec woccb, 
    //    arma::Mat<Tc> &P1a, arma::Mat<Tc> &P1b, 
    //    arma::Mat<Tc> &P2ab);

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

    /* Getters */
    const size_t& get_nz(bool alpha) { return alpha ? m_orb_a.m_nz : m_orb_b.m_nz; }
    const size_t& get_ne(bool alpha) { return alpha ? m_orb_a.m_nelec : m_orb_b.m_nelec; }
    const arma::field<arma::Mat<Tc> >& get_fX(bool alpha)  { return alpha ? m_orb_a.m_fX  : m_orb_b.m_fX;  }
    const arma::field<arma::Mat<Tc> >& get_X(bool alpha)   { return alpha ? m_orb_a.m_X   : m_orb_b.m_X;   }
    const arma::field<arma::Mat<Tc> >& get_Y(bool alpha)   { return alpha ? m_orb_a.m_Y   : m_orb_b.m_Y;   }
    const arma::field<arma::Mat<Tc> >& get_Q(bool alpha)   { return alpha ? m_orb_a.m_Q   : m_orb_b.m_Q;   }
    const arma::field<arma::Mat<Tc> >& get_R(bool alpha)   { return alpha ? m_orb_a.m_R   : m_orb_b.m_R;   }
    const arma::field<arma::Mat<Tc> >& get_wxP(bool alpha) { return alpha ? m_orb_a.m_wxP : m_orb_b.m_wxP; }
    
};

} // namespace libgnme

#endif // LIBGNME_WICK_USCF_H
