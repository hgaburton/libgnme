#include <cassert>
#include <iomanip>
#include <libgnme/utils/lowdin_pair.h>
#include "wick_uscf.h"
#include "one_body_uscf.h"
#include "two_body_uscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::add_one_body(arma::Mat<Tf> &F) 
{
    add_one_body(F,F);
}


template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::add_one_body(arma::Mat<Tf> &Fa, arma::Mat<Tf> &Fb) 
{
    // Setup control variable to indicate one-body initialised
    m_one_body = true;
    // Define new integral object
    m_one_body_int = new one_body_uscf<Tc,Tf,Tb>(m_orba, m_orbb, Fa, Fb);
}


template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::add_two_body(arma::Mat<Tb> &V)
{
    // Setup control variable to indicate two-body initialised 
    m_two_body = true;
    // Define new integral object
    m_two_body_int = new two_body_uscf<Tc,Tf,Tb>(m_orba, m_orbb, V);
}


template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate(
    bitset &bxa, bitset &bxb, 
    bitset &bwa, bitset &bwb,
    Tc &S, Tc &V)
{
    // Get excitation indices
    arma::umat xahp, xbhp, wahp, wbhp; 
    int pxa, pxb, pwa, pwb;
    m_orba.m_refx.m_bs.excitation(bxa, xahp, pxa);
    m_orbb.m_refx.m_bs.excitation(bxb, xbhp, pxb);
    m_orba.m_refw.m_bs.excitation(bwa, wahp, pwa);
    m_orbb.m_refw.m_bs.excitation(bwb, wbhp, pwb);

    // Call original functionality
    evaluate(xahp, xbhp, wahp, wbhp, S, V);

    // Get parity 
    int parity = pxa * pxb * pwa * pwb;
               
    // Multiply matrix elements by parity
    S *= ((Tc) parity);
    V *= ((Tc) parity);
}


template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate_rdm1(
    bitset &bxa, bitset &bxb, 
    bitset &bwa, bitset &bwb,
    Tc &S, arma::Mat<Tc> &Pa, arma::Mat<Tc> &Pb)
{
    // Get excitation indices
    arma::umat xahp, xbhp, wahp, wbhp; 
    int pxa, pxb, pwa, pwb;
    m_orba.m_refx.m_bs.excitation(bxa, xahp, pxa);
    m_orbb.m_refx.m_bs.excitation(bxb, xbhp, pxb);
    m_orba.m_refw.m_bs.excitation(bwa, wahp, pwa);
    m_orbb.m_refw.m_bs.excitation(bwb, wbhp, pwb);

    // Get parity 
    int parity = pxa * pxb * pwa * pwb;

    // Get spin overlaps
    Tc sa = 0.0, sb = 0.0;
    this->spin_overlap(xahp, wahp, sa, true);
    this->spin_overlap(xbhp, wbhp, sb, false);
    S = ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS * sa * sb;

    // Get occupied orbitals to simplify density matrix computation
    arma::uvec occ_xa = arma::join_cols(m_orba.m_refx.m_core, bxa.occ()+m_orba.m_refx.m_ncore);
    arma::uvec occ_xb = arma::join_cols(m_orbb.m_refx.m_core, bxb.occ()+m_orbb.m_refx.m_ncore);
    arma::uvec occ_wa = arma::join_cols(m_orba.m_refw.m_core, bwa.occ()+m_orba.m_refw.m_ncore);
    arma::uvec occ_wb = arma::join_cols(m_orbb.m_refw.m_core, bwb.occ()+m_orbb.m_refw.m_ncore);

    // Treat each spin sector separately
    this->spin_rdm1(xahp, wahp, occ_xa, occ_wa, Pa, true);
    this->spin_rdm1(xbhp, wbhp, occ_xb, occ_wb, Pb, false);
               
    // Multiply matrix elements by parity
    Pa *= ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS * sb; 
    Pb *= ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS * sa; 
}


template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate_rdm12(
    bitset &bxa, bitset &bxb, 
    bitset &bwa, bitset &bwb,
    Tc &S, 
    arma::Mat<Tc> &P1a, arma::Mat<Tc> &P1b,
    arma::Mat<Tc> &P2aa, arma::Mat<Tc> &P2bb,
    arma::Mat<Tc> &P2ab) 
{
    // Get excitation indices
    arma::umat xahp, xbhp, wahp, wbhp; 
    int pxa, pxb, pwa, pwb;
    m_orba.m_refx.m_bs.excitation(bxa, xahp, pxa);
    m_orbb.m_refx.m_bs.excitation(bxb, xbhp, pxb);
    m_orba.m_refw.m_bs.excitation(bwa, wahp, pwa);
    m_orbb.m_refw.m_bs.excitation(bwb, wbhp, pwb);

    // Get parity 
    int parity = pxa * pxb * pwa * pwb;

    // Get spin overlaps
    Tc sa = 0.0, sb = 0.0;
    this->spin_overlap(xahp, wahp, sa, true);
    this->spin_overlap(xbhp, wbhp, sb, false);
    S = m_orba.m_redS * m_orbb.m_redS * sa * sb * ((Tc) parity);

    // Get occupied orbitals to simplify density matrix computation
    arma::uvec occ_xa = arma::join_cols(m_orba.m_refx.m_core, bxa.occ()+m_orba.m_refx.m_ncore);
    arma::uvec occ_xb = arma::join_cols(m_orbb.m_refx.m_core, bxb.occ()+m_orbb.m_refx.m_ncore);
    arma::uvec occ_wa = arma::join_cols(m_orba.m_refw.m_core, bwa.occ()+m_orba.m_refw.m_ncore);
    arma::uvec occ_wb = arma::join_cols(m_orbb.m_refw.m_core, bwb.occ()+m_orbb.m_refw.m_ncore);

    // Treat each spin sector separately
    this->spin_rdm1(xahp, wahp, occ_xa, occ_wa, P1a, true);
    this->spin_rdm1(xbhp, wbhp, occ_xb, occ_wb, P1b, false);
               
    // Treat each spin sector separately
    this->same_spin_rdm2(xahp, wahp, occ_xa, occ_wa, P2aa, true);
    this->same_spin_rdm2(xbhp, wbhp, occ_xb, occ_wb, P2bb, false);
    this->diff_spin_rdm2(xahp, xbhp, wahp, wbhp, occ_xa, occ_xb, occ_wa, occ_wb, P1a, P1b, P2ab);

    // Multiply 1RDM matrix elements by parity and reduced overlap
    P1a *= ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS * sb;
    P1b *= ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS * sa;

    // Multiply 2RDM matrix elements by parity and reduced overlap
    P2aa *= ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS * sb;
    P2bb *= ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS * sa;
    P2ab *= ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS;
}




template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate_overlap(
    arma::umat &xahp, arma::umat &xbhp,
    arma::umat &wahp, arma::umat &wbhp,
    Tc &S)
{
    // Evaluate overlap terms
    Tc sa = 0.0, sb = 0.0;
    this->spin_overlap(xahp, wahp, sa, true);
    this->spin_overlap(xbhp, wbhp, sb, false);
    // Save total overlap
    S = m_orba.m_redS * m_orbb.m_redS * sa * sb;
}

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate_one_body_spin(
    arma::umat &xhp, arma::umat &whp, 
    Tc &S, Tc &V, bool alpha)
{
    // Collect reduced overlap
    Tc redS = alpha ? m_orba.m_redS : m_orbb.m_redS;

    // Evaluate overlap terms
    Tc sspin = 0.0;
    this->spin_overlap(xhp, whp, sspin, alpha);

    // Save total spin-overlap
    S = redS * sspin;

    // Evaluate one-body terms
    Tc Vspin = 0.0;
    // Evaluate separate spin one-body terms
    this->spin_one_body(xhp, whp, Vspin, alpha);
    // Recombine and increment output
    V = redS * Vspin;
}

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate(
    arma::umat &xahp, arma::umat &xbhp,
    arma::umat &wahp, arma::umat &wbhp,
    Tc &S, Tc &V)
{
    // Evaluate overlap terms
    Tc sa = 0.0, sb = 0.0;
    this->spin_overlap(xahp, wahp, sa, true);
    this->spin_overlap(xbhp, wbhp, sb, false);
    // Save total overlap
    S = m_orba.m_redS * m_orbb.m_redS * sa * sb;

    // Save any constant term
    V = S * m_Vc;

    // Evaluate one-body term if present
    if(m_one_body)
    {
        // Temporary variables
        Tc Va = 0.0, Vb = 0.0;
        // Evaluate separate spin one-body terms
        this->spin_one_body(xahp, wahp, Va, true);
        this->spin_one_body(xbhp, wbhp, Vb, false);
        // Recombine and increment output
        V += m_orba.m_redS * m_orbb.m_redS * (Va * sb + Vb * sa);
    }

    // Evaluate two-body term if present
    if(m_two_body)
    {
        // Temporary variables
        Tc Vaa = 0.0, Vbb = 0.0, Vab = 0.0;
        // Same spin terms
        this->same_spin_two_body(xahp, wahp, Vaa, true);
        this->same_spin_two_body(xbhp, wbhp, Vbb, false);
        // Different spin terms
        this->diff_spin_two_body(xahp, xbhp, wahp, wbhp, Vab);
        // Recombine
        V += 0.5 * m_orba.m_redS * m_orbb.m_redS * (Vaa * sb + Vbb * sa + 2.0 * Vab);
    }
}

} // namespace libgnme
