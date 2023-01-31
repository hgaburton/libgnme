#include <cassert>
#include <iomanip>
#include <libgnme/utils/lowdin_pair.h>
#include "wick_rscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::evaluate(
    bitset &bxa, bitset &bxb, 
    bitset &bwa, bitset &bwb,
    Tc &S, Tc &V)
{
    // Get excitation indices
    arma::umat xahp, xbhp, wahp, wbhp; 
    int pxa, pxb, pwa, pwb;
    m_bref.excitation(bxa, xahp, pxa);
    m_bref.excitation(bxb, xbhp, pxb);
    m_bref.excitation(bwa, wahp, pwa);
    m_bref.excitation(bwb, wbhp, pwb);

    // Call original functionality
    evaluate(xahp, xbhp, wahp, wbhp, S, V);

    // Get parity 
    int parity = pxa * pxb * pwa * pwb;
               
    // Multiply matrix elements by parity
    S *= ((Tc) parity);
    V *= ((Tc) parity);
}


template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::evaluate_rdm1(
    bitset &bxa, bitset &bxb, 
    bitset &bwa, bitset &bwb,
    Tc &S,
    arma::Mat<Tc> &Pa, arma::Mat<Tc> &Pb)
{
    // Get excitation indices
    arma::umat xahp, xbhp, wahp, wbhp; 
    int pxa, pxb, pwa, pwb;
    m_bref.excitation(bxa, xahp, pxa);
    m_bref.excitation(bxb, xbhp, pxb);
    m_bref.excitation(bwa, wahp, pwa);
    m_bref.excitation(bwb, wbhp, pwb);

    // Get parity 
    int parity = pxa * pxb * pwa * pwb;

    // Get spin overlaps
    Tc sa = 0.0, sb = 0.0;
    spin_overlap(xahp, wahp, sa);
    spin_overlap(xbhp, wbhp, sb);
    S = m_orb.m_redS * m_orb.m_redS * sa * sb * ((Tc) parity);

    // Get occupied orbitals to simplify density matrix computation
    arma::uvec occ_xa = arma::join_cols(m_core,bxa.occ()+m_orb.m_ncore);
    arma::uvec occ_xb = arma::join_cols(m_core,bxb.occ()+m_orb.m_ncore);
    arma::uvec occ_wa = arma::join_cols(m_core,bwa.occ()+m_orb.m_ncore);
    arma::uvec occ_wb = arma::join_cols(m_core,bwb.occ()+m_orb.m_ncore);

    // Treat each spin sector separately
    spin_rdm1(xahp, wahp, occ_xa, occ_wa, Pa);
    spin_rdm1(xbhp, wbhp, occ_xb, occ_wb, Pb);
               
    // Multiply matrix elements by parity
    Pa *= ((Tc) parity) * m_orb.m_redS * m_orb.m_redS * sb; 
    Pb *= ((Tc) parity) * m_orb.m_redS * m_orb.m_redS * sa; 
}


template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::evaluate_overlap(
    arma::umat &xahp, arma::umat &xbhp,
    arma::umat &wahp, arma::umat &wbhp,
    Tc &S)
{
    // Evaluate overlap terms
    Tc sa = 0.0, sb = 0.0;
    spin_overlap(xahp, wahp, sa);
    spin_overlap(xbhp, wbhp, sb);
    // Save total overlap
    S = m_orb.m_redS * m_orb.m_redS * sa * sb;
}

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::evaluate_one_body_spin(
    arma::umat &xhp, arma::umat &whp, 
    Tc &S, Tc &V)
{
    // Evaluate overlap terms
    Tc sspin = 0.0;
    spin_overlap(xhp, whp, sspin);

    // Save total spin-overlap
    S = m_orb.m_redS * sspin;

    // Evaluate one-body terms
    Tc Vspin = 0.0;
    // Evaluate separate spin one-body terms
    spin_one_body(xhp, whp, Vspin);
    // Recombine and increment output
    V = m_orb.m_redS * Vspin;
}

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::evaluate(
    arma::umat &xahp, arma::umat &xbhp,
    arma::umat &wahp, arma::umat &wbhp,
    Tc &S, Tc &V)
{
    // Evaluate overlap terms
    Tc sa = 0.0, sb = 0.0;
    spin_overlap(xahp, wahp, sa);
    spin_overlap(xbhp, wbhp, sb);
    // Save total overlap
    S = m_orb.m_redS * m_orb.m_redS * sa * sb;

    // Save any constant term
    V = S * m_Vc;

    // Evaluate one-body term if present
    if(m_one_body)
    {
        // Temporary variables
        Tc Va = 0.0, Vb = 0.0;
        // Evaluate separate spin one-body terms
        spin_one_body(xahp, wahp, Va);
        spin_one_body(xbhp, wbhp, Vb);
        // Recombine and increment output
        V += m_orb.m_redS * m_orb.m_redS * (Va * sb + Vb * sa);
    }

    // Evaluate two-body term if present
    if(m_two_body)
    {
        // Temporary variables
        Tc Vaa = 0.0, Vbb = 0.0, Vab = 0.0;
        // Same spin terms
        same_spin_two_body(xahp, wahp, Vaa);
        same_spin_two_body(xbhp, wbhp, Vbb);
        // Different spin terms
        diff_spin_two_body(xahp, xbhp, wahp, wbhp, Vab);
        // Recombine
        V += 0.5 * m_orb.m_redS * m_orb.m_redS * (Vaa * sb + Vbb * sa + 2.0 * Vab);
    }
}


template class wick_rscf<double, double, double>;
template class wick_rscf<std::complex<double>, double, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
