#include <cassert>
#include <iomanip>
#include <libgnme/utils/lowdin_pair.h>
#include "wick_uscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate_overlap(
    arma::umat &xahp, arma::umat &xbhp,
    arma::umat &wahp, arma::umat &wbhp,
    Tc &S)
{
    // Evaluate overlap terms
    Tc sa = 0.0, sb = 0.0;
    spin_overlap(xahp, wahp, sa, true);
    spin_overlap(xbhp, wbhp, sb, false);
    // Save total overlap
    S = m_orb_a.m_redS * m_orb_b.m_redS * sa * sb;
}

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate_one_body_spin(
    arma::umat &xhp, arma::umat &whp, 
    Tc &S, Tc &V, bool alpha)
{
    // Collect reduced overlap
    Tc redS = alpha ? m_orb_a.m_redS : m_orb_b.m_redS;

    // Evaluate overlap terms
    Tc sspin = 0.0;
    spin_overlap(xhp, whp, sspin, alpha);

    // Save total spin-overlap
    S = redS * sspin;

    // Evaluate one-body terms
    Tc Vspin = 0.0;
    // Evaluate separate spin one-body terms
    spin_one_body(xhp, whp, Vspin, alpha);
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
    spin_overlap(xahp, wahp, sa, true);
    spin_overlap(xbhp, wbhp, sb, false);
    // Save total overlap
    S = m_orb_a.m_redS * m_orb_b.m_redS * sa * sb;

    // Save any constant term
    V = S * m_Vc;

    // Evaluate one-body term if present
    if(m_one_body)
    {
        // Temporary variables
        Tc Va = 0.0, Vb = 0.0;
        // Evaluate separate spin one-body terms
        spin_one_body(xahp, wahp, Va, true);
        spin_one_body(xbhp, wbhp, Vb, false);
        // Recombine and increment output
        V += m_orb_a.m_redS * m_orb_b.m_redS * (Va * sb + Vb * sa);
    }

    // Evaluate two-body term if present
    if(m_two_body)
    {
        // Temporary variables
        Tc Vaa = 0.0, Vbb = 0.0, Vab = 0.0;
        // Same spin terms
        same_spin_two_body(xahp, wahp, Vaa, true);
        same_spin_two_body(xbhp, wbhp, Vbb, false);
        // Different spin terms
        diff_spin_two_body(xahp, xbhp, wahp, wbhp, Vab);
        // Recombine
        V += 0.5 * m_orb_a.m_redS * m_orb_b.m_redS * (Vaa * sb + Vbb * sa + 2.0 * Vab);
    }
}

//template<typename Tc, typename Tf, typename Tb>
//void wick_uscf<Tc,Tf,Tb>::evaluate_1rdm(
//    arma::umat &xahp, arma::umat &xbhp,
//    arma::umat &wahp, arma::umat &wbhp,
//    Tc &S, arma::Mat<Tc> &P)
//{
//    // Evaluate overlap terms
//    Tc sa = 0.0, sb = 0.0;
//    spin_overlap(xahp, wahp, sa, true);
//    spin_overlap(xbhp, wbhp, sb, false);
//    // Save total overlap
//    S = m_orb_a.m_redS * m_orb_b.m_redS * sa * sb;
//
//    // Evaluate spin RDMs
//    arma::Mat<Tc> Pa, Pb;
//    spin_1rdm(xahp, wahp, Pa, true);
//    spin_1rdm(xbhp, wbhp, Pb, false);
//
//    // Combine to get full 1RDM
//    P = m_orb_a.m_redS * m_orb_b.m_redS * (Pa * sb + sa * Pb);
//}



template class wick_uscf<double, double, double>;
template class wick_uscf<std::complex<double>, double, double>;
template class wick_uscf<std::complex<double>, std::complex<double>, double>;
template class wick_uscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
