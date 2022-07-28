#include <cassert>
#include <iomanip>
#include "wick_rscf.h"
#include "lowdin_pair.h"

namespace libgnme {

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
    S = m_redS * m_redS * sa * sb;
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
    S = m_redS * sspin;

    // Evaluate one-body terms
    Tc Vspin = 0.0;
    // Evaluate separate spin one-body terms
    spin_one_body(xhp, whp, Vspin);
    // Recombine and increment output
    V = m_redS * Vspin;
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
    S = m_redS * m_redS * sa * sb;

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
        V += m_redS * m_redS * (Va * sb + Vb * sa);
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
        V += 0.5 * m_redS * m_redS * (Vaa * sb + Vbb * sa + 2.0 * Vab);
    }
}


template class wick_rscf<double, double, double>;
template class wick_rscf<std::complex<double>, double, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
