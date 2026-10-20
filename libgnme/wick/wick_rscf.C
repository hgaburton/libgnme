#include <cassert>
#include <iomanip>
#include <libgnme/utils/lowdin_pair.h>
#include "wick_rscf.h"
#include "one_body_rscf.h"
#include "two_body_rscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::add_one_body(arma::Mat<Tf> &F) 
{
    // Setup control variable to indicate one-body initialised
    m_one_body = true;
    // Define new integral object
    m_one_body_int = new one_body_rscf<Tc,Tf,Tb>(m_orba, F);

    // Add to the other near-zero corners
    for(size_t k=0; k < m_corners.size(); k++)
        m_corners[k]->add_one_body(F);
}

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::add_two_body(arma::Mat<Tb> &V)
{
    // Setup control variable to indicate two-body initialised
    m_two_body = true;
    // Define new integral object
    m_two_body_int = new two_body_rscf<Tc,Tf,Tb>(m_orba, V);

    // Add to the other near-zero corners
    for(size_t k=0; k < m_corners.size(); k++)
        m_corners[k]->add_two_body(V);
}

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::init_corners()
{
    // Nothing to do without near-zero overlaps
    size_t nc = m_orba.m_ncorner;
    if(nc == 1) return;

    // Corner orbitals shared by both spins. The first corner is evaluated by this object.
    std::vector<wick_orbitals<Tc,Tb> *> orb(nc);
    for(size_t i=0; i < nc; i++)
    {
        orb[i] = new wick_orbitals<Tc,Tb>(m_orba, i);
        m_corner_orbs.push_back(orb[i]);
    }

    // Evaluators for the other combinations of alpha and beta corners
    for(size_t ia=0; ia < nc; ia++)
    for(size_t ib=0; ib < nc; ib++)
    {
        if(ia == 0 and ib == 0) continue;
        m_corners.push_back(new wick_uscf<Tc,Tf,Tb>(*orb[ia], *orb[ib], m_Vc));
        m_corner_b.push_back(ib);
    }
}

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::evaluate(
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
void wick_rscf<Tc,Tf,Tb>::evaluate_rdm1(
    bitset &bxa, bitset &bxb, 
    bitset &bwa, bitset &bwb,
    Tc &S, arma::Mat<Tc> &P1)
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
    arma::Mat<Tc> Pa(m_nmo, m_nmo, arma::fill::zeros);
    arma::Mat<Tc> Pb(m_nmo, m_nmo, arma::fill::zeros);
    this->spin_rdm1(xahp, wahp, occ_xa, occ_wa, Pa, true);
    this->spin_rdm1(xbhp, wbhp, occ_xb, occ_wb, Pb, false);
               
    // Multiply matrix elements by parity
    P1 = ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS * (sb * Pa + sa * Pb);

    // Add the other near-zero corners
    for(size_t k=0; k < m_corners.size(); k++)
    {
        Tc Sk = 0.0;
        arma::Mat<Tc> Pak(m_nmo, m_nmo), Pbk(m_nmo, m_nmo);
        m_corners[k]->evaluate_rdm1(bxa, bxb, bwa, bwb, Sk, Pak, Pbk);
        S += Sk; P1 += Pak + Pbk;
    }
}


template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::evaluate_rdm12(
    bitset &bxa, bitset &bxb, 
    bitset &bwa, bitset &bwb,
    Tc &S, 
    arma::Mat<Tc> &P1, arma::Mat<Tc> &P2) 
{
    // Get excitation indices and parity
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
    arma::Mat<Tc> Pa(m_nmo, m_nmo, arma::fill::zeros);
    arma::Mat<Tc> Pb(m_nmo, m_nmo, arma::fill::zeros);
    this->spin_rdm1(xahp, wahp, occ_xa, occ_wa, Pa, true);
    this->spin_rdm1(xbhp, wbhp, occ_xb, occ_wb, Pb, false);
               
    // Multiply 1RDM matrix elements by parity
    P1 = ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS * (sb * Pa + sa * Pb);

    // Temporary variables for 2RDM
    arma::Mat<Tc> tmpP(m_nmo*m_nmo,m_nmo*m_nmo);
    // Treat each spin sector separately
    this->same_spin_rdm2(xahp, wahp, occ_xa, occ_wa, tmpP, true);
    P2 += ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS * sb * tmpP;
    this->same_spin_rdm2(xbhp, wbhp, occ_xb, occ_wb, tmpP, false);
    P2 += ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS * sa * tmpP;
    this->diff_spin_rdm2(xahp, xbhp, wahp, wbhp, occ_xa, occ_xb, occ_wa, occ_wb, Pa, Pb, tmpP);
    P2 += ((Tc) parity) * m_orba.m_redS * m_orbb.m_redS * tmpP;

    // Add the other near-zero corners
    for(size_t k=0; k < m_corners.size(); k++)
    {
        Tc Sk = 0.0;
        arma::Mat<Tc> P1ak(m_nmo, m_nmo), P1bk(m_nmo, m_nmo);
        arma::Mat<Tc> P2aak(m_nmo*m_nmo, m_nmo*m_nmo), P2bbk(m_nmo*m_nmo, m_nmo*m_nmo);
        arma::Mat<Tc> P2abk(m_nmo*m_nmo, m_nmo*m_nmo);
        m_corners[k]->evaluate_rdm12(bxa, bxb, bwa, bwb, Sk, P1ak, P1bk, P2aak, P2bbk, P2abk);
        S += Sk; P1 += P1ak + P1bk; P2 += P2aak + P2bbk + P2abk;
    }
}


template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::evaluate_overlap(
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

    // Add the other near-zero corners
    for(size_t k=0; k < m_corners.size(); k++)
    {
        Tc Sk = 0.0;
        m_corners[k]->evaluate_overlap(xahp, xbhp, wahp, wbhp, Sk);
        S += Sk;
    }
}

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::evaluate_one_body_spin(
    arma::umat &xhp, arma::umat &whp, 
    Tc &S, Tc &V)
{
    // Evaluate overlap terms
    Tc sspin = 0.0;
    this->spin_overlap(xhp, whp, sspin, true);

    // Save total spin-overlap
    S = m_orba.m_redS * sspin;

    // Evaluate one-body terms
    Tc Vspin = 0.0;
    // Evaluate separate spin one-body terms
    this->spin_one_body(xhp, whp, Vspin, true);
    // Recombine and increment output
    V = m_orba.m_redS * Vspin;

    // Add the other near-zero corners of the alpha spin
    for(size_t k=0; k < m_corners.size(); k++)
    {
        if(m_corner_b[k] != 0) continue;
        Tc Sk = 0.0, Vk = 0.0;
        m_corners[k]->evaluate_one_body_spin(xhp, whp, Sk, Vk, true);
        S += Sk; V += Vk;
    }
}

template<typename Tc, typename Tf, typename Tb>
void wick_rscf<Tc,Tf,Tb>::evaluate(
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

    // Add the other near-zero corners
    for(size_t k=0; k < m_corners.size(); k++)
    {
        Tc Sk = 0.0, Vk = 0.0;
        m_corners[k]->evaluate(xahp, xbhp, wahp, wbhp, Sk, Vk);
        S += Sk; V += Vk;
    }
}

template class wick_rscf<double, double, double>;
template class wick_rscf<std::complex<double>, double, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, double>;
template class wick_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
