#include <cassert>
#include <iomanip>
#include <libgnme/utils/lowdin_pair.h>
#include "wick_uscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::add_one_body(arma::Mat<Tf> &F) 
{
    add_one_body(F,F);
}


template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::add_one_body(arma::Mat<Tf> &Fa, arma::Mat<Tf> &Fb) 
{
    if(m_pair == nullptr) init_pair();
    m_pair->add_one_body(
        arma::conv_to<arma::Mat<Tc> >::from(Fa), arma::conv_to<arma::Mat<Tc> >::from(Fb));
}


template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::add_two_body(arma::Mat<Tb> &V)
{
    if(m_pair == nullptr) init_pair();
    m_pair->add_two_body(V);
}

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::init_pair()
{
    m_pair = new wick_pair<Tc,Tb>(m_orba.m_metric, m_orba.m_refx, m_orba.m_refw, 
                                  m_orbb.m_refx, m_orbb.m_refw, m_Vc);
}

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::init_corners()
{
    // Nothing to do without near-zero overlaps
    size_t na = m_orba.m_ncorner, nb = m_orbb.m_ncorner;
    if(na * nb == 1) return;

    // Corner orbitals for each spin. The first corner is evaluated by this object.
    std::vector<wick_orbitals<Tc,Tb> *> orba(na), orbb(nb);
    for(size_t ia=0; ia < na; ia++)
    {
        orba[ia] = new wick_orbitals<Tc,Tb>(m_orba, ia);
        m_corner_orbs.push_back(orba[ia]);
    }
    for(size_t ib=0; ib < nb; ib++)
    {
        orbb[ib] = new wick_orbitals<Tc,Tb>(m_orbb, ib);
        m_corner_orbs.push_back(orbb[ib]);
    }

    // Evaluators for the other combinations of alpha and beta corners
    for(size_t ia=0; ia < na; ia++)
    for(size_t ib=0; ib < nb; ib++)
    {
        if(ia == 0 and ib == 0) continue;
        m_corners.push_back(new wick_uscf<Tc,Tf,Tb>(*orba[ia], *orbb[ib], m_Vc));
        m_corner_a.push_back(ia);
        m_corner_b.push_back(ib);
    }
}


template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate(
    bitset &bxa, bitset &bxb, 
    bitset &bwa, bitset &bwb,
    Tc &S, Tc &V)
{
    if(m_pair == nullptr) init_pair();
    m_pair->evaluate(bxa, bxb, bwa, bwb, S, V);
}


template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate_dense(
    std::vector<bitset> &bxa, std::vector<bitset> &bxb, 
    std::vector<bitset> &bwa, std::vector<bitset> &bwb,
    arma::Mat<Tc> &S, arma::Mat<Tc> &V)
{
    if(m_pair == nullptr) init_pair();
    m_pair->evaluate_dense(bxa, bxb, bwa, bwb, S, V);
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

    // Add the other near-zero corners
    for(size_t k=0; k < m_corners.size(); k++)
    {
        Tc Sk = 0.0;
        arma::Mat<Tc> Pak(m_nmo, m_nmo), Pbk(m_nmo, m_nmo);
        m_corners[k]->evaluate_rdm1(bxa, bxb, bwa, bwb, Sk, Pak, Pbk);
        S += Sk; Pa += Pak; Pb += Pbk;
    }
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

    // Add the other near-zero corners
    for(size_t k=0; k < m_corners.size(); k++)
    {
        Tc Sk = 0.0;
        arma::Mat<Tc> P1ak(m_nmo, m_nmo), P1bk(m_nmo, m_nmo);
        arma::Mat<Tc> P2aak(m_nmo*m_nmo, m_nmo*m_nmo), P2bbk(m_nmo*m_nmo, m_nmo*m_nmo);
        arma::Mat<Tc> P2abk(m_nmo*m_nmo, m_nmo*m_nmo);
        m_corners[k]->evaluate_rdm12(bxa, bxb, bwa, bwb, Sk, P1ak, P1bk, P2aak, P2bbk, P2abk);
        S += Sk; P1a += P1ak; P1b += P1bk; P2aa += P2aak; P2bb += P2bbk; P2ab += P2abk;
    }
}




template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate_overlap(
    arma::umat &xahp, arma::umat &xbhp,
    arma::umat &wahp, arma::umat &wbhp,
    Tc &S)
{
    if(m_pair == nullptr) init_pair();
    m_pair->evaluate_overlap(xahp, xbhp, wahp, wbhp, S);
}

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate_one_body_spin(
    arma::umat &xhp, arma::umat &whp, 
    Tc &S, Tc &V, bool alpha)
{
    if(m_pair == nullptr) init_pair();
    m_pair->evaluate_one_body_spin(xhp, whp, S, V, alpha);
}

template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::evaluate(
    arma::umat &xahp, arma::umat &xbhp,
    arma::umat &wahp, arma::umat &wbhp,
    Tc &S, Tc &V)
{
    if(m_pair == nullptr) init_pair();
    m_pair->evaluate(xahp, xbhp, wahp, wbhp, S, V);
}

template class wick_uscf<double, double, double>;
template class wick_uscf<std::complex<double>, double, double>;
template class wick_uscf<std::complex<double>, std::complex<double>, double>;
template class wick_uscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
