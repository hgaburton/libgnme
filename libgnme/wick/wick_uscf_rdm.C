#include <cassert>
#include <algorithm>
#include <libgnme/utils/linalg.h>
#include "wick_uscf.h"

namespace libgnme {


template<typename Tc, typename Tf, typename Tb>
void wick_uscf<Tc,Tf,Tb>::diff_spin_rdm2(
        arma::umat xahp, arma::umat xbhp, 
        arma::umat wahp, arma::umat wbhp, 
        arma::uvec xocca, arma::uvec xoccb, 
        arma::uvec wocca, arma::uvec woccb, 
        arma::Mat<Tc> &P1a, arma::Mat<Tc> &P1b, 
        arma::Mat<Tc> &P2ab)
{
    // Check dimensions of RDM1
    assert(P1a.n_rows == m_nmo);
    assert(P1a.n_cols == m_nmo);
    assert(P1b.n_rows == m_nmo);
    assert(P1b.n_cols == m_nmo);

    // Temporary RDM-2
    assert(P2ab.n_rows == m_nmo * m_nmo);
    assert(P2ab.n_cols == m_nmo * m_nmo);
    P2ab.zeros();

    // Use 1-RDM to compute different spin 2-RDM
    for(size_t ip=0; ip<xocca.n_elem; ip++)
    for(size_t ir=0; ir<xoccb.n_elem; ir++)
    for(size_t iq=0; iq<wocca.n_elem; iq++)
    for(size_t is=0; is<woccb.n_elem; is++)
    {
        size_t p = xocca(ip), r = xoccb(ir), q = wocca(iq), s = woccb(is);
        P2ab(p*m_nmo+q, r*m_nmo+s) += P1a(q,p) * P1b(s,r);
    }
}

template class wick_uscf<double, double, double>;
template class wick_uscf<std::complex<double>, double, double>;
template class wick_uscf<std::complex<double>, std::complex<double>, double>;
template class wick_uscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
