#include <cassert>
#include <libgnme/utils/lowdin_pair.h>
#include "slater_rscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void slater_rscf<Tc,Tf,Tb>::evaluate_overlap(
    arma::Mat<Tc> Cx, arma::Mat<Tc> Cw, Tc &Ov)
{
    // Check the input
    assert(Cx.n_rows == m_nbsf); assert(Cx.n_cols == m_nelec);
    assert(Cw.n_rows == m_nbsf); assert(Cw.n_cols == m_nelec);

    // Compute overlap 
    Tc spinOv = arma::det(Cx.t() * m_metric * Cw);
    Ov = spinOv * spinOv;
}

template<typename Tc, typename Tf, typename Tb>
void slater_rscf<Tc,Tf,Tb>::evaluate(
    arma::Mat<Tc> Cx, arma::Mat<Tc> Cw,
    Tc &Ov, Tc &H)
{
    // Check the input
    assert(Cx.n_rows == m_nbsf); assert(Cx.n_cols == m_nelec);
    assert(Cw.n_rows == m_nbsf); assert(Cw.n_cols == m_nelec);

    // Zero the output
    H = 0.0; Ov = 0.0;

    // Lowdin Pair
    arma::Col<Tc> Sxx(m_nelec); Sxx.zeros();
    arma::Col<Tc> inv_Sxx(m_nelec, arma::fill::zeros);
    size_t nZeros = 0;
    arma::uvec zeros(Sxx.n_elem);
    libgnme::lowdin_pair(Cx, Cw, Sxx, m_metric);

    // Compute reduced overlap
    Tc redOv = 1.0;
    libgnme::reduced_overlap(Sxx, inv_Sxx, redOv, nZeros, zeros);

    // Zeroth order terms
    Ov = (nZeros == 0) ? redOv * redOv : 0.0;
    H  = (nZeros == 0) ? m_Vc : 0.0;

    // Return early if no one- or two-body terms
    if(!m_one_body and !m_two_body) 
        return;
    
    // Compute the required elements
    if(nZeros == 0)
    {   
        // Save non-zero overlap
        Ov = redOv * redOv;

        // Construct co-density matrices
        arma::Mat<Tc> xwW = Cw * arma::diagmat(inv_Sxx) * Cx.t();

        // Add one-body element
        if(m_one_body)
            H += 2.0 * arma::dot(m_F, xwW.st());

        // Add two-body element
        if(m_two_body) 
        {
            for(size_t p=0; p < m_nbsf; p++)
            for(size_t q=0; q < m_nbsf; q++)
            for(size_t r=0; r < m_nbsf; r++)
            for(size_t s=0; s < m_nbsf; s++)
            {
                H += 2.0 * m_II(p*m_nbsf+q, r*m_nbsf+s) * xwW(q,p) * xwW(s,r)
                   - 1.0 * m_II(p*m_nbsf+s, r*m_nbsf+q) * xwW(q,p) * xwW(s,r);
            }
        }
    }
    // Only consider these elements if we have two-body term
    // NOTE this corresponds to a zero in alpha AND beta channel, 
    // so a total of two zero-overlap orbitals
    else if(nZeros == 1 && m_two_body)
    {   
        // Construct co-density matrices
        arma::Mat<Tc> xwP = Cw.col(zeros(0)) * Cx.col(zeros(0)).t();

        // Add two-body element
        for(size_t p=0; p < m_nbsf; p++)
        for(size_t q=0; q < m_nbsf; q++)
        for(size_t r=0; r < m_nbsf; r++)
        for(size_t s=0; s < m_nbsf; s++)
        {
            H += m_II(p*m_nbsf+q, r*m_nbsf+s) * xwP(q,p) * xwP(s,r);
        }
    }

    // Account for reduced overlap 
    H *= redOv * redOv;
}

template class slater_rscf<double, double, double>;
template class slater_rscf<std::complex<double>, double, double>;
template class slater_rscf<std::complex<double>, std::complex<double>, double>;
template class slater_rscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
