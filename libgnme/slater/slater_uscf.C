#include <cassert>
#include <libgnme/utils/lowdin_pair.h>
#include "slater_uscf.h"

namespace libgnme {

template<typename Tc, typename Tf, typename Tb>
void slater_uscf<Tc,Tf,Tb>::evaluate_overlap(
    arma::Mat<Tc> Cxa, arma::Mat<Tc> Cxb,
    arma::Mat<Tc> Cwa, arma::Mat<Tc> Cwb,
    Tc &Ov)
{
    assert(Cxa.n_rows == m_nbsf); assert(Cxa.n_cols == m_nalpha);
    assert(Cxb.n_rows == m_nbsf); assert(Cxb.n_cols == m_nbeta);
    assert(Cwa.n_rows == m_nbsf); assert(Cwa.n_cols == m_nalpha);
    assert(Cxb.n_rows == m_nbsf); assert(Cxb.n_cols == m_nbeta);

    // Compute overlap 
    Ov = arma::det(Cxa.t() * m_metric * Cwa) * 
         arma::det(Cxb.t() * m_metric * Cwb);
}

template<typename Tc, typename Tf, typename Tb>
void slater_uscf<Tc,Tf,Tb>::evaluate(
    arma::Mat<Tc> Cxa, arma::Mat<Tc> Cxb,
    arma::Mat<Tc> Cwa, arma::Mat<Tc> Cwb,
    Tc &Ov, Tc &H)
{
    assert(Cxa.n_rows == m_nbsf); assert(Cxa.n_cols == m_nalpha);
    assert(Cxb.n_rows == m_nbsf); assert(Cxb.n_cols == m_nbeta);
    assert(Cwa.n_rows == m_nbsf); assert(Cwa.n_cols == m_nalpha);
    assert(Cxb.n_rows == m_nbsf); assert(Cxb.n_cols == m_nbeta);

    // Zero the output
    H = 0.0; Ov = 0.0;

    // Lowdin Pair
    arma::Col<Tc> Sxx_a(m_nalpha); Sxx_a.zeros();
    arma::Col<Tc> Sxx_b(m_nbeta); Sxx_b.zeros();
    arma::Col<Tc> inv_Sxx_a(m_nalpha, arma::fill::zeros);
    arma::Col<Tc> inv_Sxx_b(m_nbeta, arma::fill::zeros);
    size_t nZeros_a = 0, nZeros_b = 0;
    arma::uvec zeros_a(Sxx_a.n_elem), zeros_b(Sxx_b.n_elem);
    libgnme::lowdin_pair(Cxa, Cwa, Sxx_a, m_metric);
    libgnme::lowdin_pair(Cxb, Cwb, Sxx_b, m_metric);

    // Compute reduced overlap
    Tc redOv_a = 1.0, redOv_b = 1.0;
    libgnme::reduced_overlap(Sxx_a, inv_Sxx_a, redOv_a, nZeros_a, zeros_a);
    libgnme::reduced_overlap(Sxx_b, inv_Sxx_b, redOv_b, nZeros_b, zeros_b);

    // Zeroth order terms
    Ov = ((nZeros_a + nZeros_b) == 0) ? redOv_a * redOv_b : 0.0;
    H  = ((nZeros_a + nZeros_b) == 0) ? m_Vc : 0.0;

    // Return early if no one- or two-body terms
    if(!m_one_body and !m_two_body) 
        return;
    
    // Compute the required elements
    if((nZeros_a + nZeros_b) == 0)
    {   
        // Save non-zero overlap
        Ov = redOv_a * redOv_b;

        // Construct co-density matrices
        arma::Mat<Tc> xwWa = Cwa * arma::diagmat(inv_Sxx_a) * Cxa.t();
        arma::Mat<Tc> xwWb = Cwb * arma::diagmat(inv_Sxx_b) * Cxb.t();

        // Add one-body element
        if(m_one_body)
            H += arma::dot(m_Fa, xwWa.st()) + arma::dot(m_Fb, xwWb.st());

        // Add two-body element
        if(m_two_body) 
        {
            for(size_t p=0; p < m_nbsf; p++)
            for(size_t q=0; q < m_nbsf; q++)
            for(size_t r=0; r < m_nbsf; r++)
            for(size_t s=0; s < m_nbsf; s++)
            {
                H += 0.5 * m_II(p*m_nbsf+q, r*m_nbsf+s) * xwWa(q,p) * xwWa(s,r)
                   - 0.5 * m_II(p*m_nbsf+s, r*m_nbsf+q) * xwWa(q,p) * xwWa(s,r);
                H += 0.5 * m_II(p*m_nbsf+q, r*m_nbsf+s) * xwWb(q,p) * xwWb(s,r)
                   - 0.5 * m_II(p*m_nbsf+s, r*m_nbsf+q) * xwWb(q,p) * xwWb(s,r);
                H += 1.0 * m_II(p*m_nbsf+q, r*m_nbsf+s) * xwWa(q,p) * xwWb(s,r);
            }
        }
    }
    else if((nZeros_a + nZeros_b) == 1)
    {   
        // Construct co-density matrices
        arma::Mat<Tc> xwP(m_nbsf,m_nbsf,arma::fill::zeros);
        arma::Mat<Tc> xwWs(m_nbsf,m_nbsf,arma::fill::zeros); // Same spin as P
        arma::Mat<Tc> xwWd(m_nbsf,m_nbsf,arma::fill::zeros); // Diff spin to P
        if(nZeros_a == 1)
        {
            xwP  = Cwa.col(zeros_a(0)) * Cxa.col(zeros_a(0)).t();
            xwWs = Cwa * arma::diagmat(inv_Sxx_a) * Cxa.t();
            xwWd = Cwb * arma::diagmat(inv_Sxx_b) * Cxb.t();
        }
        else
        {
            xwP  = Cwb.col(zeros_b(0)) * Cxb.col(zeros_b(0)).t();
            xwWs = Cwb * arma::diagmat(inv_Sxx_b) * Cxb.t();
            xwWd = Cwa * arma::diagmat(inv_Sxx_a) * Cxa.t();
        }

        // Add one-body element
        if(m_one_body)
        {
            arma::Mat<Tf> &F = (nZeros_a == 1) ? m_Fa : m_Fb;
            H += arma::dot(F, xwP.st());
        }

        // Add two-body element
        if(m_two_body) 
        {
            for(size_t p=0; p < m_nbsf; p++)
            for(size_t q=0; q < m_nbsf; q++)
            for(size_t r=0; r < m_nbsf; r++)
            for(size_t s=0; s < m_nbsf; s++)
            {
                H += m_II(p*m_nbsf+q, r*m_nbsf+s) * xwP(q,p) * xwWs(s,r)
                   - m_II(p*m_nbsf+s, r*m_nbsf+q) * xwP(q,p) * xwWs(s,r);
                H += m_II(p*m_nbsf+q, r*m_nbsf+s) * xwP(q,p) * xwWd(s,r);
            }
        }
    }
    // Only consider these elements if we have two-body term
    else if((nZeros_a + nZeros_b) == 2 && m_two_body)
    {   
        // Construct co-density matrices
        arma::Mat<Tc>  xwP1(m_nbsf,m_nbsf,arma::fill::zeros);
        arma::Mat<Tc> xwP2J(m_nbsf,m_nbsf,arma::fill::zeros);
        arma::Mat<Tc> xwP2K(m_nbsf,m_nbsf,arma::fill::zeros);
        if(nZeros_a == 2)
        {
            xwP1  = Cwa.col(zeros_a(0)) * Cxa.col(zeros_a(0)).t();
            xwP2J = Cwa.col(zeros_a(1)) * Cxa.col(zeros_a(1)).t();
            xwP2K = Cwa.col(zeros_a(1)) * Cxa.col(zeros_a(1)).t();
        }
        else if(nZeros_b == 2)
        {
            xwP1  = Cwb.col(zeros_b(0)) * Cxb.col(zeros_b(0)).t();
            xwP2J = Cwb.col(zeros_b(1)) * Cxb.col(zeros_b(1)).t();
            xwP2K = Cwb.col(zeros_b(1)) * Cxb.col(zeros_b(1)).t();
        }
        else 
        {
            xwP1  = Cwa.col(zeros_a(0)) * Cxa.col(zeros_a(0)).t();
            xwP2J = Cwb.col(zeros_b(0)) * Cxb.col(zeros_b(0)).t();
        }

        // Add two-body element
        if(m_two_body)
        {
            for(size_t p=0; p < m_nbsf; p++)
            for(size_t q=0; q < m_nbsf; q++)
            for(size_t r=0; r < m_nbsf; r++)
            for(size_t s=0; s < m_nbsf; s++)
            {
                H += m_II(p*m_nbsf+q, r*m_nbsf+s) * xwP1(q,p) * xwP2J(s,r)
                   - m_II(p*m_nbsf+s, r*m_nbsf+q) * xwP1(q,p) * xwP2K(s,r);
            }
        }
    }

    // Account for reduced overlap 
    H *= redOv_a * redOv_b;
}

template class slater_uscf<double, double, double>;
template class slater_uscf<std::complex<double>, double, double>;
template class slater_uscf<std::complex<double>, std::complex<double>, double>;
template class slater_uscf<std::complex<double>, std::complex<double>, std::complex<double> >;

} // namespace libgnme
