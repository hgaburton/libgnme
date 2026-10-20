#ifndef LIBGNME_WICK_PAIR_H
#define LIBGNME_WICK_PAIR_H

#include <vector>
#include <cstdint>
#include <armadillo>
#include "reference_state.h"

namespace libgnme {

/** \brief Intermediates of the extended nonorthogonal Wick's theorem for one ordered pair of
           reference determinants, in the layout of wick_kernels.h

    The occupied orbitals of the bra and ket are Lowdin paired. The rows of the contraction
    labels are the active virtual orbitals of the bra followed by the active occupied orbitals
    of the ket, and the columns are the active occupied orbitals of the bra followed by the
    active virtual orbitals of the ket. Different numbers of active occupied or virtual orbitals
    in the bra and ket are padded with zeros to the larger number.

    Near-zero orbital overlaps 1e-8 < |S_k| < 1e-1 are evaluated from the corners 1/S_k -> 0 and
    1/S_k -> 1, with the combination c = ca * ncb + cb of the alpha corner ca and beta corner cb.
    Each combination has its own row of offsets and scalars.

    \tparam Tc Type defining orbital coefficients
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tb>
class wick_pair
{
private:
    /** \brief Lowdin-paired orbitals and fundamental contractions of one spin at one corner **/
    struct spin_pairing
    {
        Tc phase;  //!< Phase of the orbital pairing
        Tc redS;   //!< Reduced overlap, including the near-zero corner weights
        size_t m;  //!< Number of zero-overlap orbital pairs
        arma::Mat<Tc> M[2];  //!< Co-density matrices
        arma::Mat<Tc> X[2];  //!< X fundamental contractions
        arma::Mat<Tc> Y[2];  //!< Y fundamental contractions
        arma::Mat<Tc> CX[2]; //!< Row contraction coefficients
        arma::Mat<Tc> XC[2]; //!< Column contraction coefficients
    };

    size_t m_n;       //!< Number of contraction labels
    size_t m_nocc[2]; //!< Number of padded active occupied orbitals of each spin
    size_t m_nox[2];  //!< Number of active occupied orbitals of the bra of each spin
    size_t m_now[2];  //!< Number of active occupied orbitals of the ket of each spin
    double m_Vc;      //!< Constant component
    std::vector<spin_pairing> m_pair[2]; //!< Orbital pairing of each spin at each corner

    std::vector<int64_t> m_info; //!< Offsets of the intermediates of each corner combination
    std::vector<Tc> m_slab;      //!< Intermediates of all corner combinations
    std::vector<Tc> m_scal;      //!< Scalar intermediates of each corner combination

public:
    /** \brief Constructor from the reference states of each spin
        \param metric Overlap matrix of the basis functions
        \param xa High-spin reference state of the bra
        \param wa High-spin reference state of the ket
        \param xb Low-spin reference state of the bra
        \param wb Low-spin reference state of the ket
        \param Vc Constant contribution [default=0]
     **/
    wick_pair(
        const arma::Mat<Tb> &metric,
        const reference_state<Tc> &xa, const reference_state<Tc> &wa,
        const reference_state<Tc> &xb, const reference_state<Tc> &wb,
        double Vc=0);

    /** \brief Destructor **/
    virtual ~wick_pair() { }

    /** \name Routines to add one- or two-body operators to the object **/
    ///@{

    /** \brief Add a one-body operator with spin-unrestricted integrals
        \param Fa One-body integrals for high-spin component in AO basis
        \param Fb One-body integrals for low-spin component in AO basis
     **/
    void add_one_body(const arma::Mat<Tc> &Fa, const arma::Mat<Tc> &Fb);

    /** \brief Add a two-body operator with spin-restricted integrals
        \param V Two-body integrals in AO basis, (ij|kl) = V(i*nbsf+j,k*nbsf+l)
     **/
    void add_two_body(arma::Mat<Tb> &V);
    ///@}

    /** \brief Evaluate the overlap and matrix element using particle-hole representation. The
               holes and particles keep their order, so these are the matrix elements of the
               excitation operators in that order.
        \param xa_hp Particle-hole indices for bra alpha excitation
        \param xb_hp Particle-hole indices for bra beta  excitation
        \param wa_hp Particle-hole indices for ket alpha excitation
        \param wb_hp Particle-hole indices for ket beta  excitation
        \param S Output overlap matrix element
        \param V Output matrix element
     **/
    void evaluate(
        const arma::umat &xa_hp, const arma::umat &xb_hp,
        const arma::umat &wa_hp, const arma::umat &wb_hp,
        Tc &S, Tc &V) const;

    /** \brief Evaluate the overlap using particle-hole representation **/
    void evaluate_overlap(
        const arma::umat &xa_hp, const arma::umat &xb_hp,
        const arma::umat &wa_hp, const arma::umat &wb_hp,
        Tc &S) const;

    /** \brief Evaluate the overlap and one-body matrix element of one spin sector
        \param xhp Particle-hole indices for bra excitation
        \param whp Particle-hole indices for ket excitation
        \param S Output overlap of the spin sector
        \param V Output one-body matrix element of the spin sector
        \param alpha True for high-spin; False for low-spin
     **/
    void evaluate_one_body_spin(
        const arma::umat &xhp, const arma::umat &whp, Tc &S, Tc &V, bool alpha) const;

    /** \name Access to the intermediates in the layout of wick_kernels.h **/
    ///@{
    size_t ncorner() const { return m_pair[0].size() * m_pair[1].size(); }
    const int64_t *info(size_t c) const { return m_info.data() + c * m_info.size() / ncorner(); }
    const Tc *scal(size_t c) const { return m_scal.data() + c * m_scal.size() / ncorner(); }
    const std::vector<Tc> &slab() const { return m_slab; }
    ///@}

private:
    /** \brief Index of an active orbital of the bra or ket in the occupied-first order of the
               determinant records
        \param p Index of the orbital in the active space of the reference
        \param ket True for an orbital of the ket; False for the bra
        \param alpha True for high-spin; False for low-spin
     **/
    int label(size_t p, bool ket, bool alpha) const
    {
        size_t s = alpha ? 0 : 1, no = ket ? m_now[s] : m_nox[s];
        return (p < no) ? p : m_nocc[s] + p - no;
    }

    /** \brief Determinant record from particle-hole indices
        \param d Output record of WICK_DHEAD + 4 * kmax entries
        \param kmax Largest excitation rank of a record
        \param ahp High-spin particle-hole indices
        \param bhp Low-spin particle-hole indices
        \param ket True for an excitation of the ket; False for the bra
     **/
    void record(int *d, int kmax, const arma::umat &ahp, const arma::umat &bhp, bool ket) const;

    /** \brief Offsets of the intermediates of one corner combination **/
    size_t layout(int64_t *info, size_t c, size_t start) const;

    /** \brief Two-body intermediates of one corner combination **/
    void write_two_body(Tc *slab, Tc *scal, const int64_t *info, size_t c, arma::Mat<Tb> &V);

    /** \brief Precombined one-column intermediates without zero-overlap pairs **/
    void combine(const int64_t *info);
};

} // namespace libgnme

#endif // LIBGNME_WICK_PAIR_H
