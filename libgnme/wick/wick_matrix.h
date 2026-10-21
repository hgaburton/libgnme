#ifndef LIBGNME_WICK_MATRIX_H
#define LIBGNME_WICK_MATRIX_H

#include <cassert>
#include <stdexcept>
#include <vector>
#include <cstdint>
#include <armadillo>
#include <libgnme/utils/bitset.h>

namespace libgnme {

/** \brief Hamiltonian and overlap matrices between excitations of many nonorthogonal reference
           determinants with the extended nonorthogonal Wick's theorem

    The intermediates of every ordered pair of references are built once, after which each
    matrix element depends only on the excitation ranks and the number of zero-overlap
    orbital pairs, not on the basis size. Matrix elements between excitations of the same
    reference use the Slater-Condon rules on occupation bit strings when the orbitals of the 
    reference are orthonormal, and the Wick's theorem otherwise.

    References are spin-unrestricted with the occupied orbitals first. A determinant is given
    by its reference and alpha and beta bitsets over the MOs of that reference, with the
    rightmost element representing the first MO, as for wick_uscf.

    \tparam Tc Type defining orbital coefficients
    \tparam Tf Type defining one-body matrix elements
    \tparam Tb Type defining basis functions
    \ingroup gnme_wick
 **/
template<typename Tc, typename Tf, typename Tb>
class wick_matrix
{
private:
    /* Useful constants */
    const size_t m_nbsf;   //!< Number of basis functions
    const size_t m_nmo;    //!< Number of (linearly independent) MOs
    const size_t m_nalpha; //!< Number of high-spin electrons
    const size_t m_nbeta;  //!< Number of low-spin electrons
    const arma::Mat<Tb> &m_metric; //!< Overlap matrix of the basis functions
    double m_Vc; //!< Constant component

    /* Reference determinants */
    std::vector<arma::Mat<Tc> > m_Ca; //!< High-spin orbital coefficients of each reference
    std::vector<arma::Mat<Tc> > m_Cb; //!< Low-spin orbital coefficients of each reference
    std::vector<bool> m_orthonormal;  //!< Whether the orbitals of each reference are orthonormal

    /* Operators */
    arma::Mat<Tc> m_Fa; //!< High-spin one-body integrals in AO basis
    arma::Mat<Tc> m_Fb; //!< Low-spin one-body integrals in AO basis
    arma::Mat<Tb> *m_V = nullptr; //!< Two-body integrals in AO basis

    /* Intermediates of every ordered pair of references */
    bool m_built = false;            //!< Control variable if the intermediates are built
    std::vector<int64_t> m_pairinfo; //!< Offsets of the intermediates of each pair
    std::vector<Tc> m_slab;          //!< Intermediates of all pairs
    std::vector<Tc> m_scal;          //!< Scalar intermediates of each pair

public:
    /** \brief Constructor for the object
        \param nbsf Number of basis functions
        \param nmo Number of linearly independent molecular orbitals
        \param nalpha Number of high-spin electrons
        \param nbeta Number of low-spin electrons
        \param metric Overlap matrix of the basis functions
        \param Vc Constant contribution [default=0]
     **/
    wick_matrix(
        const size_t nbsf, const size_t nmo, const size_t nalpha, const size_t nbeta,
        const arma::Mat<Tb> &metric, double Vc=0) :
        m_nbsf(nbsf), m_nmo(nmo), m_nalpha(nalpha), m_nbeta(nbeta), m_metric(metric), m_Vc(Vc)
    {
        if(nmo > 128)
            throw std::runtime_error("wick_matrix: At most 128 molecular orbitals are supported");
        assert(metric.n_rows == nbsf);
        assert(metric.n_cols == nbsf);
    }

    /** \brief Destructor **/
    virtual ~wick_matrix() { }

    /** \brief Add a reference determinant
        \param Ca High-spin orbital coefficients [nbsf,nmo], occupied orbitals first
        \param Cb Low-spin orbital coefficients [nbsf,nmo], occupied orbitals first
     **/
    void add_reference(const arma::Mat<Tc> &Ca, const arma::Mat<Tc> &Cb);

    /** \name Routines to add one- or two-body operators to the object **/
    ///@{

    /** \brief Add a one-body operator with spin-restricted integrals
        \param F One-body integrals in AO basis
     **/
    void add_one_body(arma::Mat<Tf> &F);

    /** \brief Add a one-body operator with spin-unrestricted integrals
        \param Fa One-body integrals for high-spin component in AO basis
        \param Fb One-body integrals for low-spin component in AO basis
     **/
    void add_one_body(arma::Mat<Tf> &Fa, arma::Mat<Tf> &Fb);

    /** \brief Add a two-body operator with spin-restricted integrals. The integrals must be
               kept until the intermediates are built.
        \param V Two-body integrals in AO basis. These are represented as matrices in chemists
                  notation, e.g. (ij|kl) = V(i*nbsf+j,k*nbsf+l)
     **/
    void add_two_body(arma::Mat<Tb> &V);
    ///@}

    /** \brief Construct the intermediates of every ordered pair of references. This is called
               by evaluate if needed.
     **/
    void build();

    /** \brief Evaluate the Hamiltonian and overlap matrices between determinants
        \param ref Reference of each determinant
        \param ba High-spin occupation bitset of each determinant
        \param bb Low-spin occupation bitset of each determinant
        \param S Output overlap matrix
        \param H Output Hamiltonian matrix
     **/
    void evaluate(
        const std::vector<size_t> &ref,
        std::vector<bitset> &ba, std::vector<bitset> &bb,
        arma::Mat<Tc> &S, arma::Mat<Tc> &H);
};

} // namespace libgnme

#endif // LIBGNME_WICK_MATRIX_H
