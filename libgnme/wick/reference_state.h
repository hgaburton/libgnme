#ifndef LIBGNME_REFERENCE_STATE_H
#define LIBGNME_REFERENCE_STATE_H

#include <armadillo> 

namespace libgnme {

template<typename Tc>
class reference_state
{
public:
    const size_t m_nbsf;  //!< Number of basis functions
    const size_t m_nmo;   //!< Number of MOs
    const size_t m_nelec; //!< Number of electrons
    const size_t m_nact;  //!< Number of active orbitals
    const size_t m_ncore; //!< Number of core orbitals
    const arma::Mat<Tc> m_C; //!< Orbital coefficients

    reference_state(
        const size_t nbsf, const size_t nmo, const size_t nelec,
        arma::Mat<Tc> &C) : 
    m_nbsf(nbsf), m_nmo(nmo), m_nelec(nelec), m_C(C), m_nact(nmo), m_ncore(0)
    { }

    reference_state(
        const size_t nbsf, const size_t nmo, const size_t nelec, 
        const size_t nact, const size_t ncore, arma::Mat<Tc> &C) : 
    m_nbsf(nbsf), m_nmo(nmo), m_nelec(nelec), m_C(C), m_nact(nact), m_ncore(ncore)
    { 
        assert(ncore + nact <= nmo);
    }

    virtual ~reference_state() { }
};

} // libgnme

#endif // LIBGNME_REFERENCE_STATE_H
