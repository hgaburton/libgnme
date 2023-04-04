#ifndef LIBGNME_BITSET_TOOLS_H
#define LIBGNME_BITSET_TOOLS_H

#include "bitset.h"

namespace libgnme {

/** \brief Generate list of bitsets for all configurations of electrons in orbitals
    \param nelec Total number of electrons
    \param norb  Total number of orbitals
 **/
std::vector<bitset> fci_bitset_list(size_t nelec, size_t norb);

} // namespace libgnme

#endif // LIBGNME_BITSET_TOOLS_H
