#include "bitset.h"

namespace libgnme {

std::vector<bitset> fci_bitset_list(size_t nelec, size_t norb)
{
    // Get reference bitset
    bitset b(std::pow(2,nelec)-1, norb); 

    std::vector<bitset> v_bs;
    do { v_bs.push_back(b); } while(b.next_fci());

    return v_bs;
}

} // namespace libgnme
