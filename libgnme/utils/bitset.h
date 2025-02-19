#ifndef LIBGNME_BITSET_H
#define LIBGNME_BITSET_H

#include <armadillo>
#include <cassert>

namespace libgnme {

/** \brief Store electronic configuration as a bitset 
    \ingroup gnme_utils
 **/
class bitset 
{
private:
    std::vector<uint8_t> m_v; //!< Vector of bool values representing the bitset
    size_t m_size; //!< Integer for the size of the bitset (total num. of orbitals)

public:
    /** \brief Default constructor **/
    bitset() : m_size(0) { };

    /** \brief Copy constructor 
        \param other Bitset to be copied
     **/
    bitset(const bitset &other) : m_v(other.m_v), m_size(other.m_size) { };

    /** \brief Constructor from bool vector
        \param bs Vector of bools representing target bitset
     **/
    bitset(std::vector<uint8_t> bs) : m_v(bs), m_size(bs.size()) { };

    /** \brief Constructor from integer value in binary representation
        \param n Target value
        \param m Total number of bits
     **/
    bitset(int n, int m) : m_v(m,0), m_size(m)
    {
        // Find highest power of two
        int dummy = 1;
        int dmax = 1;
        while(n / dummy > 0) 
        { 
            dummy *= 2; 
            dmax  += 1; 
        }
        if(dmax - 2 > int(m_size))
            throw std::runtime_error("bitset: Integer value exceeds bitset size");
        
        // Convert integer to a bit string representation
        dummy = 1;
        for(size_t i=0; i<m_size; i++)
        {
            m_v[m_size-i-1] = (n % (dummy * 2)) / dummy;
            dummy *= 2;
        }
    }

    /** \brief Flip bit between true/false
        \param i Index of bit to be flipped (0 is the far right)
     **/
    void flip(size_t i) { m_v[m_size-1-i] = not m_v[m_size-1-i]; };

    /** \brief Print the bitset representation **/
    void print()
    {
        for(size_t i=0; i<m_size; i++)  std::cout << m_v[i];
        std::cout << std::endl;
    };
    
    /** \brief Count number of set bits **/
    size_t count(size_t min=0, size_t max=0) 
    {
        if((min==0) and (max==0))
            return std::accumulate(m_v.begin(), m_v.end(), 0);
        int sum = 0;
        for(size_t i=min; i<max; i++) sum += m_v[m_size-1-i];
        return sum;
    }; 

    /** \brief Get integer representation of bitset **/
    int get_int()
    {
        int n = 0, dummy = 1; 
        for(size_t i=0; i < m_size; i++) 
        {
            if(m_v[m_size-i-1]) n += dummy;
            dummy *= 2;
        }
        return n;
    };

    /** \brief Bitwise AND operation **/
    bitset operator& (const bitset& other) 
    {
        bitset tmp(0,m_size);
        for(size_t i=0; i<m_size; i++) tmp.m_v[i] = m_v[i] & other.m_v[i];
        return tmp;
    };

    /** \brief Bitwise OR operation **/
    bitset operator| (const bitset& other) 
    {
        bitset tmp(0,m_size);
        for(size_t i=0; i<m_size; i++) tmp.m_v[i] = m_v[i] | other.m_v[i];
        return tmp;
    };

    /** \brief Bitwise XOR operation **/
    bitset operator^ (const bitset& other) 
    {
        bitset tmp(0,m_size);
        for(size_t i=0; i<m_size; i++) tmp.m_v[i] = m_v[i] ^ other.m_v[i];
        return tmp;
    };

    /** \brief Ger parity relative to another bitset
        \param other  Bitset representing excitation from current object
        \param parity Parity of the excitation
     **/
    int parity(bitset &other)
    {
        arma::ivec diff(m_size, arma::fill::zeros);
        for(size_t i=0; i<m_size; i++) 
            diff(m_size-1-i) = other.m_v[i] - m_v[i];

        // Define hole-particle array
        arma::umat hp = arma::join_rows(arma::find(diff==-1), arma::reverse(arma::find(diff==1)));

        // Get parity relative to reference
        int par = 1;
        bitset tmp(*this); // Temporary bitset copy
        for(size_t i=0; i<hp.n_rows; i++)
        {
            size_t h = hp(i,0), p = hp(i,1);
            tmp.flip(h); tmp.flip(p);
            par *= std::pow(-1, ((bitset) (*this) & tmp).count(h,p));
        }
        return par;
    };

    /** \brief Identify excitation indices between a pair of bitsets 
        \param other  Bitset representing excitation from current object
        \param hp     Output particle-hole indices
        \param parity Parity of the excitation
     **/
    void excitation(bitset &other, arma::umat &hp, int &par)
    {
        arma::ivec diff(m_size, arma::fill::zeros);
        for(size_t i=0; i<m_size; i++) 
            diff(m_size-1-i) = other.m_v[i] - m_v[i];

        // Define hole-particle array
        hp = arma::join_rows(arma::find(diff==-1), arma::reverse(arma::find(diff==1)));

        // Get parity relative to reference
        par = 1;
        bitset tmp(*this); // Temporary bitset copy
        for(size_t i=0; i<hp.n_rows; i++)
        {
            size_t h = hp(i,0), p = hp(i,1);
            tmp.flip(h); tmp.flip(p);
            par *= std::pow(-1, ((bitset) (*this) & tmp).count(h,p));
        }
    };

    /** \brief Get vector of occupied orbitals **/
    arma::uvec occ()
    {
        size_t n = count(), it=0;
        arma::uvec v_occ(n, arma::fill::zeros);
        for(size_t i=0; i<m_size; i++) 
        {
            if(m_v[m_size-i-1]) v_occ(it++) = i;
            if(it==n) break;
        }
        return v_occ;
    }

    /** \brief Iterate this bitset to the next FCI configuration **/
    bool next_fci() { return std::next_permutation(m_v.begin(), m_v.end()); }
};


} // namespace libgnme

#endif // LIBGNME_BITSET_H
