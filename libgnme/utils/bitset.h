#ifndef LIBGNME_BITSET_H
#define LIBGNME_BITSET_H

#include <armadillo>
#include <cassert>

namespace libgnme {

class bitset 
{
private:
    std::vector<bool> m_v;
    size_t m_size;

public:
    // Default constructor
    bitset() : m_size(0) { };

    // Copy constructor
    bitset(const bitset &other) : m_v(other.m_v), m_size(other.m_size) { };

    // Construct from bool vector
    bitset(std::vector<bool> bs) : m_v(bs), m_size(bs.size()) { };

    // Construct from integer value
    bitset(int n, int m) : m_size(m), m_v(m,0)
    {
        // Find highest power of two
        int dummy = 1, dmax = 1;
        bool divisible = (n / 1 > 0);
        while(n / dummy > 0) { dummy *= 2; dmax  += 1; }
        assert(dmax - 2 < int(m_size));
        
        // Convert integer to a bit string representation
        dummy = 1;
        for(size_t i=0; i<m_size; i++)
        {
            m_v[m_size-i-1] = (n % (dummy * 2)) / dummy;
            dummy *= 2;
        }
    }

    // Access element of the bitset
    //const bool& operator()(size_t index) const;
    //bool& operator()(size_t index);
    

    void flip(size_t i) { m_v[m_size-1-i] = not m_v[m_size-1-i]; };

    void print()
    {
        for(size_t i=0; i<m_size; i++)  std::cout << m_v[i];
        std::cout << std::endl;
    };
    
    
    size_t count(size_t min=0, size_t max=0) 
    {
        if(min==0 and max==0)
            return std::accumulate(m_v.begin(), m_v.end(), 0);
        int sum = 0;
        for(size_t i=min; i<max; i++) sum += m_v[m_size-1-i];
        return sum;
    }; 

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


    bitset operator& (const bitset& other) 
    {
        bitset tmp(0,m_size);
        for(size_t i=0; i<m_size; i++) tmp.m_v[i] = m_v[i] & other.m_v[i];
        return tmp;
    };

    bitset operator| (const bitset& other) 
    {
        bitset tmp(0,m_size);
        for(size_t i=0; i<m_size; i++) tmp.m_v[i] = m_v[i] | other.m_v[i];
        return tmp;
    };
    bitset operator^ (const bitset& other) 
    {
        bitset tmp(0,m_size);
        for(size_t i=0; i<m_size; i++) tmp.m_v[i] = m_v[i] ^ other.m_v[i];
        return tmp;
    };

    void excitation(bitset &other, arma::umat &hp, int &parity)
    {
        arma::ivec diff(m_size, arma::fill::zeros);
        for(size_t i=0; i<m_size; i++) 
            diff(m_size-1-i) = other.m_v[i] - m_v[i];

        // Define hole-particle array
        hp = arma::join_rows(arma::find(diff==-1), arma::reverse(arma::find(diff==1)));

        // Get parity relative to reference
        parity = 1;
        bitset tmp(*this); // Temporary bitset copy
        for(size_t i=0; i<hp.n_rows; i++)
        {
            size_t h = hp(i,0), p = hp(i,1);
            tmp.flip(h); tmp.flip(p);
            parity *= std::pow(-1, ((bitset) (*this) & tmp).count(h,p));
        }
    };

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

    bool next_fci() { return std::next_permutation(m_v.begin(), m_v.end()); }
};


} // namespace libgnme

#endif // LIBGNME_BITSET_H
