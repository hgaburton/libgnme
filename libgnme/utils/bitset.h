#ifndef LIBGNME_BITSET_H
#define LIBGNME_BITSET_H

namespace libgnme {

class bitset 
{
protected:
    std::vector<bool> m_v;

public:
    bitset() { };

    bitset(std::vector<bool> bs) : m_v(bs)
    { 
        size_t L = bs.size();
        m_v.resize(L);
        for(size_t i=0; i<L; i++) m_v[i] = bs[i];
    };

    bitset(arma::ivec occ) 
    {
        size_t L = occ.n_elem;
        m_v.resize(L);
        for(size_t i=0; i<L; i++)
        {
            if(occ(i) == 1) m_v[L-i-1] = 1;
            else m_v[L-i-1] = 0;
        }
    }

    bitset(int n, int d)
    {
        // Find highest power of two
        int dummy = 1, dmax = 1;
        bool divisible = (n / 1 > 0);
        while(n / dummy > 0) 
        {
            dummy *= 2; dmax  += 1;
        }
        assert(dmax - 2 < d);
        
        // Convert integer to a bit string representation
        m_v.resize(d);
        dummy = 1;
        for(size_t i=0; i<d; i++)
        {
            m_v[d-i-1] = (n % (dummy * 2)) / dummy;
            dummy *= 2;
        }
    };
    
    int get_int()
    {
        int n = 0, dummy = 1, L = m_v.size();
        for(size_t i=0; i < L; i++) 
        {
            if(m_v[L-i-1]) n += dummy;
            dummy *= 2;
        }
        return n;
    };

    void print()
    {
        for(size_t i=0; i<m_v.size(); i++)  std::cout << m_v[i];
        std::cout << std::endl;
    };

    int size() { return m_v.size(); };

    int count() { return std::accumulate(m_v.begin(), m_v.end(), 0); };

    bitset operator & (const bitset& other) 
    {
        size_t L = size();

        bitset tmp;
        tmp.m_v.resize(L);
        for(size_t i=0; i<L; i++) tmp.m_v[i] = m_v[i] & other.m_v[i];

        return tmp;
    }

    int parity(bitset &other)
    {
        int ss = other.get_int() - get_int();
        if(ss > 0) 
        {
            bitset bs_diff(ss, size());
            int p = std::pow(-1, ((bitset) other & bs_diff).count());
            return p;
        } 
        else if (ss == 0)
        {
            return 1;
        }
        else
        {
            bitset bs_diff(-ss, size());
            return std::pow(-1, ((bitset) (*this) & bs_diff).count());
        }
    }


    arma::umat excitation(bitset &other)
    {
        // Find particle and hole indices
        int s = size();
        arma::ivec diff(s, arma::fill::zeros);
        for(size_t i=0; i<s; i++) diff(s-1-i) = other.m_v[i] - m_v[i];

        // Return particle/hole indices in arma format
        return arma::join_rows(arma::find(diff == -1), arma::reverse(arma::find(diff == 1)));
    }


    bool next()
    {
        return std::next_permutation(m_v.begin(), m_v.end());
    }

};

}

#endif // LIBGNME_BITSET_H
