# LibGNME 
A C++ library for evaluating non-orthogonal matrix elements in electronic structure.
<ul>
<li>Hugh G. A. Burton (2022&ndash;)</li>
</ul>

If you incorporate this code in your work, please consider citing the following work:
<ol reversed>
<li>"Generalized nonorthogonal matrix elements. II: Extension to arbitrary excitations"; <a href="https://doi.org/10.1063/5.0122094"><i>J. Chem. Phys.</i> <b>157</b>, 204109 (2022)</a></li>

<li>"Generalized nonorthogonal matrix elements: Unifying Wick’s theorem and the Slater–Condon rules"; <a href="https://doi.org/10.1063/5.0045442"><i>J. Chem. Phys.</i> <b>154</b>, 144109 (2021)</a>
</ol>

## Installation
### Prerequisites
The libGNME package requires a set of standard libraries:
1. LAPACK, BLAS or Intel MKL, with <tt>$MKLROOT</tt> set.
2. OpenMP
3. CMake (version 3.12 or higher)

### Compiling the library
The configure script can be run depending on the choice of compiler:
1. <tt>./configure [intel/gcc/pgi] --with-openmp</tt> 
2. <tt>cd build</tt>
3. <tt>make install</tt>

Following installation, the test suite can be executed from the <tt>build/</tt> directory by running <tt>ctest</tt>.

### Compiling programs using LibGNME
As LibGNME produces dynamic libraries, compiling a program (e.g. <tt>mycalc.C</tt>) requires these to be linked at the compilation state. The example below illustrates this for a typical program <tt>mycalc.C</tt>
```
icpc -Wall -std=c++11 -fopenmp -I${GNMEROOT}/external/armadillo-10.1.2/include -I${GNMEROOT} -L${MKLROOT}/lib/intel64_lin -L${GNMEROOT}/lib -g mycalc.C -lgnme_wick -lgnme_utils -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64  -omycalc
```
Here, <tt>${GNMEROOT}</tt> is an environment variable that points towards the root directory of your LibGNME installation. 
Subsequently running your compiled progarm <tt>mycalc</tt> requires <tt>${GNMEROOT}/lib</tt> to be included in the <tt>$LD_LIBRARY_PATH</tt>.

## Code structure
The primary functionality of LibGNME is to compute matrix elements between non-orthogonal Slater determiants. This can be achieved using either extended nonorthogonal Wick's theory, or using the more computationally expensive Slater&ndash;Condon rules. 

The code is split into three different libraries:

### 1. gnme_utils
Support functions including:
<ul>
<li>Generalised eigenvalue problem solver;</.li>
<li>Biorthogonalisation using Lowdin pairing;</li>
<li>Two-electron integral transformation;</li>
<li>General linear algebra routines.</li>
</ul>

#### libgnme::bitset
To avoid dependencies on other libraries, LibGNME also comes with a built-in bitstring representation for electronic configurations. These bitsets store an electronic configuration using a <tt>std::vector<bool></tt> data type, with the rightmost element representing the first molecular orbital. 
A bitset object can be constructed as:
```
#include <libgnme/bitset/utils.h>

// Initialise directly from std::vector<bool> representation
libgnme::bitset b(std::vector<bool>({0,0,0,1,1,1}));

// Initialise from integer value
size_t nbit = 6;
size_t nval = 7;
libgnme::bitset b(nval,6);

// Print the bitset 
b.print();
```

### 2. gnme_wick
Compute matrix elements using the generalised nonorthogonal Wick's theorem. 

Here, the computation is divided into two types of objects:
1. <tt>wick_orbitals</tt>: Construct a biorthogonalised set of reference orbitals, and compute corresponding contractions.
2. <tt>wick_rscf</tt> and <tt>wick_uscf</tt>: Objects that build one- or two-electron matrix elements using <tt>wick_orbitals</tt> and atomic orbital integrals.

Once these objects have been defined, a given matrix element can be requested by defining the bra and ket excitation using either a bitset representation or a list of single particle excitations. 

For example, consider the coupling term between two excitations from reference states with coefficients <tt>Cx</tt> and <tt>Cw</tt>, with <tt>ne</tt> electrons, <tt>nbsf</tt> basis functions, and <tt>nmo</tt> molecular orbitals. Here, <tt>S</tt> and <tt>h1e</tt> are Armadillo matrices containing the AO overlap and one-electron integrals, respectively, and <tt>II</tt> is an Armadillo matrix containing the two-electron integrals with the indexing <tt>II(i\*nbsf+j,k\*nbsf+l) = (ij|kl)</tt>. The Hamiltonian coupling is computed as:
```
// Setup the biorthogonalized orbital pair
libgnme::wick_orbitals<double,double> orbs(nbsf, nmo, neleca, Cx, Cw, S);

// Setup the matrix builder object
libgnme::wick_rscf<double,double,double> mb(orbs, S, enuc);

// Add one- and two-body contributions
mb.add_one_body(h1e);
mb.add_two_body(II);

// Define bitsets for occupations of bra and ket state
// This coupling corresponds to spin-down single excitation in bra (x) and beta double excitation in ket (w).
libgnme::bitset bxa(std::vector<bool>({0,0,0,1,1,1});
libgnme::bitset bxb(std::vector<bool>({0,1,0,0,1,1});
libgnme::bitset bwa(std::vector<bool>({0,0,0,1,1,1});
libgnme::bitset bwb(std::vector<bool>({0,1,1,0,0,1});

// Intialise temporary variables
double Hwx = 0.0, Swx = 0.0;

// Evaluate matrix element
mb.evaluate(bxa, bxb, bwa, bwb, Swx, Hwx);
```
The <tt>wick_uscf</tt> object differs only in that it can compute matrix elements for unrestricted reference states with different molecular orbitals for different spins. This object is initialised using two <tt>wick_orbtals</tt> objects for the high- and low-spin orbitals as e.g. ```libgnme::wick_uscf<double,double,double> mb(orba, orbb, S, enuc);```

### 3. gnme_slater
Alternatively, the LibGNME library can also be used to compute matrix elements using the older generalised Slater&ndash;Condon rules. This achieved using the similar objects <tt>slater_rscf</tt> and <tt>slater_uscf</tt> objects that take the AO integrals as an input. For example, the nonorthogonal coupling term illustrated about could be computed as
```
// Setup the matrix builder object
libgnme::slater_rscf<double,double,double> mb(nbsf, nmo, neleca, nelecb, S, enuc);

// Add one- and two-body contributions
mb.add_one_body(h1e);
mb.add_two_body(II);

// Set occupation vectors
arma::uvec xocca({0,1,2}), xoccb({0,1,4});
arma::uvec wocca({0,1,2}), woccb({0,3,4});

// Intialise temporary variables
double Hwx = 0.0, Swx = 0.0;

// Evaluate matrix element
mb.evaluate(Cx.cols(xocca), Cx.cols(xoccb), Cw.cols(wocca), Cw.cols(woccb), Swx, Hwx);
```
