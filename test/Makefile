CXX_FLAGS=g++ -Wall -std=c++11 -O3
CXX_INCLUDE=-I/u/thchem/newc6131/code/libnome/include -I/u/thchem/newc6131/code/libnome/external/armadillo-10.1.2/include
CXX_LD_PATH=-L/opt/intel/compilers_and_libraries_2019.1.144/linux/mkl/lib/intel64_lin -L/opt/intel/compilers_and_libraries_2019.1.144/linux/compiler/lib/intel64_lin -L/u/thchem/newc6131/code/libnome/lib 

make: 
	${CXX_FLAGS} ${CXX_INCLUDE} ${CXX_LD_PATH} -g wick_one_body.C -lnome -lmkl_core -lmkl_intel_thread -liomp5 -lmkl_intel_lp64 -olibnome_wick_one_body

test: 
	./libnome_wick_one_body
