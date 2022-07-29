CXX_FLAGS=icpc -Wall -std=c++11 -g3
CXX_INCLUDE=-I/u/thchem/newc6131/code/libgnme -I/u/thchem/newc6131/code/libgnme/external/armadillo-10.1.2/include
CXX_LD_PATH=-L/opt/intel/compilers_and_libraries_2019.1.144/linux/mkl/lib/intel64_lin -L/opt/intel/compilers_and_libraries_2019.1.144/linux/compiler/lib/intel64_lin -L/u/thchem/newc6131/code/libgnme/lib 

all: wick_one_body wick_two_body wick_two_body_time wick_rscf_two_body_time

wick_one_body:
	${CXX_FLAGS} ${CXX_INCLUDE} ${CXX_LD_PATH} -g wick_one_body.C -lgnme_wick -lgnme_utils -lgnme_slater -lmkl_core -lmkl_intel_thread -liomp5 -lmkl_intel_lp64 -olibgnme_wick_one_body

wick_two_body:
	${CXX_FLAGS} ${CXX_INCLUDE} ${CXX_LD_PATH} -g wick_two_body.C -lgnme -lmkl_core -lmkl_intel_thread -liomp5 -lmkl_intel_lp64 -olibgnme_wick_two_body

wick_two_body_time:
	${CXX_FLAGS} ${CXX_INCLUDE} ${CXX_LD_PATH} -g wick_two_body_timing.C -lgnme -lmkl_core -lmkl_intel_thread -liomp5 -lmkl_intel_lp64 -olibgnme_wick_two_body_time

wick_rscf_two_body_time:
	${CXX_FLAGS} ${CXX_INCLUDE} ${CXX_LD_PATH} -g wick_rscf_two_body_timing.C -lgnme -lmkl_core -lmkl_intel_thread -liomp5 -lmkl_intel_lp64 -olibgnme_wick_rscf_two_body_time

test: 
	./libgnme_wick_one_body
