include(CheckCXXSourceCompiles)

if(WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_REQUIRED_FLAGS_SAVED ${CMAKE_REQUIRED_FLAGS})
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS} ${OpenMP_CXX_FLAGS})
    check_cxx_source_compiles("
        int main() {
            double x[128];
            #pragma omp simd
            for(int i = 0; i < 128; i++) x[i] = 0.0;
            return 0; }"
        WITH_OPENMP4_SIMD)
    check_cxx_source_compiles("
        int a = 0;
        void f(int b) {
            #pragma omp atomic write
            a = b;
        }
        int main() { return 0; }"
        WITH_OPENMP4_ATOMIC)
    set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_SAVED})
endif(WITH_OPENMP)
