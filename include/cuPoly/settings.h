// cuPoly - A GPGPU-based library for doing polynomial arithmetic on RLWE-based cryptosystems
// Copyright (C) 2017-2021, Pedro G. M. R. Alves - pedro.alves@ic.unicamp.br

// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.


#ifndef SETTINGS_H
#define SETTINGS_H
#include <stdio.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <time.h>
#include <map>
#include <cuPoly/tool/version.h>
#include "cuPolyConfig.h"

/**
 * @brief   Supported operators by CUDAEngine::execute_polynomial_op_by_int
 */
enum add_mode_t {ADD, SUB, MUL, MULMUL, ADDADD};
/**
 * @brief      Supported RNS Bases
 */
enum bases {QBase, BBase, QBBase, TBase, QTBBase};
/**
 * @brief      DGT possible directions
 */
enum dgt_direction{FORWARD, INVERSE};

#define HAMMINGWEIGHT_H 64

////////////////////////////////////////////////////////////////////////////////
// Typedefs and definitions
////////////////////////////////////////////////////////////////////////////////

/**
 * @brief  A 128 bits unsigned integer data type
 * 
 * It is compound by two 64 bits unsigned integers.
 * 
 * @param lo The lowest bits
 * @param hi The highest bits
 */
typedef struct {
  uint64_t lo = 0;
  uint64_t hi = 0;
} uint128_t;

#ifndef SEL
#define SEL(A, B, C) ((-(C) & ((A) ^ (B))) ^ (A))
#endif
/**
 * @brief The basic data type for cuPoly.
 * 
 * It provides the arithmetic required by the DGT.
 */
typedef struct GaussianInteger{
    uint64_t re; //<! The real part
    uint64_t imag;//<! The imaginary part
    inline bool operator==(const GaussianInteger& b){
        return (re == b.re) && (imag == b.imag);
    };
    __host__ __device__ inline void write(int i, uint64_t x) {
      re   = SEL(re, x, (i == 0));
      imag = SEL(imag, x, (i == 1));
    };
} GaussianInteger;

/**
 * @brief       A map of n-th roots.
 * 
 * Stores n-th roots of i for different coprimes.
 */
extern std::map<uint64_t, std::map<int, GaussianInteger>> NTHROOT;

/**
 * @brief       A map of inverses of n-th roots.
 * 
 * Stores inverses of n-th roots of i for different coprimes.
 */
extern std::map<uint64_t, std::map<int, GaussianInteger>> INVNTHROOT;

/**
 * @brief       A map of primitive roots
 * 
 * Stores primitive roots for different coprimes.
 */
extern std::map<uint64_t, int> PROOTS;

// default block size
#define ADDBLOCKXDIM 512 //!< A suggestion for a default CUDA block size. @fixme We should avoid this.
#define FPERROR 1e-9 //!< A tolerance epsilon for HPS floating-point division.

// RNS
#ifdef BFV_ENGINE_MODE
#define MAX_COPRIMES 30 //!< The maximum quantity of coprimes that may be used
#endif
#ifdef CKKS_ENGINE_MODE
#define MAX_COPRIMES 200 //!< The maximum quantity of coprimes that may be used in total.
#endif
#define MAX_COPRIMES_IN_A_BASE MAX_COPRIMES/2  //!< The maximum quantity of coprimes that may be used in a single base.

//
extern uint64_t COPRIMES_45_BUCKET[]; //!< A list of 45-bits coprimes supported for RNS.
extern uint64_t COPRIMES_48_BUCKET[]; //!< A list of 48-bits coprimes supported for RNS.
extern uint64_t COPRIMES_50_BUCKET[]; //!< A list of 50-bits coprimes supported for RNS.
extern uint64_t COPRIMES_52_BUCKET[]; //!< A list of 52-bits coprimes supported for RNS.
extern uint64_t COPRIMES_55_BUCKET[]; //!< A list of 55-bits coprimes supported for RNS.
extern uint64_t COPRIMES_62_BUCKET[]; //!< A list of 62-bits coprimes supported for RNS.
extern uint64_t COPRIMES_63_BUCKET[]; //!< A list of 63-bits coprimes supported for RNS.

extern const uint32_t COPRIMES_45_BUCKET_SIZE; //!< The size of COPRIMES_45_BUCKET.
extern const uint32_t COPRIMES_48_BUCKET_SIZE; //!< The size of COPRIMES_48_BUCKET.
extern const uint32_t COPRIMES_50_BUCKET_SIZE; //!< The size of COPRIMES_50_BUCKET.
extern const uint32_t COPRIMES_52_BUCKET_SIZE; //!< The size of COPRIMES_52_BUCKET.
extern const uint32_t COPRIMES_55_BUCKET_SIZE; //!< The size of COPRIMES_55_BUCKET.
extern const uint32_t COPRIMES_62_BUCKET_SIZE; //!< The size of COPRIMES_62_BUCKET.
extern const uint32_t COPRIMES_63_BUCKET_SIZE; //!< The size of COPRIMES_63_BUCKET.

extern uint64_t COPRIMES_BUCKET[]; //!< A list of coprimes supported for RNS used during the execution.
extern uint32_t COPRIMES_BUCKET_SIZE; //!< The size of COPRIMES_BUCKET.

// Auxiliar methods
extern double compute_time_ms(struct timespec start,struct timespec stop);
extern uint64_t get_cycles();

/**
 * @brief      Macro for checking cuda errors following a cuda launch or api call
 *
 */
#define cudaCheckError() {                                          \
 cudaError_t e = cudaGetLastError();                                 \
 if( e == cudaErrorInvalidDevicePointer)   \
   fprintf(stderr, "Cuda failure %s:%d: '%s' (%d)\n",__FILE__,__LINE__,cudaGetErrorString(e), e);           \
 else if(e != cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s' (%d)\n",__FILE__,__LINE__,cudaGetErrorString(e), e);           \
    exit(1);                                                                 \
 }                                                                      \
}

// Discrete gaussian setup
#define GAUSSIAN_STD_DEVIATION (float)3.2 //!< Standard deviation for the discrete gaussian sampling.
#define GAUSSIAN_BOUND (float)0 //!< Bound for the discrete gaussian sampling.

#endif
