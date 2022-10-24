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


#include <cuPoly/cuda/cudaengine.h>
#include <functional>
#include <numeric>

ZZ CUDAEngine::RNSProduct;
ZZ CUDAEngine::RNSBProduct;
std::vector<uint64_t> CUDAEngine::RNSPrimes;
std::vector<ZZ> CUDAEngine::RNSMpi;
std::vector<uint64_t> CUDAEngine::RNSInvMpi;
std::vector<uint64_t> CUDAEngine::RNSBPrimes;
int CUDAEngine::N = 0;
int CUDAEngine::is_init = false;
uint64_t *CUDAEngine::RNSCoprimes;
uint64_t CUDAEngine::t;
#ifdef CKKS_ENGINE_MODE
int CUDAEngine::scalingfactor = -1;
int CUDAEngine::dnum = 1;
#endif

// \todo These shall be moved to inside CUDAEngine
uint32_t COPRIMES_BUCKET_SIZE; //!< The size of COPRIMES_55_BUCKET.

///////////////
// Main base //
///////////////
__constant__ uint64_t d_RNSCoprimes[MAX_COPRIMES];
__constant__ int      d_RNSQNPrimes;// Used primes

// HPS basis extension from B to Q and fast_conv_B_to_Q
__constant__ int      d_RNSBNPrimes;// Used primes
// rho_ckks_rns_rid and polynomial_basis_ext_B_to_Q
__constant__ uint64_t d_RNSBqi[MAX_COPRIMES_IN_A_BASE]; // Stores (B) \pmod qPi;
// HPS basis extension from B to Q and fast_conv_B_to_Q
__constant__ uint64_t d_RNSInvModBbi[MAX_COPRIMES_IN_A_BASE]; // Stores (B/bi)^-1 \pmod bi;

#ifdef CKKS_ENGINE_MODE

// HPS basis extension from B to Q and fast_conv_B_to_Q
uint64_t *d_RNSBbiqi; // Stores (B/bi) \pmod qi;
                                                                                   // 
// HPS basis extension  and fast_conv_Q_to_B
uint64_t *d_RNSInvModQqi; // Stores (Q/bi)^-1 \pmod qi; for each level

// HPS basis extension Q to B and fast_conv_Q_to_B
uint64_t *d_RNSQqibi; // Stores (Q/qi) \pmod bi; ; for each level

// CKKS rescale
uint64_t *d_RNSInvModqlqi; // Stores (ql)^-1 \pmod qi;

// fast_conv_B_to_Q and xi_ckks_rns_rid
__constant__ uint64_t d_RNSInvModBqi[MAX_COPRIMES_IN_A_BASE]; // Stores (B) \pmod qPi;

// rho_ckks_rns_rid
uint64_t *d_RNShatQQi; // Stores \hat{Qj} \pmod qi;

#endif

#ifdef BFV_ENGINE_MODE

// HPS basis extension from B to Q and fast_conv_B_to_Q
__constant__ uint64_t d_RNSBbiqi[MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE]; // Stores (B/bi) \pmod qi;
                                                                                   // 
// HPS basis extension  and fast_conv_Q_to_B
__constant__ uint64_t d_RNSInvModQqi[MAX_COPRIMES_IN_A_BASE]; // Stores (Q/bi)^-1 \pmod qi;

// HPS basis extension Q to B and fast_conv_Q_to_B
__constant__ uint64_t d_RNSQqibi[MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE]; // Stores (Q/qi) \pmod bi;

// rho_bfv_rns_rid
__constant__ uint64_t d_RNSQqi[MAX_COPRIMES_IN_A_BASE]; // Stores (Q/qi) \pmod qi;

// HPS basis extension from Q to B
__constant__ uint64_t d_RNSQbi[MAX_COPRIMES_IN_A_BASE]; // Stores (Q) \pmod bi;

// HPS simple scaling
__constant__ uint64_t d_RNST; // Stores T

// HPS complex scaling
__constant__ uint64_t d_RNSomega_int[MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE]; //
__constant__ uint128_t d_RNSomega_frac[MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE]; //
__constant__ uint64_t d_RNSlambda[MAX_COPRIMES_IN_A_BASE]; // 
__constant__ uint64_t d_RNStomega_int[MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE]; //
__constant__ uint128_t d_RNStomega_frac[MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE]; //
#endif

int round_up_blocks(int a, int b){
  return (a % b == 0? a / b: a / b + 1);
}

/**
 * @brief      Negates each coefficient of a
 *
 * @param      b          { parameter_description }
 * @param[in]  a          { parameter_description }
 * @param[in]  N          { parameter_description }
 * @param[in]  NResidues  The n residues
 */
__global__ void polynomial_negate(
  GaussianInteger *b,
  const GaussianInteger *a,
  const int N,
  const int NResidues ){

  // One thread per polynomial coefficient on 1D-blocks
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  const int size = N * NResidues;

  if(tid < size ){
    GaussianInteger x = a[tid];
    b[tid] = {
      x.re == 0? 
        0 :
        d_RNSCoprimes[rid] - (x.re),
      x.imag == 0? 
        0 :
        d_RNSCoprimes[rid] - (x.imag)
    };
  }

}

__host__ void CUDAEngine::execute_polynomial_negate(
    GaussianInteger *b,
    const GaussianInteger *a,
    const int base,
    Context *ctx ){
  const int N = CUDAEngine::N;
  const int NResidues = get_n_residues(base);
  const int size = N * NResidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  polynomial_negate<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b,
    a,
    N,
    NResidues);
  cudaCheckError();
}

/**
 * @brief       Operate over each coefficient of a by an int x and write to b
 * 
 * @param[out] b         [description]
 * @param[in]  a         [description]
 * @param[in]  x         [description]
 * @param[in]  OP        [description]
 * @param[in]  N         [description]
 * @param[in]  NResidues [description]
 */
__global__ void polynomial_op_by_int(
  GaussianInteger *b,
  const GaussianInteger *a,
  uint64_t x,
  const int OP,
  const int N,
  const int NResidues){

  // One thread per polynomial coefficient on 1D-blocks
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  const int size = N * NResidues;

  if(tid < size ){
    //  Coalesced access to global memory. Doing this way we reduce required
    // bandwich.
    //
    switch(OP){
    case ADD:
    b[tid] = GIAdd(a[tid], (GaussianInteger){x, 0}, rid);
    break;
    case SUB:
    b[tid] = GISub(a[tid], (GaussianInteger){x, 0}, rid);
    break;
    case MUL:
    b[tid] = GIMul(a[tid], (GaussianInteger){x, 0}, rid);
    break;
    default:
    printf("Unknown operation %d\n", OP);
    break;
    }
  }
}

__host__ void CUDAEngine::execute_polynomial_op_by_int(
    GaussianInteger *b,
    GaussianInteger *a,
    const int base,
    uint64_t x,
    int OP,
    Context *ctx ){

  const int size = N * get_n_residues(base);
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  polynomial_op_by_int<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b,
    a,
    x,
    OP,
    N,
    get_n_residues(base));
  cudaCheckError();

}

/**
 * @brief       Operate over each coefficient of a by an int x and write to b
 * 
 * @param[out] b         [description]
 * @param[in]  a         [description]
 * @param[in]  x         [description]
 * @param[in]  OP        [description]
 * @param[in]  N         [description]
 * @param[in]  NResidues [description]
 */
__global__ void polynomial_double_op_by_int(
  GaussianInteger *b1,
  const GaussianInteger *a1,
  GaussianInteger *b2,
  const GaussianInteger *a2,
  uint64_t x1,
  uint64_t x2,
  const int OP,
  const int N,
  const int NResidues){

  // One thread per polynomial coefficient on 1D-blocks
  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  const int size = N * NResidues;

  if(tid < size ){
    //  Coalesced access to global memory. Doing this way we reduce required
    // bandwich.
    //
    switch(OP){
    case MULMUL:
    b1[tid] = GIMul(a1[tid], (GaussianInteger){x1, 0}, rid);
    b2[tid] = GIMul(a2[tid], (GaussianInteger){x2, 0}, rid);
    break;
    case ADDADD:
    b1[tid] = GIAdd(a1[tid], (GaussianInteger){x1, 0}, rid);
    b2[tid] = GIAdd(a2[tid], (GaussianInteger){x2, 0}, rid);
    break;
    default:
    printf("Unknown operation %d\n", OP);
    break;
    }
  }
}

__host__ void CUDAEngine::execute_polynomial_double_op_by_int(
    GaussianInteger *b1,
    GaussianInteger *a1,
    GaussianInteger *b2,
    GaussianInteger *a2,
    const int base,
    uint64_t x1,
    uint64_t x2,
    int OP,
    Context *ctx ){

  const int size = N * get_n_residues(base);
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  polynomial_double_op_by_int<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b1,
    a1,
    b2,
    a2,
    x1,
    x2,
    OP,
    N,
    get_n_residues(base));
  cudaCheckError();

}

/////////
//     //
// RNS //
//     //
/////////

/** Multiply a 128 bits uint64_t by a 64 bits uint64_t and returns the two most 
 significant words
 */
__device__ uint128_t __umul128hi(const uint128_t a, const uint64_t b){
  uint128_t c;
  c.lo = __umul64hi(a.lo, b);
  c.lo += a.hi * b;
  c.hi = __umul64hi(a.hi, b) + (c.lo < a.hi * b);

  return c;
}

#ifdef BFV_ENGINE_MODE
__global__ void simple_scaling(
  GaussianInteger *a,
  const int N){

  const int cid = threadIdx.x + blockDim.x * blockIdx.x;
  double vfrac_accum_re = 0;
  double vfrac_accum_imag = 0;
  GaussianInteger result = {0,0};  

  if(cid < N ){

    for (int rid = 0; rid < d_RNSQNPrimes; rid++){
      GaussianInteger v = a[cid + rid * N];
      
      // Integer part
      GaussianInteger vint;
      vint.re = v.re * d_RNStomega_int[rid] & (d_RNST - 1);
      vint.imag = v.imag * d_RNStomega_int[rid] & (d_RNST - 1);

      // Fractional parts
      uint128_t vtomega_frac128_re = __umul128hi(d_RNStomega_frac[rid], v.re);
      uint128_t vtomega_frac128_imag = __umul128hi(d_RNStomega_frac[rid], v.imag);
      GaussianInteger vfrac;
      vfrac.re = vtomega_frac128_re.hi & (d_RNST - 1);
      vfrac.imag = vtomega_frac128_imag.hi & (d_RNST - 1);

      vfrac_accum_re += vtomega_frac128_re.lo * __drcp_rn((uint64_t)(-1));
      vfrac_accum_imag += vtomega_frac128_imag.lo * __drcp_rn((uint64_t)(-1));
      
      result.re += (vint.re + vfrac.re) & (d_RNST - 1); 
      result.imag += (vint.imag + vfrac.imag) & (d_RNST - 1);
    }

    result.re = (result.re + llrint(vfrac_accum_re)) & (d_RNST - 1);
    result.imag = (result.imag + llrint(vfrac_accum_imag)) & (d_RNST - 1);

    a[cid] = result;

  }

}

__host__  void CUDAEngine::execute_polynomial_simple_scaling(
  GaussianInteger *a,
  Context *ctx){

  const int ADDGRIDXDIM = (
    CUDAEngine::N % ADDBLOCKXDIM == 0?
    CUDAEngine::N / ADDBLOCKXDIM :
    CUDAEngine::N / ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  simple_scaling<<< gridDim, blockDim, 0, ctx->get_stream()>>>(
    a,
    CUDAEngine::N);
  cudaCheckError();
}

__global__ void polynomial_complex_scaling(
  uint64_t *b,
  const GaussianInteger *a_Q,
  const GaussianInteger *a_B,
  const int N,
  const int nresidues_Q,
  const int nresidues_B){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int rid_B = tid / N;
  const int Boffset = rid_B + d_RNSQNPrimes;

  double vfrac_accum_re = 0;
  double vfrac_accum_imag = 0;
  GaussianInteger result = {0,0};  

  if(tid < N * nresidues_B){

    for (int rid_Q = 0; rid_Q < nresidues_Q; rid_Q++){ 

      GaussianInteger v = a_Q[cid + rid_Q * N];

      // Integer part
      GaussianInteger vint = GIMul(v, d_RNSomega_int[rid_Q + rid_B * d_RNSQNPrimes], Boffset);

      // Fractional parts
      uint128_t vfrac128_re = __umul128hi(d_RNSomega_frac[rid_Q + rid_B * d_RNSQNPrimes], v.re);
      uint128_t vfrac128_imag = __umul128hi(d_RNSomega_frac[rid_Q + rid_B * d_RNSQNPrimes], v.imag);
      //vfrac_accum_re   += vfrac128_re.lo   * __drcp_rn((uint64_t)(-1));
      //vfrac_accum_imag += vfrac128_imag.lo * __drcp_rn((uint64_t)(-1));
      vfrac_accum_re   = __fma_rn(vfrac128_re.lo,   __drcp_rn((uint64_t)(-1)), vfrac_accum_re);
      vfrac_accum_imag = __fma_rn(vfrac128_imag.lo, __drcp_rn((uint64_t)(-1)), vfrac_accum_imag);

      result = GIAdd(
        result,
        GIAdd(vint, (GaussianInteger){vfrac128_re.hi,vfrac128_imag.hi}, Boffset),
        Boffset);
    }

    result = GIAdd( result, (GaussianInteger){llrint(vfrac_accum_re), llrint(vfrac_accum_imag)}, Boffset);

    // Multiply lambda
    GaussianInteger xplbd = GIMul(
      a_B[cid + rid_B * N],
      d_RNSlambda[rid_B],
      Boffset);
     
    result = GIAdd(
      result,
      xplbd,
      Boffset);

    b[cid + rid_B * N * 2] = result.re;
    b[(cid + N)  + rid_B * N * 2] = result.imag;

  }

}

__host__  void CUDAEngine::execute_polynomial_complex_scaling(
    uint64_t *b,
    const GaussianInteger *a_Q, // \in base Q
    const GaussianInteger *a_B, // \in base B
    Context *ctx){

  const int nresidues_Q = get_n_residues(QBase);
  const int nresidues_B = get_n_residues(BBase);

  const int size = nresidues_B * CUDAEngine::N;

  assert(size > 0);
  const int ADDGRIDXDIM = (
    size%ADDBLOCKXDIM == 0?
    size/ADDBLOCKXDIM :
    size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  polynomial_complex_scaling<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b,
    a_Q,
    a_B,
    CUDAEngine::N,
    nresidues_Q,
    nresidues_B);
  cudaCheckError();
}

/////////////////////////
// CRT basis extension //
/////////////////////////

/**
 * @brief       Runs for each residue of Q and multiply its coefficients by [q/qi^-1]_qi
 * 
 * @param b           [description]
 * @param a           [description]
 * @param N           [description]
 * @param nresidues_Q [description]
 */
__global__ void polynomial_basis_ext_Q_aux(
  uint64_t *b,
  GaussianInteger *a,
  const int N,
  const int nresidues_Q){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int rid = tid / N; 

  if(tid < N * nresidues_Q ){
    GaussianInteger result = GIMul(
      a[cid + rid * N],
      d_RNSInvModQqi[rid],
      rid);

    b[cid       + rid * N * 2] = result.re;
    b[(cid + N) + rid * N * 2] = result.imag;

  }
}

/**
 * Computes the polynomial v
 * 
 * @param aux         [description]
 * @param v           [description]
 * @param N           [description]
 * @param nresidues_Q [description]
 */
__global__ void polynomial_basis_ext_Q_compute_v(
  uint64_t *v,
  uint64_t *aux, // From polynomial_basis_ext_Q_aux
  const int N,
  const int nresidues_Q){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  double value = 0;

  if(tid < N ){

    // Compute for each coefficient
    for (int rid = 0; rid < nresidues_Q; rid++)
      value = __fma_rd ((double)aux[cid + rid * N] , __drcp_rd((double)d_RNSCoprimes[rid]), value);
    
      // Round down or to the nearest if sufficiently close
      double diff = abs(value-llrint(value));
      v[tid] = (diff <= FPERROR ? llrint(value) : floor(value));
    }
    
  }

// }
// __global__ void polynomial_basis_ext_Q_compute_v(
//   uint64_t *v,
//   uint64_t *aux, // From polynomial_basis_ext_Q_aux
//   const int N,
//   const int nresidues_Q){

//   const int tid = threadIdx.x + blockDim.x * blockIdx.x;
//   const int cid = tid % N; 
//   dbldbl value_dd = make_dbldbl(0.0, 0.0);

//   if(tid < N ){

//     // Compute for each coefficient
//     for (int rid = 0; rid < nresidues_Q; rid++){
//       dbldbl a = make_dbldbl((double)aux[cid + rid * N], 0.0);
//       dbldbl b = make_dbldbl((double)d_RNSCoprimes[rid], 0.0);
//       value_dd = add_dbldbl(value_dd, div_dbldbl(a, b));
//     }

//     double value = get_dbldbl_head(value_dd) + get_dbldbl_tail(value_dd);

//     // Round down or to the nearest if sufficiently close
//     double diff = abs(value-llrint(value));
//     value = (diff <= FPERROR ? llrint(value) : floor(value));

//     v[tid] = value;
//   }

// }

__global__ void polynomial_basis_ext_Q_to_B(
  GaussianInteger *b,
  uint64_t *aux, // Outcome from polynomial_basis_ext_Q_aux
  uint64_t *v,
  const int N,
  const int nresidues_Q,
  const int nresidues_B){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % (N << 1); 
  const int rid_B = tid / (N << 1); 
  const int offset = d_RNSQNPrimes + rid_B;

  if(tid < 2 * N * nresidues_B ){

    uint64_t value = 0;
    for (int rid_Q = 0; rid_Q < nresidues_Q; rid_Q++)
      value = addmod(
        value,
        mulmod(
          aux[cid + rid_Q * N * 2],
          d_RNSQqibi[rid_Q + rid_B * d_RNSQNPrimes],
          offset
        ), 
        offset);

    value = submod(
      value,
      mulmod(v[cid], d_RNSQbi[rid_B], offset),
      offset
      );

    if(cid < N)
      b[cid + rid_B * N].re = value;
    else
      b[cid % N + rid_B * N].imag = value;
  }

}

__host__  void CUDAEngine::execute_polynomial_basis_ext_Q_to_B(
    GaussianInteger *a,
    Context *ctx){

  const int nresidues_Q = get_n_residues(QBase);
  const int nresidues_B = get_n_residues(BBase);


  const int ADDGRIDXDIM   = round_up_blocks(nresidues_Q * CUDAEngine::N * 2, ADDBLOCKXDIM);
  const int ADDGRIDXDIM_v = round_up_blocks(CUDAEngine::N * 2, ADDBLOCKXDIM);
  const int ADDGRIDXDIM_B = round_up_blocks(nresidues_B * CUDAEngine::N * 2, ADDBLOCKXDIM);

  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 gridDim_v(ADDGRIDXDIM_v);
  const dim3 gridDim_B(ADDGRIDXDIM_B);

  const dim3 blockDim(ADDBLOCKXDIM);

  // Computes v
  polynomial_basis_ext_Q_aux<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    ctx->d_aux,
    a,
    CUDAEngine::N,
    nresidues_Q);
  cudaCheckError();

  polynomial_basis_ext_Q_compute_v<<< gridDim_v, blockDim, 0, ctx->get_stream() >>>(
    ctx->d_v,
    ctx->d_aux,
    CUDAEngine::N * 2,
    nresidues_Q);
  cudaCheckError();

  // Computes the extension for each coprime of B
  polynomial_basis_ext_Q_to_B<<< gridDim_B, blockDim, 0, ctx->get_stream() >>>(
    a,
    ctx->d_aux,
    ctx->d_v,
    CUDAEngine::N,
    nresidues_Q,
    nresidues_B);
  cudaCheckError();
}

/**
 * Runs for each residue of B and multiply its coefficients by [b/bi^-1]_bi
 *
 * @param      b     [description]
 * @param      a     [description]
 * @param      N     [description]
 * @param      size  [description]
 */
__global__ void polynomial_basis_ext_B_aux(
  uint64_t *b,
  uint64_t *a,
  const int N,
  const int NResidues){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N; 
  const int offset_B = d_RNSQNPrimes;

  if(tid < N * NResidues )
    b[tid] = mulmod(
      a[tid],
      d_RNSInvModBbi[rid],
      rid + offset_B
      );

}

/**
 * Computes the polynomial v
 * 
 * @param aux         [description]
 * @param v           [description]
 * @param N           [description]
 * @param nresidues_B [description]
 */
__global__ void polynomial_basis_ext_B_compute_v( 
  uint64_t *aux,
  uint64_t *v,
  const int N,
  const int nresidues_B){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int offset_B = d_RNSQNPrimes;
  double value = 0;

  if(tid < N ){
    // Compute for each coefficient
    for (int rid = 0; rid < nresidues_B; rid++)
      value += aux[cid + rid * N] / (double)d_RNSCoprimes[rid + offset_B];

    v[tid] = (abs(value-llrint(value)) <= FPERROR? llrint(value) : floor(value));
  }

}

__global__ void polynomial_basis_ext_B_to_Q(
  GaussianInteger *b,
  uint64_t *aux,
  uint64_t *v,
  const int N,
  const int nresidues_Q,
  const int nresidues_B){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int rid_Q = tid / N; 

  if(tid < N * nresidues_Q ){

    uint64_t value = 0;
    
    // Step 1
    for (int rid_B = 0; rid_B < nresidues_B; rid_B++)
      value = addmod(
        value,
        mulmod(
          aux[cid + rid_B*N],
          d_RNSBbiqi[rid_B + rid_Q*d_RNSBNPrimes],
          rid_Q
        ),
        rid_Q
        );

    // Step 2
    value = submod(
      value,
      mulmod(
        v[cid],
        d_RNSBqi[rid_Q],
        rid_Q
        ),
      rid_Q
      );
    if(cid < (N >> 1))
      b[(cid % (N >> 1)) + rid_Q * (N >> 1)].re = value;
    else
      b[(cid % (N >> 1)) + rid_Q * (N >> 1)].imag = value;
  }

}

// todo: We do not need two declarations
__host__ void CUDAEngine::execute_polynomial_basis_ext_B_to_Q(
    GaussianInteger *b,
    uint64_t *a,
    Context *ctx){

  const int nresidues_Q = get_n_residues(QBase);
  const int nresidues_B = get_n_residues(BBase);

  const int ADDGRIDXDIM_v = round_up_blocks(CUDAEngine::N * 2, ADDBLOCKXDIM);
  const int ADDGRIDXDIM_Q = round_up_blocks(nresidues_Q * CUDAEngine::N * 2, ADDBLOCKXDIM);
  const int ADDGRIDXDIM_B = round_up_blocks(nresidues_B * CUDAEngine::N * 2, ADDBLOCKXDIM);

  const dim3 gridDim_Q(ADDGRIDXDIM_Q);
  const dim3 gridDim_v(ADDGRIDXDIM_v);
  const dim3 gridDim_B(ADDGRIDXDIM_B);

  const dim3 blockDim(ADDBLOCKXDIM);

  // Computes v
  polynomial_basis_ext_B_aux<<< gridDim_B, blockDim, 0, ctx->get_stream() >>>(
    ctx->d_aux,
    a,
    CUDAEngine::N * 2,
    nresidues_B );
  cudaCheckError();

  polynomial_basis_ext_B_compute_v<<< gridDim_v, blockDim, 0, ctx->get_stream() >>>(
    ctx->d_aux,
    ctx->d_v,
    CUDAEngine::N * 2,
    nresidues_B);
  cudaCheckError();

  // Computes the extension for each coprime of Q
  polynomial_basis_ext_B_to_Q<<< gridDim_Q, blockDim, 0, ctx->get_stream() >>>(
    b,
    ctx->d_aux,
    ctx->d_v,
    CUDAEngine::N * 2,
    nresidues_Q,
    nresidues_B);
  cudaCheckError();
}

#endif

#ifdef CKKS_ENGINE_MODE

__global__ void fast_conv_B_to_Q(
  GaussianInteger *b,
  GaussianInteger *a,
  const int N,
  const int level,
  uint64_t *RNSBbiqi){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int rid_Q = tid / N; 
  GaussianInteger val = (GaussianInteger){0, 0};

  if(tid < N * (level + 1)){
    for(int rid_B = 0; rid_B < d_RNSBNPrimes; rid_B++){
      GaussianInteger aux1 = 
          mulint_dgt(
            a[cid + (rid_B + d_RNSQNPrimes) * N],
            d_RNSInvModBbi[rid_B],
            rid_B + d_RNSQNPrimes);
      GaussianInteger aux2 = 
        mulint_dgt(aux1,
          RNSBbiqi[rid_B + rid_Q*d_RNSBNPrimes],
          rid_Q);
      val = GIAdd(
        val,
        aux2,
        rid_Q);
    }

    mulint_dgt(
      &b[cid + rid_Q * N], // Output
      GISub(
        a[cid + rid_Q * N],
        val,
        rid_Q),
      d_RNSInvModBqi[rid_Q],
      rid_Q);

  }
}

__host__ void CUDAEngine::execute_approx_modulus_reduction(
  Context *ctx,
  GaussianInteger *a,
  GaussianInteger *b,
  int level){

  const int N = CUDAEngine::N;
  const int NResiduesQ = level + 1;
  const int size = N * NResiduesQ;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

    fast_conv_B_to_Q<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
      a,
      b,
      N,
      level,
      d_RNSBbiqi) ;
    cudaCheckError();

}


__global__ void ckks_rescale(
  GaussianInteger *a,
  GaussianInteger *b,
  const int level,
  const int N,
  const int nresidues_Q,
  uint64_t *RNSInvModqlqi){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int rid = tid / N; 

  if(tid < N * level){
    uint64_t s = RNSInvModqlqi[rid + level * d_RNSQNPrimes];

    mulint_dgt(
      &a[cid + rid * N], // Output
      GISub(
        a[cid + rid * N],
        a[cid + level * N], // Refactor this to use shared memory!
        rid),
      s,
      rid);
    
    mulint_dgt(
      &b[cid + rid * N], // Output
      GISub(
        b[cid + rid * N],
        b[cid + level * N], // Refactor this to use shared memory!
        rid),
      s,
      rid);
  }
}

__host__ void CUDAEngine::execute_ckks_rescale(
  GaussianInteger *a,
  GaussianInteger *b,
  const int level,
  Context *ctx){

  const int N = CUDAEngine::N;
  const int NResiduesQ = level + 1;
  const int size = N * NResiduesQ;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

    ckks_rescale<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
      a,
      b,
      level,
      N,
      NResiduesQ,
      d_RNSInvModqlqi);
    cudaCheckError();

}


__global__ void fast_conv_Q_to_B(
  GaussianInteger *a,
  const int N,
  const int level,
  uint64_t *RNSQqibi,
  uint64_t *RNSInvModQqi){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int cid = tid % N; 
  const int rid_B = tid / N; 
  GaussianInteger val = (GaussianInteger){0, 0};

  if(tid < N * d_RNSBNPrimes){
    for(int rid_Q = 0; rid_Q <= level; rid_Q++){
      GaussianInteger aux = 
          mulint_dgt(
            a[cid + rid_Q * N],
            RNSInvModQqi[rid_Q + level * d_RNSQNPrimes],
            rid_Q);
      aux.re %= d_RNSCoprimes[rid_B   + d_RNSQNPrimes];
      aux.imag %= d_RNSCoprimes[rid_B + d_RNSQNPrimes];
      val = GIAdd(
        val,
        mulint_dgt(aux,
          RNSQqibi[level * d_RNSQNPrimes * d_RNSBNPrimes + rid_Q * d_RNSBNPrimes + rid_B], //  [level][rid_Q][rid_B],
          rid_B + d_RNSQNPrimes),
        rid_B + d_RNSQNPrimes);
    }
    a[cid + (rid_B + d_RNSQNPrimes) * N] = val;
  }
}


__host__ void CUDAEngine::execute_approx_modulus_raising(
  Context *ctx,
  GaussianInteger *a,
  int level){

  const int N = CUDAEngine::N;
  const int NResiduesB = get_n_residues(BBase);
  const int size = N * NResiduesB;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

    fast_conv_Q_to_B<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
      a,
      N,
      level,
      d_RNSQqibi,
      d_RNSInvModQqi) ;
    cudaCheckError();
}

#endif

#ifdef BFV_ENGINE_MODE
/**
 * @brief     Apply \f$\xi()\f$.
 * 
 * @param[out] b         output
 * @param[in]  a         input
 * @param[in]  N         Degree of each residue
 * @param[in]  NResidues Quantity of residues
 */
__global__ void xi_rns_rid(
  GaussianInteger **b,
  GaussianInteger *a,
  const int N,
  const int NResidues){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  const int cid = tid % N;

  if(tid < N * NResidues)
    mulint_dgt(
      &b[rid][cid + rid*N],
      a[cid + rid*N],
      d_RNSInvModQqi[rid],
      rid
      );
  
}

__host__ void CUDAEngine::execute_xi_rns(
  GaussianInteger **b,
  GaussianInteger *a,
  Context *ctx){

  // const int k = NResidues;
  const int N = CUDAEngine::N;
  const int NResidues = get_n_residues(QBase);
  const int size = N * NResidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  // #pragma omp parallel for
  // for(int i = 0; i < k; i ++){
    xi_rns_rid<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
      b,
      a,
      N,
      NResidues) ;
    cudaCheckError();
  // }

}

/**
 * @brief     Apply \f$\rho()\f$ as in the BFV.
 * 
 * @param[out] b         output
 * @param[in]  a         input
 * @param[in]  N         Degree of each residue
 * @param[in]  NResidues Quantity of residues
 */
__global__ void rho_bfv_rns_rid(
  GaussianInteger **b,
  const GaussianInteger *a,
  const int N,
  const int NResidues){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  const int cid = tid % N;

  if(tid < N * NResidues)
      mulint_dgt(
        &b[rid][cid + rid * N],
        a[cid + rid * N],
        d_RNSQqi[rid],
        rid
        );
}

__host__ void CUDAEngine::execute_rho_bfv_rns(
  GaussianInteger **b,
  GaussianInteger *a,
  Context *ctx){

  const int N = CUDAEngine::N;
  const int NResidues = get_n_residues(QBase);
  const int size = N * NResidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  rho_bfv_rns_rid<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b,
    a,
    N,
    NResidues);
  cudaCheckError();

}
#endif

#ifdef CKKS_ENGINE_MODE
/**
 * @brief     Apply \f$\rho()\f$ as in the CKKS.
 * 
 * @param[out] b         output
 * @param[in]  a         input
 * @param[in]  N         Degree of each residue
 * @param[in]  NResidues Quantity of residues
 */
__global__ void rho_ckks_rns_rid(
  GaussianInteger *b,
  const GaussianInteger *a,
  const int N,
  const int NResidues){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  const int cid = tid % N;

  if(tid < N * NResidues)
      mulint_dgt(
        &b[cid + rid * N],
        a[cid + rid * N],
        d_RNSBqi[rid],
        rid
      );
}

__host__ void CUDAEngine::execute_rho_ckks_rns(
  GaussianInteger *b,
  GaussianInteger *a,
  Context *ctx){

  const int N = CUDAEngine::N;
  const int NResidues = get_n_residues(QBase);
  const int size = N * NResidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);


  rho_ckks_rns_rid<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b,
    a,
    N,
    NResidues);
  cudaCheckError();

}

/**
 * @brief     Apply \f$\xi()\f$ as in the CKKS.
 * 
 * @param[out] b         output
 * @param[in]  a         input
 * @param[in]  N         Degree of each residue
 * @param[in]  NResidues Quantity of residues
 */
__global__ void xi_ckks_rns_rid(
  GaussianInteger *b,
  const GaussianInteger *a,
  const int N,
  const int NResidues){

  const int tid = threadIdx.x + blockDim.x * blockIdx.x;
  const int rid = tid / N;
  const int cid = tid % N;

  if(tid < N * NResidues)
      mulint_dgt(
        &b[cid + rid * N],
        a[cid + rid * N],
        d_RNSInvModBqi[rid],
        rid
        );
}

__host__ void CUDAEngine::execute_xi_ckks_rns(
  GaussianInteger *b,
  GaussianInteger *a,
  Context *ctx){

  const int N = CUDAEngine::N;
  const int NResidues = get_n_residues(QBase);
  const int size = N * NResidues;
  const int ADDGRIDXDIM = (size%ADDBLOCKXDIM == 0? size/ADDBLOCKXDIM : size/ADDBLOCKXDIM + 1);
  const dim3 gridDim(ADDGRIDXDIM);
  const dim3 blockDim(ADDBLOCKXDIM);

  xi_ckks_rns_rid<<< gridDim, blockDim, 0, ctx->get_stream() >>>(
    b,
    a,
    N,
    NResidues);
  cudaCheckError();

}

#endif

//////////////////////////
//                      //
//   Init / Destroy RNS //
//                      //
//////////////////////////

#ifdef CKKS_ENGINE_MODE
void fill_ckks_coprimes_bucket(int k, int kl, int prec){
  switch(prec){

    case 45:
    COPRIMES_BUCKET[0] = COPRIMES_63_BUCKET[0]; // The first element
    assert(COPRIMES_45_BUCKET_SIZE >= (k + kl - 1));
    memcpy(&COPRIMES_BUCKET[1], &COPRIMES_45_BUCKET[0], (k-1) * sizeof(uint64_t)); // Q Base
    memcpy(&COPRIMES_BUCKET[k], &COPRIMES_45_BUCKET[k-1], kl    * sizeof(uint64_t)); // B Base
    break;

    case 48:
    COPRIMES_BUCKET[0] = COPRIMES_63_BUCKET[0]; // The first element
    assert(COPRIMES_55_BUCKET_SIZE >= (k + kl -1));
    memcpy(&COPRIMES_BUCKET[1], &COPRIMES_48_BUCKET[0], (k-1) * sizeof(uint64_t)); // Q Base
    memcpy(&COPRIMES_BUCKET[k], &COPRIMES_55_BUCKET[k-1], kl    * sizeof(uint64_t)); // B Base
    break;

    case 52:
    COPRIMES_BUCKET[0] = COPRIMES_63_BUCKET[0]; // The first element
    assert(COPRIMES_55_BUCKET_SIZE >= (k + kl -1));
    memcpy(&COPRIMES_BUCKET[1], &COPRIMES_52_BUCKET[0], (k-1) * sizeof(uint64_t)); // Q Base
    memcpy(&COPRIMES_BUCKET[k], &COPRIMES_55_BUCKET[k-1], kl    * sizeof(uint64_t)); // B Base
    break;

    case 55:
    COPRIMES_BUCKET[0] = COPRIMES_63_BUCKET[0]; // The first element
    // COPRIMES_BUCKET_SIZE = COPRIMES_55_BUCKET_SIZE + 1;
    assert(COPRIMES_55_BUCKET_SIZE >= (k + kl -1));
    memcpy(&COPRIMES_BUCKET[1], &COPRIMES_55_BUCKET[0], (k-1) * sizeof(uint64_t)); // Q Base
    memcpy(&COPRIMES_BUCKET[k], &COPRIMES_55_BUCKET[k-1], kl    * sizeof(uint64_t)); // B Base
    break;

    default:
    throw std::runtime_error("cuPoly can't handle this precision.");
  }
  COPRIMES_BUCKET_SIZE = (k+kl);
}
#endif

// Multiply all elements of q with index j such that a <= j < b
template <class T>
ZZ multiply_subset(std::vector<T> q, int a, int b){
  ZZ accum = to_ZZ(1);
  for(int i = a; i < b; i++)
    accum *= to_ZZ(q[i]);
  return accum;
}

// Multiply all elements of q with index j such that a <= j < b and j != c
template <class T>
ZZ multiply_subset_except(std::vector<T> q, int a, int b, int c){
  ZZ accum = multiply_subset(q, a, b);
  return accum / q[c];
}

// Selects a set of coprimes to be used as the main and secondary bases
__host__ void CUDAEngine::gen_rns_primes(
  unsigned int k,       // length of |q|
  unsigned int kl){     // length of |b|  

  // Loads to COPRIMES_BUCKET all the coprimes that may be used 
  #ifdef BFV_ENGINE_MODE
  COPRIMES_BUCKET_SIZE = min(MAX_COPRIMES, COPRIMES_63_BUCKET_SIZE);
  memcpy(&COPRIMES_BUCKET[0], &COPRIMES_63_BUCKET[0], COPRIMES_BUCKET_SIZE * sizeof(uint64_t));
  #else
  fill_ckks_coprimes_bucket(k, kl, scalingfactor);
  #endif

  // Select the main base
  RNSProduct = to_ZZ(1);
  ostringstream os;
  os << "Q base: ";
  for(unsigned int i = 0; RNSPrimes.size() < k; i++){
    assert(
      RNSPrimes.size() < MAX_COPRIMES_IN_A_BASE &&
      RNSPrimes.size() < COPRIMES_BUCKET_SIZE);
    RNSPrimes.push_back(COPRIMES_BUCKET[RNSPrimes.size()]);
    assert(RNSPrimes.back() > 0);
    RNSProduct *= RNSPrimes.back();
    
    os << RNSPrimes.back() << " ";
  }
  os << std::endl;

  // Select the secondary base
  RNSBProduct = to_ZZ(1);
  os << "B base: ";
  for(unsigned int i = 0; i < kl; i++){ // |B| == |q| + 1 
    assert(
      RNSBPrimes.size() < MAX_COPRIMES_IN_A_BASE &&
      RNSPrimes.size() + RNSBPrimes.size() < COPRIMES_BUCKET_SIZE);
    RNSBPrimes.push_back(COPRIMES_BUCKET[RNSPrimes.size() + RNSBPrimes.size()]);
    assert(RNSBPrimes.back() > 0);
    RNSBProduct *= RNSBPrimes.back();

    os << RNSBPrimes.back() << " ";
  }
  os << std::endl;
  
  // Copy to device
  assert(RNSPrimes.size() < MAX_COPRIMES_IN_A_BASE && RNSPrimes.size() > 0);
  assert(RNSBPrimes.size() < MAX_COPRIMES_IN_A_BASE && RNSBPrimes.size() > 0);
  assert(RNSPrimes.size() + RNSBPrimes.size() + 1 < MAX_COPRIMES);
  
  cudaMemcpyToSymbol(
    d_RNSCoprimes,
    &RNSPrimes[0],
    RNSPrimes.size() * sizeof(uint64_t),
    0);
  cudaCheckError();

  cudaMemcpyToSymbol(
    d_RNSCoprimes,
    &RNSBPrimes[0],
    RNSBPrimes.size() * sizeof(uint64_t),
    RNSPrimes.size() * sizeof(uint64_t)); // offset
  cudaCheckError();

  int vsize = RNSPrimes.size(); // Do we need this
  cudaMemcpyToSymbol(
    d_RNSQNPrimes,
    &vsize,
    sizeof(int));
    cudaCheckError();
    
    vsize = RNSBPrimes.size(); // Do we need this
    cudaMemcpyToSymbol(
      d_RNSBNPrimes,
      &vsize,
    sizeof(int));
  cudaCheckError();

  ///////////////////////
  cudaDeviceSynchronize();
  cudaCheckError();

  os << "q: " << RNSProduct << " ( " << NumBits(RNSProduct) << " bits )" << std::endl;
  os << "B: " << RNSBProduct << " ( " << NumBits(RNSBProduct) << " bits )" << std::endl;
  os << "|q| == " << RNSPrimes.size() << std::endl;
  os << "|B| == " << RNSBPrimes.size() << std::endl;
  Logger::getInstance()->log_info(os.str().c_str());

}

__host__ void CUDAEngine::precompute(){
  const int k = RNSPrimes.size();
  const int kl = RNSBPrimes.size();
  
  std::vector<uint64_t> Qqi;
  std::vector<uint64_t> Qbi;
  std::vector<uint64_t> Bqi;
  std::vector<uint64_t> Qqibi;
  std::vector<uint64_t> InvBbi;
  std::vector<uint64_t> Bbiqi;
  std::vector<uint64_t> InvModBqi;

  #ifdef BFV_ENGINE_MODE
  // HPS
  uint64_t  omega_int  [MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE];
  uint128_t omega_frac [MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE];
  uint64_t  lambda     [MAX_COPRIMES_IN_A_BASE];
  uint64_t  tomega_int [MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE];
  uint128_t tomega_frac[MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE];
  #else
  std::vector<uint64_t> hatQQi;
  uint64_t Invqlqi [MAX_COPRIMES_IN_A_BASE][MAX_COPRIMES_IN_A_BASE];
  uint64_t InvQqi  [MAX_COPRIMES_IN_A_BASE][MAX_COPRIMES_IN_A_BASE];
  #endif

  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  // Compute Q/qi mod qi and it's inverse
  for(auto p : RNSPrimes){
    ZZ pi = to_ZZ(p);

    RNSMpi.push_back(RNSProduct/pi);
    
    // rho_bfv_rns_rid
    Qqi.push_back(conv<uint64_t>(RNSMpi.back() % pi));
    
    // HPS basis extension  and fast_conv_Q_to_B
    RNSInvMpi.push_back(conv<uint64_t>(NTL::InvMod(to_ZZ(Qqi.back()), pi)));

    //  B % qi
    // rho_ckks_rns_rid and polynomial_basis_ext_B_to_Q
    Bqi.push_back(conv<uint64_t>(RNSBProduct % pi));

    //  (B % qi)^-1 mod qi
    // fast_conv_B_to_Q and xi_ckks_rns_rid
    InvModBqi.push_back(
      conv<uint64_t>(NTL::InvMod(RNSBProduct % pi, pi))
    );
  }

  for(auto p : RNSBPrimes){
    ZZ bi = to_ZZ(p);

    // Compute Q mod bi
    Qbi.push_back(conv<uint64_t>(RNSProduct % bi));
  
    // Compute B/bi and it's inverse
    InvBbi.push_back(conv<uint64_t>(NTL::InvMod(RNSBProduct / bi % bi, bi)));
  }

  // HPS basis extension from B to Q and fast_conv_B_to_Q
  for(auto qi : RNSPrimes)
    for(auto bi : RNSBPrimes)
      Bbiqi.push_back(conv<uint64_t>(RNSBProduct / to_ZZ(bi) % to_ZZ(qi)));    

  
  #ifdef BFV_ENGINE_MODE
  // HPS basis extension Q to B and fast_conv_Q_to_B
  for(auto bi : RNSBPrimes)
    for(auto qi : RNSPrimes)
      Qqibi.push_back(RNSProduct / qi % bi);
  #else
    for(int l = 0; l < k; l++){ // Level
      for(int i = 0; i < k; i++){ // Q residue
        ZZ qi = to_ZZ(RNSPrimes[i]);
        if( i < k )
          InvQqi[l][i] = conv<uint64_t>(
            InvMod(
              multiply_subset_except(RNSPrimes, 0, l+1, i) % qi,
              qi));
        for(int j = 0; j < kl; j++){ // B residue
          ZZ pi = to_ZZ(RNSBPrimes[j]);
  
          Qqibi.push_back(
            conv<uint64_t>(
              multiply_subset_except(RNSPrimes, 0, l+1, i) % pi)
            );
        }
      }
    }
  #endif

  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  // CKKS' 
  #ifdef CKKS_ENGINE_MODE
    // alpha = (L + 1) /dnum
    int alpha = k / CUDAEngine::dnum;
    std::vector<ZZ> Qj(CUDAEngine::dnum, to_ZZ(1));
    for(int j = 0; j < CUDAEngine::dnum; j++)
      for(int i = j * alpha; i < (j + 1) * alpha; i++)
        Qj[j] *= RNSPrimes[i];

    std::vector<ZZ> hatQj(CUDAEngine::dnum, to_ZZ(1));
    for(int j = 0; j < CUDAEngine::dnum; j++)
      for(int i = 0; i < CUDAEngine::dnum; i++)
        if( i != j)
          hatQj[j] *= Qj[i];

    for(int j = 0; j < CUDAEngine::dnum; j++)
      for(int i = 0; i < k; i++)
        hatQQi.push_back(conv<uint64_t>(hatQj[j] % RNSPrimes[i]));

    // Compute ql^{-1} mod qi
    for(unsigned int l = 0; l < RNSPrimes.size();l++)
      for(unsigned int i = 0; i < RNSPrimes.size();i++)
        if(l != i)
          Invqlqi[l][i] = conv<uint64_t>(
            NTL::InvMod(to_ZZ(RNSPrimes[l]) % to_ZZ(RNSPrimes[i]),
            to_ZZ(RNSPrimes[i])));
  #endif
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  RR::SetPrecision(512);

  #ifdef BFV_ENGINE_MODE
  int idx = 0;
  for(unsigned int i = 0; i < RNSBPrimes.size();i++){
    lambda[i] = conv<uint64_t>(
      to_ZZ(CUDAEngine::t) * 
      NTL::InvMod(
        RNSBProduct * RNSProduct / RNSBPrimes[i] % RNSBPrimes[i],
        RNSBPrimes[i]) * to_ZZ(RNSBProduct / RNSBPrimes[i]) % RNSBPrimes[i]);

    for(unsigned int j = 0; j < RNSPrimes.size();j++){
      ZZ pj = to_ZZ(RNSPrimes[j]);
      ZZ aux = to_ZZ(CUDAEngine::t) * 
        NTL::InvMod((RNSProduct * RNSBProduct / pj) % pj, pj) * RNSBProduct;

      RR tmp = to_RR(aux) / to_RR(RNSPrimes[j]);
      ZZ tmp_int = NTL::FloorToZZ(tmp);

      // Shift by 128 bits
      ZZ tmp_frac_ZZ = to_ZZ((tmp - to_RR(tmp_int)) * to_RR(to_ZZ(1) << 128));
      uint128_t tmp_frac;
      tmp_frac.lo = conv<uint64_t>(tmp_frac_ZZ);
      tmp_frac.hi = conv<uint64_t>(tmp_frac_ZZ>>64);

      omega_int[idx]  = tmp_int % RNSBPrimes[i];
      omega_frac[idx] = tmp_frac;
      idx++;
    }
  }

  for(unsigned int i = 0; i < RNSPrimes.size(); i++){
    ZZ pi = to_ZZ(RNSPrimes[i]);
    ZZ aux = to_ZZ(CUDAEngine::t) * NTL::InvMod(RNSProduct / pi % pi, pi);

    RR tmp = to_RR(aux) / to_RR(pi);
    ZZ tmp_int = NTL::FloorToZZ(tmp);
    
    // Shift by 128 bits
    ZZ tmp_frac_ZZ = to_ZZ((tmp - to_RR(tmp_int)) * to_RR(to_ZZ(1) << 128));

    uint128_t tmp_frac;
    tmp_frac.lo = conv<uint64_t>(tmp_frac_ZZ);
    tmp_frac.hi = conv<uint64_t>(tmp_frac_ZZ>>64);

    tomega_int[i]  = conv<uint64_t>(tmp_int);
    tomega_frac[i] = tmp_frac;
  }
  #endif


  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  //////////////////////////////////////////////////////////////////// 
  // Linearizes InvQqi
  uint64_t *lin_InvQqi = (uint64_t*) malloc (k * k * sizeof(uint64_t));
  uint64_t *lin_Invqlqi = (uint64_t*) malloc (k * k * sizeof(uint64_t));

  // Copy to device
  
  #ifdef BFV_ENGINE_MODE
  // rho_bfv_rns_rid
  cudaMemcpyToSymbol(
    d_RNSQqi,
    &Qqi[0],
    Qqi.size() * sizeof(uint64_t));
  cudaCheckError();

  assert(Qbi.size() < MAX_COPRIMES_IN_A_BASE);
  // HPS basis extension from Q to B
  cudaMemcpyToSymbol(
    d_RNSQbi,
    &Qbi[0],
    Qbi.size() * sizeof(uint64_t));
  cudaCheckError();

  // HPS basis extension  and fast_conv_Q_to_B
  assert(RNSInvMpi.size() < MAX_COPRIMES_IN_A_BASE);
  cudaMemcpyToSymbol(
    d_RNSInvModQqi,
    &RNSInvMpi[0],
    RNSInvMpi.size() * sizeof(uint64_t));
  cudaCheckError();

  // HPS basis extension Q to B and fast_conv_Q_to_B
  assert(Qqibi.size() < MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE);
  cudaMemcpyToSymbol(
    d_RNSQqibi,
    &Qqibi[0],
    Qqibi.size() * sizeof(uint64_t));
  cudaCheckError();

  /////////////////////////
  // HPS complex scaling
  cudaMemcpyToSymbol(
    d_RNSomega_int,
    omega_int,
    sizeof(omega_int));
  cudaCheckError();

  cudaMemcpyToSymbol(
    d_RNSomega_frac,
    omega_frac,
    sizeof(omega_frac));
  cudaCheckError();

  cudaMemcpyToSymbol(
    d_RNSlambda,
    lambda,
    sizeof(lambda));
  cudaCheckError();

  cudaMemcpyToSymbol(
    d_RNStomega_int,
    tomega_int,
    sizeof(tomega_int));
  cudaCheckError();
 
  cudaMemcpyToSymbol(
    d_RNStomega_frac,
    tomega_frac,
    sizeof(tomega_frac));
  cudaCheckError();
  ///////////////////////
  ///

  assert(Bbiqi.size() < MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE);
  // HPS basis extension from B to Q and fast_conv_B_to_Q
  cudaMemcpyToSymbol(
    d_RNSBbiqi,
    &Bbiqi[0],
    Bbiqi.size() * sizeof(uint64_t));
  cudaCheckError();
  #else
  // CKKS rescale
  cudaMalloc((void**)&d_RNSInvModqlqi, k * k * sizeof(uint64_t));
  cudaCheckError();
  for(int i = 0; i < k; i++)
    for(int j = 0; j < k; j++)
      lin_Invqlqi[j + i * k] = Invqlqi[i][j];
  cudaMemcpyAsync(
    d_RNSInvModqlqi,
    lin_Invqlqi,
    k * k * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError();

  // HPS basis extension  and fast_conv_Q_to_B
  cudaMalloc((void**)&d_RNSInvModQqi, Qqibi.size() * sizeof(uint64_t));
  cudaCheckError();
  for(int i = 0; i < k; i++)
    for(int j = 0; j < k; j++)
      lin_InvQqi[j + i * k] = InvQqi[i][j];
  cudaMemcpyAsync(
    d_RNSInvModQqi,
    lin_InvQqi,
    k * k * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError();

  assert(Qqibi.size() < MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE);
  cudaMalloc((void**)&d_RNSQqibi, Qqibi.size() * sizeof(uint64_t));
  cudaCheckError();

  // HPS basis extension Q to B and fast_conv_Q_to_B
  cudaMemcpyAsync(
    d_RNSQqibi,
    &Qqibi[0],
    Qqibi.size() * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError();

  assert(InvModBqi.size() < MAX_COPRIMES_IN_A_BASE);
  // fast_conv_B_to_Q and xi_ckks_rns_rid
  cudaMemcpyToSymbol(
    d_RNSInvModBqi,
    &InvModBqi[0],
    InvModBqi.size() * sizeof(uint64_t));
  cudaCheckError();

  assert(Bbiqi.size() < MAX_COPRIMES_IN_A_BASE * MAX_COPRIMES_IN_A_BASE);
  // HPS basis extension from B to Q and fast_conv_B_to_Q
  cudaMalloc((void**)& d_RNSBbiqi, Bbiqi.size() * sizeof(uint64_t));
  cudaCheckError();
  cudaMemcpyAsync(
    d_RNSBbiqi,
    &Bbiqi[0],
    Bbiqi.size() * sizeof(uint64_t),
    cudaMemcpyHostToDevice);
  cudaCheckError();
  #endif

  // rho_ckks_rns_rid and polynomial_basis_ext_B_to_Q
  assert(Bqi.size() < MAX_COPRIMES_IN_A_BASE);
  cudaMemcpyToSymbol(
    d_RNSBqi,
    &Bqi[0],
    Bqi.size() * sizeof(uint64_t));
  cudaCheckError();

  assert(InvBbi.size() < MAX_COPRIMES_IN_A_BASE);
  cudaMemcpyToSymbol(
    d_RNSInvModBbi,
    &InvBbi[0],
    InvBbi.size()*sizeof(uint64_t));
  cudaCheckError();

  cudaDeviceSynchronize();
  cudaCheckError();
  free(lin_InvQqi);
}

//////////////////////
//                  //
//   Init / Destroy //
//                  //
//////////////////////

__host__ void CUDAEngine::init(CUDAParams p){
  

    init(p.k, p.kl, p.nphi, p.pt);
}

__host__ bool is_power2(uint64_t t){
  return t && !(t & (t - 1));
}

__host__ void CUDAEngine::init(
  const int k,
  const int kl,
  const int M,
  const uint64_t t){
  ostringstream os;

  #ifdef BFV_ENGINE_MODE
  if(!is_power2(t))
    throw std::runtime_error(std::to_string(t) + " is not a power of 2. This is not supported.");
  
  CUDAEngine::t = t;
  #endif

  // By using the folded encoding for DGT we just need half the degree
  CUDAEngine::N = (M >> 1);

  ////////////////////////
  // Generate RNSPrimes //
  ////////////////////////    
  #ifdef CKKS_ENGINE_MODE
  scalingfactor = t;
  dnum = 1;
  #endif
  gen_rns_primes(k, (kl > 0? kl : k+1));// Generates CRT's primes
  precompute();
  cudaGetSymbolAddress((void**)&CUDAEngine::RNSCoprimes, d_RNSCoprimes);
  cudaCheckError();
  is_init = true;

  ///////////////
  // Greetings //
  ///////////////
  os << "CUDAEngine initialized  = [" << CUDAEngine::N << "]" << std::endl;
  Logger::getInstance()->log_info(os.str().c_str());  
  
  /////////
  // DGT //
  /////////
  DGTEngine::init();    
}

__host__ void CUDAEngine::destroy(){
  if(!is_init)
    return;

  CUDAEngine::RNSPrimes.clear();
  CUDAEngine::RNSMpi.clear();
  CUDAEngine::RNSInvMpi.clear();
  CUDAEngine::RNSBPrimes.clear();

  #ifdef CKKS_ENGINE_MODE
  cudaFree(d_RNSBbiqi);
  cudaCheckError();
  cudaFree(d_RNSInvModQqi);
  cudaCheckError();
  cudaFree(d_RNSQqibi);
  cudaCheckError();
  cudaFree(d_RNShatQQi);
  cudaCheckError();
  #endif
  
  DGTEngine::destroy();

  cudaDeviceSynchronize();
  cudaCheckError();

  is_init = false;
}

/**
 * Return the quantity of residues for a certain base
 */
__host__ int CUDAEngine::get_n_residues(int base){
  switch(base){
    case QBase:
      return RNSPrimes.size();
    case TBase:
      return 1;
    case BBase:
      return RNSBPrimes.size();
    case QBBase:
      return RNSPrimes.size() + RNSBPrimes.size();
    case QTBBase:
      return RNSPrimes.size() + RNSBPrimes.size() + 1;
    default:
      throw std::runtime_error("Unknown base!");
  }
}
