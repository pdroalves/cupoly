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
    
#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <vector>
#include <NTL/ZZ.h>
#include <map>
#include <algorithm>
#include <sstream>
#include <cuPoly/settings.h>
#include <cuPoly/cuda/cudaengine.h>
#include <cuPoly/arithmetic/context.h>
#include <cuPoly/tool/log.h>
#include <cuPoly/cuda/dgt.h>
#include <omp.h>

NTL_CLIENT

/**
 * @brief      Defines the states supported for a poly_t object
 * 
 * There are three possible states.
 */
enum poly_states {
    HOSTSTATE,///< data is up-to-date on the host, and maybe out-of-date on the GPU
    RNSSTATE,///< data is up-to-date on the GPU, and maybe out-of-date on the host
    BOTHSTATE///< memories are sync
};

/**
 * @brief      Defines a polynomial.
 * 
 * It contains:
 * - a boolean indicating if the object was initialized,
 * - a vector of coefficients on the host,
 * - an array of residues on the device,
 * - the status of the object (related to poly_states),
 * - the base that the object lies in (related to settings.h bases enum_t).
 */
struct polynomial {
    /// a boolean indicating if the object was initialized,
	bool init = false; 
	/// a vector of coefficients on the host,
    std::vector<ZZ> coefs;
	/// an array of residues on the device,
    GaussianInteger *d_coefs = NULL;
    /// the status of the object as an element of #poly_states,
	int status = BOTHSTATE;
    /// the base that the object lies as an element of #bases
    int base = QBase;
} typedef poly_t;

/**
 * @brief      Return the size in bytes required to store
 *             all the residues of a certain base
 *             
 * @param base the base
 *
 * @return     the size in bytes required to store all the residues of a certain base
 */
__host__ size_t poly_get_residues_size(int base);

/**
 * @brief Initializes a poly_t object.
 * 
 * @param[in] ctx the context
 * @param[in, out] a The object
 * @param[in] base the base (the default value is QBase)
 */
__host__ void poly_init(Context *ctx, poly_t *a, int base = QBase);

/**
 * @brief Releases the memory related to a poly_t object
 * 
 * @param[in] a the object
 */
__host__ void poly_free(Context *ctx, poly_t *a);

/**
 * @brief Clear the memory related to a poly_t object without deallocating it.
 * 
 * Write 0's to all arrays.
 * 
 * @param ctx The context
 * @param a The object
 * 
 */
__host__ void poly_clear(Context *ctx, poly_t *a);

/**
 * @brief  Copy the coefficients from host's standard memory to device's global memory.
 * 
 * Do all the intermediate operations required (for instance, RNS and DGT). Skip if the
 * device memory already contains the most recent version of the coefficients.
 * 
 * @param ctx   The context
 * @param a     the object
 */
__host__ void poly_copy_to_device(Context *ctx, poly_t *a);

/**
 * @brief  Copy the coefficients from device's global memory to host's standard memory.
 * 
 * Do all the intermediate operations required (for instance, IDGT and IRNS). Skip if the
 * host memory already contains the most recent version of the coefficients.
 * 
 * @param ctx   The context
 * @param a     the object
 */
__host__ void poly_copy_to_host(Context *ctx, poly_t *a);

/**
 * @brief       Copy all coefficients from "a" to "b"
 * 
 * This method copy coefficient privileging the RNSSTATE, what means that if
 * the object status is RNSSTATE or BOTHSTATE it will execute the copy within device's
 * memory and change the status to RNSSTATE. Otherwise it will execute the copy on the 
 * host by **will not** call poly_copy_to_device().
 * 
 * @param ctx   The context
 * @param b     Destiny
 * @param a     Source
 */
__host__ void poly_copy(
    Context *ctx, 
    poly_t *b,
    poly_t *a);

/**
 * @brief Computes the residues of a polynomial.
 * 
 * @param ctx   The context
 * @param  a    the object
 */
GaussianInteger* poly_crt(Context *ctx, poly_t *a);

/**
 * @brief      Receives an array of residues and interpolate to recover the coefficients.

 * @param ctx   The context
 * @param  a    the object
 */
__host__ void poly_icrt(Context *ctx, poly_t *a);

/**
 * @brief       Returns the polynomial degree.
 * 
 * @param ctx   The context
 * @param  a    the object
 */
__host__ int poly_get_deg(Context *ctx, poly_t *a);

/**
 * @brief       Executes a polynomial addition.
 * 
 * Computes \f$c = a + b\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c    Outcome
 * @param[in]  a     First operand
 * @param[in]  b     Second operand
 */
__host__ void poly_add(Context *ctx, poly_t *c, poly_t *a, poly_t *b);

/**
 * @brief Compute two polynomial additions.
 *
 * Computes \f$c_1 = a_1 + b_1\f$ and \f$c_2 = a_2 + b_2\f$
 * 
 * @param[in] ctx         The context
 * @param[out] c1         Outcome of the first addition
 * @param[in]  a1         First operator of the first addition
 * @param[in]  b1         Second operator of the first addition
 * @param[out] c2         Outcome of the second addition
 * @param[in]  a2         First operator of the second addition
 * @param[in]  b2         Second operator of the second addition
 */
__host__ void poly_double_add(Context *ctx, poly_t *c1, poly_t *a1, poly_t *b1, poly_t *c2, poly_t *a2, poly_t *b2);

/**
 * @brief       Executes a polynomial subtraction.
 * 
 * Computes \f$c = a - b\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c    Outcome
 * @param[in]  a     First operand
 * @param[in]  b     Second operand
 */
__host__ void poly_sub(Context *ctx, poly_t *c, poly_t *a, poly_t *b);

/**
 * @brief       Executes a polynomial multiplication.
 * 
 * Computes \f$c = a \times b\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c    Outcome
 * @param[in]  a     First operand
 * @param[in]  b     Second operand
 */
__host__ void poly_mul(Context *ctx, poly_t *c, poly_t *a, poly_t *b);


/**
 * @brief       Executes a polynomial multiplication followed by a polynomial addition.
 * 
 * Computes \f$d = a \times b + c\f$.
 * 
 * @param ctx   The context
 * @param d     Outcome
 * @param a     First operand
 * @param b     Second operand
 * @param c     Third operand
 */
__host__ void poly_mul_add(Context *ctx, poly_t *d, poly_t *a, poly_t *b, poly_t *c);

__host__  void poly_dr2(
    Context *ctx,
    poly_t *ct21, // Outcome
    poly_t *ct22, // Outcome
    poly_t *ct23, // Outcome
    poly_t *ct01, // Operand 1
    poly_t *ct02, // Operand 1
    poly_t *ct11, // Operand 2
    poly_t *ct12);// Operand 2

/**
 * @brief Multiply each coefficient by and integer x
 * 
 * Computes \f$c = a \times x\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c     Outcome
 * @param[in]  a     The polynomial
 * @param[in]  x     The integer
 */
__host__  void poly_mul_int(Context *ctx, poly_t *c, poly_t *a, uint64_t x);
__host__  void poly_double_mul_int(
    Context *ctx, 
    poly_t *b1,
    poly_t *a1,
    poly_t *b2,
    poly_t *a2,
    uint64_t x1,
    uint64_t x2);
__host__  void poly_double_add_int(
    Context *ctx, 
    poly_t *b1,
    poly_t *a1,
    poly_t *b2,
    poly_t *a2,
    uint64_t x1,
    uint64_t x2);

/**
 * @brief Add a zero-degree polynomial
 * 
 * Computes \f$c = a \times x\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c     Outcome
 * @param[in]  a     The polynomial
 * @param[in]  x     The integer
 */
__host__  void poly_add_int(Context *ctx, poly_t *c, poly_t *a, uint64_t x);

/**
 * @brief Subtract a zero-degree polynomial
 * 
 * Computes \f$c = a \times x\f$.
 * 
 * @param[in]  ctx   The context
 * @param[out] c     Outcome
 * @param[in]  a     The polynomial
 * @param[in]  x     The integer
 */
__host__  void poly_sub_int(Context *ctx, poly_t *c, poly_t *a, uint64_t x);


/**
 * @brief Divide each coefficient by x
 * 
 * Computes \f$c = \frac{a}{x}\f$.
 * This operation is done exclusively at the host.
 * 
 * @param[in]   ctx   The context
 * @param       c      Outcome
 * @param       a      The polynomial
 * @param       x      The integer divisor
 */
__host__  void poly_div_by_ZZ(Context *ctx, poly_t *c, poly_t *a, ZZ x);

/**
 * @brief       Compute the modular reduction of each coefficient by x.
 * 
 * Computes \f$c = a \pmod x\f$.
 * This operation is done exclusively at the host.
 * 
 * @param ctx     The context
 * @param c      [description]
 * @param a      [description]
 * @param x      [description]
 */
__host__  void poly_mod_by_ZZ(Context *ctx, poly_t *c, poly_t *a, ZZ x);

/**
 * @brief The HPS method for scaling by \f$\frac{t}{q} \pmod t\f$.
 *
 * Executes the simple scaling procedure inplace, as described in section 2.3 so that
 * b = a*(t/q) mod t
 *
 * "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
 * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
 *
 * @param[in]     ctx    the context
 * @param[out]    b      Outcome in base b
 * @param[in]     a      Input in base q
 */
__host__ void poly_simple_scaling_tDivQ(Context *ctx, poly_t *b, poly_t *a);

/**
 * @brief The HPS method for scaling by \f$\frac{t}{q} \pmod b\f$.
 *
 * Executes the simple scaling procedure inplace, as described in section 2.4 so that
 * b = a*(t/q) mod b
 *
 * "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
 * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
 *
 * THERE IS A SERIOUS BUG: THIS METHOD DESTROYS a_Q and a_B
 * @fixme Fix this method so that it does not destroy a_Q and a_B
 *
 * @param[in]     ctx       the context
 * @param[out]    b        Outcome in base b
 * @param[in]     a_Q      Input in base q
 * @param[in]     a_B      Input in base b
 * 
 */
__host__ void poly_complex_scaling_tDivQ(Context *ctx, poly_t *b, poly_t *a_Q, poly_t *a_B);

/**
 * @brief       Computes the basis extension from base Q to base B
 * 
 * @param[in]     ctx      the context
 * @param[out]    a_B      Outcome in base b
 * @param[in]     a_Q      Input in base q
 */
__host__ void poly_basis_extension_Q_to_B(Context *ctx, poly_t *a_B, poly_t *a_Q);

/**
 * @brief       Computes the basis extension from base Q to base QB inplace
 * 
 * @param[in]     ctx      the context
 * @param[out]    a_B      Outcome in base b
 * @param[in]     a_Q      Input in base q
 */
__host__ void poly_basis_extension_Q_to_QB(Context *ctx, poly_t *b, poly_t *a);

__host__ void poly_approx_basis_reduction_QB_to_Q( Context *ctx, poly_t *a, poly_t *b, int level);
__host__ void poly_alt_approx_basis_reduction_QB_to_Q( Context *ctx, poly_t *a);
__host__ void poly_approx_basis_raising_Q_to_QB( Context *ctx, poly_t *a, poly_t *b, int level);

/**
 * @brief       Computes the basis extension from base B to base Q
 * 
 * @param[in]     ctx      the context
 * @param[out]    a_B      Outcome in base b
 * @param[in]     a_Q      Input in base q
 */

__host__ void poly_basis_extension_B_to_Q(Context *ctx, poly_t *a_B, poly_t *a_Q);

/**
 * @brief The HPS method for computing \f$\xi\f$, used by FV's homomorphic multiplication
 *
 * Section 4 of "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
 * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
 *
 * @param[in]     ctx      the context
 * @param[out]    c      Outcome of many polynomials in base q
 * @param[in]     a      Input in base q
 */
__host__ void poly_xi_bfv(Context *ctx, poly_t *c, poly_t *a);

/**                                                            *
 * @brief  The HPS method for computing \f$\xi\f$, used by CKKS's keygen
 *     
 * Section 4 of "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
 * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
 *
 * @param[in]     ctx      the context
 * @param[out]    c      Outcome of many polynomials in base q
 * @param[in]     a      Input in base q
 */
__host__ void poly_xi_ckks(Context *ctx, poly_t *c, poly_t *a);

/**
 * @brief  The HPS method for computing \f$\rho\f$, used by FV's keygen
 *     
 * Section 4 of "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
 * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
 *
 * @param[in]     ctx      the context
 * @param[out]    c      Outcome of many polynomials in base q
 * @param[in]     a      Input in base q
 */
__host__ void poly_rho_bfv(Context *ctx, poly_t *c, poly_t *a);

/**                                                            *
 * @brief  The HPS method for computing \f$\rho\f$, used by CKKS's keygen
 *     
 * Section 4 of "An Improved RNS Variant of the BFV Homomorphic Encryption Scheme",
 * from Shai Halevi, Yuriy Polyakov, and Victor Shoup
 *
 * @param[in]     ctx      the context
 * @param[out]    c      Outcome of many polynomials in base q
 * @param[in]     a      Input in base q
 */
__host__ void poly_rho_ckks(Context *ctx, poly_t *c, poly_t *a);

__host__ void poly_ckks_rescale( Context *ctx,  poly_t *a, poly_t *b, int level = 1);
/**
 * Change the representation of coefficients of a from [0, b) to [-t/2, t/2)
 * @param a        [description]
 * @param new_base [description]
 */
__host__ void poly_change_repr(poly_t *a, int new_base);

/**
 * @brief       Return the string representation of a poly_t object
 * 
 * @param ctx     The context
 * @param a [description]
 */
__host__ std::string poly_to_string(Context *ctx, poly_t *a);

/**
 * @brief       Set c as the index-th coefficient.
 * 
 * @param ctx [description]
 * @param a [description]
 * @param index [description]
 * @param c [description]
 */
__host__ void poly_set_coeff(Context *ctx, poly_t *a, unsigned int index, ZZ c);

/**
 * @brief       Returns the index-th coefficient.
 * 
 * @param ctx     The context
 * @param a     [description]
 * @param index [description]
 */
__host__ ZZ poly_get_coeff(Context *ctx, poly_t *a, int index);

/**
 * @brief       Set "a" as the nth cyclotomic polynomial
 * 
 * @param a [description]
 * @param n [description]
 */
__host__ void poly_set_nth_cyclotomic(poly_t *a, unsigned int n);

/**
 * @brief       Negate the coefficients of a polynomial a
 * 
 * @param ctx     The context
 * @param a    [description]
 */
__host__ void poly_negate(Context *ctx, poly_t *a);

/**
 * @brief       Right shift of each coefficient
 * 
 * @param ctx     The context
 * @param b       Output
 * @param a       Input
 * @param bits    Number of bits to shift
 */
__host__ void poly_right_shift(Context *ctx, poly_t *b, poly_t *a, int bits);

/**
 * @brief      Returns true if a == b, false otherwise.
 *
 * @param ctx     The context
 * @param      a     { parameter_description }
 * @param      b     { parameter_description }
 */
__host__ bool poly_are_equal(Context *ctx, poly_t *a, poly_t *b);

/**
 * @brief      Serializes a polynomial in a vector of ZZs
 *
 * @param      p     { parameter_description }
 *
 * @param ctx     The context
 * @return     { description_of_the_return_value }
 */
__host__ std::string poly_export(Context *ctx, poly_t *p);

/**
 * @brief      Returns a polynomial such that each coefficient vi lies in the ith-coefficient of v.
 *
 * @param ctx     The context
 * @param[in]  v     { parameter_description }
 */
__host__ poly_t* poly_import_residues(Context *ctx, std::string v, int base = QBase);

/**
 * @brief      Serializes a polynomial in a vector of ZZs
 *
 * @param      p     { parameter_description }
 *
 * @param ctx     The context
 * @return     { description_of_the_return_value }
 */
__host__ std::string poly_export_residues(Context *ctx, poly_t *p);

/**
 * @brief      Returns a polynomial such that each coefficient vi lies in the ith-coefficient of v.
 *
 * @param ctx     The context
 * @param[in]  v     { parameter_description }
 */
__host__ poly_t* poly_import(Context *ctx, std::string v);

/**
 * @brief      Compute the dot procut of a and b.
 * 
 * Computes \f$c = a \cdot b\f$.
 *
 * @param      ctx     { parameter_description }
 * @param      c     { parameter_description }
 * @param      a     { parameter_description }
 * @param      b     { parameter_description }
 * @param      k     { parameter_description }
 */
__host__ void poly_dot(
    Context *ctx, 
    poly_t *c,
    poly_t *a,
    poly_t *b,
    const int k);

/**
 * @brief      Return the infinity norm of p
 *
 * @param[in]  ctx     { parameter_description }
 * @param[in]  p     { parameter_description }
 */
__host__ ZZ poly_infty_norm(Context *ctx, poly_t *p);

/**
 * @brief      Return the 2-norm of p
 *
 * @param[in]  ctx     { parameter_description }
 * @param[in]  p     { parameter_description }
 */
__host__ RR poly_norm_2(Context *ctx, poly_t *p);

/**
 * @brief       Return the string representation of a residue
 * 
 * @param ctx     The context
 * @param a    the target poly_t
 * @param id    the id of the residue
 */
__host__ std::string poly_residue_to_string(Context *ctx, poly_t *a, int id);

/**
 * @brief      Select a specific residue and overwrite everything else with it
 *
 * @param[in]  ctx     A context object
 * @param[in]  p     The operand
 * @param[in]  id     The residue
 */
__host__ void poly_select_residue(Context *ctx, poly_t *p, int id);


/**
 * @brief      Returns an integer array with a particular residue of a.
 *
 * If b is allocated the outcome will be written on it, otherwise a new array will be allocated and the reference stored in b.
 *
 * @param      ctx   The context
 * @param[out]     b     The id-th residue of a
 * @param[int]      a     The polynomial
 * @param  id    The identifier
 *
 * @return     The id-th residue of a
 */
__host__ uint64_t* poly_get_residue(Context *ctx, uint64_t* b, poly_t *a, int id);

/**
 * @brief      Returns the decomposition of a in its RNS base.
 *
 * @param      ctx   The context
 * @param[in]      a     The polynomial
 *
 * @return     The residues of a in a->base base.
 */
__host__ uint64_t* poly_get_residues(Context *ctx, poly_t *a);

/**
 * @brief      Receives a polynomial represented in base QB and discard the q-residues.
 *
 * @param      ctx   The context
 * @param      p     { parameter_description }
 */
__host__ void poly_discard_qbase(Context *ctx, poly_t *p);

#endif