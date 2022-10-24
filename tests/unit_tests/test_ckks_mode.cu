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


#include <gtest/gtest.h>
#include <cuPoly/settings.h>
#include <cuPoly/arithmetic/polynomial.h>
#include <cuPoly/cuda/sampler.h>
#include <cuPoly/cuda/cudaengine.h>
#include <cuPoly/cuda/dgt.h>
#include <stdlib.h>
#include <cxxopts.hpp>
#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/ZZX.h>
#include <NTL/ZZ_pX.h>
#include <NTL/ZZ_pE.h>
#include <NTL/ZZ_pEX.h>

typedef struct{
	int k;
	int kl;
	int nphi;
	int prec;
} TestParams;

//int LOGLEVEL = INFO;
// int LOGLEVEL = DEBUG;
int LOGLEVEL = QUIET;
unsigned int NTESTS = 100;

// Focus on the arithmetic
class TestArithmetic : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	int nphi;
	Context *ctx;
	int t = 256; // Used for testing only

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		Logger::getInstance()->log_info(("k: " + std::to_string(k)).c_str());
		nphi = GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAEngine::init(k, kl, nphi, prec);// Init ClUDA
		ZZ q = CUDAEngine::RNSProduct;
		ctx = new Context();

		// Init NTL
		ZZ_p::init(q);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, nphi, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);

		// Samplers
		Sampler::init(ctx);
	}

	__host__ void TearDown()
	{
		delete ctx;
		Sampler::destroy();
		CUDAEngine::destroy();
		cudaDeviceReset();
	}
};

class TestRNS : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	int nphi;
	Context *ctx;
	int t = 256; // Used for testing only

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		Logger::getInstance()->log_info(("k: " + std::to_string(k)).c_str());
		nphi = GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAEngine::init(k, kl, nphi, prec);// Init ClUDA
		ZZ q = CUDAEngine::RNSProduct;
		ctx = new Context();

		// Init NTL
		ZZ_p::init(q);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, nphi, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);

		// Samplers
		Sampler::init(ctx);
		
	}

	__host__ void TearDown()
	{
		delete ctx;
		Sampler::destroy();
		CUDAEngine::destroy();
		cudaDeviceReset();
	}
};

class TestDGT : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	int nphi;
	Context *ctx;
	int t = 256; // Used for testing only/

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		Logger::getInstance()->log_info(("k: " + std::to_string(k)).c_str());
		nphi = GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAEngine::init(k, kl, nphi, prec);// Init ClUDA
		ZZ q = CUDAEngine::RNSProduct;
		ctx = new Context();

		// Init NTL
		ZZ_p::init(q);
		NTL::SetCoeff(NTL_Phi,0,conv<ZZ_p>(1));
		NTL::SetCoeff(NTL_Phi, nphi, conv<ZZ_p>(1));
		ZZ_pE::init(NTL_Phi);

		// Samplers
		Sampler::init(ctx);
		
	}

	__host__ void TearDown()
	{
		delete ctx;
		Sampler::destroy();
		CUDAEngine::destroy();
		cudaDeviceReset();
	}
};

//
// Focus on Sampler
class TestSampler : public ::testing::TestWithParam<TestParams> {
	protected:

	int prec;
	ZZ_pX NTL_Phi;
	int nphi;
	Context *ctx;
	int t = 256; // Used for testing only

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = (int)GetParam().k;
		int kl = (int)GetParam().kl;
		Logger::getInstance()->log_info(("k: " + std::to_string(k)).c_str());
		nphi = GetParam().nphi;
		prec = (int)GetParam().prec;

		// Init
		CUDAEngine::init(k, kl, nphi, prec);// Init ClUDA
		ctx = new Context();

		Sampler::init(ctx);
	}

	__host__ void TearDown()
	{
		delete ctx;
		CUDAEngine::destroy();
		cudaDeviceReset();
	}
};

//////////////////////////
// Basic DGT arithmetic //
//////////////////////////

uint64_t rand64(uint64_t upperbound = 18446744073709551615) {
    // Assuming RAND_MAX is 2^32-1
    uint64_t r = rand();
    r = r<<32 | rand();
    return r % upperbound;
}

#define TEST_DESCRIPTION(desc) RecordProperty("description", desc)
TEST_P(TestDGT, mod)
{
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++){
		ZZ a = (to_ZZ(rand64()) << 64) + to_ZZ(rand64());
		a %= to_ZZ(2<<prec);

		uint128_t a_cupoly;
		a_cupoly.hi = conv<uint64_t>(a>>64);
		a_cupoly.lo = conv<uint64_t>(a);

		ZZ a_ntl = to_ZZ(a_cupoly.hi);
		a_ntl <<= 64;
		a_ntl += a_cupoly.lo;

		// Test for all the initialized coprimes
		for(unsigned int i = 0; 
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			uint64_t x = mod(a_cupoly, i);
			ASSERT_EQ(x, a_ntl % to_ZZ(p));
		}

	}
}

TEST_P(TestDGT, addmod)
{
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++){

		uint64_t a_cupoly, b_cupoly;
		a_cupoly = rand64() % (1 << prec);
		b_cupoly = rand64() % (1 << prec);

		ZZ a_ntl = to_ZZ(a_cupoly);
		ZZ b_ntl = to_ZZ(b_cupoly);

		// Test for all the initialized coprimes
		for(unsigned int i = 0; 
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			ASSERT_EQ(addmod(a_cupoly, b_cupoly, i), (a_ntl + b_ntl) % to_ZZ(p));
		}

	}
}

TEST_P(TestDGT, submod)
{
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++){

		uint64_t a_cupoly, b_cupoly;
		a_cupoly = rand64() % (1 << prec);
		b_cupoly = rand64() % (1 << prec);

		ZZ a_ntl = to_ZZ(a_cupoly);
		ZZ b_ntl = to_ZZ(b_cupoly);

		// Test for all the initialized coprimes
		for(unsigned int i = 0; 
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			ASSERT_EQ(submod(a_cupoly, b_cupoly, i), (a_ntl - b_ntl) % to_ZZ(p));
		}

	}
}

TEST_P(TestDGT, mulmod)
{
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++){

		uint64_t a_cupoly, b_cupoly;
		a_cupoly = 3246018949690369621;
		b_cupoly = 31686739480305835;

		ZZ a_ntl = to_ZZ(a_cupoly);
		ZZ b_ntl = to_ZZ(b_cupoly);

	//	ASSERT_EQ(mulmod(a_cupoly, b_cupoly, 80), (a_ntl * b_ntl) % to_ZZ(COPRIMES_BUCKET[80]));

		a_cupoly = rand64() % (1 << prec);
		b_cupoly = rand64() % (1 << prec);

		a_ntl = to_ZZ(a_cupoly);
		b_ntl = to_ZZ(b_cupoly);

		// Test for all the initialized coprimes
		for(unsigned int i = 0; 
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			ASSERT_EQ(mulmod(a_cupoly, b_cupoly, i), (a_ntl * b_ntl) % to_ZZ(p));
		}

	}
}

TEST_P(TestDGT, powmod)
{
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++){

		uint64_t a_cupoly, b_cupoly;
		a_cupoly = rand64() % (1 << prec);
		b_cupoly = rand64() % (1 << prec);

		ZZ a_ntl = to_ZZ(a_cupoly);
		ZZ b_ntl = to_ZZ(b_cupoly);

		// Test for all the initialized coprimes
		for(unsigned int i = 0; 
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); 
			i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			uint64_t x = fast_pow(a_cupoly, b_cupoly, i);
			ASSERT_EQ(x, PowerMod(a_ntl, b_ntl, to_ZZ(p)));
		}

	}
}
////////////////////////////////////////////////////////////////////////////////
// GaussianIntegerArithmetic
////////////////////////////////////////////////////////////////////////////////

TEST_P(TestDGT, GIAdd) {
	// Test for all the initialized coprimes
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++)
		for(unsigned int i = 0;
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size());
			i++){

		    GaussianInteger a = {rand64((uint64_t)1<<prec), rand64((uint64_t)1<<prec)};
		    GaussianInteger b = {rand64((uint64_t)1<<prec), rand64((uint64_t)1<<prec)};
		    GaussianInteger expected = {
		      conv<uint64_t>((to_ZZ(a.re) + to_ZZ(b.re)) % COPRIMES_BUCKET[i]),
		      conv<uint64_t>((to_ZZ(a.imag) + to_ZZ(b.imag)) % COPRIMES_BUCKET[i])
		    };
		    GaussianInteger received = GIAdd(a, b, i);

		    ASSERT_EQ(
		      received.re,
		      expected.re
		      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " + " << "(" << b.re << ", " << b.imag << ")";
		    ASSERT_EQ(
		      received.imag,
		      expected.imag
		      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " + " << "(" << b.re << ", " << b.imag << ")";
		}
}

TEST_P(TestDGT, GISub) {
	// Test for all the initialized coprimes
	for(unsigned int ntest = 0; ntest < 10 * NTESTS; ntest++){
		for(unsigned int i = 0;
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size());
			i++){

		    GaussianInteger a = {rand64((uint64_t)1<<prec), rand64((uint64_t)1<<prec)};
		    GaussianInteger b = {rand64((uint64_t)1<<prec), rand64((uint64_t)1<<prec)};
		    GaussianInteger expected = {
		      conv<uint64_t>((to_ZZ(a.re) - to_ZZ(b.re)) % COPRIMES_BUCKET[i]),
		      conv<uint64_t>((to_ZZ(a.imag) - to_ZZ(b.imag)) % COPRIMES_BUCKET[i])
		    };
		    GaussianInteger received = GISub(a, b, i);

		    ASSERT_EQ(
		      received.re,
		      expected.re
		      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " - " << "(" << b.re << ", " << b.imag << ")";
		    ASSERT_EQ(
		      received.imag,
		      expected.imag
		      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " - " << "(" << b.re << ", " << b.imag << ")";
		}
  	}
}

TEST_P(TestDGT, GIMul) {
	// Test for all the initialized coprimes
	for(unsigned int test = 0; test < 10 * NTESTS; test++)
		for(unsigned int i = 0;
			i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size());
			i++){

		    GaussianInteger a = {rand64((uint64_t)1<<prec), rand64((uint64_t)1<<prec)};
		    GaussianInteger b = {rand64((uint64_t)1<<prec), rand64((uint64_t)1<<prec)};
		    GaussianInteger expected = {
		      submod(
		        mulmod(a.re, b.re, i),
		        mulmod(a.imag, b.imag, i),
		        i
		        ),
		      addmod(
		        mulmod(a.imag, b.re, i),
		        mulmod(a.re, b.imag, i),
		        i
		        )
		    };
		    GaussianInteger received = GIMul(a, b, i);

		    ASSERT_EQ(
		      received.re,
		      expected.re
		      ) << test << ") Failure! (" << a.re << ", " << a.imag << ")" << " * " << "(" << b.re
				<< ", " << b.imag << ") for rid " << i << " ( " << COPRIMES_BUCKET[i] << ")";
		    ASSERT_EQ(
		      received.imag,
		      expected.imag
		      ) << test << ") Failure! (" << a.re << ", " << a.imag << ")" << " * " << "(" << b.re
				<< ", " << b.imag << ") for rid " << i << " ( " << COPRIMES_BUCKET[i] << ")";
		}
}

/////////
// RNS //
/////////

// Tests the implementation of the RNS
TEST_P(TestRNS, RNS)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a;
		poly_init(ctx, &a);

		for(int i = 0; i < nphi;i++)
			poly_set_coeff(ctx, &a, i, to_ZZ(1));

		// to device
		poly_copy_to_device(ctx, &a);
		a.status = RNSSTATE;

		// to host
		poly_copy_to_host(ctx, &a);

		for(int i = 0; i < nphi; i++)
			ASSERT_EQ(poly_get_coeff(ctx, &a,i) , to_ZZ(1)) << "Fail at index " << i;
		for(int i = nphi; i < nphi*2; i++)
			ASSERT_EQ(poly_get_coeff(ctx, &a,i) , to_ZZ(0)) << "Fail at index " << i;
		poly_free(ctx, &a);
	}
}

// Export and import polynomials
TEST_P(TestRNS, Serialize)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a;
		poly_init(ctx, &a);

		Sampler::sample(ctx, &a, DISCRETE_GAUSSIAN);

		// to device
		poly_copy_to_device(ctx, &a);
		a.status = RNSSTATE;

		// export
		std::string s = poly_export(ctx, &a);
		poly_t *b = poly_import(ctx, s);

		// assert
		ASSERT_TRUE(poly_are_equal(ctx, &a, b));

		poly_free(ctx, &a);
		poly_free(ctx, b);
		delete b;
	}
}

// Export and import polynomials
TEST_P(TestRNS, SerializeResidues)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a;
		poly_init(ctx, &a);

		Sampler::sample(ctx, &a, DISCRETE_GAUSSIAN);

		// to device
		poly_copy_to_device(ctx, &a);
		a.status = RNSSTATE;

		// export
		std::string s = poly_export_residues(ctx, &a);
		poly_t *b = poly_import_residues(ctx, s, a.base);

		// assert
		ASSERT_TRUE(poly_are_equal(ctx, &a, b));

		poly_free(ctx, &a);
		poly_free(ctx, b);
		delete b;
	}
}

// \forall (a,b) \in R^2, <D_{w,q}(a), P_{w,q}(b)> \equiv a*b mod q
TEST_P(TestRNS, WorddecompPowersOf)
{

	for(unsigned int n = 0; n < NTESTS; n++){
		poly_t a, b, c;
		poly_init(ctx, &a);
		poly_init(ctx, &b);
		poly_init(ctx, &c);

		// Sample random polynomials
		Sampler::sample(ctx, &a, DISCRETE_GAUSSIAN);
		Sampler::sample(ctx, &b, DISCRETE_GAUSSIAN);

		// Compute rho/xi	
		const int k = CUDAEngine::get_n_residues(QBase);

		poly_t *A = new poly_t[k]; // D_{w,q} (a)
		poly_t *B = new poly_t[k]; // P_{w,q} (b)

		for(int i = 0; i < k; i++){
			poly_init(ctx, &A[i]);
			poly_init(ctx, &B[i]);
		}

		poly_rho_ckks(ctx, A, &a);
		poly_xi_ckks(ctx, B, &b);

		poly_t acc;
		poly_init(ctx, &acc);
		ZZ_p x = conv<ZZ_p>(to_ZZ(0));
		// Compute the dot product
		for(int i = 0; i < k; i++)
			poly_mul_add(ctx, &acc, &A[i], &B[i], &acc);

		// Compute a*b
		poly_mul(ctx, &c, &a, &b);

		for(int i = 0; i < nphi;i++)
			ASSERT_EQ(
				poly_get_coeff(ctx, &acc, i),
				poly_get_coeff(ctx, &c, i)
			) << "Fail at index " << i;

		poly_free(ctx, &a);
		poly_free(ctx, &b);
		poly_free(ctx, &c);
		poly_free(ctx, &acc);
		for(int i = 0; i < k; i++){
			poly_free(ctx, &A[i]);
			poly_free(ctx, &B[i]);
		}
		delete[] A;
		delete[] B;
	}
}

//
// Tests the Set/Get coeff behavior
TEST_P(TestRNS, SetCoeff)
{
	poly_t a;
	poly_init(ctx, &a);

	poly_set_coeff(ctx, &a,0,to_ZZ(42));

	ASSERT_EQ(poly_get_coeff(ctx, &a,0) , to_ZZ(42));
	poly_free(ctx, &a);
}

//
// Tests polynomial addition
TEST_P(TestArithmetic, Add)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b, c;
		poly_init(ctx, &a);
		poly_init(ctx, &b);

		// Sample random polynomials
		Sampler::sample(ctx, &a, UNIFORM);
		Sampler::sample(ctx, &b, UNIFORM);

		// Add
		poly_add(ctx, &c, &a, &b);
		
		for(int rid = 0; rid < CUDAEngine::get_n_residues(QBase); rid++){
			uint64_t *a_coefs = poly_get_residue(ctx, NULL, &a, rid);
			uint64_t *b_coefs = poly_get_residue(ctx, NULL, &b, rid);
			uint64_t *c_coefs = poly_get_residue(ctx, NULL, &c, rid);
			for(int i = 0; i < CUDAEngine::N; i++)
					ASSERT_EQ(
						(to_ZZ(a_coefs[i]) + to_ZZ(b_coefs[i])) % COPRIMES_BUCKET[rid],
						c_coefs[i]) << ntest << ") Fail at index " << i << " at rid " << rid;

			free(a_coefs);
			free(b_coefs);
			free(c_coefs);
		}

		poly_free(ctx, &a);
		poly_free(ctx, &b);
		poly_free(ctx, &c);
	}
}

//
// Tests polynomial subtraction
TEST_P(TestArithmetic, Sub)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b, c;
		poly_init(ctx, &a);
		poly_init(ctx, &b);

		// Sample random polynomials
		Sampler::sample(ctx, &a, UNIFORM);
		Sampler::sample(ctx, &b, UNIFORM);

		// Sub
		poly_sub(ctx, &c, &a, &b);
		
		for(int rid = 0; rid < CUDAEngine::get_n_residues(QBase); rid++){
			uint64_t *a_coefs = poly_get_residue(ctx, NULL, &a, rid);
			uint64_t *b_coefs = poly_get_residue(ctx, NULL, &b, rid);
			uint64_t *c_coefs = poly_get_residue(ctx, NULL, &c, rid);
			for(int i = 0; i < CUDAEngine::N; i++)
					ASSERT_EQ(
						(to_ZZ(a_coefs[i]) - to_ZZ(b_coefs[i])) % COPRIMES_BUCKET[rid],
						c_coefs[i]) << ntest << ") Fail at index " << i << " at rid " << rid;

			free(a_coefs);
			free(b_coefs);
			free(c_coefs);
		}

		poly_free(ctx, &a);
		poly_free(ctx, &b);
		poly_free(ctx, &c);
	}
}

//
// Tests polynomial multiplication
TEST_P(TestArithmetic, Mul)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b,c;
		poly_init(ctx, &a);
		poly_init(ctx, &b);

		// Sample random polynomials
		Sampler::sample(ctx, &a, UNIFORM);
		Sampler::sample(ctx, &b, UNIFORM);

		ZZX ntl_a, ntl_b, ntl_c;
		for(int i = 0; i < nphi; i++)
			SetCoeff(ntl_a, i, poly_get_coeff(ctx, &a, i));
		for(int i = 0; i < nphi; i++)
			SetCoeff(ntl_b, i, poly_get_coeff(ctx, &b, i));

		poly_mul(ctx, &c, &a, &b);
		ntl_c = ntl_a * ntl_b % conv<ZZX>(NTL_Phi);

		for(unsigned int rid = 0; rid < CUDAEngine::RNSPrimes.size(); rid++){
			uint64_t *c_coefs = poly_get_residue(ctx, NULL, &c, rid);

			for(int i = 0; i < CUDAEngine::N;i++)
				ASSERT_EQ(
					c_coefs[i],
					coeff(ntl_c, i) % CUDAEngine::RNSPrimes[rid]
					) << ntest << ") Fail at index " << i << " at rid " << rid;

			free(c_coefs);
		}
		poly_free(ctx, &a);
		poly_free(ctx, &b);
		poly_free(ctx, &c);
	}
}

TEST_P(TestArithmetic, MulAdd)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b, c, d;
		poly_init(ctx, &a);
		poly_init(ctx, &b);
		poly_init(ctx, &c);

		// Sample random polynomials
		Sampler::sample(ctx, &a, UNIFORM);
		Sampler::sample(ctx, &b, UNIFORM);
		Sampler::sample(ctx, &c, UNIFORM);

		ZZX ntl_a, ntl_b, ntl_c, ntl_d;
		for(int i = 0; i < nphi; i++){
			SetCoeff(ntl_a, i, poly_get_coeff(ctx, &a, i));
			SetCoeff(ntl_b, i, poly_get_coeff(ctx, &b, i));
			SetCoeff(ntl_c, i, poly_get_coeff(ctx, &c, i));
		}

		poly_mul_add(ctx, &d, &a, &b, &c);
		ntl_d = (ntl_a * ntl_b + ntl_c) % conv<ZZX>(NTL_Phi);

		for(unsigned int rid = 0; rid < CUDAEngine::RNSPrimes.size(); rid++){
			uint64_t *d_coefs = poly_get_residue(ctx, NULL, &d, rid);

			for(int i = 0; i < CUDAEngine::N;i++)
				ASSERT_EQ(
					d_coefs[i],
					coeff(ntl_d, i) % CUDAEngine::RNSPrimes[rid]
					) << ntest << ") Fail at index " << i << " at rid " << rid;

			free(d_coefs);
		}

		poly_free(ctx, &a);
		poly_free(ctx, &b);
		poly_free(ctx, &c);
		poly_free(ctx, &d);
	}
}

TEST_P(TestArithmetic, MulBaseB)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a_Q, b_Q;
		poly_t a_B, b_B, c_B;
		poly_init(ctx, &a_Q);
		poly_init(ctx, &b_Q);
		poly_init(ctx, &a_B, BBase);
		poly_init(ctx, &b_B, BBase);
		poly_init(ctx, &c_B, BBase);

		Sampler::sample(ctx, &a_Q, UNIFORM);
		Sampler::sample(ctx, &b_Q, UNIFORM);

		// Copy to NTL polynomials
		ZZX ntl_a, ntl_b, ntl_c;
		for(int i = 0; i < nphi; i++)
			SetCoeff(ntl_a, i, poly_get_coeff(ctx, &a_Q, i));
		for(int i = 0; i < nphi; i++)
			SetCoeff(ntl_b, i, poly_get_coeff(ctx, &b_Q, i));

		// Compute base B
		GaussianInteger *h_coefs = (GaussianInteger*) malloc (poly_get_residues_size(BBase));

		for(unsigned int i = 0; i < CUDAEngine::RNSBPrimes.size(); i++)
			for(int j = 0; j < CUDAEngine::N; j++){
				h_coefs[j + i * CUDAEngine::N].re 	= conv<uint64_t>(
					poly_get_coeff(ctx, &a_Q, j) % CUDAEngine::RNSBPrimes[i]
					);
				h_coefs[j + i * CUDAEngine::N].imag = conv<uint64_t>(
					poly_get_coeff(ctx, &a_Q, j + CUDAEngine::N) % CUDAEngine::RNSBPrimes[i]
					);
			}
		cudaStreamSynchronize(ctx->get_stream());
		cudaCheckError();
		cudaMemcpyAsync(
			a_B.d_coefs,
			h_coefs,
			poly_get_residues_size(BBase),
			cudaMemcpyHostToDevice,
			ctx->get_stream());
		a_B.status = RNSSTATE;
		DGTEngine::execute_dgt(
			a_B.d_coefs,
			BBase,
			FORWARD,
			ctx
		);
		cudaStreamSynchronize(ctx->get_stream());
		cudaCheckError();
		for(unsigned int i = 0; i < CUDAEngine::RNSBPrimes.size(); i++)
			for(int j = 0; j < CUDAEngine::N; j++){
				h_coefs[j + i * CUDAEngine::N].re 	= conv<uint64_t>(
					poly_get_coeff(ctx, &b_Q, j) % CUDAEngine::RNSBPrimes[i]
					);
				h_coefs[j + i * CUDAEngine::N].imag = conv<uint64_t>(
					poly_get_coeff(ctx, &b_Q, j + CUDAEngine::N) % CUDAEngine::RNSBPrimes[i]
					);
			}
		cudaStreamSynchronize(ctx->get_stream());
		cudaCheckError();
		cudaMemcpyAsync(
			b_B.d_coefs,
			h_coefs,
			poly_get_residues_size(BBase),
			cudaMemcpyHostToDevice,
			ctx->get_stream());
		b_B.status = RNSSTATE;
		DGTEngine::execute_dgt(
			b_B.d_coefs,
			BBase,
			FORWARD,
			ctx
		);
		cudaStreamSynchronize(ctx->get_stream());
		cudaCheckError();
		free(h_coefs);

		// Multiply in both bases
		poly_mul(ctx, &c_B, &a_B, &b_B);
		ntl_c = ntl_a * ntl_b % conv<ZZX>(NTL_Phi) ;

		/////////////////////////////
		// Copy B residues to Host //
		/////////////////////////////
		GaussianInteger* a_h_coefs = (GaussianInteger*) malloc(poly_get_residues_size(BBase));
		GaussianInteger* b_h_coefs = (GaussianInteger*) malloc(poly_get_residues_size(BBase));
		GaussianInteger* c_h_coefs = (GaussianInteger*) malloc(poly_get_residues_size(BBase));

		// IDGT
		DGTEngine::execute_dgt(
			a_B.d_coefs,
			BBase,
			INVERSE,
			ctx
		);
		DGTEngine::execute_dgt(
			b_B.d_coefs,
			BBase,
			INVERSE,
			ctx
		);
		DGTEngine::execute_dgt(
			c_B.d_coefs,
			BBase,
			INVERSE,
			ctx
		);

		cudaStreamSynchronize(ctx->get_stream());
		cudaCheckError();
	
		cudaMemcpyAsync(	
			a_h_coefs,
			a_B.d_coefs,
			poly_get_residues_size(a_B.base),
			cudaMemcpyDeviceToHost,
			ctx->get_stream() );
		cudaCheckError();

		cudaMemcpyAsync(
			b_h_coefs,
			b_B.d_coefs,
			poly_get_residues_size(b_B.base),
			cudaMemcpyDeviceToHost,
			ctx->get_stream() );
		cudaCheckError();

		cudaMemcpyAsync(
			c_h_coefs,
			c_B.d_coefs,
			poly_get_residues_size(c_B.base),
			cudaMemcpyDeviceToHost,
			ctx->get_stream() );
		cudaCheckError();

		cudaStreamSynchronize(ctx->get_stream());
		cudaCheckError();
		// Verifies base B
		// Tries the operands
		for(unsigned int rid = 0; rid < CUDAEngine::RNSBPrimes.size(); rid++)
			for(int i = 0; i < nphi/2; i++){
				ASSERT_EQ(
					a_h_coefs[i + rid * (CUDAEngine::N)].re % CUDAEngine::RNSBPrimes[rid],
					poly_get_coeff(ctx, &a_Q, i) % CUDAEngine::RNSBPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
				if(a_h_coefs[i + rid * (CUDAEngine::N)].imag % CUDAEngine::RNSBPrimes[rid] !=
					poly_get_coeff(ctx, &a_Q, i + nphi/2) % CUDAEngine::RNSBPrimes[rid]){

					std::cout << std::endl << "FAILED: a = Polynomial(coef=[" << 
						poly_residue_to_string(ctx, &a_B, rid) << "])" << std::endl;
					std::cout << "FAILED: b = Polynomial(coef=[" << 
						poly_residue_to_string(ctx, &b_B, rid) << "])" << std::endl<< std::endl;
					std::cout << "FAILED: received r = Polynomial(coef=[" << 
						poly_residue_to_string(ctx, &c_B, rid) << "])" << std::endl<< std::endl;
				}
				ASSERT_EQ(
					a_h_coefs[i + rid * (CUDAEngine::N)].imag % CUDAEngine::RNSBPrimes[rid],
					poly_get_coeff(ctx, &a_Q, i + nphi/2) % CUDAEngine::RNSBPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
			}
		for(unsigned int rid = 0; rid < CUDAEngine::RNSBPrimes.size(); rid++)
			for(int i = 0; i < nphi/2; i++){
				ASSERT_EQ(
					b_h_coefs[i + rid * (CUDAEngine::N)].re % CUDAEngine::RNSBPrimes[rid],
					poly_get_coeff(ctx, &b_Q, i) % CUDAEngine::RNSBPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
				ASSERT_EQ(
					b_h_coefs[i + rid * (CUDAEngine::N)].imag % CUDAEngine::RNSBPrimes[rid],
					poly_get_coeff(ctx, &b_Q, i + nphi/2) % CUDAEngine::RNSBPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
			}
		// Tries the result
		for(unsigned int rid = 0; rid < CUDAEngine::RNSBPrimes.size(); rid++)
			for(int i = 0; i < nphi/2; i++){
				ASSERT_EQ(
					c_h_coefs[i + rid * (CUDAEngine::N)].re % CUDAEngine::RNSBPrimes[rid],
					coeff(ntl_c, i) % CUDAEngine::RNSBPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
				ASSERT_EQ(
					c_h_coefs[i + rid * (CUDAEngine::N)].imag % CUDAEngine::RNSBPrimes[rid],
					coeff(ntl_c, i + nphi/2) % CUDAEngine::RNSBPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
			}

		// Release 
		poly_free(ctx, &a_Q);
		poly_free(ctx, &b_Q);
		poly_free(ctx, &a_B);
		poly_free(ctx, &b_B);
		poly_free(ctx, &c_B);
		free(a_h_coefs);
		free(b_h_coefs);
		free(c_h_coefs);
	}
}

// Tests polynomial multiplication by an 32 bits int
TEST_P(TestArithmetic, MulByInt)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a,c;
		uint64_t b;
		poly_init(ctx, &a);

		// Sample random polynomials
		// int degree = (rand() % nphi);
		Sampler::sample(ctx, &a, DISCRETE_GAUSSIAN);
		b = (rand() % t);

		ZZ_pX ntl_a, ntl_c;
		for(int i = 0; i < nphi; i++)
			SetCoeff(ntl_a, i, conv<ZZ_p>(poly_get_coeff(ctx, &a, i)));

		poly_mul_int(ctx, &c, &a, b);
		ntl_c = ntl_a * b % conv<ZZ_pX>(NTL_Phi) ;

		for(int i = 0; i < nphi;i++)
			ASSERT_EQ(poly_get_coeff(ctx, &c,i), conv<ZZ>(coeff(ntl_c, i))) << "Fail at index " << i;

		poly_free(ctx, &a);
		poly_free(ctx, &c);
	}
}

//
// Tests samples from the uniform distribution
TEST_P(TestSampler, Uniform)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		Sampler::sample(ctx, &a, UNIFORM);

		// Recovers RNS's residues and calls IRNS
		poly_copy_to_host(ctx, &a);

		// Verify consistency along residues
		for(int i = 0; i < CUDAEngine::N; i++){
			ZZ sum = to_ZZ(0);
			for(int j = 0; j < CUDAEngine::get_n_residues(a.base); j++){
				ASSERT_LT(ctx->h_coefs[i + j * CUDAEngine::N].re, COPRIMES_BUCKET[j]);
				ASSERT_LT(ctx->h_coefs[i + j * CUDAEngine::N].imag, COPRIMES_BUCKET[j]);
				sum += to_ZZ(ctx->h_coefs[i + j * CUDAEngine::N].re) + to_ZZ(ctx->h_coefs[i + j * CUDAEngine::N].imag);
			}
			ASSERT_GT(sum, to_ZZ(0));
		}

		poly_free(ctx, &a);
	}
}

double compute_norm(GaussianInteger* a, int length, int64_t p){
	ZZ aux = to_ZZ(0);
	for(int i = 0; i < length; i++){
		ZZ v1 = to_ZZ(a[i].re);
		ZZ v2 = to_ZZ(a[i].imag);

		v1 = (v1 < p/2 ? v1 : v1 - p);
		v2 = (v2 < p/2 ? v2 : v2 - p);

		aux += v1 * v1 + v2 * v2;
	}
	return conv<double>(sqrt(to_RR(aux)));
}

//
// Tests samples from the narrow distribution
TEST_P(TestSampler, Narrow)
{
	double avgnorm = 0;
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		Sampler::sample(ctx, &a, NARROW);

		poly_copy_to_host(ctx, &a);

		// Verify the narrow
		for(int i = 0; i < CUDAEngine::N; i++){
			ASSERT_TRUE(
				ctx->h_coefs[i].re == 0 ||
				ctx->h_coefs[i].re == 1 ||
				ctx->h_coefs[i].re == COPRIMES_BUCKET[0] - 1);
			ASSERT_TRUE(
				ctx->h_coefs[i].imag == 0 ||
				ctx->h_coefs[i].imag == 1 ||
				ctx->h_coefs[i].imag == COPRIMES_BUCKET[0] - 1);
		}
		avgnorm += compute_norm(ctx->h_coefs, CUDAEngine::N, COPRIMES_BUCKET[0]);

		// Verify consistency along residues
		for(int i = 0; i < CUDAEngine::N; i++)
			for(int j = 0; j < CUDAEngine::get_n_residues(a.base); j++){
				ASSERT_EQ(
					(int64_t) (ctx->h_coefs[i].re <= 1? ctx->h_coefs[i].re : -1),
					(int64_t) (ctx->h_coefs[i + j * CUDAEngine::N].re <= 1?
						ctx->h_coefs[i + j * CUDAEngine::N].re : -1)
					) << "Inconsistency at index " << i << " and rid " << j;
				ASSERT_EQ(
					(int64_t) (ctx->h_coefs[i].imag <= 1? ctx->h_coefs[i].imag : -1),
					(int64_t) (ctx->h_coefs[i + j * CUDAEngine::N].imag <= 1?
						ctx->h_coefs[i + j * CUDAEngine::N].imag : -1)
					) << "Inconsistency at index " << i << " and rid " << j;
			}

		poly_free(ctx, &a);
	}
	avgnorm /= NTESTS;
	std::cout << "Average norm-2: " << avgnorm << std::endl;
}

//
// Tests samples from the discrete gaussian distribution
TEST_P(TestSampler, DiscreteGaussian)
{
	double avgnorm = 0;
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		Sampler::sample(ctx, &a, DISCRETE_GAUSSIAN);

		poly_copy_to_host(ctx, &a);
		
		// Verify the range
		uint64_t acc = 0;
		for(int i = 0; i < CUDAEngine::N; i++){
			ASSERT_LE(
				(int64_t) (ctx->h_coefs[i].re < 100?
					ctx->h_coefs[i].re :
					ctx->h_coefs[i].re - COPRIMES_BUCKET[0]),
				GAUSSIAN_STD_DEVIATION * 6 + GAUSSIAN_BOUND);
			ASSERT_LE(
				(int64_t) (ctx->h_coefs[i].imag < 100?
					ctx->h_coefs[i].imag :
					ctx->h_coefs[i].imag - COPRIMES_BUCKET[0]),
				GAUSSIAN_STD_DEVIATION * 6 + GAUSSIAN_BOUND);
			ASSERT_GE(
				(int64_t) (ctx->h_coefs[i].re < 100?
					ctx->h_coefs[i].re :
					ctx->h_coefs[i].re - COPRIMES_BUCKET[0]),
				-(GAUSSIAN_STD_DEVIATION * 6 + GAUSSIAN_BOUND));
			ASSERT_GE(
				(int64_t) (ctx->h_coefs[i].imag < 100?
					ctx->h_coefs[i].imag :
					ctx->h_coefs[i].imag - COPRIMES_BUCKET[0]),
				-(GAUSSIAN_STD_DEVIATION * 6 + GAUSSIAN_BOUND));
			acc += ctx->h_coefs[i].re + ctx->h_coefs[i].imag;
		}
		ASSERT_GT(acc, 0);
		avgnorm += compute_norm(ctx->h_coefs, CUDAEngine::N, COPRIMES_BUCKET[0]);

		// Verify consistency along residues
		for(int i = 0; i < CUDAEngine::N; i++)
			for(int j = 0; j < CUDAEngine::get_n_residues(a.base); j++){
				ASSERT_EQ(
					(int64_t) (ctx->h_coefs[i].re < 100?
						ctx->h_coefs[i].re :  ctx->h_coefs[i].re - COPRIMES_BUCKET[0]),
					(int64_t) (ctx->h_coefs[i + j * CUDAEngine::N].re < 100?
						ctx->h_coefs[i + j * CUDAEngine::N].re :
						 ctx->h_coefs[i + j * CUDAEngine::N].re - COPRIMES_BUCKET[j])
					) << "Inconsistency at index " << i << " and rid " << j;
				ASSERT_EQ(
					(int64_t) (ctx->h_coefs[i].imag < 100?
						ctx->h_coefs[i].imag :  ctx->h_coefs[i].imag - COPRIMES_BUCKET[0]),
					(int64_t) (ctx->h_coefs[i + j * CUDAEngine::N].imag < 100?
						ctx->h_coefs[i + j * CUDAEngine::N].imag :
						 ctx->h_coefs[i + j * CUDAEngine::N].imag - COPRIMES_BUCKET[j])
					) << "Inconsistency at index " << i << " and rid " << j;
			}

		poly_free(ctx, &a);
	}
	avgnorm /= NTESTS;
	std::cout << "Average norm-2: " << avgnorm << std::endl;
}

// Tests samples from the binary distribution
TEST_P(TestSampler, HammingWeight)
{
	double avgnorm = 0;
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		Sampler::sample(ctx, &a, HAMMINGWEIGHT);

		poly_copy_to_host(ctx, &a);

		// Verify the hamming weight
		int count = 0;
		for(int i = 0; i < CUDAEngine::N; i++){
			count += (ctx->h_coefs[i].re != 0);
			count += (ctx->h_coefs[i].imag != 0);
		}
		ASSERT_EQ(count, std::min(HAMMINGWEIGHT_H, nphi));
		avgnorm += compute_norm(ctx->h_coefs, CUDAEngine::N, COPRIMES_BUCKET[0]);

		// Verify consistency along residues
		for(int i = 0; i < CUDAEngine::N; i++)
			for(int j = 0; j < CUDAEngine::get_n_residues(a.base); j++){
				ASSERT_EQ(
					(int64_t) (ctx->h_coefs[i].re <= 1? ctx->h_coefs[i].re : -1),
					(int64_t) (ctx->h_coefs[i + j * CUDAEngine::N].re <= 1?
						ctx->h_coefs[i + j * CUDAEngine::N].re : -1)
					) << "Inconsistency at index " << i << " and rid " << j;
				ASSERT_EQ(
					(int64_t) (ctx->h_coefs[i].imag <= 1? ctx->h_coefs[i].imag : -1),
					(int64_t) (ctx->h_coefs[i + j * CUDAEngine::N].imag <= 1?
						ctx->h_coefs[i + j * CUDAEngine::N].imag : -1)
					) << "Inconsistency at index " << i << " and rid " << j;
			}

		poly_free(ctx, &a);
	}
	avgnorm /= NTESTS;
	std::cout << "Average norm-2: " << avgnorm << std::endl;
}

// Tests samples from the binary distribution
TEST_P(TestSampler, Binary)
{
	double avgnorm = 0;
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		Sampler::sample(ctx, &a, BINARY);

		poly_copy_to_host(ctx, &a);

		// Verify the hamming weight
		for(int i = 0; i < CUDAEngine::N; i++){
			ASSERT_TRUE(ctx->h_coefs[i].re == 0 || ctx->h_coefs[i].re == 1);
			ASSERT_TRUE(ctx->h_coefs[i].imag == 0 || ctx->h_coefs[i].imag == 1);
		}
		avgnorm += compute_norm(ctx->h_coefs, CUDAEngine::N, COPRIMES_BUCKET[0]);

		// Verify consistency along residues
		for(int i = 0; i < CUDAEngine::N; i++)
			for(int j = 0; j < CUDAEngine::get_n_residues(a.base); j++){
				ASSERT_EQ(
					ctx->h_coefs[i].re,
					ctx->h_coefs[i + j * CUDAEngine::N].re
					) << "Inconsistency at index " << i << " and rid " << j;
				ASSERT_EQ(
					ctx->h_coefs[i].imag,
					ctx->h_coefs[i + j * CUDAEngine::N].imag
					) << "Inconsistency at index " << i << " and rid " << j;
			}

		poly_free(ctx, &a);
	}
	avgnorm /= NTESTS;
	std::cout << "Average norm-2: " << avgnorm << std::endl;
}

//
//Defines for which parameters set cuPoly will be tested.
//It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<TestParams> params = ::testing::Values(
	// {   k, kl, nphi, prec},
	(TestParams){2, 5, 128, 45},
	(TestParams){2, 5, 2048, 45},
	(TestParams){3, 5, 4096, 45},
	(TestParams){5, 5, 8192, 45},
	(TestParams){2, 5, 32, 48},
	(TestParams){2, 5, 128, 48},
	(TestParams){2, 5, 2048, 48},
	(TestParams){3, 5, 4096, 48},
	(TestParams){5, 5, 8192, 48},
	(TestParams){2, 5, 128, 55},
	(TestParams){2, 5, 2048, 55},
	(TestParams){3, 5, 4096, 55},
	(TestParams){5, 5, 8192, 55}
	);

std::string printParamName(::testing::TestParamInfo<TestParams> p){
	TestParams params = p.param;

	return std::to_string(params.nphi) +
	"_k" + std::to_string(params.k) + "_kl" + std::to_string(params.kl) +
	"_prec" + std::to_string(params.prec);
}

INSTANTIATE_TEST_CASE_P(cuPolyCKKSInstantiation,
	TestArithmetic,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(cuPolyCKKSInstantiation,
	TestRNS,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(cuPolyCKKSInstantiation,
	TestDGT,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(cuPolyCKKSInstantiation,
	TestSampler,
	params,
	printParamName
);

int main(int argc, char **argv) {

  //////////////////////////
  ////////// Google tests //
  //////////////////////////
  std::cout << "Testing cuPoly " << GET_CUPOLY_VERSION() << std::endl;
  ::testing::InitGoogleTest(&argc, argv);
  
  return RUN_ALL_TESTS();
}
