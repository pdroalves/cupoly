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
	int logq2;
	int nphi;
} TestParams;

//int LOGLEVEL = INFO;
int LOGLEVEL = QUIET;
unsigned int NTESTS = 100;

// Focus on the arithmetic
class TestArithmetic : public ::testing::TestWithParam<TestParams> {
	protected:

	ZZ q;
	uint64_t t;
	ZZ_pX NTL_Phi;
	int nphi;
	Context *ctx;

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = ceil((float)GetParam().logq2 / 63);
		Logger::getInstance()->log_info(("k: " + std::to_string(k)).c_str());
		nphi = GetParam().nphi;
		t = 256;

		// Init
		CUDAEngine::init(k, k + 1, nphi, t);// Init ClUDA
		q = CUDAEngine::RNSProduct;
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

	ZZ q;
	uint64_t t;
	ZZ_pX NTL_Phi;
	int nphi;
	Context *ctx;

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = ceil((float)GetParam().logq2 / 63);
		Logger::getInstance()->log_info(("k: " + std::to_string(k)).c_str());
		nphi = GetParam().nphi;
		t = 256;

		// Init
		CUDAEngine::init(k, k + 1, nphi, t);// Init ClUDA
		q = CUDAEngine::RNSProduct;
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

	ZZ q;
	uint64_t t;
	ZZ_pX NTL_Phi;
	int nphi;
	Context *ctx;

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = ceil((float)GetParam().logq2 / 63);
		Logger::getInstance()->log_info(("k: " + std::to_string(k)).c_str());
		nphi = GetParam().nphi;
		t = 256;

		// Init
		CUDAEngine::init(k, k + 1, nphi, t);// Init ClUDA
		q = CUDAEngine::RNSProduct;
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

	ZZ q;
	ZZ_pX NTL_Phi;
	int nphi;
	Context *ctx;

	public:
	// Test arithmetic functions
	__host__ void SetUp(){
		cudaDeviceReset();

		srand(0);
		NTL::SetSeed(conv<ZZ>(0));
		Logger::getInstance()->set_mode(LOGLEVEL);

		// TestParams
		int k = ceil((float)GetParam().logq2 / 63);
		Logger::getInstance()->log_info(("k: " + std::to_string(k)).c_str());
		nphi = GetParam().nphi;
		uint64_t t = 256;

		// Init
		CUDAEngine::init(k, k + 1, nphi, t);// Init CUDA
		q = CUDAEngine::RNSProduct;
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

uint64_t rand64() {
    // Assuming RAND_MAX is 2^32-1
    uint64_t r = rand();
    r = r<<32 | rand();
    r = r<<32 | rand();
    return r;
}

#define TEST_DESCRIPTION(desc) RecordProperty("description", desc)
TEST_P(TestDGT, mod)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		// Sample a random 63 bits uint64_t
		uint128_t a;
		a.hi = rand64() >> 2;
		a.lo = rand64();

		ZZ a_ntl = to_ZZ(a.hi);
		a_ntl <<= 64;
		a_ntl += a.lo;

		// Test for all the initialized coprimes
		for(unsigned int i = 0; i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			uint64_t x = mod(a, i);
			ASSERT_EQ(x, a_ntl % to_ZZ(p));
		}

	}
}

TEST_P(TestDGT, addmod)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		// Sample a random 63 bits uint64_t
		uint64_t a = rand64() >> 1;
		// Sample a random 63 bits uint64_t
		uint64_t b = rand64() >> 1;

		ZZ a_ntl = to_ZZ(a);
		ZZ b_ntl = to_ZZ(b);

		// Test for all the initialized coprimes
		for(unsigned int i = 0; i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			ASSERT_EQ(addmod(a, b, i), (a_ntl + b_ntl) % to_ZZ(p));
		}

	}
}

TEST_P(TestDGT, submod)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		// Sample a random 63 bits uint64_t
		uint64_t a = rand64() >> 1;
		// Sample a random 63 bits uint64_t
		uint64_t b = rand64() >> 1;

		ZZ a_ntl = to_ZZ(a);
		ZZ b_ntl = to_ZZ(b);

		// Test for all the initialized coprimes
		for(unsigned int i = 0; i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			ASSERT_EQ(submod(a, b, i), (a_ntl - b_ntl) % to_ZZ(p));
		}

	}
}

TEST_P(TestDGT, mulmod)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		// Sample a random 63 bits uint64_t
		uint64_t a = rand64() >> 1;
		// Sample a random 63 bits uint64_t
		uint64_t b = rand64() >> 1;

		ZZ a_ntl = to_ZZ(a);
		ZZ b_ntl = to_ZZ(b);

		// Test for all the initialized coprimes
		for(unsigned int i = 0; i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			ASSERT_EQ(mulmod(a, b, i), (a_ntl * b_ntl) % to_ZZ(p));
		}

	}
}

TEST_P(TestDGT, powmod)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		// Sample a random 63 bits uint64_t
		uint64_t a = rand64() >> 1;
		// Sample a random 63 bits uint64_t
		uint64_t b = rand64() >> 1;

		ZZ a_ntl = to_ZZ(a);
		ZZ b_ntl = to_ZZ(b);

		// Test for all the initialized coprimes
		for(unsigned int i = 0; i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); i++){
			uint64_t p = (uint64_t) COPRIMES_BUCKET[i];
			uint64_t x = fast_pow(a, b, i);
			ASSERT_EQ(x, PowerMod(a_ntl, b_ntl, to_ZZ(p)));
		}

	}
}
////////////////////////////////////////////////////////////////////////////////
// GaussianIntegerArithmetic
////////////////////////////////////////////////////////////////////////////////

TEST_P(TestDGT, GIAdd) {
  for(unsigned int i = 0; i < NTESTS; i++){


	// Test for all the initialized coprimes
	for(unsigned int i = 0; i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); i++){

	    GaussianInteger a = {rand64(), rand64()};
	    GaussianInteger b = {rand64(), rand64()};
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
}

TEST_P(TestDGT, GISub) {
  for(unsigned int i = 0; i < NTESTS; i++){

	// Test for all the initialized coprimes
	for(unsigned int i = 0; i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); i++){

	    GaussianInteger a = {rand64(), rand64()};
	    GaussianInteger b = {rand64(), rand64()};
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
  for(unsigned int i = 0; i < NTESTS; i++){
	// Test for all the initialized coprimes
	for(unsigned int i = 0; i < (CUDAEngine::RNSPrimes.size() + CUDAEngine::RNSBPrimes.size()); i++){

	    GaussianInteger a = {rand64(), rand64()};
	    GaussianInteger b = {rand64(), rand64()};
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
	      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " * " << "(" << b.re << ", " << b.imag << ")";
	    ASSERT_EQ(
	      received.imag,
	      expected.imag
	      ) << "Failure! (" << a.re << ", " << a.imag << ")" << " * " << "(" << b.re << ", " << b.imag << ")";
	}
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
TEST_P(TestRNS, Serializing)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a;
		poly_init(ctx, &a);

		// int degree = (rand() % nphi);
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

		poly_rho_bfv(ctx, A, &a);
		poly_xi_bfv(ctx, B, &b);

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

TEST_P(TestRNS, SimpleBasisExtensionQuadQtoB)
{
	poly_t a_Q, a_B;
	poly_init(ctx, &a_Q);
	poly_init(ctx, &a_B, BBase);

	for(int i = 0; i < nphi;i++)
		poly_set_coeff(ctx, &a_Q, i, to_ZZ(i*i % t));

	////////////////////
	// Copy to device //
	////////////////////
	poly_copy_to_device(ctx, &a_Q);

	////////////
	// Extend //
	////////////
	poly_basis_extension_Q_to_B(ctx, &a_B, &a_Q);

	//////////////////
	// Copy to Host //
	//////////////////
	// Recover RNS's residues and call ICRT
	GaussianInteger* a_Q_h_coefs = (GaussianInteger*) malloc(poly_get_residues_size(QBase));
	DGTEngine::execute_dgt(
		a_Q.d_coefs,
		a_Q.base,
		INVERSE,
		ctx
	);

	cudaMemcpyAsync(
		a_Q_h_coefs,
		a_Q.d_coefs,
		poly_get_residues_size(a_Q.base),
		cudaMemcpyDeviceToHost,
		ctx->get_stream()
	);
  	cudaCheckError();

	GaussianInteger* a_B_h_coefs = (GaussianInteger*) malloc(poly_get_residues_size(BBase));
	DGTEngine::execute_dgt(
		a_B.d_coefs,
		a_B.base,
		INVERSE,
		ctx
	);
  
	cudaMemcpyAsync(
		a_B_h_coefs,
		a_B.d_coefs,
		poly_get_residues_size(a_B.base),
		cudaMemcpyDeviceToHost,
		ctx->get_stream()
	);
  	cudaCheckError();

	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();

	// Verifies base Q
	for(unsigned int rid = 0; rid < CUDAEngine::RNSPrimes.size(); rid++)
		for(int i = 0; i < nphi/2;i++){
			ASSERT_EQ(
				a_Q_h_coefs[i + rid * CUDAEngine::N].re % CUDAEngine::RNSPrimes[rid]  % t,
				i*i % t
				) << "Fail at index " << i << ", rid: " << rid;
			ASSERT_EQ(
				a_Q_h_coefs[i + rid * CUDAEngine::N].imag % CUDAEngine::RNSPrimes[rid]  % t,
				(i + CUDAEngine::N) * (i + CUDAEngine::N) % t
				) << "Fail at index " << i << ", rid: " << rid;
		}
	
	// Verifies base B
	for(unsigned int rid = 0; rid < CUDAEngine::RNSBPrimes.size(); rid++)
		for(int i = 0; i < nphi/2;i++){
			ASSERT_EQ(
				a_B_h_coefs[i + rid * CUDAEngine::N].re % CUDAEngine::RNSBPrimes[rid] % t,
				i*i % t
				) << "Fail at index " << i << ", rid: " << rid;
			ASSERT_EQ(
				a_B_h_coefs[i + rid * CUDAEngine::N].imag % CUDAEngine::RNSBPrimes[rid] % t,
				(i + CUDAEngine::N) * (i + CUDAEngine::N) % t
				) << "Fail at index " << i << ", rid: " << rid;
		}

	poly_free(ctx, &a_Q);
	poly_free(ctx, &a_B);
	free(a_Q_h_coefs);
	free(a_B_h_coefs);
}

TEST_P(TestRNS, SimpleBasisExtensionUniformQtoB)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a_Q, a_B, b;
		poly_init(ctx, &a_Q);
		poly_init(ctx, &a_B, BBase);
		poly_init(ctx, &b);

		Sampler::sample(ctx, &a_Q, UNIFORM);
		poly_copy(ctx, &b, &a_Q);
		
		ASSERT_TRUE(poly_are_equal(ctx, &a_Q, &b));

		// to device
		poly_copy_to_device(ctx, &a_Q);
		poly_basis_extension_Q_to_B(ctx, &a_B, &a_Q);

		//////////////////
		// Copy to Host //
		//////////////////
		// Recovers RNS's residues and calls IRNS
		GaussianInteger* a_Q_h_coefs = (GaussianInteger*) malloc(poly_get_residues_size(QBase));
		DGTEngine::execute_dgt(
			a_Q.d_coefs,
			a_Q.base,
			INVERSE,
			ctx
		);

		cudaMemcpyAsync(
			a_Q_h_coefs,
			a_Q.d_coefs,
			poly_get_residues_size(a_Q.base),
			cudaMemcpyDeviceToHost,
			ctx->get_stream()
		);
		cudaCheckError();

		GaussianInteger* a_B_h_coefs = (GaussianInteger*) malloc(poly_get_residues_size(BBase));
		DGTEngine::execute_dgt(
			a_B.d_coefs,
			a_B.base,
			INVERSE,
			ctx
		);

		cudaMemcpyAsync(
			a_B_h_coefs,
			a_B.d_coefs,
			poly_get_residues_size(a_B.base),
			cudaMemcpyDeviceToHost,
			ctx->get_stream()
		);
		cudaCheckError();

		cudaStreamSynchronize(ctx->get_stream());
		cudaCheckError();

		// Verifies base Q
		for(unsigned int rid = 0; rid < CUDAEngine::RNSPrimes.size(); rid++)
			for(int i = 0; i < nphi/2;i++){
				ASSERT_EQ(
					a_Q_h_coefs[i + rid*(CUDAEngine::N)].re % CUDAEngine::RNSPrimes[rid],
					poly_get_coeff(ctx, &b, i) % CUDAEngine::RNSPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
				ASSERT_EQ(
					a_Q_h_coefs[i + rid*(CUDAEngine::N)].imag % CUDAEngine::RNSPrimes[rid],
					poly_get_coeff(ctx, &b, i + nphi/2) % CUDAEngine::RNSPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
			}
		
		// Verifies base B
		for(unsigned int rid = 0; rid < CUDAEngine::RNSBPrimes.size(); rid++)
			for(int i = 0; i < nphi/2;i++){
				ASSERT_EQ(
					a_B_h_coefs[i + rid*(CUDAEngine::N)].re % CUDAEngine::RNSBPrimes[rid],
					poly_get_coeff(ctx, &b, i) % CUDAEngine::RNSBPrimes[rid]
				) << "Fail at index " << i << ", rid: " << rid;
				ASSERT_EQ(
					a_B_h_coefs[i + rid*(CUDAEngine::N)].imag % CUDAEngine::RNSBPrimes[rid],
					poly_get_coeff(ctx, &b, i + nphi/2) % CUDAEngine::RNSBPrimes[rid]
				) << "Fail at index " << i << ", rid: " << rid;
			}

		poly_free(ctx, &a_Q);
		poly_free(ctx, &a_B);
		poly_free(ctx, &b);
		free(a_Q_h_coefs);
		free(a_B_h_coefs);
	}
}

TEST_P(TestRNS, SimpleBasisExtensionQuadQtoQB)
{
	poly_t a_Q, a_QB;
	poly_init(ctx, &a_Q);
	poly_init(ctx, &a_QB, QBBase);

	for(int i = 0; i < nphi;i++)
		poly_set_coeff(ctx, &a_Q, i, to_ZZ(i*i % t));

	////////////////////
	// Copy to device //
	////////////////////
	poly_copy_to_device(ctx, &a_Q);

	////////////
	// Extend //
	////////////
	poly_basis_extension_Q_to_QB(ctx, &a_QB, &a_Q);

	//////////////////
	// Copy to Host //
	//////////////////
	// Recover RNS's residues and call ICRT
	GaussianInteger* a_QB_h_coefs = (GaussianInteger*) malloc(poly_get_residues_size(a_QB.base));
	DGTEngine::execute_dgt(
		a_QB.d_coefs,
		a_QB.base,
		INVERSE,
		ctx
	);

	cudaMemcpyAsync(
		a_QB_h_coefs,
		a_QB.d_coefs,
		poly_get_residues_size(a_QB.base),
		cudaMemcpyDeviceToHost,
		ctx->get_stream()
	);
  	cudaCheckError();

	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();

	// Verifies base Q
	for(unsigned int rid = 0; rid < CUDAEngine::RNSPrimes.size(); rid++)
		for(int i = 0; i < nphi/2;i++){
			ASSERT_EQ(
				a_QB_h_coefs[i + rid * CUDAEngine::N].re % CUDAEngine::RNSPrimes[rid],
				i*i % t
				) << "Fail at index " << i << ", rid: " << rid;
			ASSERT_EQ(
				a_QB_h_coefs[i + rid * CUDAEngine::N].imag % CUDAEngine::RNSPrimes[rid],
				(i + CUDAEngine::N) * (i + CUDAEngine::N) % t
				) << "Fail at index " << i << ", rid: " << rid;
		}
	
	// Verifies base B
	for(unsigned int rid = 0; rid < CUDAEngine::RNSBPrimes.size(); rid++)
		for(int i = 0; i < nphi/2;i++){
			ASSERT_EQ(
				a_QB_h_coefs[i + (rid + CUDAEngine::RNSPrimes.size()) * CUDAEngine::N].re % CUDAEngine::RNSBPrimes[rid],
				i*i % t
				) << "Fail at index " << i << ", rid: " << rid;
			ASSERT_EQ(
				a_QB_h_coefs[i + (rid + CUDAEngine::RNSPrimes.size()) * CUDAEngine::N].imag % CUDAEngine::RNSBPrimes[rid],
				(i + CUDAEngine::N) * (i + CUDAEngine::N) % t
				) << "Fail at index " << i << ", rid: " << rid;
		}

	poly_free(ctx, &a_QB);
	free(a_QB_h_coefs);
}

TEST_P(TestRNS, SimpleBasisExtensionUniformQtoQB)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a_Q, a_QB;
		poly_init(ctx, &a_Q);
		poly_init(ctx, &a_QB, QBBase);

		Sampler::sample(ctx, &a_Q, UNIFORM);

		////////////////////
		// Copy to device //
		////////////////////
		poly_copy_to_device(ctx, &a_Q);

		////////////
		// Extend //
		////////////
		poly_basis_extension_Q_to_QB(ctx, &a_QB, &a_Q);

		//////////////////
		// Copy to Host //
		//////////////////
		// Recovers RNS's residues and calls IRNS
		GaussianInteger* a_QB_h_coefs = (GaussianInteger*) malloc(poly_get_residues_size(a_QB.base));
		DGTEngine::execute_dgt(
			a_QB.d_coefs,
			a_QB.base,
			INVERSE,
			ctx
		);

		cudaMemcpyAsync(
			a_QB_h_coefs,
			a_QB.d_coefs,
			poly_get_residues_size(a_QB.base),
			cudaMemcpyDeviceToHost,
			ctx->get_stream()
		);
	  	cudaCheckError();

		cudaStreamSynchronize(ctx->get_stream());
		cudaCheckError();

		// Verifies base Q
		for(unsigned int rid = 0; rid < CUDAEngine::RNSPrimes.size(); rid++)
			for(int i = 0; i < nphi/2;i++){
				ASSERT_EQ(
					a_QB_h_coefs[i + rid*(CUDAEngine::N)].re % CUDAEngine::RNSPrimes[rid],
					poly_get_coeff(ctx, &a_Q, i) % CUDAEngine::RNSPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
				ASSERT_EQ(
					a_QB_h_coefs[i + rid*(CUDAEngine::N)].imag % CUDAEngine::RNSPrimes[rid],
					poly_get_coeff(ctx, &a_Q, i + nphi/2) % CUDAEngine::RNSPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
			}
		
		// Verifies base B
		for(unsigned int rid = 0; rid < CUDAEngine::RNSBPrimes.size(); rid++)
			for(int i = 0; i < nphi/2;i++){
				ASSERT_EQ(
					a_QB_h_coefs[i + (rid + CUDAEngine::RNSPrimes.size())*(CUDAEngine::N)].re % CUDAEngine::RNSBPrimes[rid],
					poly_get_coeff(ctx, &a_Q, i) % CUDAEngine::RNSBPrimes[rid]
				) << "Fail at index " << i << ", rid: " << rid;
				ASSERT_EQ(
					a_QB_h_coefs[i + (rid + CUDAEngine::RNSPrimes.size())*(CUDAEngine::N)].imag % CUDAEngine::RNSBPrimes[rid],
					poly_get_coeff(ctx, &a_Q, i + nphi/2) % CUDAEngine::RNSBPrimes[rid]
				) << "Fail at index " << i << ", rid: " << rid;
			}

		poly_free(ctx, &a_QB);
		poly_free(ctx, &a_Q);
	}
}

// TEST_P(TestRNS, SimpleBasisExtensionQuadBtoQ)
// {
// 	poly_t a, a_Q, a_B;
// 	poly_init(ctx, &a);
// 	poly_init(ctx, &a_Q);
// 	poly_init(ctx, &a_B, BBase);

// 	for(int i = 0; i < nphi;i++)
// 		poly_set_coeff(ctx, &a, i, to_ZZ(i*i % t));
// 	poly_basis_extension_Q_to_B(ctx, &a_B, &a);

// 	////////////
// 	// Extend //
// 	////////////
// 	poly_basis_extension_B_to_Q(ctx, &a_Q, &a_B);

// 	///////////
// 	// Check //
// 	///////////
// 	for(int i = 0; i < nphi;i++)
// 		ASSERT_EQ(
// 			poly_get_coeff(ctx, &a,i),
// 			poly_get_coeff(ctx, &a_Q,i)) << "Fail at index " << i;

// 	poly_free(ctx, &a);
// 	poly_free(ctx, &a_Q);
// 	poly_free(ctx, &a_B);
// }

// TEST_P(TestRNS, SimpleBasisExtensionUniformBtoQ)
// {
// 	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
// 		poly_t a, a_Q, a_B;
// 		poly_init(ctx, &a);
// 		poly_init(ctx, &a_Q);
// 		poly_init(ctx, &a_B, BBase);

// 		Sampler::sample(ctx, &a, UNIFORM);
// 		poly_basis_extension_Q_to_B(ctx, &a_B, &a);

// 		////////////
// 		// Extend //
// 		////////////
// 		poly_basis_extension_B_to_Q(ctx, &a_Q, &a_B);

// 		///////////
// 		// Check //
// 		///////////
// 		for(int i = 0; i < nphi;i++)
// 			ASSERT_EQ(
// 				poly_get_coeff(ctx, &a,i),
// 				poly_get_coeff(ctx, &a_Q,i)) << "Fail at index " << i;

// 		poly_free(ctx, &a);
// 		poly_free(ctx, &a_Q);
// 		poly_free(ctx, &a_B);
// 	}
// }

TEST_P(TestRNS, SimpleScalingtDivQ)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a, b, c;
		poly_init(ctx, &a);
		poly_init(ctx, &b);
		poly_init(ctx, &c);

		Sampler::sample(ctx, &a, UNIFORM);
		// for(int i = 0; i < nphi;i++)
		// 	poly_set_coeff(ctx, &a, i, to_ZZ(i*i % t));

		poly_copy(ctx, &c, &a);
		ASSERT_TRUE(poly_are_equal(ctx, &a, &c));

		// Scaling
		poly_simple_scaling_tDivQ(ctx, &b, &a);

		//////////////////
		// Copy to Host //
		//////////////////
		a.status = RNSSTATE;

		// Verifies base Q
		for(unsigned int rid = 0; rid < CUDAEngine::RNSPrimes.size(); rid++)
			for(int i = 0; i < nphi; i++){
				ASSERT_EQ(
					poly_get_coeff(ctx, &a, i),
					poly_get_coeff(ctx, &c, i)
					) << "Fail at index " << i << ", rid: " << rid;
			}

		// Verifies base t
		for(unsigned int rid = 0; rid < CUDAEngine::RNSPrimes.size(); rid++)
			for(int i = 0; i < nphi; i++){
				ASSERT_EQ(
					poly_get_coeff(ctx, &b, i),
					to_ZZ(round(to_RR(poly_get_coeff(ctx, &a, i)) * t / to_RR(q))) % to_ZZ(t)
					) << "Fail at index " << i << ", rid: " << rid;
			}

		poly_free(ctx, &a);
		poly_free(ctx, &b);
		poly_free(ctx, &c);
	}
}

TEST_P(TestRNS, ComplexBasisScalingUniform)
{
	Context ctx_B;
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){
		poly_t a_Q, a_B, b, c;
		poly_init(ctx, &a_Q);
		poly_init(&ctx_B, &a_B, BBase);
		poly_init(ctx, &b);
		poly_init(ctx, &c);

		Sampler::sample(ctx, &a_Q, UNIFORM);

		poly_copy(ctx, &c, &a_Q);
		ASSERT_TRUE(poly_are_equal(ctx, &a_Q, &c));

		// Extend to base B
		poly_basis_extension_Q_to_B(ctx, &a_B, &a_Q);

		//////////////////
		// Copy to Host //
		//////////////////
		// Recovers RNS's residues and call IRNS
		GaussianInteger* a_B_h_coefs = (GaussianInteger*) malloc(poly_get_residues_size(BBase));
		GaussianInteger* a_Q_h_coefs = (GaussianInteger*) malloc(poly_get_residues_size(QBase));

		DGTEngine::execute_dgt(
			a_Q.d_coefs,
			a_Q.base,
			INVERSE,
			ctx
		);
		DGTEngine::execute_dgt(
			a_B.d_coefs,
			a_B.base,
			INVERSE,
			ctx
		);

		cudaMemcpyAsync(
			a_Q_h_coefs,
			a_Q.d_coefs,
			poly_get_residues_size(a_Q.base),
			cudaMemcpyDeviceToHost,
			ctx->get_stream()
		);
		cudaCheckError();

		cudaMemcpyAsync(
			a_B_h_coefs,
			a_B.d_coefs,
			poly_get_residues_size(a_B.base),
			cudaMemcpyDeviceToHost,
			ctx->get_stream()
		);
		cudaCheckError();

		cudaStreamSynchronize(ctx->get_stream());
		cudaCheckError();

		// Verifies base Q
		for(unsigned int rid = 0; rid < CUDAEngine::RNSPrimes.size(); rid++)
			for(int i = 0; i < nphi/2; i++){
				ASSERT_EQ(
					a_Q_h_coefs[i + rid*(CUDAEngine::N)].re % CUDAEngine::RNSPrimes[rid],
					poly_get_coeff(ctx, &c, i) % CUDAEngine::RNSPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
				ASSERT_EQ(
					a_Q_h_coefs[i + rid*(CUDAEngine::N)].imag % CUDAEngine::RNSPrimes[rid],
					poly_get_coeff(ctx, &c, i + nphi/2) % CUDAEngine::RNSPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
			}

		// Verifies base B
		for(unsigned int rid = 0; rid < CUDAEngine::RNSBPrimes.size(); rid++)
			for(int i = 0; i < nphi/2; i++){
				ASSERT_EQ(
					a_B_h_coefs[i + rid*(CUDAEngine::N)].re % CUDAEngine::RNSBPrimes[rid],
					poly_get_coeff(ctx, &c, i) % CUDAEngine::RNSBPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
				ASSERT_EQ(
					a_B_h_coefs[i + rid*(CUDAEngine::N)].imag % CUDAEngine::RNSBPrimes[rid],
					poly_get_coeff(ctx, &c, i + nphi/2) % CUDAEngine::RNSBPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
			}

		// At this point the basis extension is asserted!
		DGTEngine::execute_dgt(
			a_Q.d_coefs,
			a_Q.base,
			FORWARD,
			ctx
		);
		DGTEngine::execute_dgt(
			a_B.d_coefs,
			a_B.base,
			FORWARD,
			ctx
		);

		//////////////////////////////
		// Verifies the second step //
		//////////////////////////////
		poly_complex_scaling_tDivQ(ctx, &b, &a_Q, &a_B);

		cudaStreamSynchronize(ctx->get_stream());
		cudaCheckError();

		for(int i = 0; i < nphi; i++){
			ZZ r = NTL::abs(poly_get_coeff(ctx, &b, i) - t * poly_get_coeff(ctx, &c,i)/q);
			ASSERT_LT(r, 2) << "Fail at index " << i;
		}

		poly_free(ctx, &a_Q);
		poly_free(ctx, &a_B);
		poly_free(ctx, &b);
		poly_free(ctx, &c);
		free(a_Q_h_coefs);
		free(a_B_h_coefs);
	}
}

//
// Tests the Set/Get coeff behavior
TEST_P(TestArithmetic, SetCoeff)
{
	poly_t a;
	poly_init(ctx, &a);

	poly_set_coeff(ctx, &a,0,to_ZZ(42));

	ASSERT_EQ(poly_get_coeff(ctx, &a,0) , to_ZZ(42));
	poly_free(ctx, &a);
}


/////////////////////////////
// Size-variable integers  //
/////////////////////////////

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
		
		for(int i = 0; i < nphi;i++)
			ASSERT_EQ(
				(poly_get_coeff(ctx, &a,i)  + poly_get_coeff(ctx, &b,i)) % q,
				poly_get_coeff(ctx, &c,i)) << "Fail at index " << i;

		poly_free(ctx, &a);
		poly_free(ctx, &b);
		poly_free(ctx, &c);
	}
}

// Tests polynomial Subtraction
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

		for(int i = 0; i < nphi;i++)
			ASSERT_EQ(
				(poly_get_coeff(ctx, &a,i) - poly_get_coeff(ctx, &b,i)) % q,
				poly_get_coeff(ctx, &c,i)) << "Fail at index " << i;

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
		ntl_c = ntl_a * ntl_b % conv<ZZX>(NTL_Phi) ;

		for(int i = 0; i < nphi; i++)
			ASSERT_EQ(poly_get_coeff(ctx, &c, i) % q, coeff(ntl_c, i) % q) << "Fail at index " << i;

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

		ZZ_pX ntl_a, ntl_b, ntl_c, ntl_d;
		for(int i = 0; i < nphi; i++){
			SetCoeff(ntl_a, i, conv<ZZ_p>(poly_get_coeff(ctx, &a, i)));
			SetCoeff(ntl_b, i, conv<ZZ_p>(poly_get_coeff(ctx, &b, i)));
			SetCoeff(ntl_c, i, conv<ZZ_p>(poly_get_coeff(ctx, &c, i)));
		}

		poly_mul_add(ctx, &d, &a, &b, &c);
		ntl_d = (ntl_a * ntl_b + ntl_c) % conv<ZZ_pX>(NTL_Phi) ;

		for(int i = 0; i < nphi; i++)
			ASSERT_EQ(
				poly_get_coeff(ctx, &d,i) % q,
				conv<ZZ>(coeff(ntl_d, i))
			) << "Fail at index " << i;

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

		// Extend to base B
		poly_basis_extension_Q_to_B(ctx, &a_B, &a_Q);
		poly_basis_extension_Q_to_B(ctx, &b_B, &b_Q);

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
	
		cudaMemcpy(	
			a_h_coefs,
			a_B.d_coefs,
			poly_get_residues_size(a_B.base),
			cudaMemcpyDeviceToHost );
		cudaCheckError();

		cudaMemcpy(
			b_h_coefs,
			b_B.d_coefs,
			poly_get_residues_size(b_B.base),
			cudaMemcpyDeviceToHost );
		cudaCheckError();

		cudaMemcpy(
			c_h_coefs,
			c_B.d_coefs,
			poly_get_residues_size(c_B.base),
			cudaMemcpyDeviceToHost );
		cudaCheckError();

		// Verifies base B
		// Tries the operands
		for(unsigned int rid = 0; rid < CUDAEngine::RNSBPrimes.size(); rid++)
			for(int i = 0; i < nphi/2; i++){
				ASSERT_EQ(
					a_h_coefs[i + rid * (CUDAEngine::N)].re % CUDAEngine::RNSBPrimes[rid],
					poly_get_coeff(ctx, &a_Q, i) % CUDAEngine::RNSBPrimes[rid]
					) << "Fail at index " << i << ", rid: " << rid;
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
		int b;
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
			ASSERT_EQ(poly_get_coeff(ctx, &c,i) % q, conv<ZZ>(coeff(ntl_c, i))) << "Fail at index " << i;

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

		for(int i = 0; i < nphi; i++){
			ASSERT_GE(poly_get_coeff(ctx, &a,i), 0);
			ASSERT_LT(poly_get_coeff(ctx, &a,i), q);
		}

		poly_free(ctx, &a);
	}
}

//
// Tests samples from the narrow distribution
TEST_P(TestSampler, Narrow)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		Sampler::sample(ctx, &a, NARROW);

		for(int i = 0; i < nphi; i++)
			ASSERT_TRUE(
				poly_get_coeff(ctx, &a,i) == 0 ||
				poly_get_coeff(ctx, &a,i) == 1 ||
				poly_get_coeff(ctx, &a,i) == 2
				);

		poly_free(ctx, &a);
	}
}

//
// Tests samples from the discrete gaussian distribution
TEST_P(TestSampler, DiscreteGaussian)
{

	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		Sampler::sample(ctx, &a, DISCRETE_GAUSSIAN);

		ZZ acc = to_ZZ(0);
		for(int i = 0; i < nphi; i++){
			ASSERT_LE(
				poly_get_coeff(ctx, &a, i),
				GAUSSIAN_STD_DEVIATION * 6 + GAUSSIAN_BOUND );
			acc += poly_get_coeff(ctx, &a, i);
		}
		ASSERT_GT(acc, 0);

		poly_free(ctx, &a);
	}
}

// Tests samples from the binary distribution
TEST_P(TestSampler, HammingWeight)
{
	for(unsigned int ntest = 0; ntest < NTESTS; ntest++){

		poly_t a;

		// Sample random polynomials
		Sampler::sample(ctx, &a, HAMMINGWEIGHT);

		int count = 0;
		for(int i = 0; i < nphi; i++)
			count += (poly_get_coeff(ctx, &a, i) > 0);
		ASSERT_EQ(count, std::min(HAMMINGWEIGHT_H, nphi));

		poly_free(ctx, &a);
	}
}

//
//Defines for which parameters set cuPoly will be tested.
//It executes each test for all pairs on phis X qs (Cartesian product)
::testing::internal::ParamGenerator<TestParams> params = ::testing::Values(
	// {   logq2, nphi},
	(TestParams){30, 32},
	(TestParams){120, 32},
	(TestParams){130, 32},
	(TestParams){340, 32},
	(TestParams){30, 128},
	(TestParams){120, 128},
	(TestParams){120, 256},
	(TestParams){30, 2048},
	(TestParams){120, 2048},
	(TestParams){340, 2048},
	(TestParams){170, 4096},
	(TestParams){170, 8192},
	(TestParams){250, 8192},
	(TestParams){340, 8192}
	);

std::string printParamName(::testing::TestParamInfo<TestParams> p){
	TestParams params = p.param;

	return std::to_string(params.nphi) +
	"_q" + std::to_string(params.logq2);
}

INSTANTIATE_TEST_CASE_P(cuPolyInstantiation,
	TestArithmetic,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(cuPolyInstantiation,
	TestRNS,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(cuPolyInstantiation,
	TestDGT,
	params,
	printParamName
);

INSTANTIATE_TEST_CASE_P(cuPolyInstantiation,
	TestSampler,
	params,
	printParamName
);

int main(int argc, char **argv) {
  /////////////////////////
  // Command line parser //
  ////////////////////////
  // cxxopts::Options options("cupoly_test", "This program executes a sequence of unittests to assert the correctness of cuPoly.");
  // options.add_options()
	 //  ("l,log", "log level: INFO, DEBUG, QUIET, VERBOSE", cxxopts::value<std::string>()->default_value("QUIET"))
	 //  ("n,ntests", "quantity of repetions that should be performed", cxxopts::value<int>()->default_value("5"))
	 //  ;
  // auto result = options.parse(argc, argv);
  
  // std::cout << "Settings: " << std::endl << "Log level: " << result["log"].as<std::string>() << std::endl << "ntests: " << result["ntests"].as<int>() << std::endl; 
  // NTESTS = result["ntests"].as<int>();
  // if(result["log"].as<std::string>() == std::string("INFO"))
  // 	LOGLEVEL = INFO;
  // else if(result["log"].as<std::string>() == std::string("QUIET"))
  // 	LOGLEVEL = QUIET;
  // else if(result["log"].as<std::string>() == std::string("VERBOSE"))
  // 	LOGLEVEL = VERBOSE;
  // else if(result["log"].as<std::string>() == std::string("DEBUG"))
  // 	LOGLEVEL = DEBUG;

  //////////////////////////
  ////////// Google tests //
  //////////////////////////
  std::cout << "Testing cuPoly " << GET_CUPOLY_VERSION() << std::endl;
  ::testing::InitGoogleTest(&argc, argv);
  
  return RUN_ALL_TESTS();
}
