
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


#include <cuPoly/arithmetic/polynomial.h>
#include <iterator>

uint64_t get_cycles() {
  unsigned int hi, lo;
  asm (
    "cpuid\n\t"/*serialize*/
    "rdtsc\n\t"/*read the clock*/
    "mov %%edx, %0\n\t"
    "mov %%eax, %1\n\t"
    : "=r" (hi), "=r" (lo):: "%rax", "%rbx", "%rcx", "%rdx"
  );
  return ((uint64_t) lo) | (((uint64_t) hi) << 32);
}

//////////////
// Internal //
//////////////
__host__ int isPowerOfTwo (unsigned int x)
{
  return ((x != 0) && ((x & (~x + 1)) == x));
}

//////////////
//////////////
//////////////

__host__ size_t poly_get_residues_size(int base){

	return (CUDAEngine::N) * CUDAEngine::get_n_residues(base) *	sizeof(GaussianInteger);
}

__host__  void poly_init(Context *ctx, poly_t *a, int base){
	if(a->init)
		return;
	assert(ctx);
	Logger::getInstance()->log_debug("poly_init");

	assert(CUDAEngine::is_init);

	//
	a->base = base; // By default

	// Memory allocation on the GPU is synchronous. So, first we allocate all memory that
	// we need and then call asynchronous functions.

	// Device
	cudaMalloc((void**)&a->d_coefs, poly_get_residues_size(base));
	cudaCheckError();

	cudaMemsetAsync(a->d_coefs, 0, poly_get_residues_size(base), ctx->get_stream());
	cudaCheckError();

	// Host
	a->coefs = std::vector<ZZ>(CUDAEngine::N * 2);

	a->init = true;
}

__host__  void poly_free(Context *ctx, poly_t *a){
	if(!a || !a->init)
		return;

	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();

	// RNS residues
	a->coefs.clear();
	cudaFree(a->d_coefs);
	cudaCheckError();

	a->d_coefs = NULL;
	a->init = false;
}

__host__  void poly_clear(Context *ctx, poly_t *a){
	assert(CUDAEngine::is_init);
	poly_init(ctx, a);

	a->coefs.clear();
	
	cudaMemsetAsync(a->d_coefs,0, poly_get_residues_size(a->base), ctx->get_stream());
	cudaCheckError();

	a->status = BOTHSTATE;
}

__host__  void poly_copy_to_device(Context *ctx, poly_t *a){
	if(a->status == RNSSTATE || a->status == BOTHSTATE) // Already copied
		return;
	Logger::getInstance()->log_debug("poly_copy_to_device");
	
	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	// Computes RNS's residues and copy to the GPU
	poly_crt(ctx, a);

	cudaMemcpyAsync(
		a->d_coefs,
		ctx->h_coefs,
		poly_get_residues_size(a->base),
		cudaMemcpyHostToDevice,
		ctx->get_stream()
	);
	cudaCheckError();
  	
	DGTEngine::execute_dgt(
		a->d_coefs,
		a->base,
		FORWARD,
		ctx
	);

	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();

	a->status = BOTHSTATE;
	return;
}

__host__ void poly_copy_to_host(Context *ctx, poly_t *a){
	if(a->status == HOSTSTATE || a->status == BOTHSTATE) // Already copied
		return;

	Logger::getInstance()->log_debug("poly_copy_to_host");

	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	//////////////////
	// Copy to Host //
	//////////////////
	cudaMemcpyAsync(
		ctx->d_aux_coefs,
		a->d_coefs,
		poly_get_residues_size(a->base),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream() );

	DGTEngine::execute_dgt(
		ctx->d_aux_coefs,
		a->base,
		INVERSE,
		ctx
	);

	// Recovers RNS's residues and calls IRNS
	cudaMemcpyAsync(
		ctx->h_coefs,
		ctx->d_aux_coefs,
		poly_get_residues_size(a->base),
		cudaMemcpyDeviceToHost,
		ctx->get_stream()
	);

	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();


	//////////
	// IRNS //
	//////////
	poly_icrt(ctx, a);

	//
	a->status = BOTHSTATE;
	return;
}

__host__ GaussianInteger* poly_crt(Context *ctx, poly_t *a){
	assert(a->status == HOSTSTATE || a->status == BOTHSTATE);
	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);
	assert(a->coefs.size() <= (unsigned int)CUDAEngine::N * 2);
	assert(a->base == QBase || a->base == QBBase);

    // The product of those prime numbers should be larger than the potentially
    // 	largest coefficient of polynomial c, that we will obtain as a result of
    // 	a computation for accurate recovery through IRNS.
    //
    //  produtorio_{i=1}^n (pi) > n*q^2
	memset(ctx->h_coefs, 0, poly_get_residues_size(a->base));

	// #pragma omp parallel for
	for(int rid = 0; rid < CUDAEngine::get_n_residues(a->base); rid++){
		uint64_t prime = COPRIMES_BUCKET[rid];
		for(unsigned int cid = 0; cid < (unsigned int) CUDAEngine::N; cid++){
			// Fold
			ctx->h_coefs[cid + rid * CUDAEngine::N].re = (
				cid < a->coefs.size() ? 
				a->coefs[cid] % prime : 0
				);
			ctx->h_coefs[cid + rid * CUDAEngine::N].imag = (
				cid + CUDAEngine::N < a->coefs.size() ? 
				a->coefs[cid + CUDAEngine::N] % prime : 0
				);		}
	}

	return ctx->h_coefs;
}

__host__  void poly_icrt(Context *ctx, poly_t *a){
	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	a->coefs.clear();
	a->coefs.resize(CUDAEngine::N * 2);
	std::fill(a->coefs.begin(), a->coefs.end(), 0);

	if(a->base != QBase && a->base != TBase)
		throw std::runtime_error("Unknown base");

	for(int rid = 0; rid < CUDAEngine::get_n_residues(a->base); rid++){
		GaussianInteger *residue = &ctx->h_coefs[rid * CUDAEngine::N];

		// Iterate over residues' coefficients
		// #pragma omp parallel for
		for(int cid = 0; cid < CUDAEngine::N; cid++){
			// "untwist"
			GaussianInteger xi = residue[cid];

			// DGT untwist
			// if(a->base != TBase)
			// xi = GIMul(
			// 	xi,
			// 	DGTEngine::h_invnthroot[CUDAEngine::N][cid + rid * CUDAEngine::N],
			// 	rid);

      		ZZ c = to_ZZ(xi.re);
			switch(a->base){
				case QBase:
				c *= CUDAEngine::RNSInvMpi[rid];
				c %= to_ZZ(CUDAEngine::RNSPrimes[rid]);
				a->coefs[cid] += CUDAEngine::RNSMpi[rid] * c;
				break;
				case TBase:
				a->coefs[cid]  = c; 
				break;
				default:
				throw std::runtime_error("Unknown base");
			}

			c = to_ZZ(xi.imag);
			switch(a->base){
				case QBase:
				c *= CUDAEngine::RNSInvMpi[rid];
				c %= to_ZZ(CUDAEngine::RNSPrimes[rid]);
				a->coefs[cid + CUDAEngine::N] += CUDAEngine::RNSMpi[rid] * c;
				break;
				case TBase:
				a->coefs[cid + CUDAEngine::N] = c; 
				break;
				default:
				throw std::runtime_error("Unknown base");
			}
		}

	}
	// Reduce each coefficient by M
	// #pragma omp parallel for
	for(int cid = 0; cid < CUDAEngine::N * 2; cid++){
		switch(a->base){
			case QBase:
			a->coefs[cid] %= CUDAEngine::RNSProduct;
			break;
			case TBase:
			a->coefs[cid] %= to_ZZ(CUDAEngine::t);
			break;			
			default:
			throw std::runtime_error("Unknown base");
		}
	}

}

__host__ int poly_get_deg(Context *ctx, poly_t *a){
	if(a->status == RNSSTATE)
		poly_copy_to_host(ctx, a);

	assert(a->status == HOSTSTATE || a->status == BOTHSTATE);

	while(a->coefs.size() > 0 && a->coefs.back() == 0)
		a->coefs.pop_back();
	return a->coefs.size() - 1;
}

#ifdef BFV_ENGINE_MODE
__host__ void poly_simple_scaling_tDivQ(
	Context *ctx,
	poly_t *b,
	poly_t *a){
	assert(CUDAEngine::is_init);
	assert(ctx);
	assert(a->init);
	assert(CUDAEngine::N > 0);
	assert(a->base == QBase);

	poly_copy_to_device(ctx, a);
	poly_init(ctx, b);
	
	// Since base B is always bigger than base Q, we shall temporary
	// store data in a_B
	cudaMemcpyAsync(
		b->d_coefs,
		a->d_coefs,
		poly_get_residues_size(QBase),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream()
	);
	cudaCheckError();
	
	DGTEngine::execute_dgt(
		b->d_coefs,
		QBase,
		INVERSE,
		ctx
	);

	CUDAEngine::execute_polynomial_simple_scaling(
		b->d_coefs,
		ctx);

	b->status = RNSSTATE;
	b->base = TBase;

}

__host__ void poly_complex_scaling_tDivQ(
	Context *ctx, 
	poly_t *b_Q, // Output in base Q
	poly_t *a_Q, // Base Q
	poly_t *a_B){ // Base B        
    
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);
	assert(ctx);
	assert(a_Q->init);
	assert(a_B->init);
	assert(b_Q->init);
	assert(a_Q->base == QBase);
	assert(a_B->base == BBase);

	poly_copy_to_device(ctx, a_Q);
	poly_copy_to_device(ctx, a_B);

	// 
	DGTEngine::execute_dgt(
		a_Q->d_coefs,
		a_Q->base,
		INVERSE,
		ctx
	);
	DGTEngine::execute_dgt(
		a_B->d_coefs,
		a_B->base,
		INVERSE,
		ctx
	);

	CUDAEngine::execute_polynomial_complex_scaling(
		ctx->d_coefs_B,
		a_Q->d_coefs,
		a_B->d_coefs,
		ctx
		);

	CUDAEngine::execute_polynomial_basis_ext_B_to_Q(
		b_Q->d_coefs,
		ctx->d_coefs_B,
		ctx
		);

	DGTEngine::execute_dgt(
		b_Q->d_coefs,
		QBase,
		FORWARD,
		ctx
	);

	b_Q->status = RNSSTATE;
	b_Q->base = QBase;
}


__host__ void poly_basis_extension_Q_to_B(
	Context *ctx, // Default context
	poly_t *a_B, // Output in base B
	poly_t *a_Q){ // Input in base Q

	assert(CUDAEngine::is_init);
	assert(ctx);
	assert(a_Q->init);
	assert(a_B->init);
	assert(CUDAEngine::N > 0);
	assert(a_Q->base == QBase);
	assert(a_B->base == BBase);

	poly_copy_to_device(ctx, a_Q);
	
	// Since base B is always greater than base Q we may temporary
	// store data in a_B and save one DGT call
	cudaMemcpyAsync(
		a_B->d_coefs,
		a_Q->d_coefs,
		poly_get_residues_size(QBase),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream()
		);
	cudaCheckError();

	// 
	DGTEngine::execute_dgt(
		a_B->d_coefs,
		QBase,
		INVERSE,
		ctx
	);

	CUDAEngine::execute_polynomial_basis_ext_Q_to_B(
		a_B->d_coefs,
		ctx
	);

	// 
	DGTEngine::execute_dgt(
		a_B->d_coefs,
		BBase,
		FORWARD,
		ctx
	);

	// 
	a_B->status = RNSSTATE;
	a_B->base = BBase;
}

__host__ void poly_basis_extension_Q_to_QB(
	Context *ctx, // Default context
	poly_t *b, // Output
	poly_t *a){ // Operand in base Q

	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);
	assert(ctx);
	assert(a->init);
	assert(a->base == QBase);
	assert(b->base == QBBase);

	poly_copy_to_device(ctx, a);
	
	// Since base B is always greater than base Q we may temporary
	// store data in a_B and save one DGT call
	cudaMemcpyAsync(
		ctx->d_aux_coefs,
		a->d_coefs,
		poly_get_residues_size(QBase),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream()
		);
	cudaCheckError();

	// 
	DGTEngine::execute_dgt(
		ctx->d_aux_coefs,
		QBase,
		INVERSE,
		ctx
	);

	CUDAEngine::execute_polynomial_basis_ext_Q_to_B(
		ctx->d_aux_coefs,
		ctx
	);

	// 
	DGTEngine::execute_dgt(
		ctx->d_aux_coefs,
		BBase,
		FORWARD,
		ctx
	);

	// Copy new residues to b
	cudaMemcpyAsync(
		b->d_coefs,
		a->d_coefs,
		poly_get_residues_size(QBase),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream()
		);
	cudaCheckError();
	cudaMemcpyAsync(
		b->d_coefs + CUDAEngine::get_n_residues(QBase) * CUDAEngine::N,
		ctx->d_aux_coefs,
		poly_get_residues_size(BBase),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream()
		);
	cudaCheckError();


	// 
	b->status = RNSSTATE;
	b->base = QBBase;
}
#endif

#ifdef CKKS_ENGINE_MODE
__host__ void poly_approx_basis_reduction_QB_to_Q(
	Context *ctx, // Default context
	poly_t *a, // Output in base Q
	poly_t *b, // Input in base QB
	int level){

	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);
	assert(ctx);
	assert(a->init);
	assert(a->base == QBase);
	assert(b->base == QBBase);
	assert(level > 0);

	poly_copy_to_device(ctx, b);

	cudaMemcpyAsync(
		ctx->d_aux_coefs,
		b->d_coefs,
		poly_get_residues_size(QBBase),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream()
	);
	DGTEngine::execute_dgt(
		ctx->d_aux_coefs,
		QBBase,
		INVERSE,
		ctx
	);

	CUDAEngine::execute_approx_modulus_reduction(
		ctx,
		a->d_coefs,
		ctx->d_aux_coefs,
		level);

	DGTEngine::execute_dgt(
		a->d_coefs,
		QBase,
		FORWARD,
		ctx
	);
	a->status = RNSSTATE;
}

__host__ void poly_approx_basis_raising_Q_to_QB(
	Context *ctx, // Default context
	poly_t *a,  // Output in base QB
	poly_t *b,  // Input in base QB
	int level){

	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);
	assert(ctx);
	assert(a->init);
	assert(a->base == QBBase);
	assert(b->base == QBase);
	assert(level > 0);

	poly_copy(ctx, a, b);

	DGTEngine::execute_dgt(
		a->d_coefs,
		QBase,
		INVERSE,
		ctx
	);

	CUDAEngine::execute_approx_modulus_raising(ctx, a->d_coefs, level);

	DGTEngine::execute_dgt(
		a->d_coefs,
		QBBase,
		FORWARD,
		ctx
	);
}
#endif

#ifdef BFV_ENGINE_MODE

__host__ void poly_xi_bfv(
	Context *ctx,
	poly_t *c,
	poly_t *a){
	assert(CUDAEngine::is_init);
	assert(a->init);
	assert(CUDAEngine::N > 0);
	assert(a->base == QBase);

  	const int k = CUDAEngine::get_n_residues(a->base);

	// Keep track of each element of xi
	for(int i = 0; i < k; i++)
		ctx->h_b[i] = c[i].d_coefs;

	cudaMemcpyAsync(
		ctx->d_b,
		ctx->h_b,
		k * sizeof(GaussianInteger*),
		cudaMemcpyHostToDevice,
		ctx->get_stream()
		);
  	cudaCheckError();

	poly_copy_to_device(ctx, a);

	////////////////////
	// Computes xi() //
	////////////////////

	CUDAEngine::execute_xi_rns(
		ctx->d_b,
		a->d_coefs,
		ctx
		);

	for(int i = 0; i < k; i++){
		c[i].status = RNSSTATE;
		c[i].base = QBase;
	}
}

__host__ void poly_rho_bfv(
	Context *ctx, 
	poly_t *c,
	poly_t *a){
	assert(CUDAEngine::is_init);
	assert(a->init);
	assert(CUDAEngine::N > 0);
	// If a lies in QB the B-residues will return as zero
	assert(a->base == QBase || a->base == QBBase);

  	const int k = CUDAEngine::get_n_residues(a->base);

	// Keep track of each element of xi
	for(int i = 0; i < k; i++){
		poly_init(ctx, &c[i], a->base);
		ctx->h_b[i] = c[i].d_coefs;
	}

	cudaMemcpy(
		ctx->d_b,
		ctx->h_b,
		k * sizeof(GaussianInteger*),
		cudaMemcpyHostToDevice);
	poly_copy_to_device(ctx, a);

	////////////////////
	// Computes xi() //
	////////////////////

	CUDAEngine::execute_rho_bfv_rns(
		ctx->d_b,
		a->d_coefs,
		ctx
		);

	for(int i = 0; i < k; i++){
		c[i].status = RNSSTATE;
		c[i].base = a->base;
	}
}
#endif

#ifdef CKKS_ENGINE_MODE

__host__ void poly_xi_ckks(
	Context *ctx, 
	poly_t *c,
	poly_t *a){
	assert(CUDAEngine::is_init);
	assert(a->init);
	assert(CUDAEngine::N > 0);
	assert(a->base == QBase);

  	const int k = CUDAEngine::get_n_residues(a->base);
  	poly_init(ctx, c);
	////////////////////
	// Computes xi() //
	////////////////////

	CUDAEngine::execute_xi_ckks_rns(
		c->d_coefs,
		a->d_coefs,
		ctx
		);

	c->status = RNSSTATE;
}

__host__ void poly_rho_ckks(
	Context *ctx, 
	poly_t *c,
	poly_t *a){
	assert(CUDAEngine::is_init);
	assert(a->init);
	assert(CUDAEngine::N > 0);
	// If a lies in QB the B-residues will return as zero
	assert(a->base == QBase || a->base == QBBase);

  	const int k = CUDAEngine::get_n_residues(a->base);

	////////////////////
	// Computes xi() //
	////////////////////

	CUDAEngine::execute_rho_ckks_rns(
		c->d_coefs,
		a->d_coefs,
		ctx
		);

	c->status = RNSSTATE;
}

__host__ void poly_ckks_rescale(
	Context *ctx, 
	poly_t *a,
	poly_t *b,
	int level){
	assert(CUDAEngine::is_init);
	assert(a->init);
	assert(CUDAEngine::N > 0);
	assert(a->base == QBase);

  	poly_copy_to_device(ctx, a);

	////////////////////
	// Computes xi() //
	////////////////////

	DGTEngine::execute_dgt(
		a->d_coefs,
		QBase,
		INVERSE,
		ctx
	);

	DGTEngine::execute_dgt(
		b->d_coefs,
		QBase,
		INVERSE,
		ctx
	);

	CUDAEngine::execute_ckks_rescale(
		a->d_coefs,
		b->d_coefs,
		level,
		ctx
		);

	DGTEngine::execute_dgt(
		a->d_coefs,
		QBase,
		FORWARD,
		ctx
	);

	DGTEngine::execute_dgt(
		b->d_coefs,
		QBase,
		FORWARD,
		ctx
	);

	a->status = RNSSTATE;
}
#endif

__host__  void poly_add(Context *ctx, poly_t *c, poly_t *a, poly_t *b){
	assert(CUDAEngine::is_init);
	assert(a->init);
	assert(b->init);
	assert(CUDAEngine::N > 0);
	assert(a->base == b->base);

	poly_copy_to_device(ctx, a);
	poly_copy_to_device(ctx, b);

	poly_init(ctx, c, a->base);

	DGTEngine::execute_add_dgt(
		c->d_coefs,
		a->d_coefs,
		b->d_coefs,
		a->base,
		ctx->get_stream()
		);

	c->status = RNSSTATE;
	c->base = a->base;
}

__host__  void poly_double_add(
	Context *ctx,
	poly_t *c1,
	poly_t *a1,
	poly_t *b1,
	poly_t *c2,
	poly_t *a2,
	poly_t *b2){
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);
	assert(a1->init);
	assert(b1->init);
	assert(a2->init);
	assert(b2->init);
	assert(a1->base == b1->base);
	assert(a2->base == b2->base);
	assert(a1->base == b2->base);

	poly_copy_to_device(ctx, a1);
	poly_copy_to_device(ctx, b1);
	poly_copy_to_device(ctx, a2);
	poly_copy_to_device(ctx, b2);

	poly_init(ctx, c1);
	poly_init(ctx, c2);

	DGTEngine::execute_double_add_dgt(
		c1->d_coefs,
		a1->d_coefs,
		b1->d_coefs,
		c2->d_coefs,
		a2->d_coefs,
		b2->d_coefs,
		a1->base,
		ctx->get_stream()
		);

	c1->status = RNSSTATE;
	c1->base = a1->base;
	c2->status = RNSSTATE;
	c2->base = a2->base;
}

__host__  void poly_sub(Context *ctx, poly_t *c, poly_t *a, poly_t *b){
	assert(CUDAEngine::is_init);
	assert(a->init);
	assert(b->init);
	assert(CUDAEngine::N > 0);
	assert(a->base == b->base);

	poly_init(ctx, c, a->base);
	poly_copy_to_device(ctx, a);
	poly_copy_to_device(ctx, b);

	DGTEngine::execute_sub_dgt(
		c->d_coefs,
		a->d_coefs,
		b->d_coefs,
		a->base,
		ctx->get_stream()
		);

	c->status = RNSSTATE;
	c->base = a->base;
}

__host__  void poly_mul(Context *ctx, poly_t *c, poly_t *a, poly_t *b){
	assert(CUDAEngine::is_init);
	assert(a->init);
	assert(b->init);
	assert(CUDAEngine::N > 0);
	assert(a->base == b->base);

	poly_copy_to_device(ctx, a);
	poly_copy_to_device(ctx, b);

	poly_init(ctx, c, a->base);

	DGTEngine::execute_mul_dgt_gi(
		c->d_coefs,
		a->d_coefs,
		b->d_coefs,
		a->base,
		ctx->get_stream()
		);

	c->status = RNSSTATE;
	c->base = a->base;
}

__host__  void poly_mul_add(
	Context *ctx,
	poly_t *d,
	poly_t *a,
	poly_t *b,
	poly_t *c){
	assert(CUDAEngine::is_init);
	assert(a->init);
	assert(b->init);
	assert(c->init);
	assert(CUDAEngine::N > 0);
	assert(a->base == b->base);
	assert(a->base == c->base);

	poly_copy_to_device(ctx, a);
	poly_copy_to_device(ctx, b);
	poly_copy_to_device(ctx, c);

	poly_init(ctx, d);
	d->base = a->base;

	DGTEngine::execute_muladd_dgt(
		d->d_coefs,
		a->d_coefs,
		b->d_coefs,
		c->d_coefs,
		d->base,
		ctx->get_stream()
		);

	d->status = RNSSTATE;

}


__host__  void poly_dr2(
	Context *ctx,
	poly_t *ct21, // Outcome
	poly_t *ct22, // Outcome
	poly_t *ct23, // Outcome
	poly_t *ct01, // Operand 1
	poly_t *ct02, // Operand 1
	poly_t *ct11, // Operand 2
	poly_t *ct12){// Operand 2
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);
	assert(ct01->init);
	assert(ct02->init);
	assert(ct11->init);
	assert(ct12->init);
	assert(ct01->base == ct02->base);
	assert(ct11->base == ct12->base);
	assert(ct01->base == ct11->base);

	poly_copy_to_device(ctx, ct01);
	poly_copy_to_device(ctx, ct02);
	poly_copy_to_device(ctx, ct11);
	poly_copy_to_device(ctx, ct12);

	poly_init(ctx, ct21);
	poly_init(ctx, ct22);
	poly_init(ctx, ct23);

	DGTEngine::execute_dr2_dgt(
		ct21->d_coefs,
		ct22->d_coefs,
		ct23->d_coefs,
		ct01->d_coefs,
		ct02->d_coefs,
		ct11->d_coefs,
		ct12->d_coefs,
		ct01->base,
		ctx->get_stream()
		);

	ct21->status = RNSSTATE;
	ct22->status = RNSSTATE;
	ct23->status = RNSSTATE;

	ct21->base = ct01->base;
	ct22->base = ct01->base;
	ct23->base = ct01->base;

}

__host__  void poly_double_mul_int(
	Context *ctx, 
	poly_t *b1,
	poly_t *a1,
	poly_t *b2,
	poly_t *a2,
	uint64_t x1,
	uint64_t x2){
	
	assert(a1->init);
	assert(a2->init);
	assert(a1->base == a2->base);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	poly_copy_to_device(ctx, a1);
	poly_copy_to_device(ctx, a2);
	poly_init(ctx, b1);
	poly_init(ctx, b2);

	CUDAEngine::execute_polynomial_double_op_by_int(
		b1->d_coefs,
		a1->d_coefs,
		b2->d_coefs,
		a2->d_coefs,
		a1->base,
		x1,
		x2,
		MULMUL,
    	ctx
		);

	b1->status = RNSSTATE;
	b1->base = a1->base;
	b2->status = RNSSTATE;
	b2->base = a1->base;
}

__host__  void poly_double_add_int(
	Context *ctx, 
	poly_t *b1,
	poly_t *a1,
	poly_t *b2,
	poly_t *a2,
	uint64_t x1,
	uint64_t x2){
	
	assert(a1->init);
	assert(a2->init);
	assert(a1->base == a2->base);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	poly_copy_to_device(ctx, a1);
	poly_copy_to_device(ctx, a2);
	poly_init(ctx, b1);
	poly_init(ctx, b2);

	CUDAEngine::execute_polynomial_double_op_by_int(
		b1->d_coefs,
		a1->d_coefs,
		b2->d_coefs,
		a2->d_coefs,
		a1->base,
		x1,
		x2,
		ADDADD,
    	ctx
		);

	b1->status = RNSSTATE;
	b1->base = a1->base;
	b2->status = RNSSTATE;
	b2->base = a1->base;
}

__host__  void poly_mul_int(
	Context *ctx, 
	poly_t *b,
	poly_t *a,
	uint64_t x){
	
	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	poly_copy_to_device(ctx, a);
	poly_init(ctx, b);

	CUDAEngine::execute_polynomial_op_by_int(
		b->d_coefs,
		a->d_coefs,
		a->base,
		x,
		MUL,
    	ctx
		);

	b->status = RNSSTATE;
	b->base = a->base;
}

__host__  void poly_add_int(
	Context *ctx, 
	poly_t *b,
	poly_t *a,
	uint64_t x){
	
	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	poly_copy_to_device(ctx, a);
	poly_init(ctx, b);

	CUDAEngine::execute_polynomial_op_by_int(
		b->d_coefs,
		a->d_coefs,
		a->base,
		x,
		ADD,
    	ctx
		);

	b->status = RNSSTATE;
	b->base = a->base;
}

__host__  void poly_sub_int(
	Context *ctx, 
	poly_t *b,
	poly_t *a,
	uint64_t x){
	
	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	poly_copy_to_device(ctx, a);
	poly_init(ctx, b);

	CUDAEngine::execute_polynomial_op_by_int(
		b->d_coefs,
		a->d_coefs,
		a->base,
		x,
		SUB,
    	ctx
		);

	b->status = RNSSTATE;
	b->base = a->base;
}

__host__  void poly_div_by_ZZ(Context *ctx, poly_t *c, poly_t *a, ZZ x){
	const unsigned int deg_a = poly_get_deg(ctx, a);

	// #pragma omp parallel for
	for(unsigned int i = 0; i <= deg_a; i++){
		poly_set_coeff(
			ctx,
			c,
			i,
			(poly_get_coeff(ctx, a, i)/x));
	}
}

__host__  void poly_mod_by_ZZ(Context *ctx, poly_t *c, poly_t *a, ZZ x){
	const unsigned int deg_a = poly_get_deg(ctx, a);

	// #pragma omp parallel for
	for(unsigned int i = 0; i <= deg_a; i++){
		poly_set_coeff(
			ctx,
			c,
			i,
			(poly_get_coeff(ctx, a, i) % x));
	}
}

__host__  std::string poly_to_string(Context *ctx, poly_t *a){
	std::ostringstream oss;
	for(int i = 0; i <= poly_get_deg(ctx, a); i++)
		oss << poly_get_coeff(ctx, a, i) << ", ";
	return oss.str();
}

__host__  void poly_set_coeff(
	Context *ctx,
	poly_t *a,
	unsigned int index,
	ZZ c){

	if(!a->init)
      poly_init(ctx, a);

	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	poly_copy_to_host(ctx, a);

	c = (c < 0 ? CUDAEngine::RNSProduct + c : c); // Adjust negatives
	if(index >= a->coefs.size())
		a->coefs.push_back(c);
	else
		a->coefs[index] = c;
	a->status = HOSTSTATE;
}

__host__  ZZ poly_get_coeff(Context *ctx, poly_t *a, int index){
	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	poly_copy_to_host(ctx, a);

	if((unsigned int)index >= a->coefs.size())
		return to_ZZ(0);
	else{
		return a->coefs[index];	
	}
}

__host__ void poly_copy(Context *ctx, poly_t *b, poly_t *a){
	poly_init(ctx, b, a->base);

	if(a->status == RNSSTATE || a->status == BOTHSTATE){
		cudaMemcpyAsync(
			b->d_coefs,
			a->d_coefs,
			CUDAEngine::N * CUDAEngine::get_n_residues(a->base) * sizeof(GaussianInteger),
			cudaMemcpyDeviceToDevice,
			ctx->get_stream());
		b->status = RNSSTATE;
	}else{
		for(int i = 0; i <= poly_get_deg(ctx, a); i++)
			poly_set_coeff(ctx, b, i, poly_get_coeff(ctx, a, i));
		b->status = HOSTSTATE;
	}
}

__host__ void poly_negate(Context *ctx, poly_t *a){
		
	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	poly_copy_to_device(ctx, a);

	CUDAEngine::execute_polynomial_negate(
		a->d_coefs,
		a->d_coefs,
		a->base,
		ctx);
	a->status = RNSSTATE;
}

__host__ void poly_right_shift(Context *ctx, poly_t *b, poly_t *a, int bits){
	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	poly_init(ctx, b);

	/////////////////
	// Runs in CPU //
	/////////////////
    int deg_a = poly_get_deg(ctx, a);
    // #pragma omp parallel for
    for(int i = 0; i <= deg_a; i++)
        poly_set_coeff(ctx, b, i, poly_get_coeff(ctx, a,i) >> bits );
    
}

__host__ bool poly_are_equal(Context *ctx, poly_t *a, poly_t *b){
	assert(a->init);
	assert(b->init);

	if(poly_get_deg(ctx, a) != poly_get_deg(ctx, b))
		return false;
	
	for(int i = 0; i < poly_get_deg(ctx, a); i++)
		if(
			poly_get_coeff(ctx, a, i) % CUDAEngine::RNSProduct != 
			poly_get_coeff(ctx, b, i) % CUDAEngine::RNSProduct)
			return false;
	return true;
}

__host__ std::string poly_export_residues(Context *ctx, poly_t *a){
	//////////////////
	// Copy to Host //
	//////////////////
	cudaMemcpyAsync(
		ctx->d_aux_coefs,
		a->d_coefs,
		poly_get_residues_size(a->base),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream() );
	cudaCheckError();

	DGTEngine::execute_dgt(
		ctx->d_aux_coefs,
		a->base,
		INVERSE,
		ctx
	);

	// Recovers RNS's residues and calls IRNS
	cudaMemcpyAsync(
		ctx->h_coefs,
		ctx->d_aux_coefs,
		poly_get_residues_size(a->base),
		cudaMemcpyDeviceToHost,
		ctx->get_stream()
	);

	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();

	int64_t *h_coefs = (int64_t*) malloc (2 * CUDAEngine::N * CUDAEngine::get_n_residues(a->base) * sizeof(int64_t));
	for(int rid = 0; rid < CUDAEngine::get_n_residues(a->base); rid++){
		for(int cid = 0; cid < CUDAEngine::N; cid++)
			h_coefs[cid + 2 * CUDAEngine::N * rid] =
				ctx->h_coefs[cid + rid * CUDAEngine::N].re;
		for(int cid = 0; cid < CUDAEngine::N; cid++)
			h_coefs[(cid + CUDAEngine::N) + 2 * CUDAEngine::N * rid] =
				ctx->h_coefs[cid + rid * CUDAEngine::N].imag;
			// if(h_coefs[cid + rid * CUDAEngine::N] > COPRIMES_BUCKET[rid] / 2)
				// h_coefs[cid + rid * CUDAEngine::N] -= COPRIMES_BUCKET[rid];
		}
   	
   	// To string
	// Convert all but the last element to avoid a trailing ","
	std::stringstream result;
	std::copy(
		&h_coefs[0],
		&h_coefs[2 * CUDAEngine::N * CUDAEngine::get_n_residues(a->base)],
		std::ostream_iterator<int64_t>(result, " "));
	string s = result.str();
	s = s.substr(0, s.length()-1);

	free(h_coefs);
   	return s.c_str();
}

__host__ poly_t* poly_import_residues(Context *ctx, std::string s, int base){
	// String to vector
	std::stringstream iss(s);

	int64_t val;
	std::vector<int64_t> v;
	while ( iss >> val )
	  v.push_back( val );

	// Vector to poly_t
	for(unsigned int i = 0; i < v.size(); i++){
		int cid = i % (2 * CUDAEngine::N);
		int rid = i / (2 * CUDAEngine::N);
		// uint64_t p = COPRIMES_BUCKET[rid];
		if(cid < CUDAEngine::N)
			ctx->h_coefs[(cid % CUDAEngine::N) + rid * CUDAEngine::N].re   = v[i];
			// ctx->h_coefs[i].re   = (v[i] < p/2? v[i] : p - v[i]);
		else
			ctx->h_coefs[(cid % CUDAEngine::N) + rid * CUDAEngine::N].imag = v[i];
			// ctx->h_coefs[i].imag = (v[i] < p/2? v[i] : p - v[i]);
	}

	poly_t *a = new poly_t;
	poly_init(ctx, a, base);
	cudaMemcpyAsync(
		a->d_coefs,
		ctx->h_coefs,
		poly_get_residues_size(a->base),
		cudaMemcpyHostToDevice,
		ctx->get_stream()
	);
	cudaCheckError();
  	
	DGTEngine::execute_dgt(
		a->d_coefs,
		a->base,
		FORWARD,
		ctx
	);

	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();

	a->status = RNSSTATE;
   	return a;
}


__host__ std::string poly_export(Context *ctx, poly_t *p){
	std::vector<ZZ> v;
	std::ostringstream oss;

	// To vector
	for (int i = 0; i <= poly_get_deg(ctx, p); i++)
   		v.push_back( poly_get_coeff(ctx, p, i));
   	
   	// To string
	if (!v.empty()){
		// Convert all but the last element to avoid a trailing ","
		for(unsigned int i = 0; i < v.size()-1; i++){
			oss << v[i] << ",";
		}

		// Now add the last element with no delimiter
		oss << v.back();
	}
   	return oss.str();
}

__host__ poly_t* poly_import(Context *ctx, std::string s){
	// String to vector
	std::vector<ZZ> v;
	if(s.length() > 0){
		std::string value = "";
		for(int i = s.length(); i >= -1; i--)
			if(i == -1 || s[i] == ','){
				v.insert(v.begin(), to_ZZ(value.c_str()));
				value = "";
			}else if(s[i] != ' ')
				value = s[i] + value;
	}

	// Vector to poly_t
	poly_t *p = new poly_t;
	poly_init(ctx, p);

	for (unsigned int i = 0; i < v.size(); i++)
   		poly_set_coeff(ctx, p, i, v[i]);
   	return p;
}

__host__ void poly_dot(
	Context *ctx,
	poly_t *c,
	poly_t *a,
	poly_t *b,
	const int k){
	// Dot product

	poly_mul(ctx, c, &a[0], &b[0]);
	for(int i = 1; i < k; i++)
		poly_mul_add(ctx, c, &a[i], &b[i], c);

}

__host__ ZZ poly_infty_norm(Context *ctx, poly_t *p){
	ZZ x = to_ZZ(0);
	#ifdef BFV_ENGINE_MODE
		for(int i = 0; i <= poly_get_deg(ctx, p); i++)
			if(poly_get_coeff(ctx, p, i) > x)
				x = poly_get_coeff(ctx, p, i);
	#else
		uint64_t *v = poly_get_residue(ctx, NULL, p, 0);
		for(int i = 0; i <= poly_get_deg(ctx, p); i++){
			int64_t aux = (
				v[i] < COPRIMES_BUCKET[0]/2 ? v[i] : (int64_t)v[i] - (int64_t)COPRIMES_BUCKET[0]
				);
			if(aux > x)
				x = to_ZZ(aux);
		}
	#endif
	return x;
}

__host__ RR poly_norm_2(Context *ctx, poly_t *p){
	ZZ x = to_ZZ(0);
	#ifdef BFV_ENGINE_MODE
		for(int i = 0; i <= poly_get_deg(ctx, p); i++)
			x += poly_get_coeff(ctx, p, i) * poly_get_coeff(ctx, p, i);
	#else
		uint64_t *v = poly_get_residue(ctx, NULL, p, 0);
		for(int i = 0; i <= poly_get_deg(ctx, p); i++){
			ZZ aux = to_ZZ(
				v[i] < COPRIMES_BUCKET[0]/2 ? v[i] : (COPRIMES_BUCKET[0] - v[i])
				);
			x += aux * aux;
		}
	#endif
	return NTL::sqrt(to_RR(x));
}

__host__ std::string poly_residue_to_string(Context *ctx, poly_t *a, int id){

	Logger::getInstance()->log_debug("poly_residue_to_string");

	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	//////////////////
	// Copy to Host //
	//////////////////
	cudaMemcpyAsync(
		ctx->d_aux_coefs,
		a->d_coefs,
		poly_get_residues_size(a->base),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream() );
	cudaCheckError();

	DGTEngine::execute_dgt(
		ctx->d_aux_coefs,
		a->base,
		INVERSE,
		ctx
	);

	// Recovers RNS's residues and calls IRNS
	cudaMemcpyAsync(
		ctx->h_coefs,
		ctx->d_aux_coefs,
		poly_get_residues_size(a->base),
		cudaMemcpyDeviceToHost,
		ctx->get_stream()
	);
	cudaCheckError();

	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();

	// 
	std::ostringstream oss;
	for(int i = 0; i < 2 * CUDAEngine::N; i++)
		oss << ( 
			i < CUDAEngine::N ?
			ctx->h_coefs[id * CUDAEngine::N + i].re :
			ctx->h_coefs[id * CUDAEngine::N + (i % CUDAEngine::N)].imag) << ",";
	return oss.str();
}

__host__ uint64_t* poly_get_residue(Context *ctx, uint64_t* b, poly_t *a, int id){

	Logger::getInstance()->log_debug("poly_get_residue");

	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	poly_copy_to_device(ctx, a);

	//////////////////
	// Copy to Host //
	//////////////////
	cudaMemcpyAsync(
		ctx->d_aux_coefs,
		a->d_coefs,
		poly_get_residues_size(a->base),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream() );
	cudaCheckError();

	DGTEngine::execute_dgt(
		ctx->d_aux_coefs,
		a->base,
		INVERSE,
		ctx
	);

	// Recovers RNS's residues and calls IRNS
	cudaMemcpyAsync(
		ctx->h_coefs,
		ctx->d_aux_coefs,
		poly_get_residues_size(a->base),
		cudaMemcpyDeviceToHost,
		ctx->get_stream()
	);
	cudaCheckError();

	cudaStreamSynchronize(ctx->get_stream());
	cudaCheckError();

	uint64_t* r;
	if(b)
		r = b;
	else
		r = (uint64_t*) malloc (2 * CUDAEngine::N * sizeof(uint64_t));

	// 
	for(int i = 0; i < 2 * CUDAEngine::N; i++)
		r[i] = ( 
			i < CUDAEngine::N ?
			ctx->h_coefs[id * CUDAEngine::N + i].re :
			ctx->h_coefs[id * CUDAEngine::N + (i % CUDAEngine::N)].imag
		);
	return r;
}

__host__ uint64_t* poly_get_residues(Context *ctx, poly_t *a){

	Logger::getInstance()->log_debug("poly_get_residues");

	assert(a->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);

	uint64_t* r = (uint64_t*) malloc (poly_get_residues_size(a->base));

	// 
	for(int i = 0; i < CUDAEngine::get_n_residues(a->base); i++)
		poly_get_residue(ctx, r + i * CUDAEngine::N, a, i);
	return r;
}

__host__ void poly_select_residue(Context *ctx, poly_t *p, int id){
	assert(p->init);
	assert(CUDAEngine::is_init);
	assert(CUDAEngine::N > 0);
	
	poly_copy_to_device(ctx, p);

	// IDGT
	DGTEngine::execute_dgt(
		p->d_coefs,
		p->base,
		INVERSE,
		ctx
	);

	for(int i = 0; i < CUDAEngine::get_n_residues(p->base); i++)
		if(i != id){
			cudaMemcpyAsync(
				p->d_coefs + i * CUDAEngine::N,
				p->d_coefs + id * CUDAEngine::N,
				CUDAEngine::N * sizeof(GaussianInteger),
				cudaMemcpyDeviceToDevice,
				ctx->get_stream());
			cudaCheckError();
		}

	DGTEngine::execute_dgt(
		p->d_coefs,
		p->base,
		FORWARD,
		ctx
	);
}

__host__ void poly_discard_qbase(Context *ctx, poly_t *p){
	poly_copy_to_device(ctx, p);

	if(p->base == BBase)
		return;

	assert(p->base == QBBase);
	cudaMemcpyAsync(
		p->d_coefs,
		p->d_coefs + CUDAEngine::get_n_residues(QBase) * CUDAEngine::N,
		poly_get_residues_size(BBase),
		cudaMemcpyDeviceToDevice,
		ctx->get_stream());
	cudaCheckError();
	p->base = BBase;
}
