Known-Issues
============

cuPoly is not free of bugs (as expected). Following we mention the most relevant ones we are aware of:

1. Method simple_scaling_tDivQ(), at src/cuda/cudaengine.cu, is not overflow-safe when t is not a power of 2. The multiplication between v and d_RNStomega_int may result in an overflow, which requires careful handling not done at the current version. Until this is fixed, we inserted a constraint at CUDAEngine::init().
2. There is a floating-point division at CUDAEngine::polynomial_basis_ext_Q_compute_v() that may fail when we work close to the 64-bit boundaries of double precision. It's not clear that this will not break correctness in some cases. 
3. This implementation was conceived as a proof of concept, so there are parts of it that are obviously insecure and make it entirely unfit for production. For instance, at sampler.h we hardcode a SEED. This was necessary during the early days of cuPoly for debugging, but the SEED generation needs to be refactored ASAP.
4. There is something wrong in CMakeLists.txt. Sometimes the compiler won't link correctly with the CUDA toolkit, but it works if you call one more time ``make``.
5. poly_t::coefs is not needed anymore. It can be removed to reduce memory consumption and initialization time.