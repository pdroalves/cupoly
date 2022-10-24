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

#include<cuPoly/arithmetic/context.h>

Context::Context(){
  assert(CUDAEngine::is_init);
  
  cudaStreamCreate(&stream);
  cudaCheckError();

  const int k = CUDAEngine::get_n_residues(QBase);
  h_b = (GaussianInteger**) malloc (k * sizeof(GaussianInteger*));
  cudaMalloc((void**)&d_b, k * sizeof(GaussianInteger*));
  cudaCheckError();

  /////////////////////////
  // Pre-allocate memory //
  /////////////////////////
  
  // RNS
  h_coefs = (GaussianInteger*) malloc(poly_get_residues_size(QBBase));
  cudaMalloc((void**)&d_aux_coefs, poly_get_residues_size(QBBase));
  cudaCheckError();
  cudaMemset(d_aux_coefs, 0, poly_get_residues_size(QBBase));
  cudaCheckError();

  // HPS
  cudaMalloc((void**)&d_v, 2 * CUDAEngine::N * sizeof(uint64_t));
  cudaCheckError();
  cudaMalloc((void**)&d_aux, 
    2 * CUDAEngine::N * CUDAEngine::get_n_residues(QBBase) * sizeof(uint64_t));
  cudaCheckError();
  cudaMalloc((void**)&d_coefs_B, poly_get_residues_size(BBase));
  cudaCheckError();
  cudaMalloc((void**)&d_tmp_data,
    CUDAEngine::N * CUDAEngine::get_n_residues(QBBase) * sizeof(GaussianInteger));
  cudaCheckError();
  cudaMemset(d_tmp_data, 0, poly_get_residues_size(QBBase));
  cudaCheckError();

  cudaMemset(d_v, 0, 2 * CUDAEngine::N * sizeof(uint64_t));
  cudaCheckError();
  cudaMemset(d_aux,
    0,
    2 * CUDAEngine::N * CUDAEngine::get_n_residues(QBBase) * sizeof(uint64_t));
  cudaCheckError();
  cudaMemset(d_coefs_B, 0, poly_get_residues_size(BBase));
  cudaCheckError();

  // Sampler
  h_aux = (uint64_t*) malloc (
    2 * CUDAEngine::N * CUDAEngine::get_n_residues(QBBase) * sizeof(uint64_t)
    );
}