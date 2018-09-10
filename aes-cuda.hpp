/*
 * aes-cuda.hpp
 *
 *  Created on: Sep 4, 2018
 *      Author: hb4ch
 */

#ifndef AES_CUDA_HPP_
#define AES_CUDA_HPP_

#include "misc.hpp"

#define EXPANDED_KEY_SIZE 176;//16 bytes * 11 rounds = 176 bytes
#define EXPANDED_KEY_SIZE_INT 44;//blocksize * ( numrounds + 1 )

#define NUM_THREADS 256

//Device Globals

__device__ const unsigned char dev_sbox[256] = { SBOX };
__device__ const unsigned char g_GF_2[256] = { GF2 };
__device__ unsigned int g_expanded_key[44];
__device__ unsigned char g_counter_initial[16] = {0};
// Dummy counter
// 16bytes | 128bits

//Device kernel for SubBytes step
__device__ void SubBytes(unsigned char *state);
__device__ void ShiftRows(unsigned char *state);
__device__ void AddRoundKey(unsigned char *state, uint *w_);
__device__ void MixColumns(unsigned char *state);

__global__ void cuda_main_kernel(unsigned char* text,
		unsigned int num_blocks, unsigned char * destination);
__global__ void transfer(unsigned char * d1_ptr, unsigned char * d2_ptr);

#endif /* AES_CUDA_HPP_ */
