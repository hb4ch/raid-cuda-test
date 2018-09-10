/*
 * aes-cuda.cpp
 *
 *  Created on: Sep 4, 2018
 *      Author: hb4ch
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>

#include "aes-cuda.hpp"
#include "misc.hpp"

__device__ void SubBytes(unsigned char *state) {
	state[0]  = dev_sbox[state[0]];
	state[1]  = dev_sbox[state[1]];
	state[2]  = dev_sbox[state[2]];
	state[3]  = dev_sbox[state[3]];
	state[4]  = dev_sbox[state[4]];
	state[5]  = dev_sbox[state[5]];
	state[6]  = dev_sbox[state[6]];
	state[7]  = dev_sbox[state[7]];
	state[8]  = dev_sbox[state[8]];
	state[9]  = dev_sbox[state[9]];
	state[10] = dev_sbox[state[10]];
	state[11] = dev_sbox[state[11]];
	state[12] = dev_sbox[state[12]];
	state[13] = dev_sbox[state[13]];
	state[14] = dev_sbox[state[14]];
	state[15] = dev_sbox[state[15]];
}

//Device kernel for ShiftRows step
__device__ void ShiftRows(unsigned char *state) {
	unsigned char temp = state[1];

        //NOTE: column-major ordering
	// 0 1 2 3 --> 0 1 2 3  | 0  4  8  12 --> 0   4  8 12
	// 0 1 2 3 --> 1 2 3 0  | 1  5  9  13 --> 5   9 13  1
	// 0 1 2 3 --> 2 3 0 1  | 2  6  10 14 --> 10 14  2  6
	// 0 1 2 3 --> 3 0 1 2  | 3  7  11 15 --> 15  3  7 11
	state[1]  = state[5];
	state[5]  = state[9];
	state[9]  = state[13];
	state[13] = temp;

	temp = state[2];
	state[2]  = state[10];
	state[10] = temp;
	temp = state[6];
	state[6]  = state[14];
	state[14] = temp;

	temp = state[3];
	state[3]  = state[15];
	state[15] = state[11];
	state[11] = state[7];
	state[7]  = temp;
}

//Device kernel for AddRoundKey step
__device__ void AddRoundKey(unsigned char *state, uint *w_) {
	unsigned int w = w_[0];
	state[3]  = state[3] ^ (w & 0xFF);
	w >>= 8;
	state[2]  = state[2] ^ (w & 0xFF);
	w >>= 8;
	state[1]  = state[1] ^ (w & 0xFF);
	w >>= 8;
	state[0]  = state[0] ^ (w & 0xFF);

	w = w_[1];
	state[7]  = state[7] ^ (w & 0xFF);
	w >>= 8;
	state[6]  = state[6] ^ (w & 0xFF);
	w >>= 8;
	state[5]  = state[5] ^ (w & 0xFF);
	w >>= 8;
	state[4]  = state[4] ^ (w & 0xFF);

	w = w_[2];
	state[11] = state[11] ^ (w & 0xFF);
	w >>= 8;
	state[10] = state[10] ^ (w & 0xFF);
	w >>= 8;
	state[9]  = state[9]  ^ (w & 0xFF);
	w >>= 8;
	state[8]  = state[8]  ^ (w & 0xFF);

	w = w_[3];
	state[15] = state[15] ^ (w & 0xFF);
	w >>= 8;
	state[14] = state[14] ^ (w & 0xFF);
	w >>= 8;
	state[13] = state[13] ^ (w & 0xFF);
	w >>= 8;
	state[12] = state[12] ^ (w & 0xFF);
}

//Device kernel for MixColumns step
//See "Efficient Software Implementation of AES on 32-bit platforms"
__device__ void MixColumns(unsigned char *state) {
	unsigned char x[4];
	x[0] = state[0];
	x[1] = state[1];
	x[2] = state[2];
	x[3] = state[3];
	unsigned char * y = (unsigned char *)&state[0];
	y[0] = x[1] ^ x[2] ^ x[3];
	y[1] = x[0] ^ x[2] ^ x[3];
	y[2] = x[0] ^ x[1] ^ x[3];
	y[3] = x[0] ^ x[1] ^ x[2];
	x[0] = g_GF_2[x[0]];
	x[1] = g_GF_2[x[1]];
	x[2] = g_GF_2[x[2]];
	x[3] = g_GF_2[x[3]];
	y[0] ^= x[0] ^ x[1];
	y[1] ^= x[1] ^ x[2];
	y[2] ^= x[2] ^ x[3];
	y[3] ^= x[3] ^ x[0];

	x[0] = state[4];
	x[1] = state[5];
	x[2] = state[6];
	x[3] = state[7];
	y = (unsigned char *)&state[4];
	y[0] = x[1] ^ x[2] ^ x[3];
	y[1] = x[0] ^ x[2] ^ x[3];
	y[2] = x[0] ^ x[1] ^ x[3];
	y[3] = x[0] ^ x[1] ^ x[2];
	x[0] = g_GF_2[x[0]];
	x[1] = g_GF_2[x[1]];
	x[2] = g_GF_2[x[2]];
	x[3] = g_GF_2[x[3]];
	y[0] ^= x[0] ^ x[1];
	y[1] ^= x[1] ^ x[2];
	y[2] ^= x[2] ^ x[3];
	y[3] ^= x[3] ^ x[0];

	x[0] = state[8];
	x[1] = state[9];
	x[2] = state[10];
	x[3] = state[11];
	y = (unsigned char *)&state[8];
	y[0] = x[1] ^ x[2] ^ x[3];
	y[1] = x[0] ^ x[2] ^ x[3];
	y[2] = x[0] ^ x[1] ^ x[3];
	y[3] = x[0] ^ x[1] ^ x[2];
	x[0] = g_GF_2[x[0]];
	x[1] = g_GF_2[x[1]];
	x[2] = g_GF_2[x[2]];
	x[3] = g_GF_2[x[3]];
	y[0] ^= x[0] ^ x[1];
	y[1] ^= x[1] ^ x[2];
	y[2] ^= x[2] ^ x[3];
	y[3] ^= x[3] ^ x[0];

	x[0] = state[12];
	x[1] = state[13];
	x[2] = state[14];
	x[3] = state[15];
	y = (unsigned char *)&state[12];
	y[0] = x[1] ^ x[2] ^ x[3];
	y[1] = x[0] ^ x[2] ^ x[3];
	y[2] = x[0] ^ x[1] ^ x[3];
	y[3] = x[0] ^ x[1] ^ x[2];
	x[0] = g_GF_2[x[0]];
	x[1] = g_GF_2[x[1]];
	x[2] = g_GF_2[x[2]];
	x[3] = g_GF_2[x[3]];
	y[0] ^= x[0] ^ x[1];
	y[1] ^= x[1] ^ x[2];
	y[2] ^= x[2] ^ x[3];
	y[3] ^= x[3] ^ x[0];
}



__global__ void cuda_main_kernel(unsigned char* text,
		unsigned int num_blocks, unsigned char * destination) {

	// unsigned int * key = g_expanded_key;
	//get thread id
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int tid = y * (blockDim.x * gridDim.x) + x; // offset

	if (tid > num_blocks)
		return;

	//Initial state is the block number + initial counter
	unsigned char state[16];
	state[15] = tid & 0xFF;
	state[14] = (tid >> 8) & 0xFF;
	state[13] = (tid >> 16) & 0xFF;
	state[12] = (tid >> 24) & 0xFF;
	state[11] = 0;
	state[10] = 0;
	state[9] = 0;
	state[8] = 0;
	state[7] = 0;
	state[6] = 0;
	state[5] = 0;
	state[4] = 0;
	state[3] = 0;
	state[2] = 0;
	state[1] = 0;
	state[0] = 0;

	// Copy our state into private memory
	unsigned char temp, temp2;
	unsigned char overflow = 0;

	if (tid < num_blocks) {
		for (int i = 15; i != -1; i--) {
			temp = g_counter_initial[i];
			temp2 = state[i];
			state[i] += temp + overflow;
			overflow = ((int) temp2 + (int) temp + (int) overflow > 255);
		}
	}

	AddRoundKey(state, &g_expanded_key[0]);
	for (int i = 1; i < 10; i++) {
		SubBytes(state);
		ShiftRows(state);
		MixColumns(state);
		AddRoundKey(state, &g_expanded_key[4 * i]);
	}
	SubBytes(state);
	ShiftRows(state);
	AddRoundKey(state, &g_expanded_key[4 * 10]);

	if (tid < num_blocks) {

		destination[(tid << 4) + 0] = text[(tid << 4) + 0] ^ state[0];
		destination[(tid << 4) + 1] = text[(tid << 4) + 1] ^ state[1];
		destination[(tid << 4) + 2] = text[(tid << 4) + 2] ^ state[2];
		destination[(tid << 4) + 3] = text[(tid << 4) + 3] ^ state[3];
		destination[(tid << 4) + 4] = text[(tid << 4) + 4] ^ state[4];
		destination[(tid << 4) + 5] = text[(tid << 4) + 5] ^ state[5];
		destination[(tid << 4) + 6] = text[(tid << 4) + 6] ^ state[6];
		destination[(tid << 4) + 7] = text[(tid << 4) + 7] ^ state[7];
		destination[(tid << 4) + 8] = text[(tid << 4) + 8] ^ state[8];
		destination[(tid << 4) + 9] = text[(tid << 4) + 9] ^ state[9];
		destination[(tid << 4) + 10] = text[(tid << 4) + 10] ^ state[10];
		destination[(tid << 4) + 11] = text[(tid << 4) + 11] ^ state[11];
		destination[(tid << 4) + 12] = text[(tid << 4) + 12] ^ state[12];
		destination[(tid << 4) + 13] = text[(tid << 4) + 13] ^ state[13];
		destination[(tid << 4) + 14] = text[(tid << 4) + 14] ^ state[14];
		destination[(tid << 4) + 15] = text[(tid << 4) + 15] ^ state[15];

	}
	// printf("Kernel done!\n");

}

__global__ void transfer(unsigned char * d1_ptr, unsigned char * d2_ptr) {

	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int offset = y * (blockDim.x * gridDim.x) + x; // offset
	*(d2_ptr + offset) = *(d1_ptr + offset);
	// Copy data from d1_ptr to d2_ptr concurrently
}
