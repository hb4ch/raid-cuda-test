/*
 * raid.cu
 *
 *  Created on: Sep 10, 2018
 *      Author: hb4ch
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <stdio.h>

#define W 8
#define NW (1 << W) /* In other words, NW equals 2 to the w-th power */

__device__ uint8_t gfexp[16] = { 1, 2, 4, 8, 3, 6, 12, 11, 5, 10, 7, 14, 15, 13,
		9, 0 };
__device__ uint8_t gflog[16] = { 0, 0, 1, 4, 2, 8, 5, 10, 3, 14, 9, 7, 6, 13,
		11, 12 };
__device__ uint8_t gfmul_16[16][16] = { { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0 }, { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }, {
		0, 2, 4, 6, 8, 10, 12, 14, 3, 1, 7, 5, 11, 9, 15, 13 }, { 0, 3, 6, 5,
		12, 15, 10, 9, 11, 8, 13, 14, 7, 4, 1, 2 }, { 0, 4, 8, 12, 3, 7, 11, 15,
		6, 2, 14, 10, 5, 1, 13, 9 }, { 0, 5, 10, 15, 7, 2, 13, 8, 14, 11, 4, 1,
		9, 12, 3, 6 }, { 0, 6, 12, 10, 11, 13, 7, 1, 5, 3, 9, 15, 14, 8, 2, 4 },
		{ 0, 7, 14, 9, 15, 8, 1, 6, 13, 10, 3, 4, 2, 5, 12, 11 }, { 0, 8, 3, 11,
				6, 14, 5, 13, 12, 4, 15, 7, 10, 2, 9, 1 }, { 0, 9, 1, 8, 2, 11,
				3, 10, 4, 13, 5, 12, 6, 15, 7, 14 }, { 0, 10, 7, 13, 14, 4, 9,
				3, 15, 5, 8, 2, 1, 11, 6, 12 }, { 0, 11, 5, 14, 10, 1, 15, 4, 7,
				12, 2, 9, 13, 6, 8, 3 }, { 0, 12, 11, 7, 5, 9, 14, 2, 10, 6, 1,
				13, 15, 3, 4, 8 }, { 0, 13, 9, 4, 1, 12, 8, 5, 2, 15, 11, 6, 3,
				14, 10, 7 }, { 0, 14, 15, 1, 13, 3, 2, 12, 9, 7, 6, 8, 4, 10,
				11, 5 },
		{ 0, 15, 13, 2, 9, 6, 4, 11, 1, 14, 12, 3, 8, 7, 5, 10 }, };



__device__ uint8_t gf_add(uint8_t a, uint8_t b) {
	return a ^ b;
}

__device__ uint8_t gf_sub(uint8_t a, uint8_t b) {
	return gf_add(a, b);
}

__device__ uint8_t gf_mul(uint8_t a, uint8_t b) {
	int sum_log;
	if (a == 0 || b == 0) {
		return 0;
	}
	//	sum_log = (gflog[a] + gflog[b]) % (NW-1);
	sum_log = gflog[a] + gflog[b];
	if (sum_log >= NW - 1) {
		sum_log -= NW - 1;
	}
	return gfexp[sum_log];
}

__device__ uint8_t gf_mul(uint8_t a, uint8_t b, uint8_t *gflog,
		uint8_t *gfexp) {
	int sum_log;
	if (a == 0 || b == 0) {
		return 0;
	}
	//	sum_log = (gflog[a] + gflog[b]) % (NW-1);
	sum_log = gflog[a] + gflog[b];
	if (sum_log >= NW - 1) {
		sum_log -= NW - 1;
	}
	return gfexp[sum_log];
}

__device__ uint8_t gf_mul_bit(uint8_t a, uint8_t b) {
	uint8_t sum_log;
	while (b) {
		if (b & 1) {
			sum_log ^= a;
		}
		a = (a << 1) ^ (a & 0x80 ? 0x1d : 0);
		b >>= 1;
	}
	return sum_log;
}

__device__ uint8_t gf_mul_bit(uint8_t a, uint8_t b, uint8_t *gflog,
		uint8_t *gfexp) {
	uint8_t sum_log;
	while (b) {
		if (b & 1) {
			sum_log ^= a;
		}
		a = (a << 1) ^ (a & 0x80 ? 0x1d : 0);
		b >>= 1;
	}
	return sum_log;
}

__device__ uint8_t gf_div(uint8_t a, uint8_t b) {
	int diff_log;
	if (a == 0) {
		return 0;
	}
	/* Can’t divide by 0 */
	if (b == 0) {
		return -1;
	}
	//	diff_log = (gflog[a] - gflog[b]) % (NW-1);
	diff_log = gflog[a] - gflog[b];
	if (diff_log < 0) {
		diff_log += NW - 1;
	}
	return gfexp[diff_log];
}

__device__ uint8_t gf_div(uint8_t a, uint8_t b, uint8_t *gflog,
		uint8_t *gfexp) {
	int diff_log;
	if (a == 0) {
		return 0;
	}
	/* Can’t divide by 0 */
	if (b == 0) {
		return -1;
	}
	//	diff_log = (gflog[a] - gflog[b]) % (NW-1);
	diff_log = gflog[a] - gflog[b];
	if (diff_log < 0) {
		diff_log += NW - 1;
	}
	return gfexp[diff_log];
}

__device__ uint8_t gf_pow(uint8_t a, uint8_t power) {
	int pow_log = (gflog[a] * power) % (NW - 1);
	return gfexp[pow_log];
}

__device__ uint8_t gf_pow(uint8_t a, uint8_t power, uint8_t *gflog,
		uint8_t *gfexp) {
	int pow_log = (gflog[a] * power) % (NW - 1);
	return gfexp[pow_log];
}

__global__ void raid_cuda_encode(uint8_t * plain, uint8_t * p, uint8_t * q, size_t stride)
{
	unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int tid = y * (blockDim.x * gridDim.x) + x;

	if(tid > stride)
		return;
	// native data chunks is 8
	uint8_t * start_points[8];
	for(int i = 0; i < 8; i++) {
		start_points[i] = plain + (stride * i);
	}

	p[tid] = 0;
	for(int i = 0; i < 8; i++) {
		p[tid] ^= *(start_points[i] + tid);
	}
	// Parity p is calculated.
	q[tid] = 0;
	for(uint8_t i = 0; i < 8; i++) {
		q[tid] ^= gf_mul(gf_pow(2, i), *(start_points[i] + tid));
	}
	// Parity q is calculated.

}


