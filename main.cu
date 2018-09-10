/*
 * main.cpp
 *
 *  Created on: Sep 4, 2018
 *      Author: hb4ch
 */

#define HYPER_Q 1

#include <cstdio>
#include <cstdlib>
#include <sys/stat.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "benchmark.hpp"
#include "misc.hpp"
#include "raid.hpp"

unsigned char original_key[16] = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06,
		0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, };
// Dummy key for testing
void test_raid(int argc, char ** argv) {
	if (argc != 4) {
		std::printf("Wrong usage.\n");
		exit(0);
	}

	size_t bs = atoi(argv[2]);
	bs *= 1024;
	// KBs

	int hq = atoi(argv[3]);
	char * file_name = argv[1];
	FILE * fp = fopen(file_name, "r");

	if (!fp) {
		std::printf("Error reading file.\n");
		exit(0);
	}

	struct stat st;
	stat(file_name, &st);
	size_t file_size = st.st_size;

	if (file_size % 128 != 0) {
		std::printf("Your file needs padding to align with 16 bytes.\n");
		exit(0);
	}

	//unsigned char * fb = (unsigned char * )malloc(file_size);
	unsigned char * h_fb;
	checkCudaErrors(cudaMallocHost((void ** )&h_fb, file_size));
	fread(h_fb, 1, file_size, fp);
	fclose(fp);

	std::printf("File read successful.\n");
	// File is now read in properly.

	raid_benchmark rb(bs, file_size / bs, file_size, (uint8_t * )h_fb, hq);
	rb.start();
	cudaFree(h_fb);

}
void test_aes(int argc, char ** argv) {
	if (argc != 4) {
		std::printf("Wrong usage.\n");
		exit(0);
	}

	size_t bs = atoi(argv[2]);
	bs *= 1024;
	// KBs

	int hq = atoi(argv[3]);
	char * file_name = argv[1];
	FILE * fp = fopen(file_name, "r");

	if (!fp) {
		std::printf("Error reading file.\n");
		exit(0);
	}

	struct stat st;
	stat(file_name, &st);
	size_t file_size = st.st_size;

	if (file_size % 128 != 0) {
		std::printf("Your file needs padding to align with 16 bytes.\n");
		exit(0);
	}

	//unsigned char * fb = (unsigned char * )malloc(file_size);
	unsigned char * h_fb;
	checkCudaErrors(cudaMallocHost((void ** )&h_fb, file_size));
	fread(h_fb, 1, file_size, fp);
	fclose(fp);

	std::printf("File read successful.\n");
	// File is now read in properly.

	unsigned int * expanded_key = (unsigned int *) malloc(176);

	CPU_KeyExpansion(original_key, expanded_key);

	// Key is now expanded.
	/*
	 std::printf("Expanded key: \n");
	 for(int i = 0; i < 44; i++) {
	 std::printf("%d ", expanded_key[i]);
	 if(i % 4 == 0)
	 std::printf("\n");
	 }
	 std::printf("\n");
	 */

	std::printf("Key is now expanded.\n");

	aes_benchmark ab(expanded_key, bs, file_size / bs, file_size, h_fb, hq);
	// 4096 = 4KB is the batch_size

	ab.start();
	// Here we go ...
	free(expanded_key);
	cudaFree(h_fb);
}

int main(int argc, char ** argv) {
	//test_aes(argc, argv);
	test_raid(argc, argv);
	return 0;
}

