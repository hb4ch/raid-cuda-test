/*
 * benchmark.cpp
 *
 *  Created on: Sep 4, 2018
 *      Author: hb4ch
 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <sys/time.h>

#include "benchmark.hpp"
#include "misc.hpp"
#include "aes-cuda.hpp"
#include "raid.hpp"

size_t GetTimeMS() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return (int64_t) tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

void aes_benchmark::start() {

	std::printf("[AES Benchmark started.]\n");
	std::printf("File size = %lld \n", this->file_size);

	unsigned char * d_fb;
	unsigned char * d_result;
	unsigned char * h_result;

	checkCudaErrors(cudaMalloc((void ** )&d_fb, file_size));
	// Device file_buffer malloc'd
	checkCudaErrors(cudaMalloc((void ** )&d_result, file_size));
	// Device result buffer malloc'd
	checkCudaErrors(cudaMallocHost((void ** )&h_result, file_size));
	// Host result buffer malloc'd (pinned)
	checkCudaErrors(
			cudaMemcpyToSymbol(g_expanded_key, &expanded_key[0], 176, size_t(0),
					cudaMemcpyHostToDevice));
	// copy expanded key to device.

	unsigned char * temp_d_result = d_result;
	unsigned char * temp_d_fb = d_fb;
	unsigned char * temp_file_buffer = file_buffer;
	unsigned char * temp_h_result = h_result;

	size_t total_batches = file_size / batch_size;

	std::printf("total_batches = %d\n", total_batches);
	std::printf("batch_size = %d\n", batch_size);
	std::printf("Cuda block num = %d\n", batch_size / 16);


	size_t begin_ts;
	begin_ts = GetTimeMS();
	std::printf("HYPER_Q = %d\n", this->hyper_q);
	dim3 nblock(((file_size / 16) + 32*64 - 1) / (32*64), 128);

	if (hyper_q > 1) {

		int stream_num = hyper_q;
		int batches_per_stream = total_batches / stream_num;

		cudaStream_t streams[stream_num];
		for (int i = 0; i < stream_num; i++)
			cudaStreamCreate(&streams[i]);
		// Async create stream
		for (int i = 0; i < batches_per_stream; i++) {
			for (int j = 0; j < stream_num; j++) {

				checkCudaErrors(
						cudaMemcpyAsync(temp_d_fb, temp_file_buffer, batch_size,
								cudaMemcpyHostToDevice, streams[j]));

				cuda_main_kernel<<<batch_size / 16 / 1024, 1024>>>
						(temp_d_fb, batch_size / 16, temp_d_result);

				checkCudaErrors(
						cudaMemcpyAsync(temp_h_result, temp_d_result,
								batch_size, cudaMemcpyDeviceToHost,
								streams[j]));

				temp_d_result += batch_size;
				temp_d_fb += batch_size;
				temp_h_result += batch_size;
				temp_file_buffer += batch_size;
			}
			checkCudaErrors(cudaDeviceSynchronize());
		}

		for (int i = 0; i < stream_num; i++)
			cudaStreamDestroy(streams[i]);

	} else {

		for (int i = 0; i < total_batches; i++) {
			checkCudaErrors(
					cudaMemcpy(temp_d_fb, temp_file_buffer, batch_size,
							cudaMemcpyHostToDevice));
			// Copy one batch of data to device


			cuda_main_kernel<<<batch_size / 16 / 1024, 1024>>>
			//cuda_main_kernel<<<nblock , 64>>>
				(temp_d_fb, batch_size / 16, temp_d_result);

			 /*ctr_encrypt_nofrag_perword<<<nblock, 64>>>
					 ((uint8_t *)temp_d_fb, (uint8_t *)g_expanded_key, total_size / 16);*/

			// transfer<<<batch_size / 4 , 1>>>(temp_d_fb, temp_d_result);
			// Running kernel to cipher a 16 byte cipher block
			// Each CUDA block process a 16 byte cipher block
			checkCudaErrors(cudaThreadSynchronize());
			// Barrier
			checkCudaErrors(
					cudaMemcpy(temp_h_result, temp_d_result, batch_size,
							cudaMemcpyDeviceToHost));
			// Fetch the result back
			temp_d_result += batch_size;
			temp_d_fb += batch_size;
			temp_h_result += batch_size;
			temp_file_buffer += batch_size;

		}
	}
	// Now the kernel is expected to process the given file_buffer into result_buffer
	// One batch is infused into device, processed and then fetched back.
	// The loop stops until the file is done ciphering.
	std::printf("Loop done!\n");

	size_t end_ts = GetTimeMS();
	double thruput = (total_size / (1024 * 1024 * 1024 + 0.0))
			/ ((end_ts - begin_ts) / 1000.0);
	double latency = (end_ts - begin_ts) / (total_batches / 1000.0);
	printf("total time: %lf s\n", (end_ts - begin_ts) / 1000.0);
	printf("Thruput: %lf GB/s, latency: %lf us/batch \n", thruput, latency);
	printf("[End benchmark]\n");

	checkCudaErrors(cudaFreeHost(h_result));
	checkCudaErrors(cudaFree(d_fb));
	checkCudaErrors(cudaFree(d_result));
	// Leave file_buffer alone
	// It is freed in superclass d'tor
}

void raid_benchmark::start()
{

	std::printf("[Raid Benchmark started.]\n");
	std::printf("File size = %lld \n", this->file_size);

	uint8_t * d_fb;
	uint8_t * d_result_p;
	uint8_t * d_result_q;

	uint8_t * h_result_p;
	uint8_t * h_result_q;

	checkCudaErrors(cudaMalloc((void ** )&d_fb, file_size));
	// Device file_buffer malloc'd
	checkCudaErrors(cudaMalloc((void ** )&d_result_p, file_size));
	// Native data block is 8
	// Device result buffer malloc'd
	checkCudaErrors(cudaMalloc((void ** )&d_result_q, file_size));

	checkCudaErrors(cudaMallocHost((void ** )&h_result_p, file_size));
	checkCudaErrors(cudaMallocHost((void ** )&h_result_q, file_size));


	uint8_t * temp_d_result_p = d_result_p;
	uint8_t * temp_d_result_q = d_result_q;

	uint8_t * temp_d_fb = d_fb;
	uint8_t * temp_file_buffer = (uint8_t * )file_buffer;

	uint8_t * temp_h_result_p = h_result_p;
	uint8_t * temp_h_result_q = h_result_q;

	size_t total_batches = file_size / batch_size;

	std::printf("total_batches = %d\n", total_batches);
	std::printf("batch_size = %d\n", batch_size);

	size_t begin_ts;
	begin_ts = GetTimeMS();
	std::printf("HYPER_Q = %d\n", this->hyper_q);

	if (hyper_q > 1) {

		int stream_num = hyper_q;
		int batches_per_stream = total_batches / stream_num;

		cudaStream_t streams[stream_num];
		for (int i = 0; i < stream_num; i++)
			cudaStreamCreate(&streams[i]);
		// Async create stream
		for (int i = 0; i < batches_per_stream; i++) {
			for (int j = 0; j < stream_num; j++) {

				checkCudaErrors(
						cudaMemcpyAsync(temp_d_fb, temp_file_buffer, batch_size,
								cudaMemcpyHostToDevice, streams[j]));

				raid_cuda_encode<<<batch_size / 8 / 512, 512>>>
					(temp_d_fb, temp_d_result_p, temp_d_result_q, batch_size / 8);

				checkCudaErrors(
						cudaMemcpyAsync(temp_h_result_p, temp_d_result_p,
								batch_size, cudaMemcpyDeviceToHost,
								streams[j]));
				checkCudaErrors(
						cudaMemcpyAsync(temp_h_result_q, temp_d_result_q,
								batch_size, cudaMemcpyDeviceToHost,
								streams[j]));

				temp_d_result_p += batch_size;
				temp_d_result_q += batch_size;
				temp_d_fb += batch_size;
				temp_h_result_p += batch_size;
				temp_h_result_q += batch_size;
				temp_file_buffer += batch_size;
			}
			checkCudaErrors(cudaDeviceSynchronize());
		}

		for (int i = 0; i < stream_num; i++)
			cudaStreamDestroy(streams[i]);

	} else {

		for (int i = 0; i < total_batches; i++) {
			checkCudaErrors(
					cudaMemcpy(temp_d_fb, temp_file_buffer, batch_size,
							cudaMemcpyHostToDevice));
			// Copy one batch of data to device

			raid_cuda_encode<<<batch_size / 8 / 512, 512>>>
				(temp_d_fb, temp_d_result_p, temp_d_result_q, batch_size / 8);

			checkCudaErrors(cudaThreadSynchronize());
			// Barrier
			checkCudaErrors(
					cudaMemcpy(temp_h_result_p, temp_d_result_p, batch_size,
							cudaMemcpyDeviceToHost));
			checkCudaErrors(
					cudaMemcpy(temp_h_result_q, temp_d_result_q, batch_size,
							cudaMemcpyDeviceToHost));
			// Fetch the result back
			temp_d_result_p += batch_size;
			temp_d_result_q += batch_size;

			temp_d_fb += batch_size;
			temp_h_result_p += batch_size;
			temp_h_result_q += batch_size;

			temp_file_buffer += batch_size;

		}
	}
	// Now the kernel is expected to process the given file_buffer into result_buffer
	// One batch is infused into device, processed and then fetched back.
	// The loop stops until the file is done ciphering.
	std::printf("Loop done!\n");

	size_t end_ts = GetTimeMS();
	double thruput = (total_size / (1024 * 1024 * 1024 + 0.0))
			/ ((end_ts - begin_ts) / 1000.0);
	double latency = (end_ts - begin_ts) / (total_batches / 1000.0);
	printf("total time: %lf s\n", (end_ts - begin_ts) / 1000.0);
	printf("Thruput: %lf GB/s, latency: %lf us/batch \n", thruput, latency);
	printf("[End benchmark]\n");

	checkCudaErrors(cudaFreeHost(h_result_p));
	checkCudaErrors(cudaFreeHost(h_result_q));
	checkCudaErrors(cudaFree(d_fb));
	checkCudaErrors(cudaFree(d_result_p));
	checkCudaErrors(cudaFree(d_result_q));

	// Leave file_buffer alone
	// It is freed in superclass d'tor
}

