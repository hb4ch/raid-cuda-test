/*
 * benchmark.hpp
 *
 *  Created on: Sep 4, 2018
 *      Author: hb4ch
 */

#ifndef BENCHMARK_HPP_
#define BENCHMARK_HPP_

#include <cstring>
#include <iostream>
#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>

class basic_benchmark {
public:
	size_t batch_size;
	size_t total_batches;
	size_t total_size;
	size_t file_size;

	int hyper_q;
	unsigned char * file_buffer;

	basic_benchmark(size_t bs, size_t tb, size_t fs, unsigned char * fb, int hq)
		: total_size(bs * tb)
		, total_batches(tb)
		, batch_size(bs)
		, file_size(fs)
		, file_buffer(fb)
	    , hyper_q(hq) {}

	virtual ~basic_benchmark() {}
};

class aes_benchmark : basic_benchmark {
public:
	unsigned int * expanded_key;

	aes_benchmark(unsigned int * ek,
			size_t bs, size_t tb, size_t fs, unsigned char * fb, int hq)
		: expanded_key(ek)
		, basic_benchmark(bs, tb, fs, fb, hq) {}

	void start();

	virtual ~aes_benchmark() {}

};

class raid_benchmark : basic_benchmark {
public:

	raid_benchmark(size_t bs, size_t tb, size_t fs, unsigned char * fb, int hq) :
	basic_benchmark(bs, tb, fs, fb, hq) {}

	void start();
	virtual ~raid_benchmark() {}
};

#endif /* BENCHMARK_HPP_ */
