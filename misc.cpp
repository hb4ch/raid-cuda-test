/*
 * misc.cpp
 *
 *  Created on: Sep 4, 2018
 *      Author: hb4ch
 */

#include "misc.hpp"

void print_block_hex(unsigned int * block) {
	for (int i = 0; i < 4; ++i) {
		for (int j = 0; j < 4; ++j) {
			std::printf("%4x ", block[i + 4 * j]);
		}
		std::printf("\n");
	}
	std::printf("\n");
}

unsigned int CPU_SubWord(unsigned int w) {
	unsigned int i = (sbox[(w >> 24) & 0xFF] << 24)
			| (sbox[(w >> 16) & 0xFF] << 16);
	i |= (sbox[(w >> 8) & 0xFF] << 8) | sbox[w & 0xFF];
	return i;
}

unsigned int CPU_RotWord(unsigned int w) {
	unsigned char temp = (w >> 24) & 0xFF;
	return ((w << 8) | temp);
}

void CPU_KeyExpansion(unsigned char* key, unsigned int* w) {
	unsigned int temp;
	int i = 0;

	for (i = 0; i < KEY_SIZE; i++) {
		w[i] = (key[4 * i] << 24) | (key[4 * i + 1] << 16)
				| (key[4 * i + 2] << 8) | key[4 * i + 3];
	}

	for (; i < BLOCK_SIZE * (NUM_ROUNDS + 1); i++) {
		temp = w[i - 1];
		if (i % KEY_SIZE == 0) {
			temp = CPU_SubWord(CPU_RotWord(temp)) ^ CPU_Rcon[i / KEY_SIZE];
		}
		w[i] = w[i - KEY_SIZE] ^ temp;
	}
}
