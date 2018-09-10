/*
 * raid.hpp
 *
 *  Created on: Sep 10, 2018
 *      Author: hb4ch
 */

#ifndef RAID_HPP_
#define RAID_HPP_

__global__ void raid_cuda_encode(uint8_t * plain, uint8_t * p, uint8_t * q, size_t stride);


#endif /* RAID_HPP_ */
