/*****************************************************************************/
// File         : hemory.hpp
// Author       : Zhihua Ban
// Cotact       : sawpara@126.com
// Last Revised : 2017-1-18
/*****************************************************************************/
// Copyright 2017 Zhihua Ban. All rights reserved.
/*****************************************************************************/
// Desc : memory management cuda runtime applications
/*****************************************************************************/


#ifndef __HEMORY_HPP__
#define __HEMORY_HPP__


#include "../memory.hpp"
#include "cuda_runtime_api.h"
#include "cudart_util.h"


namespace su{ namespace gpu{
	
	class Hem
	{
	public:
		/*!
		\param sz, Size of the requested memory allocation.
		\param ss, The alignment value, which must be an integer power of 2.
		           We do not use it here, because cudaMallocHost does not 
							 support it now.
		*/
		static void *malloc(size_t sz, size_t ss){
			void *ptr = NULL;
			CUDART_CALL_CHECK(cudaMallocHost((void**)(&ptr), sz));
			return ptr;
		}

		/*!
		\param pd, Pointer to release.
		*/
		static void  release(void *ptr){
			CUDART_CALL_CHECK(cudaFreeHost(ptr));
		}
	}; // end class Hem

}} // end namespace su.


#endif