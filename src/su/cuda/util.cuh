/*****************************************************************************/
// File         : util.cuh
// Author       : Zhihua Ban
// Cotact       : sawpara@126.com
// Last Revised : 2017-1-19
/*****************************************************************************/
// Copyright 2017 Zhihua Ban. All rights reserved.
/*****************************************************************************/
// Desc : util for cuda kernels
/*****************************************************************************/

#ifndef __KERNEL_UTIL_CUH__
#define __KERNEL_UTIL_CUH__

namespace su{ namespace gpu{

	static __device__ __forceinline__ unsigned int laneid(){
		unsigned int i;
		asm("mov.u32 %0, %laneid;" : "=r"(i));
		return i;
	}

}}

#endif