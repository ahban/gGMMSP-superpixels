/*****************************************************************************/
// File         : cudart_util.hpp
// Author       : Zhihua Ban
// Cotact       : sawpara@126.com
// Last Revised : 2017-1-18
/*****************************************************************************/
// Copyright 2017 Zhihua Ban. All rights reserved.
/*****************************************************************************/
// Desc : utility for cuda application.
/*****************************************************************************/


#ifndef __CUDART_UTIL_H__
#define __CUDART_UTIL_H__

#include "cuda_runtime_api.h"
#include <stdio.h>
#include <stdlib.h>

#define CUDART_CALL_CHECK(err) su::gpu::_cudart_error(err, __FILE__, __LINE__)
#define CUDART_LAST_CHECK      su::gpu::_cudart_error(cudaGetLastError(), __FILE__, __LINE__)

namespace su{namespace gpu{
  
  /*****************************************************************************/
  // CUDA Runtime error check
  /*****************************************************************************/
  static const char *_cudart_error_string(cudaError_t err)
  {
  	switch (err)
  	{
  	case cudaSuccess:
  		return "cudaSuccess";
  
  	case cudaErrorMissingConfiguration:
  		return "cudaErrorMissingConfiguration";
  
  	case cudaErrorMemoryAllocation:
  		return "cudaErrorMemoryAllocation";
  
  	case cudaErrorInitializationError:
  		return "cudaErrorInitializationError";
  
  	case cudaErrorLaunchFailure:
  		return "cudaErrorLaunchFailure";
  
  	case cudaErrorPriorLaunchFailure:
  		return "cudaErrorPriorLaunchFailure";
  
  	case cudaErrorLaunchTimeout:
  		return "cudaErrorLaunchTimeout";
  
  	case cudaErrorLaunchOutOfResources:
  		return "cudaErrorLaunchOutOfResources";
  
  	case cudaErrorInvalidDeviceFunction:
  		return "cudaErrorInvalidDeviceFunction";
  
  	case cudaErrorInvalidConfiguration:
  		return "cudaErrorInvalidConfiguration";
  
  	case cudaErrorInvalidDevice:
  		return "cudaErrorInvalidDevice";
  
  	case cudaErrorInvalidValue:
  		return "cudaErrorInvalidValue";
  
  	case cudaErrorInvalidPitchValue:
  		return "cudaErrorInvalidPitchValue";
  
  	case cudaErrorInvalidSymbol:
  		return "cudaErrorInvalidSymbol";
  
  	case cudaErrorMapBufferObjectFailed:
  		return "cudaErrorMapBufferObjectFailed";
  
  	case cudaErrorUnmapBufferObjectFailed:
  		return "cudaErrorUnmapBufferObjectFailed";
  
  	case cudaErrorInvalidHostPointer:
  		return "cudaErrorInvalidHostPointer";
  
  	case cudaErrorInvalidDevicePointer:
  		return "cudaErrorInvalidDevicePointer";
  
  	case cudaErrorInvalidTexture:
  		return "cudaErrorInvalidTexture";
  
  	case cudaErrorInvalidTextureBinding:
  		return "cudaErrorInvalidTextureBinding";
  
  	case cudaErrorInvalidChannelDescriptor:
  		return "cudaErrorInvalidChannelDescriptor";
  
  	case cudaErrorInvalidMemcpyDirection:
  		return "cudaErrorInvalidMemcpyDirection";
  
  	case cudaErrorAddressOfConstant:
  		return "cudaErrorAddressOfConstant";
  
  	case cudaErrorTextureFetchFailed:
  		return "cudaErrorTextureFetchFailed";
  
  	case cudaErrorTextureNotBound:
  		return "cudaErrorTextureNotBound";
  
  	case cudaErrorSynchronizationError:
  		return "cudaErrorSynchronizationError";
  
  	case cudaErrorInvalidFilterSetting:
  		return "cudaErrorInvalidFilterSetting";
  
  	case cudaErrorInvalidNormSetting:
  		return "cudaErrorInvalidNormSetting";
  
  	case cudaErrorMixedDeviceExecution:
  		return "cudaErrorMixedDeviceExecution";
  
  	case cudaErrorCudartUnloading:
  		return "cudaErrorCudartUnloading";
  
  	case cudaErrorUnknown:
  		return "cudaErrorUnknown";
  
  	case cudaErrorNotYetImplemented:
  		return "cudaErrorNotYetImplemented";
  
  	case cudaErrorMemoryValueTooLarge:
  		return "cudaErrorMemoryValueTooLarge";
  
  	case cudaErrorInvalidResourceHandle:
  		return "cudaErrorInvalidResourceHandle";
  
  	case cudaErrorNotReady:
  		return "cudaErrorNotReady";
  
  	case cudaErrorInsufficientDriver:
  		return "cudaErrorInsufficientDriver";
  
  	case cudaErrorSetOnActiveProcess:
  		return "cudaErrorSetOnActiveProcess";
  
  	case cudaErrorInvalidSurface:
  		return "cudaErrorInvalidSurface";
  
  	case cudaErrorNoDevice:
  		return "cudaErrorNoDevice";
  
  	case cudaErrorECCUncorrectable:
  		return "cudaErrorECCUncorrectable";
  
  	case cudaErrorSharedObjectSymbolNotFound:
  		return "cudaErrorSharedObjectSymbolNotFound";
  
  	case cudaErrorSharedObjectInitFailed:
  		return "cudaErrorSharedObjectInitFailed";
  
  	case cudaErrorUnsupportedLimit:
  		return "cudaErrorUnsupportedLimit";
  
  	case cudaErrorDuplicateVariableName:
  		return "cudaErrorDuplicateVariableName";
  
  	case cudaErrorDuplicateTextureName:
  		return "cudaErrorDuplicateTextureName";
  
  	case cudaErrorDuplicateSurfaceName:
  		return "cudaErrorDuplicateSurfaceName";
  
  	case cudaErrorDevicesUnavailable:
  		return "cudaErrorDevicesUnavailable";
  
  	case cudaErrorInvalidKernelImage:
  		return "cudaErrorInvalidKernelImage";
  
  	case cudaErrorNoKernelImageForDevice:
  		return "cudaErrorNoKernelImageForDevice";
  
  	case cudaErrorIncompatibleDriverContext:
  		return "cudaErrorIncompatibleDriverContext";
  
  	case cudaErrorPeerAccessAlreadyEnabled:
  		return "cudaErrorPeerAccessAlreadyEnabled";
  
  	case cudaErrorPeerAccessNotEnabled:
  		return "cudaErrorPeerAccessNotEnabled";
  
  	case cudaErrorDeviceAlreadyInUse:
  		return "cudaErrorDeviceAlreadyInUse";
  
  	case cudaErrorProfilerDisabled:
  		return "cudaErrorProfilerDisabled";
  
  	case cudaErrorProfilerNotInitialized:
  		return "cudaErrorProfilerNotInitialized";
  
  	case cudaErrorProfilerAlreadyStarted:
  		return "cudaErrorProfilerAlreadyStarted";
  
  	case cudaErrorProfilerAlreadyStopped:
  		return "cudaErrorProfilerAlreadyStopped";
  
  		/* Since CUDA 4.0*/
  	case cudaErrorAssert:
  		return "cudaErrorAssert";
  
  	case cudaErrorTooManyPeers:
  		return "cudaErrorTooManyPeers";
  
  	case cudaErrorHostMemoryAlreadyRegistered:
  		return "cudaErrorHostMemoryAlreadyRegistered";
  
  	case cudaErrorHostMemoryNotRegistered:
  		return "cudaErrorHostMemoryNotRegistered";
  
  		/* Since CUDA 5.0 */
  	case cudaErrorOperatingSystem:
  		return "cudaErrorOperatingSystem";
  
  	case cudaErrorPeerAccessUnsupported:
  		return "cudaErrorPeerAccessUnsupported";
  
  	case cudaErrorLaunchMaxDepthExceeded:
  		return "cudaErrorLaunchMaxDepthExceeded";
  
  	case cudaErrorLaunchFileScopedTex:
  		return "cudaErrorLaunchFileScopedTex";
  
  	case cudaErrorLaunchFileScopedSurf:
  		return "cudaErrorLaunchFileScopedSurf";
  
  	case cudaErrorSyncDepthExceeded:
  		return "cudaErrorSyncDepthExceeded";
  
  	case cudaErrorLaunchPendingCountExceeded:
  		return "cudaErrorLaunchPendingCountExceeded";
  
  	case cudaErrorNotPermitted:
  		return "cudaErrorNotPermitted";
  
  	case cudaErrorNotSupported:
  		return "cudaErrorNotSupported";
  
  		/* Since CUDA 6.0 */
  	case cudaErrorHardwareStackError:
  		return "cudaErrorHardwareStackError";
  
  	case cudaErrorIllegalInstruction:
  		return "cudaErrorIllegalInstruction";
  
  	case cudaErrorMisalignedAddress:
  		return "cudaErrorMisalignedAddress";
  
  	case cudaErrorInvalidAddressSpace:
  		return "cudaErrorInvalidAddressSpace";
  
  	case cudaErrorInvalidPc:
  		return "cudaErrorInvalidPc";
  
  	case cudaErrorIllegalAddress:
  		return "cudaErrorIllegalAddress";
  
  		/* Since CUDA 6.5*/
  	case cudaErrorInvalidPtx:
  		return "cudaErrorInvalidPtx";
  
  	case cudaErrorInvalidGraphicsContext:
  		return "cudaErrorInvalidGraphicsContext";
  
  	case cudaErrorStartupFailure:
  		return "cudaErrorStartupFailure";
  
  	case cudaErrorApiFailureBase:
  		return "cudaErrorApiFailureBase";
  
  		/* Since CUDA 8.0*/
  	case cudaErrorNvlinkUncorrectable:
  		return "cudaErrorNvlinkUncorrectable";
  	}
  
  	return "<unknown>";
  }
  
  static void _cudart_error(cudaError_t err, const char *const file, int const line){
  	if (cudaSuccess != err){
  		fprintf(stderr, "CUDA RT ERROR at %s : %d : \"%s\" \n", file, line, _cudart_error_string(err));
  		cudaDeviceReset();
  		exit(EXIT_FAILURE);
  	}
  }
  
    
  /*****************************************************************************/
  // device management
  /*****************************************************************************/
  
  static int warp_size(int device = 0){
  	int ws;
  	CUDART_CALL_CHECK(cudaDeviceGetAttribute(&ws, cudaDevAttrWarpSize, device));
  	return ws;
  }
  
  static int get_num_sm_cores(int major, int minor){
    switch ((major<<4)+minor){
      case 0x20: return 32;  // Fermi   (SM 2.0) GF100 class
      case 0x21: return 48;  // Fermi   (SM 2.1) GF10x class
      case 0x30: return 192; // Kepler  (SM 3.0) GK10x class
      case 0x32: return 192; // Kepler  (SM 3.2) GK10x class
      case 0x35: return 192; // Kepler  (SM 3.5) GK11x class
      case 0x37: return 192; // Kepler  (SM 3.7) GK21x class
      case 0x50: return 128; // Maxwell (SM 5.0) GM10x class
      case 0x52: return 128; // Maxwell (SM 5.2) GM20x class
      case 0x53: return 128; // Maxwell (SM 5.3) GM20x class
      case 0x60: return 64;  // Pascal  (SM 6.0) GP100 class
      case 0x61: return 128; // Pascal  (SM 6.1) GP10x class
      case 0x62: return 128; // Pascal  (SM 6.2) GP10x class
      default: return -1;
    }
  }

  static int find_best_gpu(){
    int num_gpus = 0;
    if (cudaSuccess != cudaGetDeviceCount(&num_gpus)){
      return -1;
    }
    double best_perf = 0;
    int best_gpu = 0;
    for (int i = 0; i < num_gpus; i++){
      cudaDeviceProp prop;
      CUDART_CALL_CHECK(cudaGetDeviceProperties(&prop, i));
      double curt_perf = 1.0*prop.clockRate*get_num_sm_cores(prop.major, prop.minor)*prop.multiProcessorCount;
      if (curt_perf > best_perf){
        best_perf = curt_perf;
        best_gpu = i;
      }
    }
    return best_gpu;
  }


}} // end namespace

#endif
