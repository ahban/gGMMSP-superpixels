/*****************************************************************************/
// File         : colors.cuh
// Author       : Zhihua Ban
// Cotact       : sawpara@126.com
// Last Revised : 2017-1-19
/*****************************************************************************/
// Copyright 2017 Zhihua Ban. All rights reserved.
/*****************************************************************************/
// Desc : color manipulation for c/c++ with NVIDIA GPU and nvcc
/*****************************************************************************/

#ifndef __COLORS_CUH__
#define __COLORS_CUH__

#include "../pixel.hpp"
#include "../colors.h"

namespace su{
	namespace gpu{
    
	  void bgr2lab(PixI *lab, int lab_steps, PixI* bgr, int bgr_steps, int W, int H);

  } // end namespace
} // end namespace

#endif