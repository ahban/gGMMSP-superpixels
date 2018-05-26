#ifndef __GISP_UTIL_H__
#define __GISP_UTIL_H__

namespace sp{
	static __device__ __forceinline__ void dranging(int &XXB, int &XXE, int &YYB, int &YYE, int W, int H, int n_x, int n_y, int k_x, int k_y, int v_x, int v_y, int rl, int ru){
		if ((k_x - 1) <= 0)         XXB = 0;  else XXB = (k_x - 1) * v_x + rl;
		if ((k_x + 1) >= (n_x - 1)) XXE = W;	  else XXE = (k_x + 2) * v_x + rl;
		if ((k_y - 1) <= 0)         YYB = 0;  else YYB = (k_y - 1) * v_y + ru;
		if ((k_y + 1) >= (n_y - 1)) YYE = H;  	else YYE = (k_y + 2) * v_y + ru;
	}

	static __device__ __forceinline__ void dranging(int &XXB, int &YYB, int k_x, int k_y, int v_x, int v_y, int rl, int ru){
		if ((k_x - 1) <= 0) XXB = 0; else XXB = (k_x - 1) * v_x + rl;
		if ((k_y - 1) <= 0) YYB = 0; else YYB = (k_y - 1) * v_y + ru;
	}
}


#endif