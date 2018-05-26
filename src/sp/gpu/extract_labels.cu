#include <inttypes.h>
#include <cfloat>

#include "gisp_util.cuh"


namespace sp{

	__global__ void kernel_extract_labels(
		int32_t *hard, int hard_steps,
		float   *  oR, int   oR_steps, int oR_lysize,
		int W, int H, int v_x, int v_y, int n_x, int n_y, int rl, int ru
	){
#define t_x 1
#define t_y 1
		int x = threadIdx.x + blockIdx.x*blockDim.x;
		int y = threadIdx.y + blockIdx.y*blockDim.y;
		if (x >= W || y >= H){
			return;
		}
		int ilabel_x = (x - rl) / v_x; if (ilabel_x == n_x) ilabel_x = n_x - 1;
		int ilabel_y = (y - ru) / v_y; if (ilabel_y == n_y) ilabel_y = n_y - 1;

		float max_dense = -FLT_MAX;
		int final_label = -1;

		for (int dy = -t_y; dy <= t_y; dy++){
			for (int dx = -t_x; dx <= t_x; dx++){
				const int al_x = ilabel_x + dx;
				const int al_y = ilabel_y + dy;
				if (al_x < 0 || al_y < 0 || al_x >= n_x || al_y >= n_y){
					continue;
				}
				const int al_k = al_y*n_x + al_x;

				int xxb, yyb;
				dranging(xxb, yyb, al_x, al_y, v_x, v_y, rl, ru);
				float cur_dense = oR[(y - yyb)*oR_steps + x - xxb + al_k*oR_lysize];
				if (max_dense < cur_dense){
					max_dense = cur_dense;
					final_label = al_k;
				}
			}
		}
		hard[y*hard_steps + x] = final_label;
#undef t_x
#undef t_y
	}

	void gpu_extract_labels(
		int32_t *hard, int hard_steps,
		float   *  oR, int   oR_steps, int oR_lysize,
		int W, int H, int v_x, int v_y, int rl, int ru
	){
		int n_x = W / v_x;
		int n_y = H / v_y;

#define _BX 16
#define _BY 8

		dim3 blocks(_BX, _BY);
		dim3 grids;
		grids.x = (W + blocks.x - 1) / blocks.x;
		grids.y = (H + blocks.y - 1) / blocks.y;

		kernel_extract_labels<<<grids, blocks>>>(hard, hard_steps, oR, oR_steps, oR_lysize, W, H, v_x, v_y, n_x, n_y, rl, ru);

#undef _BX
#undef _BY
	}

}