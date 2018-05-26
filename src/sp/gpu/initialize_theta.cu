#include "su/pixel.hpp"

namespace sp{

	// float   *MG, int MG_steps, cols = 32
	//  0,  1,  2,  3,  4,   5,   6,   7,   8,   9,    10,    11,    12,    13,    14,    15,    16,    17,  18,  19,  20,  21,  22
	// mx, my, ml, ma, mb, isx, isy, isl, isa, isb, vxy00, vxy01, vxy10, vxy11, vab00, vab01, vab10, vab11, isd, xxb, yyb, xxe, yye
	template<int _BX/*must be 32*/, int _BY>
	__global__ void kernel_init_theta(
		float    *MG, int MG_steps,
		su::PixI *iC, int iC_steps,
		int W, int H, int v_x, int v_y, int n_x, int n_y, int xhr, int yhr, int hxs, int hys,
		float   isx, float   isy, float   isl, float   isa, float isb,
		float vxy00, float vxy01, float vxy10, float vxy11,
		float vab00, float vab01, float vab10, float vab11, float isd
		){
		int tx = threadIdx.x; // 0-31
		int ty = threadIdx.y; // 

		__shared__ float smem[_BY][_BX];

		int k_x = blockIdx.x;
		int k_y = blockIdx.y*_BY + ty;

		int k = k_x + k_y*n_x;

		int fx = (xhr + k_x*v_x + hxs);
		int fy = (yhr + k_y*v_y + hys);

		//  0,  1,  2,  3,  4,   5,   6,   7,   8,   9,    10,    11,    12,    13,    14,    15,    16,    17,  18
		// mx, my, ml, ma, mb, isx, isy, isl, isa, isb, vxy00, vxy01, vxy10, vxy11, vab00, vab01, vab10, vab11, isd
		if (k_x < n_x && k_y < n_y) {
			su::PixI pix = iC[fy*iC_steps + fx]; // permute?
			smem[ty][0] = fx;
			smem[ty][1] = fy;
			smem[ty][2] = pix.f0();
			smem[ty][3] = pix.f1();
			smem[ty][4] = pix.f2();// 5 is not initialized

			smem[ty][5] = isx;
			smem[ty][6] = isy;
			smem[ty][7] = isl;
			smem[ty][8] = isa;
			smem[ty][9] = isb;
			smem[ty][10] = vxy00;
			smem[ty][11] = vxy01;
			smem[ty][12] = vxy10;
			smem[ty][13] = vxy11;
			smem[ty][14] = vab00;
			smem[ty][15] = vab01;
			smem[ty][16] = vab10;
			smem[ty][17] = vab11;
			smem[ty][18] = isd;
		}
		__syncthreads();

		if (k_x >= n_x || k_y >= n_y){
			return;
		}

		if (tx < 19)
			MG[k*MG_steps + tx] = smem[ty][tx];
	}

	void gpu_init_theta(float *MG, int MG_steps, su::PixI *iC, int iC_steps, int W, int H, int v_x, int v_y, float sl, float sa, float sb){

		int n_x = W / v_x;
		int n_y = H / v_y;

		float isx = 1. / (v_x*v_x); // x
		float isy = 1. / (v_y*v_y); // y
		float isl = 1. / (sl*sl); // l
		float isa = 1. / (sa*sa); // a
		float isb = 1. / (sb*sb); // b

		float vxy00 = 1., vxy01 = 0.; // direction on x
		float vxy10 = 0., vxy11 = 1.; // direction on y

		float vab00 = 1., vab01 = 0.; // direction on a
		float vab10 = 0., vab11 = 1.; // direction on b

		float isd = sqrt(isx * isy * isl * isa * isb);

		int xhr = (W - n_x*v_x) >> 1;
		int yhr = (H - n_y*v_y) >> 1;
		int hxs = v_x >> 1;
		int hys = v_y >> 1;

#define _BX 32
#define _BY 4

		dim3 blocks(_BX, _BY);
		dim3 grids;
		grids.x = (n_x);
		grids.y = (n_y + blocks.y - 1) / blocks.y;

		kernel_init_theta<_BX, _BY> << <grids, blocks >> >(MG, MG_steps, iC, iC_steps, W, H, v_x, v_y, n_x, n_y, xhr, yhr, hxs, hys, isx, isy, isl, isa, isb, vxy00, vxy01, vxy10, vxy11, vab00, vab01, vab10, vab11, isd);

#undef _BX
#undef _BY
	}

}