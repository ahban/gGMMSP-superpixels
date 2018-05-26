#include <cfloat>

#include "su/pixel.hpp"
#include "gisp_util.cuh"
#include "blocksizes.hpp"

#include "su/cuda/cudart_util.h"

using su::PixI;

namespace sp{
	// one block per superpixel
	// \param MG
	//  0,  1,  2,  3,  4,   5,   6,   7,   8,   9,    10,    11,    12,    13,    14,    15,    16,    17,  18,  19,  20,  21,  22
	// mx, my, ml, ma, mb, isx, isy, isl, isa, isb, vxy00, vxy01, vxy10, vxy11, vab00, vab01, vab10, vab11, isd, xxb, yyb, xxe, yye
	template<int _BX, int _BY, int _NW = ((_BX*_BY) >> 5)>
	__global__ void kernel_update_R(
		float    *oR, int oR_steps, int oR_lysize,
		su::PixI *iC, int iC_steps,
		float    *MG, int MG_steps,
		int W, int H, int v_x, int v_y, int n_x, int n_y, int rl, int ru
	){
#define epsilon_t (FLT_MIN*9.f)
#define t_x 1
#define t_y 1
#define ntx 3
#define nty 3
		__shared__ float smemM[ntx*nty][32];
		__shared__ float smemR[ntx*nty][_BY][_BX];

		int ix = threadIdx.x; // thread id x
		int iy = threadIdx.y; // thread id y

		int ik_x = blockIdx.x; // one superpixel per block
		int ik_y = blockIdx.y; // one superpixel per block

		int wx = (iy*_BX + ix) & 31; // thread id in a warp, do i need use asm() to get warp id?
		int wy = (iy*_BX + ix) >> 5; // warp id in a block

		int ok_x = 0;
		int ok_y = 0;
		int ok = 0;

    // load theta using warps
#if 1

#define LOADM3(cdx,cdy,nextld) ok_x=ik_x+cdx; ok_y=ik_y+cdy; if(ok_x<0||ok_x>=n_x||ok_y<0||ok_y>=n_y){goto nextld;} if (wx<24)smemM[(cdy+t_y)*ntx + cdx+t_x][wx] = MG[(ok_y*n_x + ok_x)*MG_steps+wx];
#define LOADM2(cdx,cdy       ) ok_x=ik_x+cdx; ok_y=ik_y+cdy; if(ok_x<0||ok_x>=n_x||ok_y<0||ok_y>=n_y){   continue;} if (wx<24)smemM[(cdy+t_y)*ntx + cdx+t_x][wx] = MG[(ok_y*n_x + ok_x)*MG_steps+wx];
#define LOADM4(cdx,cdy,_nid,_nws) LOADM3(cdx, cdy, nextn##_nid##_##_nws); nextn##_nid##_##_nws:

		if (_NW == 1){ for (int dy = -t_y; dy <= t_y; dy++){ for (int dx = -t_x; dx <= t_x; dx++){	LOADM2(dx, dy);}}}

		if (_NW == 2){ // 2 warps per block
			if (wy == 0){
				LOADM4(-1, -1, 0, 2);	LOADM4( 0, -1, 1, 2);	LOADM4( 1, -1, 2, 2);	LOADM4(-1,  0, 3, 2);	LOADM4( 0,  0, 4, 2);
			}
			else{ // wy == 1
				LOADM4( 1, 0, 5, 2);	LOADM4(-1, 1, 6, 2);		LOADM4( 0, 1, 7, 2);	LOADM4( 1, 1, 8, 2);
			}
		}


		if (_NW == 3){
			switch (wy)
			{
			  case 0:LOADM4(-1, -1, 0, 3); LOADM4(0, -1, 1, 3); LOADM4( 1, -1, 2, 3); break;
			  case 1:LOADM4(-1,  0, 3, 3); LOADM4(1,  0, 5, 3); LOADM4( 0,  0, 4, 3); break;
			  case 2:LOADM4(-1,  1, 6, 3); LOADM4( 0, 1, 7, 3); LOADM4( 1,  1, 8, 3); break;	
			}
		}

	  if (_NW == 4){
			switch (wy)
			{
			  case 0:LOADM4(-1, -1, 0, 4); LOADM4(0, -1, 1, 4); break;
			  case 1:LOADM4( 1, -1, 2, 4); LOADM4(-1, 0, 3, 4); break;
			  case 2:LOADM4( 1,  0, 5, 4); LOADM4( 0, 0, 4, 4); break;
			  case 3:LOADM4(-1,  1, 6, 4); LOADM4( 0, 1, 7, 4); LOADM4( 1,  1, 8, 4); break;	
			}
		}

		if (_NW == 5){
			switch (wy)
			{
			  case 0:LOADM4(-1, -1, 0, 5); LOADM4(0, -1, 1, 5); break;
			  case 1:LOADM4( 1, -1, 2, 5); LOADM4(-1, 0, 3, 5); break;
			  case 2:LOADM4( 1,  0, 5, 5); LOADM4( 0, 0, 4, 5); break;
			  case 3:LOADM4(-1,  1, 6, 5); LOADM4( 0, 1, 7, 5); break;	
			  case 4:LOADM4( 1,  1, 8, 5); break;
			}
		}

		if (_NW == 6){
			switch (wy)
			{
			  case 0:LOADM4(-1, -1, 0, 6); LOADM4(0, -1, 1, 6); break;
			  case 1:LOADM4( 1, -1, 2, 6); LOADM4(-1, 0, 3, 6); break;
			  case 2:LOADM4( 1,  0, 5, 6); LOADM4( 0, 0, 4, 6); break;
			  case 3:LOADM4(-1,  1, 6, 6); break;
			  case 4:LOADM4( 0,  1, 7, 6); break;
			  case 5:LOADM4( 1,  1, 8, 6); break;
			}
		}
		if (_NW == 7){ 
			switch (wy)
			{
			  case 0:LOADM4(-1, -1, 0, 7); LOADM4(0, -1, 1, 7); break;
			  case 1:LOADM4( 1, -1, 2, 7); LOADM4(-1, 0, 3, 7); break;
			  case 2:LOADM4( 1,  0, 5, 7); break;
				case 3:LOADM4( 0,  0, 4, 7); break;
			  case 4:LOADM4(-1,  1, 6, 7); break;
				case 5:LOADM4( 0,  1, 7, 7); break;
				case 6:LOADM4( 1,  1, 8, 7); break;
			}
		}
		if (_NW == 8){
			switch (wy)
			{
			  case 0:LOADM4(-1, -1, 0, 8); LOADM4(0, -1, 1, 8); break;
			  case 1:LOADM4( 1, -1, 2, 8); break;
			  case 2:LOADM4(-1,  0, 3, 8); break;
				case 3:LOADM4( 1,  0, 5, 8); break;
			  case 4:LOADM4( 0,  0, 4, 8); break;
				case 5:LOADM4(-1,  1, 6, 8); break;
				case 6:LOADM4( 0,  1, 7, 8); break;
				case 7:LOADM4( 1,  1, 8, 8); break;
			}
		}
		if (_NW > 8){
			switch (wy)
			{
			  case 0:LOADM4(-1, -1, 0, 9); break;
			  case 1:LOADM4( 0, -1, 1, 9); break;
			  case 2:LOADM4( 1, -1, 2, 9); break;
				case 3:LOADM4(-1,  0, 3, 9); break;
			  case 4:LOADM4( 1,  0, 5, 9); break;
				case 5:LOADM4( 0,  0, 4, 9); break;
				case 6:LOADM4(-1,  1, 6, 9); break;
				case 7:LOADM4( 0,  1, 7, 9); break;
				case 8:LOADM4( 1,  1, 8, 9); break;
			}
		}		
#endif

		__syncthreads();

		int sf_xb = (ik_x == 0) ? 0 : (rl + ik_x*v_x); // float begin of current superpixel 
		int sf_yb = (ik_y == 0) ? 0 : (ru + ik_y*v_y); // float begin of current superpixel
		//int sf_xa = (sf_xb >> 3) << 3; // aligned begin. aligned with 8*4 = 32Byte = a L2 transaction

		int sf_xe = (ik_x == (n_x - 1)) ? W : (rl + (ik_x + 1)*v_x); // float end+1 of current superpixel
		int sf_ye = (ik_y == (n_y - 1)) ? H : (ru + (ik_y + 1)*v_y); // float end+1 of current superpixel

		float ffi0 = 0, ffi1 = 0;
		float ffo0 = 0, ffo1 = 0;
		float ff = 0;
		float d_xy = 0, d_l = 0, d_ab = 0, D = 0;
		float sum_R = 0, sum_exists = 0;

		float   isx = 0, isy = 0, isl = 0, isa = 0, isb = 0;
		float vxy00 = 0, vxy01 = 0, vxy10 = 0, vxy11 = 0;
		float vab00 = 0, vab01 = 0, vab10 = 0, vab11 = 0;
		float   isd = 0;

		float mx = 0, my = 0, ml = 0, ma = 0, mb = 0;
		int xxb = 0, yyb = 0;

		// scanning each pixel in this superpixel
		for (int y = sf_yb + iy; y < sf_ye; y += _BY){
			for (int x = sf_xb + ix; x < sf_xe; x += _BX){

				su::PixI px = iC[y*iC_steps + x];
				sum_exists = 0; sum_R = 0;

				// for its neighboring superpixels of pixel (x,y), update the R
				//  0,  1,  2,  3,  4,   5,   6,   7,   8,   9,    10,    11,    12,    13,    14,    15,    16,    17,  18,  19,  20,  21,  22
				// mx, my, ml, ma, mb, isx, isy, isl, isa, isb, vxy00, vxy01, vxy10, vxy11, vab00, vab01, vab10, vab11, isd, xxb, yyb, xxe, yye
				for (int dy = -t_y; dy <= t_y; dy++){
					for (int dx = -t_x; dx <= t_x; dx++){
						ok_x = ik_x + dx;
						ok_y = ik_y + dy;
						if (ok_x < 0 || ok_x >= n_x || ok_y < 0 || ok_y >= n_y)
							continue;

						mx    = smemM[(dy + t_y)*ntx + dx + t_x][ 0];
						my    = smemM[(dy + t_y)*ntx + dx + t_x][ 1];
						ml    = smemM[(dy + t_y)*ntx + dx + t_x][ 2];
						ma    = smemM[(dy + t_y)*ntx + dx + t_x][ 3];
						mb    = smemM[(dy + t_y)*ntx + dx + t_x][ 4];
						isx   = smemM[(dy + t_y)*ntx + dx + t_x][ 5];
						isy   = smemM[(dy + t_y)*ntx + dx + t_x][ 6];
						isl   = smemM[(dy + t_y)*ntx + dx + t_x][ 7];
						isa   = smemM[(dy + t_y)*ntx + dx + t_x][ 8];
						isb   = smemM[(dy + t_y)*ntx + dx + t_x][ 9];
						vxy00 = smemM[(dy + t_y)*ntx + dx + t_x][10];
						vxy01 = smemM[(dy + t_y)*ntx + dx + t_x][11];
						vxy10 = smemM[(dy + t_y)*ntx + dx + t_x][12];
						vxy11 = smemM[(dy + t_y)*ntx + dx + t_x][13];
						vab00 = smemM[(dy + t_y)*ntx + dx + t_x][14];
						vab01 = smemM[(dy + t_y)*ntx + dx + t_x][15];
						vab10 = smemM[(dy + t_y)*ntx + dx + t_x][16];
						vab11 = smemM[(dy + t_y)*ntx + dx + t_x][17];
						isd   = smemM[(dy + t_y)*ntx + dx + t_x][18];

						ffi0 = x - mx; ffi1 = y - my;
						ffo0 = ffi0 * vxy00 + ffi1 * vxy01; ffo0 = ffo0 * ffo0;
						ffo1 = ffi0 * vxy10 + ffi1 * vxy11; ffo1 = ffo1 * ffo1;
						d_xy = ffo0 * isx + ffo1 * isy;

						// l
						ff = px.f0() - ml; ff = ff*ff;
						d_l = ff * isl;

						// a, b
						ffi0 = px.f1() - ma;
						ffi1 = px.f2() - mb;
						ffo0 = ffi0 * vab00 + ffi1 * vab01; ffo0 = ffo0 * ffo0;
						ffo1 = ffi0 * vab10 + ffi1 * vab11; ffo1 = ffo1 * ffo1;
						d_ab = ffo0 *   isa + ffo1 * isb;

						D = (d_xy + d_l + d_ab)*float(-0.5);

						smemR[(dy + t_y)*ntx + t_x + dx][iy][ix] = exp(D) * isd;

						sum_R += exp(D) * isd;
						sum_exists += 1.f;
					}
				}
				if (sum_R < epsilon_t){
					for (int dy = -t_y; dy <= t_y; dy++){
						for (int dx = -t_x; dx <= t_x; dx++){
							ok_x = ik_x + dx;
							ok_y = ik_y + dy;
							if (ok_x < 0 || ok_x >= n_x || ok_y < 0 || ok_y >= n_y){
								continue;
							}
							ok = ok_y*n_x + ok_x;
							dranging(xxb, yyb, ok_x, ok_y, v_x, v_y, rl, ru);
							oR[(y - yyb)*oR_steps + (x - xxb) + ok*oR_lysize] = float(1) / float(sum_exists);
						}
					}
				}
				else{
					for (int dy = -t_y; dy <= t_y; dy++){
						for (int dx = -t_x; dx <= t_x; dx++){
							ok_x = ik_x + dx;
							ok_y = ik_y + dy;
							if (ok_x < 0 || ok_x >= n_x || ok_y < 0 || ok_y >= n_y)
								continue;
							ok = ok_y*n_x + ok_x;
							dranging(xxb, yyb, ok_x, ok_y, v_x, v_y, rl, ru);
							oR[(y - yyb)*oR_steps + (x - xxb) + ok*oR_lysize] = smemR[(dy + t_y)*ntx + t_x + dx][iy][ix] / sum_R;
						}
					}
				}
			}
		} // end scan each pixel in this superpixel 

#undef epsilon_t
#undef t_x 
#undef t_y 
#undef ntx 
#undef nty 
#undef LOADM3
#undef LOADM2
#undef LOADM4
	}

	void gpu_update_R(
		float    *oR, int oR_steps, int oR_lysize,
		su::PixI *iC, int iC_steps,
		float    *MG, int MG_steps,
		int W, int H, int v_x, int v_y, int n_x, int n_y, int rl, int ru
		){
#define _BX 16
#define _BY 8
		dim3 blocks(_BX, _BY);
		dim3 grid(n_x, n_y);
		kernel_update_R<_BX, _BY> <<<grid, blocks >> >(oR, oR_steps, oR_lysize, iC, iC_steps, MG, MG_steps, W, H, v_x, v_y, n_x, n_y, rl, ru);
		CUDART_LAST_CHECK;
#undef _BX
#undef _BY	
	}


	
}// end namespace sp
