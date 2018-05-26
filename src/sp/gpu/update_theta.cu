#include "su/pixel.hpp"

#include "gisp_util.cuh"

#include "GISP.hpp"

#include <fstream>

using su::PixI;

namespace sp{

	__device__ __forceinline__ void tDNSymE2x2(float a00, float a01, float a11, float &e0, float &e1, float &v00, float &v01, float &v10, float &v11){
		float const zero = (float)0, one = (float)1, half = (float)0.5;
		float c2 = half * (a00 - a11), s2 = a01;
		float maxAbsComp = max(abs(c2), abs(s2));
		if (maxAbsComp > zero){
			c2 /= maxAbsComp;  // in [-1,1]
			s2 /= maxAbsComp;  // in [-1,1]
			float length = sqrt(c2 * c2 + s2 * s2);
			c2 /= length;
			s2 /= length;
			if (c2 > zero){
				c2 = -c2;	s2 = -s2;
			}
		}
		else{
			c2 = -one; s2 = zero;
		}
		float s = sqrt(half * (one - c2));  // >= 1/sqrt(2)
		float c = half * s2 / s;

		float csqr = c * c, ssqr = s * s, mid = s2 * a01;
		e0 = csqr * a00 + mid + ssqr * a11;
		e1 = csqr * a11 - mid + ssqr * a00;

		v00 = c;	v01 = s;
		v10 = -s; 	v11 = c;
	}

	// one theta one block
	// _NW : number of warps per block
	template<int _BX, int _BY, int _NW = ((_BX*_BY) >> 5)>
	__global__ void kernel_update_theta(
		float *oR, int oR_steps, int oR_lysize,
		PixI  *iC, int iC_steps,
		float *MG, int MG_steps,
		int W, int H, int v_x, int v_y, int n_x, int n_y, int rl, int ru, float e_s, float e_c
	){
#define t_x 1
#define t_y 1

		int i = 0;

#define __SHFREDUC(_var) for (i = 16; i > 0; i = (i >> 1)) _var += __shfl_xor(_var, i, 32);
#define __SUMARIZE(_var) smemMG[wy][wx] = _var; __syncthreads();          _var = 0; for (i = 0; i < _NW; i++){_var+=smemMG[i][wx];} __syncthreads();
#define __SUM0WARP(_var) smemMG[wy][wx] = _var; __syncthreads();if(wy==0){_var = 0; for (i = 0; i < _NW; i++) _var+=smemMG[i][wx];} __syncthreads();
    
		int ix = threadIdx.x;
		int iy = threadIdx.y;

		int wx = (iy*_BX + ix) & 31; // thread id in a warp, can use asm();
		int wy = (iy*_BX + ix) >> 5; // warp id in a block

		__shared__ float smemMG[_NW][32];

		int k_x = blockIdx.x;
    int k_y = blockIdx.y;
		int ik = k_x + k_y * n_x;
    
		int XXB = 0, XXE = 0, YYB = 0, YYE = 0;
		float mx = 0, my = 0, ml = 0, ma = 0, mb = 0, md = 0;
		int x = 0; int y = 0;

		dranging(XXB, XXE, YYB, YYE, W, H, n_x, n_y, k_x, k_y, v_x, v_y, rl, ru);
    
		for (y = YYB + iy; y < YYE; y += _BY){
			for (x = XXB + ix; x < XXE; x += _BX){
				const float RV = oR[(y - YYB)*oR_steps + (x - XXB) + ik*oR_lysize];// ik*oR_lysize can be moved out of this loop
				const su::PixI pix = iC[y*iC_steps + x];
				mx += RV*x;
				my += RV*y;
				ml += RV*pix.f0();
				ma += RV*pix.f1();
				mb += RV*pix.f2();
				md += RV;
			}
		}

    __SHFREDUC(md);__SUMARIZE(md);
		
    if (md < 1.f)
      return;
    
		// for each warp 
		__SHFREDUC(mx);	__SHFREDUC(my);	__SHFREDUC(ml);
		__SHFREDUC(ma);	__SHFREDUC(mb);	
    
    __SUMARIZE(mx); __SUMARIZE(my); __SUMARIZE(ml);
    __SUMARIZE(ma); __SUMARIZE(mb); 

		md = 1.f / md;
		mx = mx * md; my = my * md; ml = ml * md;
		ma = ma * md;	mb = mb * md;

		//////////////////////////////////////////////////////////////////////////
		// sigma 
		float xy00 = 0, xy01 = 0, xy11 = 0, ab00 = 0, ab01 = 0, ab11 = 0;
		float sl = 0, tp0 = 0, tp1 = 0;    
		
		for (y = YYB + iy; y < YYE; y += _BY){
			for (x = XXB + ix; x < XXE; x += _BX){
				const float RV     = oR[(y - YYB)*oR_steps + (x - XXB) + ik*oR_lysize];// has been aligned
				const su::PixI pix = iC[y*iC_steps + x]; 

				tp0 = x - mx; tp1 = y - my;

				xy00 += RV * tp0 * tp0;
				xy01 += RV * tp0 * tp1;
				xy11 += RV * tp1 * tp1;

				tp0 = pix.f0() - ml;
				sl += RV * tp0 * tp0;

				tp0 = pix.f1() - ma;
				tp1 = pix.f2() - mb;
				ab00 += RV * tp0 * tp0;
				ab01 += RV * tp0 * tp1;
				ab11 += RV * tp1 * tp1;
			}
		}
		__SHFREDUC(xy00); __SHFREDUC(xy01); __SHFREDUC(xy11);
		__SHFREDUC(ab00); __SHFREDUC(ab01); __SHFREDUC(ab11);
		__SHFREDUC(sl);

    
		__SUM0WARP(xy00); __SUM0WARP(xy01); __SUM0WARP(xy11);
		__SUM0WARP(ab00); __SUM0WARP(ab01); __SUM0WARP(ab11);
		__SUM0WARP(sl);
		
		float vxy00 = 0, vxy01 = 0, vxy10 = 0, vxy11 = 0, isx = 0, isy = 0;
		float vab00 = 0, vab01 = 0, vab10 = 0, vab11 = 0, isa = 0, isb = 0;
		float isl = 0, isd = 0;
    
		if (ix == 0 && iy == 0){
			xy00 = xy00 * md; xy01 = xy01 * md; xy11 = xy11 * md;
			tDNSymE2x2(xy00, xy01, xy11, isx, isy, vxy00, vxy01, vxy10, vxy11);
			if (isx < e_s) { isx = e_s; } isx = 1. / isx;
			if (isy < e_s) { isy = e_s; } isy = 1. / isy;

			ab00 = ab00 * md; ab01 = ab01 * md; ab11 = ab11 * md;
			tDNSymE2x2(ab00, ab01, ab11, isa, isb, vab00, vab01, vab10, vab11);
			if (isa < e_c) { isa = e_c; } isa = 1. / isa;
			if (isb < e_c) { isb = e_c; } isb = 1. / isb;

			sl = sl * md;
			if (sl < e_c){ sl = e_c; }isl = 1. / sl;

			isd = sqrt(isl * isx * isy * isa * isb);

			//  0,  1,  2,  3,  4,   5,   6,   7,   8,   9,    10,    11,    12,    13,    14,    15,    16,    17,  18,  19,  20,  21,  22
			// mx, my, ml, ma, mb, isx, isy, isl, isa, isb, vxy00, vxy01, vxy10, vxy11, vab00, vab01, vab10, vab11, isd, xxb, yyb, xxe, yye
			smemMG[0][ 0] = mx;
			smemMG[0][ 1] = my;
			smemMG[0][ 2] = ml;
			smemMG[0][ 3] = ma;
			smemMG[0][ 4] = mb;
			smemMG[0][ 5] = isx;
			smemMG[0][ 6] = isy;
			smemMG[0][ 7] = isl;
			smemMG[0][ 8] = isa;
			smemMG[0][ 9] = isb;
			smemMG[0][10] = vxy00;
			smemMG[0][11] = vxy01;
			smemMG[0][12] = vxy10;
			smemMG[0][13] = vxy11;
			smemMG[0][14] = vab00;
			smemMG[0][15] = vab01;
			smemMG[0][16] = vab10;
			smemMG[0][17] = vab11;
			smemMG[0][18] = isd;
		}
		__syncthreads();
		if (wy == 0){
			if (wx < 19)
				MG[ik*MG_steps + wx] = smemMG[0][wx];
		}
#undef __SHFREDUC
#undef t_x
#undef t_y
#undef __SUMARIZE
#undef __SUM0WARP
	}

	void gpu_update_theta(
		float    *oR, int oR_steps, int oR_lysize,
		su::PixI *iC, int iC_steps,
		float    *MG, int MG_steps,
		int W, int H, int v_x, int v_y, int n_x, int n_y, int rl, int ru, float e_s, float e_c
	){
#define _BX 16
#define _BY 8
		dim3 blocks(_BX, _BY);
		dim3 grid(n_x, n_y);
		kernel_update_theta<_BX, _BY> <<<grid, blocks >> >(oR, oR_steps, oR_lysize, iC, iC_steps, MG, MG_steps, W, H, v_x, v_y, n_x, n_y, rl, ru, e_s, e_c);
#undef _BY
#undef _BX
	}

}