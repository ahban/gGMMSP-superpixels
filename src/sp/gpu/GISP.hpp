
#ifndef __GPU_GMMSP_HPP__
#define __GPU_GMMSP_HPP__

#include "su/cuda/datrix.hpp"
#include "su/cuda/colors.cuh"

#include "su/pixel.hpp"
#include <inttypes.h>

using su::PixI;

//#define BENCH_ID 2

namespace sp{
	// basic types for GMMSP on GPU

	typedef su::gpu::Hat<float   > HatF; // theta, R
	typedef su::gpu::Hat<int32_t > HatL; // label type
	typedef su::gpu::Hat<su::PixI> HatP; // image 

	typedef su::gpu::Dat<float   > DatF; // theta, R
	typedef su::gpu::Dat<int32_t > DatL; // label type
	typedef su::gpu::Dat<su::PixI> DatP; // image 

	// pure C/C++ interfaces for GMMSP on GPU

	void gpu_init_theta(float *MG, int MG_steps, su::PixI *iC, int iC_steps, int W, int H, int v_x, int v_y, float sl, float sa, float sb);
	void gpu_update_R(
		float    *oR, int oR_steps, int oR_lysize,
		su::PixI *iC, int iC_steps,
		float    *MG, int MG_steps,
		int W, int H, int v_x, int v_y, int n_x, int n_y, int rl, int ru
	);
	void gpu_update_theta(
		float    *oR, int oR_steps, int oR_lysize,
		su::PixI *iC, int iC_steps,
		float    *MG, int MG_steps,
		int W, int H, int v_x, int v_y, int n_x, int n_y, int rl, int ru, float e_s, float e_c
	);
	
	void gpu_extract_labels(
		int32_t *hard, int hard_steps,
		float   *  oR, int   oR_steps, int oR_lysize,
		int W, int H, int v_x, int v_y, int rl, int ru
	);

	// class for superpixel segmentation
	class GISP{
	public:

		void segmentation_S(HatP &bgr, int v_xi, int v_yi, int Ti, float e_ci, float e_si, float lambda_ci){
			//////////////////////////////////////////////////////////////////////////
			// initialize parameters
			W = int(bgr.cols); 
			H = int(bgr.rows);
			lambda_c = lambda_ci;
			T = Ti;
			float sl = lambda_c;
			float sa = lambda_c;
			float sb = lambda_c;

			v_x = v_xi;
			v_y = v_yi;

			e_c = e_ci;	e_s = e_si;
			v_x = v_xi; v_y = v_yi;

			//////////////////////////////////////////////////////////////////////////
			int ntx = 2 * 1 + 1;
			int nty = 2 * 1 + 1;


			int n_x = W / v_x;
			int n_y = H / v_y;

			K = n_x * n_y;

			int rdw = W - n_x*v_x;
			int rdh = H - n_y*v_y;

			int rl = rdw >> 1;
			int ru = rdh >> 1;

			// theta cell size
			int tcw = v_x * ntx + rdw;
			int tch = v_y * nty + rdh;

			//////////////////////////////////////////////////////////////////////////
			// memory allcation
			oL.create(H, W);

			doL .create(H, W);
			dlab.create(H, W);
			dbgr.create(H, W);
			dMG.create(K, 32);
			doR.create(tch, tcw, K);
			doR.clear();

			//////////////////////////////////////////////////////////////////////////
			// method
			dbgr.upload(bgr);
			su::gpu::bgr2lab(dlab.data, dlab.steps, dbgr.data, dbgr.steps, W, H);

			//////////////////////////////////////////////////////////////////////////
			// gmmsp			
			gpu_init_theta(dMG.data, dMG.steps, dlab.data, dlab.steps, W, H, v_x, v_y, sl, sa, sb);
			gpu_update_R(doR.data, doR.steps, doR.steps*doR.rows, dlab.data, dlab.steps, dMG.data, dMG.steps, W, H, v_x, v_y, n_x, n_y, rl, ru);
			for (int it = 1; it < T; it++){
				gpu_update_theta(doR.data, doR.steps, doR.steps*doR.rows, dlab.data, dlab.steps, dMG.data, dMG.steps, W, H, v_x, v_y, n_x, n_y, rl, ru, e_s, e_c);
				gpu_update_R(doR.data, doR.steps, doR.steps*doR.rows, dlab.data, dlab.steps, dMG.data, dMG.steps, W, H, v_x, v_y, n_x, n_y, rl, ru);
			}
			gpu_extract_labels(doL.data, doL.steps, doR.data, doR.steps, doR.steps*doR.rows, W, H, v_x, v_y, rl, ru);

			doL.download(oL);

			//cudaEvent_t start, stop;
			//float time;
			//cudaEventCreate(&start);
			//cudaEventCreate(&stop);
			//int niterations = 100;
			//float max_time = 0, men_time = 0, min_time = FLT_MAX;
			//
			//max_time = 0, men_time = 0, min_time = FLT_MAX;
			//for (int b = 0; b < niterations; b++){
			//	cudaEventRecord(start, 0);
      //#if   BENCH_ID == 0
			//	gpu_update_theta(doR.data, doR.steps, doR.steps*doR.rows, dlab.data, dlab.steps, dMG.data, dMG.steps, W, H, v_x, v_y, n_x, n_y, rl, ru, e_s, e_c);
      //#elif BENCH_ID == 1
			//	gpu_update_R(doR.data, doR.steps, doR.steps*doR.rows, dlab.data, dlab.steps, dMG.data, dMG.steps, W, H, v_x, v_y, n_x, n_y, rl, ru);
      //#elif BENCH_ID == 2
      //  gpu_extract_labels(doL.data, doL.steps, doR.data, doR.steps, doR.steps*doR.rows, W, H, v_x, v_y, rl, ru);
      //#endif
			//	cudaEventRecord(stop, 0);
			//	cudaEventSynchronize(stop);
			//	cudaEventElapsedTime(&time, start, stop);
			//	men_time += time;
			//	if (time < min_time){ min_time = time; }
			//	if (time > max_time){ max_time = time; }
			//}
			//men_time = men_time / niterations;
			//printf("%2d, %0.06f, %0.06f, %0.06f\n", v_xi, men_time, men_time - min_time, max_time - men_time);
		}

	public:
		int W, H;

	public:
		int v_x, v_y, K;
		int T;
		float e_c, e_s;
		float lambda_c;

	public:
		HatL oL; // output label on host

		DatP dlab; // CIELab color space on device
		DatP dbgr; //    RGB color space on device
		DatL doL;  // output label on device
		DatF doR;  // post probability of superpixel k for pixel i on device
		DatF dMG;  // $\theta$ on device
	};



}// end namespace sp




#endif