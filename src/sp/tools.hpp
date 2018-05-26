/*****************************************************************************/
// File         : tools.hpp
// Author       : Zhihua Ban
// Cotact       : sawpara@126.com
// Last Revised : 2017-1-19
/*****************************************************************************/
// Copyright 2017 Zhihua Ban. All rights reserved.
/*****************************************************************************/
// Desc : tools for superpixel manipulation
/*****************************************************************************/

#ifndef __TOOLS_HPP__
#define __TOOLS_HPP__

#include <vector>
#include "su/matrix.hpp"
#include "su/pixel.hpp"
#include "sp/util.hpp"
namespace sp{
	class Util{
	public:

		static std::vector<su::PixI> random_color(int K){
			std::vector<su::PixI> res(K);
			for (int k = 0; k < K; k++){
				res[k] = su::PixI(rand() % 256, rand() % 256, rand() % 256);
			}
			return res;
		}
		
		template<class _ML>
		static int relabel(_ML &lab, const int ns){

			SU_ASSERT(lab.layers == 1, "matrix");
			int W = lab.width;
			int H = lab.height;
			// find min and max
			int lab_min = lab(0, 0);
			int lab_max = lab(0, 0);
			for (int y = 0; y < H; y++){
				for (int x = 0; x < W; x++){
					if (lab_max < lab(y, x))
						lab_max = lab(y, x);
					if (lab_min > lab(y, x))
						lab_min = lab(y, x);
				}
			}
			std::vector<int> m_map;
			// construct map
			int *p_map = NULL;
			if (lab_max < 0){
				lab_max = 0;
			}
			if (lab_min < 0){
				m_map.resize(lab_max + 1 - lab_min);
				p_map = &(m_map[-lab_min]);
			}
			else{
				m_map.resize(lab_max + 1);
				p_map = &(m_map[0]);
			}
			m_map.assign(m_map.size(), -1);

			int new_lab = ns;
			for (int y = 0; y < H; y++){
				for (int x = 0; x < W; x++){
					if (p_map[lab(y, x)] < 0){
						p_map[lab(y, x)] = new_lab;
						new_lab++;
					}
					lab(y, x) = p_map[lab(y, x)];
				}
			}

			return new_lab - ns;
		}
	};

	class Draw{
	public:
		template<class _MP, class _ML>
		static _MP contour(const _MP &img, const _ML &lab, const std::vector<su::PixI> &color){
			int W = img.cols;
			int H = img.rows;

			_MP res;
			res = img;

			int dx[] = { -1, 0, 1, 0 };
			int dy[] = { 0, -1, 0, 1 };
			int nd = sizeof(dx) / sizeof(*dx);

			for (int y = 0; y < H; y++){
				for (int x = 0; x < W; x++){
					bool is_boundary = false;

					// check if it is a boundary
					for (int d = 0; d < nd; d++){
						int tx = x + dx[d];
						int ty = y + dy[d];
						if (tx < 0 || ty < 0 || tx >= W || ty >= H)
							continue;
						if (lab(y, x) != lab(ty, tx)){
							is_boundary = true;
							break;
						}
					}

					if (is_boundary){
						res(y, x) = color[lab(y, x)];
					}
				}
			}
			return res;
		}

		template<class _MP, class _ML>
		static _MP contour(const _MP &img, const _ML &lab, const su::PixI &color){
			int W = int(img.cols);
			int H = int(img.rows);

			_MP res;
			res.create(img.rows, img.cols);
			res = img;

			int dx[] = { /*-1, 0,*/ 1, 0 };
			int dy[] = { /*0, -1,*/ 0, 1 };
			int nd = sizeof(dx) / sizeof(*dx);

			for (int y = 0; y < H; y++){
				for (int x = 0; x < W; x++){
					bool is_boundary = false;

					// check if it is a boundary
					for (int d = 0; d < nd; d++){
						int tx = x + dx[d];
						int ty = y + dy[d];
						if (tx < 0 || ty < 0 || tx >= W || ty >= H)
							continue;
						if (lab(y, x) != lab(ty, tx)){
							is_boundary = true;
							break;
						}
					}
					if (is_boundary){
						res(y, x) = color;
					}
				}
			}
			return res;
		}

		template<class _MP, class _ML>
		static _MP paintcs(const _ML &lab, const std::vector<su::PixI> &color){
			int W = lab.cols;
			int H = lab.rows;
			_MP res;
			res.create(H, W);
			
			for (int y = 0; y < H; y++){
				for (int x = 0; x < W; x++){
					res(y, x) = color[lab(y, x)];
				}
			}
			return res;
		}


		template<class _MP, class _ML>
		static _MP meanval(const _MP &img, const _ML &lab){
			int W = img.cols;
			int H = img.rows;

			_ML nlab(lab);

			int K = Util::relabel(nlab, 0);


			
			std::vector<double> mv0(K); mv0.assign(K, 0);
			std::vector<double> mv1(K); mv1.assign(K, 0);
			std::vector<double> mv2(K); mv2.assign(K, 0);
			std::vector<int> count(K); count.assign(K, 0);

			_MP res(H, W);

			for (int y = 0; y < H; y++){
				for (int x = 0; x < W; x++){
					mv0[nlab(y,x)] += img(y,x).u0();
					mv1[nlab(y,x)] += img(y,x).u1();
					mv2[nlab(y,x)] += img(y,x).u2();
					count[nlab(y, x)] ++;
				}
			}
			for (int k = 0; k < K; k++){
				mv0[k] = mv0[k] / double(count[k]);
				mv1[k] = mv1[k] / double(count[k]);
				mv2[k] = mv2[k] / double(count[k]);
			}

			for (int y = 0; y < H; y++){
				for (int x = 0; x < W; x++){
					res(y, x) = su::PixI(mv0[nlab(y, x)], mv1[nlab(y, x)], mv2[nlab(y, x)]);
				}
			}
			return res;
		}

	};

} // end namespace 

#endif
