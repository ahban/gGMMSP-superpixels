/*****************************************************************************/
// File         : datrix.hpp
// Author       : Zhihua Ban
// Cotact       : sawpara@126.com
// Last Revised : 2017-1-18
/*****************************************************************************/
// Copyright 2017 Zhihua Ban. All rights reserved.
/*****************************************************************************/
// Desc : device matrix for cuda application
// Hat needs c++11 support 
/*****************************************************************************/


#ifndef __DATRIX_HPP__
#define __DATRIX_HPP__

#include "../matrix.hpp"

#include "hemory.hpp"
#include "cudart_util.h"
#include "cuda_runtime_api.h"

#include <iostream>
#include <iomanip>
#include <cstring>
#include <cmath>

namespace su{ namespace gpu{
	
	// needs c++11 support
	template<class _T = int, size_t _S = _DEFAULT_AIGNED_WIDTH_BYTES, class _HEM = Hem>
	using Hat = su::Mat<_T, _S, _HEM>;


	template<class _T = int, size_t _S = _DEFAULT_AIGNED_WIDTH_BYTES>
	class Dat
	{
	public:
		Dat();
		Dat(const size_t _rows, const size_t _cols, const size_t _layers = 1);

		Dat(const Dat<_T, _S> &_obj);
		Dat<_T, _S> operator=(const Dat<_T, _S> &_obj);
		template<class _TMM>
		Dat<_T, _S> operator=(const Mat<_T, _S, _TMM> &_obj);

		~Dat();

	public:
		void create(const size_t _rows, const size_t _cols, const size_t _layers = 1);
		void clear(){ cudaMemset(data, 0, layers*steps*rows*sizeof(_T)); }

	public:
		
		template<class _HEM>
		void upload(const su::Mat<_T, _S, _HEM> &_obj);
		template<class _HEM>
		void download(su::Mat<_T, _S, _HEM> &_obj) const;

		friend std::ostream& operator<<(std::ostream& os, const Dat<_T, _S> &_mat){
			su::Mat<_T, _S, Hem> temp;  _mat.download(temp);  os << temp;  return os;
		}

	public:
		_T *data;
		size_t layers, steps;
		union {
			size_t cols;
			size_t width;
		};
		union {
			size_t rows;
			size_t height;
		};
	};



	/*****************************************************************************/
	// implementation of Dat. Deveice matric
	/*****************************************************************************/


	template<class _T, size_t _S>
	template<class _HEM>
	void Dat<_T, _S>::download(su::Mat<_T, _S, _HEM> &_obj) const	{
		_obj.create(rows, cols, layers);
		CUDART_CALL_CHECK(cudaMemcpy(_obj.data, data, steps*rows*layers*sizeof(_T), cudaMemcpyDeviceToHost));
	}

	template<class _T, size_t _S>
	template<class _HEM>
	void Dat<_T, _S>::upload(const su::Mat<_T, _S, _HEM> &_obj){
		this->create(_obj.rows, _obj.cols, _obj.layers);
		CUDART_CALL_CHECK(cudaMemcpy(data, _obj.data, steps*rows*layers*sizeof(_T), cudaMemcpyHostToDevice));
	}

	template<class _T, size_t _S>
	Dat<_T, _S>::~Dat(){
		if (data){
			CUDART_CALL_CHECK(cudaFree(data)); data = NULL;
		}
		rows = layers = steps = cols = 0;
	}

	template<class _T, size_t _S>
	Dat<_T, _S>::Dat(){
		this->cols = this->rows = this->steps = this->layers = 0;
		this->data = NULL;
	}

	template<class _T, size_t _S>
	Dat<_T, _S>::Dat(const size_t _rows, const size_t _cols, const size_t _layers /*= 1*/){
		this->cols = this->rows = this->steps = this->layers = 0;
		this->data = NULL;
		this->create(_rows, _cols, _layers);
	}

	template<class _T, size_t _S>
	Dat<_T, _S>::Dat(const Dat<_T, _S> &_obj){
		this->cols = this->rows = this->steps = this->layers = 0;
		this->data = NULL;
		this->create(_obj.rows, _obj.cols, _obj.layers);
		CUDART_CALL_CHECK(cudaMemcpy(data, _obj.data, steps*rows*layers*sizeof(_T), cudaMemcpyDeviceToDevice));
	}

	template<class _T, size_t _S>
	Dat<_T, _S> Dat<_T, _S>::operator=(const Dat<_T, _S> &_obj){
		this->create(_obj.rows, _obj.cols, _obj.layers);
		CUDART_CALL_CHECK(cudaMemcpy(data, _obj.data, steps*rows*layers*sizeof(_T), cudaMemcpyDeviceToDevice));
		return *this;
	}

	template<class _T, size_t _S>
	template<class _TMM>
	Dat<_T, _S> Dat<_T, _S>::operator=(const Mat<_T, _S, _TMM> &_obj){
		this->create(_obj.rows, _obj.cols, _obj.layers);
		CUDART_CALL_CHECK(cudaMemcpy(data, _obj.data, steps*rows*layers*sizeof(_T), cudaMemcpyHostToDevice));
		return *this;
	}

	template<class _T, size_t _S>
	void Dat<_T, _S>::create(const size_t _rows, const size_t _cols, const size_t _layers /*= 1*/){
		if (this->cols == _cols && this->rows == _rows && _layers == this->layers){
			return;
		}
		this->rows = _rows;
		this->cols = _cols;
		this->layers = _layers;

		steps = su::aligned_words<_T, _S>(cols);

		// re-creating 
		if (data){ // data not null
			CUDART_CALL_CHECK(cudaFree(data)); data = NULL;
		}
		if (layers*rows*steps == 0){
			data = NULL; return;
		}
		CUDART_CALL_CHECK(cudaMalloc((void**)&data, layers*rows*steps*sizeof(_T)));
	}


}} // end namespace suan


#endif