#ifndef __NDARRAY_LINARG_HPP__
#define __NDARRAY_LINARG_HPP__

#include <string>
#include <typeinfo>

namespace na {

	/*   *** handle id array as column vector ***  */

	namespace linalg {
		
		struct bad_matirx_transpose{};

		template <typename T>
		class __ndArray_linalg_transpose_base : public ndArrayExpression<__ndArray_linalg_transpose_base<T>> {
		protected:
			typename std::conditional_t<T::is_leaf, const T, const T&> _u;
			__ndArray_shape shp;
			ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base::* op_br)(size_t i) const;
			std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base::* func_at)(size_t i) const;
			size_t col;
			size_t row;
			size_t* col_arr;
			size_t* row_arr;
		public:
			static constexpr bool is_leaf = false;
			__ndArray_linalg_transpose_base(T const& u, std::initializer_list<size_t> list,
				ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base::* _op_br)(size_t i) const,
				std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base::* _func_at)(size_t i) const) :
				_u(u), shp(list), op_br(_op_br), func_at(_func_at), col(0), row(0), col_arr(nullptr), row_arr(nullptr) {}

			__ndArray_linalg_transpose_base(T const& u, std::initializer_list<size_t> list,
				ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base::* _op_br)(size_t i) const,
				std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base::* _func_at)(size_t i) const,
				size_t _row, size_t _col) :
				_u(u), shp(list), op_br(_op_br), func_at(_func_at), col(_col), row(_row) {

				col_arr = (size_t*)__ndArray_allocator_instance.allocate(col * sizeof(size_t));
				row_arr = (size_t*)__ndArray_allocator_instance.allocate(row * sizeof(size_t));
				col_arr[0] = 0;
				row_arr[0] = 0;

				size_t i;
				for (i = 1; i < col; ++i) {
					col_arr[i] = col_arr[i - 1] + row;
				}

				for (i = 1; i < row; ++i) {
					row_arr[i] = row_arr[i - 1] + col;
				}
			}

			ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>> operator[](size_t i) const {return (this->*op_br)(i);}
			std::_Remove_cvref_t<decltype(std::declval<T>().at(0))> at(size_t i) const { return (this->*func_at)(i); }
			const auto& raw_shape() const { return shp; }
			~__ndArray_linalg_transpose_base() noexcept {
				if(!col_arr) __ndArray_allocator_instance.deallocate(col_arr, col);
				if(!row_arr) __ndArray_allocator_instance.deallocate(row_arr, row);
			}
		};

		template <typename T>
		class __ndArray_linalg_transpose_lv1 : public __ndArray_linalg_transpose_base<T> {
		public:
			__ndArray_linalg_transpose_lv1(T const& u) : __ndArray_linalg_transpose_base<T>(u, { 1, u.raw_shape().back() },
				static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base<T>::*)(size_t i) const>(&__ndArray_linalg_transpose_lv1<T>::operator[]),
				static_cast<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base<T>::*)(size_t i) const>(&__ndArray_linalg_transpose_lv1<T>::at)){
				assert(u.raw_shape().size() == 2);
			}
			ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>> operator[](size_t i) const {
				assert(i == 0);
				ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>> res = this->_u;
				return res;
			}
			std::_Remove_cvref_t<decltype(std::declval<T>().at(0))> at(size_t i) const { return this->_u.at(i); }
			const auto& raw_shape() const { return this->shp; }
		};


		template <typename T>
		class __ndArray_linalg_transpose_lv2_r : public __ndArray_linalg_transpose_base<T> {
		public:
			__ndArray_linalg_transpose_lv2_r(T const& u) : __ndArray_linalg_transpose_base<T>(u, { u.raw_shape().back() },
				static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base<T>::*)(size_t i) const>(&__ndArray_linalg_transpose_lv2_r<T>::operator[]),
				static_cast<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base<T>::*)(size_t i) const>(&__ndArray_linalg_transpose_lv2_r<T>::at)){
				assert(u.raw_shape().size() == 3 && u.raw_shape()[1] == u.raw_shape()[2]);
			}
			ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>> operator[](size_t i) const { return this->_u[0][i];}
			std::_Remove_cvref_t<decltype(std::declval<T>().at(0))> at(size_t i) const { return this->_u.at(i); }
			const auto& raw_shape() const { return this->shp; }
			
		};

		template <typename T>
		class __ndArray_linalg_transpose_lv2_c : public __ndArray_linalg_transpose_base<T> {
		public:
			__ndArray_linalg_transpose_lv2_c(T const& u) :__ndArray_linalg_transpose_base<T>(u, { 1, u.raw_shape().back() },
				static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base<T>::*)(size_t i) const>(&__ndArray_linalg_transpose_lv2_c<T>::operator[]),
				static_cast<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base<T>::*)(size_t i) const>(&__ndArray_linalg_transpose_lv2_c<T>::at)){
				assert(u.raw_shape()[1] = 1);
			}
			ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>> operator[](size_t i) const {
				assert(i == 0);
				ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>> res = this->_u;
				res.reshape({ this->shp.back() });
				return res;
			}
			std::_Remove_cvref_t<decltype(std::declval<T>().at(0))> at(size_t i) const { return this->_u.at(i); }
			const auto& raw_shape() const { return this->shp; }
		};
		
		template <typename T>
		class __ndArray_linalg_transpose_lv3 :public __ndArray_linalg_transpose_lv1<T> {
		public:
			__ndArray_linalg_transpose_lv3(T const& u) : __ndArray_linalg_transpose_base<T>(u, { u.raw_shape()[1], u.raw_shape()[2] / u.raw_shape()[1] },
				static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base<T>::*)(size_t i) const>(&__ndArray_linalg_transpose_lv3<T>::operator[]),
				static_cast<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base<T>::*)(size_t i) const>(&__ndArray_linalg_transpose_lv3<T>::at),
				u.raw_shape()[2] / u.raw_shape()[1], u.raw_shape()[1]) {
				assert(u.raw_shape().size() == 3);
			}
			ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>> operator[](size_t i) const
			{
				ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>> res;
				res.alloc({ this->row });
				for (size_t j = this->row - 1; j != 0; --j) {
					res.at(j) = this->_u.at(this->row_arr[j] + i);
				}
				res.at(0) = this->_u.at(this->row_arr[0] + i);
				
				return res;
			}
			std::_Remove_cvref_t<decltype(std::declval<T>().at(0))> at(size_t i) const
			{
				size_t r = i / this->row;
				size_t c = i - this->col_arr[r];
				return this->_u.at(this->row_arr[c] + r);
			}
			const auto& raw_shape() const { return this->shp; }
		};

		//transpose without dimension argument is only for matirx/vector
		template <typename T>
		__ndArray_linalg_transpose_base<T> transpose(const ndArrayExpression<T>& arr) {
			const auto& shp = arr.raw_shape();
			if (shp.size() == 2) {
				return __ndArray_linalg_transpose_base<T>(*static_cast<const T*>(&arr), { 1, arr.raw_shape().back() },
					static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv1<T>::operator[]),
					static_cast<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv1<T>::at));
			}
			else if (shp.size() == 3) {
				if (shp[1] == shp[2]) {
					return __ndArray_linalg_transpose_base<T>(*static_cast<const T*>(&arr), { arr.raw_shape().back() },
						static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv2_r<T>::operator[]),
						static_cast<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv2_r<T>::at));
				}
				else if (shp[1] == 1) {
					return __ndArray_linalg_transpose_base<T>(*static_cast<const T*>(&arr), { 1, arr.raw_shape().back() },
						static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv2_c<T>::operator[]),
						static_cast<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv2_c<T>::at));
				}
				else {
					size_t row = arr.raw_shape()[2] / arr.raw_shape()[1];
					size_t col = arr.raw_shape()[1];
					return __ndArray_linalg_transpose_base<T>(*static_cast<const T*>(&arr), { col, row },
						static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv3<T>::operator[]),
						static_cast<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv3<T>::at),
						row, col);
				}
			}
			
			throw bad_matirx_transpose();
		}
	}

	//matrix multiplication
	template <typename U, typename V>
	auto operator^ (const ndArray<U>& arr1, const ndArray<V>& arr2) -> ndArray<decltype(arr1.at(0) * arr2.at(0))> {
		assert(arr1.dim() <= 2 && arr1.dim() > 0);
		assert(arr2.dim() <= 2 && arr2.dim() > 0);

		size_t arr1_totalsize;
		size_t arr1_row;
		size_t arr1_col;

		if (arr1.dim() == 1) {
			arr1_totalsize = arr1.total_size();
			arr1_row = arr1.total_size();
			arr1_col = 1;
		}
		else {
			const __ndArray_shape& arr1_shp = arr1.raw_shape();
			arr1_totalsize = arr1_shp[2];
			arr1_row = arr1_shp[2] / arr1_shp[1];
			arr1_col = arr1_shp[1];
		}

		size_t arr2_row;
		size_t arr2_col;

		if (arr2.dim() == 1) {
			arr2_row = arr2.total_size();
			arr2_col = 1;
		}
		else {
			const __ndArray_shape& arr2_shp = arr2.raw_shape();
			arr2_row = arr2_shp[2] / arr2_shp[1];
			arr2_col = arr2_shp[1];
		}

		assert(arr1_col == arr2_row);

		ndArray<decltype(arr1.at(0)* arr2.at(0))> res;
		res.alloc({ arr1_row, arr2_col });

		for (size_t res_i = 0, row_i = 0; row_i < arr1_totalsize; res_i += arr2_col, row_i += arr1_col) {
			for (size_t j = 0; j < arr2_col; ++j) {
				//res[i][j] = arr1[i][0] * arr2[0][j];
				res.at(res_i + j) = arr1.at(row_i) * arr2.at(j);

				for (size_t k = 1, row_k = arr2_col; k < arr1_col; ++k, row_k += arr2_col) {
					//res[i][j] += arr1[i][k] * arr2[k][j];
					res.at(res_i + j) += arr1.at(row_i + k) * arr2.at(row_k + j);
				}
			}
		}

		if (arr1_row == 1 && arr2_col == 1) res.reshape({});

		return res;
	}
}

#endif