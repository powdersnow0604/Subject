#ifndef __NDARRAY_LINARG_HPP__
#define __NDARRAY_LINARG_HPP__

#include <string>

namespace na {

	/*   *** handle id array as column vector ***  */


	namespace linalg {

		extern 	__ndArray_allocator __ndArray_allocator_linalg;

		typedef enum _linarg_type : unsigned char
		{
			LINARGTYPE_VECTOR = 0,
			LINARGTYPE_VECTOR_ID = 1,
			LINARGTYPE_VECTOR_2D_R = 2,
			LINARGTYPE_VECTOR_2D_C = 3,
			LINARGTYPE_MATRIX = 4,
			LINARGTYPE_TENSOR = 5,
			LINARGTYPE_SCALAR = 6,
		}LINALGTYPE;

		template<typename T>
		LINALGTYPE type(const ndArrayExpression<T>& arr) {
			const auto& shp = arr.raw_shape();
			if (shp.size() == 1) return LINARGTYPE_SCALAR;
			if (shp.size() == 2) return LINARGTYPE_VECTOR_ID;
			else if (shp.size() == 3) {
				if (shp[1] == shp[2]) return LINARGTYPE_VECTOR_2D_R;
				else if (shp[1] == 1) return LINARGTYPE_VECTOR_2D_C;
				else return LINARGTYPE_MATRIX;
			}
			else return LINARGTYPE_TENSOR;					
		}

		#pragma region support_transpose

		// lv1: 1d array vector
		// lv2_r: 2d array row vector
		// lv2_c: 2d array column vector
		// lv3: 2d array matirx

		struct bad_matrix_transpose{
			std::string err_msg;
			bad_matrix_transpose(const char* msg) : err_msg(msg) {}
		};

		template <typename T>
		class __ndArray_linalg_transpose_base : public ndArrayExpression<__ndArray_linalg_transpose_base<T>> {
		protected:
			typename std::conditional_t<T::is_leaf, const T, const T&> _u;
			__ndArray_shape shp;
			size_t col;
			size_t row;
			size_t* col_arr;
			size_t* row_arr;
		private:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base::* op_br)(size_t i) const;
			std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base::* func_at)(size_t i) const;
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

				col_arr = (size_t*)__ndArray_allocator_linalg.allocate(col * sizeof(size_t));
				row_arr = (size_t*)__ndArray_allocator_linalg.allocate(row * sizeof(size_t));
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
				if(col_arr) __ndArray_allocator_linalg.deallocate(col_arr, col);
				if(row_arr) __ndArray_allocator_linalg.deallocate(row_arr, row);
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

		#pragma endregion

		//transpose without dimension argument is only for matirx/vector
		template <typename T>
		__ndArray_linalg_transpose_base<T> transpose(const ndArrayExpression<T>& arr) {
			
			switch (type(*static_cast<const T*>(&arr)))
			{
			case 4: {
				size_t row = arr.raw_shape()[2] / arr.raw_shape()[1];
				size_t col = arr.raw_shape()[1];
				return __ndArray_linalg_transpose_base<T>(*static_cast<const T*>(&arr), { col, row },
					static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv3<T>::operator[]),
					static_cast<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv3<T>::at),
					row, col); }
			case 5: 
				throw bad_matrix_transpose{"its tensor"};
			case 6:
				throw bad_matrix_transpose("its scalar");
			case 1:
				return __ndArray_linalg_transpose_base<T>(*static_cast<const T*>(&arr), { 1, arr.raw_shape().back() },
					static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv1<T>::operator[]),
					static_cast<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv1<T>::at));
			case 2:
				return __ndArray_linalg_transpose_base<T>(*static_cast<const T*>(&arr), { arr.raw_shape().back() },
					static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv2_r<T>::operator[]),
					static_cast<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv2_r<T>::at));
			case 3:
				return __ndArray_linalg_transpose_base<T>(*static_cast<const T*>(&arr), { 1, arr.raw_shape().back() },
					static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv2_c<T>::operator[]),
					static_cast<std::_Remove_cvref_t<decltype(std::declval<T>().at(0))>(__ndArray_linalg_transpose_base<T>::*)(size_t) const>(&__ndArray_linalg_transpose_lv2_c<T>::at));
			}

			throw bad_matrix_transpose{};
		}

		template <typename T>
		inline void transpose_matrix_block(T* dst, T* src,  size_t row, size_t col, size_t i_limit, size_t j_limit) {
			/*
			for (size_t i = i_s, i_col = i_s * col; i < i_limit; i++, i_col += col) {
				for (size_t j = j_s, j_row = i + j_s * row; j < j_limit; j++, j_row += row) {
					dst[j_row] = src[i_col + j];
				}
			}
			*/

			for (size_t i = 0, i_col = 0; i < i_limit; i++, i_col += col) {
				for (size_t j = 0, j_row = 0; j < j_limit; j++, j_row += row) {
					dst[j_row + i] = src[i_col + j];
				}
			}
		}


		#pragma region support_inner_product
		struct bad_inner_product {
			std::string err_msg;
			bad_inner_product(const char* msg) : err_msg(msg) {}
		};

		template <typename U, typename V>
		class __ndArray_inner_base;

		template <typename U, typename V>
		class __ndArray_inner_lv1_m : public __ndArray_inner_base<U, V> {
		public:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
				operator[](size_t i) const
			{
				ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
				res.alloc({ this->col });
				res.at(0) = this->u.at(i) * this->v.at(0);
				for (size_t j = this->col - 1; j != 0; --j) {
					res.at(j) = this->u.at(0) * this->v.at(j);
				}

				return res;
			}

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
				at(size_t i) const
			{
				size_t r = i / this->col;
				size_t c = i - this->temp[r];
				return this->u.at(r) * this->v.at(c);
			}
		};

		template <typename U, typename V>
		class __ndArray_inner_lv1_s : public __ndArray_inner_base<U, V> {
		public:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
				operator[](size_t i) const
			{
				ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
				res.alloc({ this->col });
				assert(i == 0);

				res.at(0) = this->u.at(0) * this->v.at(0);
				for (size_t j = this->share - 1; j != 0; --j) {
					res.at(j) += this->u.at(j) * this->v.at(j);
				}

				return res;
			}

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
				at(size_t i) const
			{
				assert(i == 0);

				std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))> sum = this->u.at(0) * this->v.at(0);

				for (size_t j = this->share - 1; j != 0; --j) {
					sum += this->u.at(j) * this->v.at(j);
				}

				return sum;
			}
		};

		template <typename U, typename V>
		class __ndArray_inner_lv2_r : public __ndArray_inner_base<U, V> {
		public:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
				operator[](size_t i) const
			{
				ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
				res.alloc({});

				res.at(0) = 0;
				for (size_t j = 0, j_row = i; j < this->share; ++j, j_row += this->row) {
					res.at(0) += this->u.at(j_row) * this->v.at(j);
				}

				return res;
			}

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
				at(size_t i) const
			{
				std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))> sum = 0;
				for (size_t j = 0, j_row = i; j < this->share; ++j, j_row += this->row) {
					sum += this->u.at(j_row) * this->v.at(j);
				}

				return sum;
			}
		};

		template <typename U, typename V>
		class __ndArray_inner_lv2_l : public __ndArray_inner_base<U, V> {
		public:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
				operator[](size_t i) const
			{
				assert(i == 0);
				ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
				res.alloc({ this->col });
				for (size_t j = this->col - 1;; --j) {
					res.at(j) = this->u.at(0) * this->v.at(j);
					for (size_t k = this->share - 1, v_k = j + k * this->col; k != 0; --k, v_k -= this->col) {
						res.at(j) += this->u.at(k) * this->v.at(v_k);
					}
					if (j == 0) break;
				}

				return res;
			}

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
				at(size_t i) const
			{
				std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))> sum = this->u.at(0) * this->v.at(i);
				for (size_t j = this->share - 1, v_j = i + j * this->col; j != 0; --j, v_j -= this->col) {
					sum += this->u.at(j) * this->v.at(v_j);
				}

				return sum;
			}
		};

		template <typename U, typename V>
		class __ndArray_inner_lv3 : public __ndArray_inner_base<U, V> {
		public:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
				operator[](size_t i) const
			{
				ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
				res.alloc({ this->col });
				for (size_t j = 0; j < this->col; ++j) {
					res.at(j) = this->u.at(i) * this->v.at(j);
					for (size_t u_col = i + this->row, k_col = j + this->col; k_col < this->v.raw_shape().back(); u_col += this->row, k_col += this->col) {
						res.at(j) += this->u.at(u_col) * this->v.at(k_col);
					}
				}

				return res;
			}

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
				at(size_t i) const
			{
				size_t r = i / this->col;
				size_t c = i - r * this->col;
				std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))> sum = this->u.at(r) * this->v.at(c);
				for (size_t j_col = c + this->col, u_col = r + this->row; j_col < this->v.raw_shape().back(); j_col += this->col, u_col += this->row) {
					sum += this->u.at(u_col) * this->v.at(j_col);
				}

				return sum;
			}
		};

		template <typename U, typename V>
		class __ndArray_inner_base : public ndArrayExpression<__ndArray_inner_base<U, V>> {
		protected:
			typename std::conditional_t<U::is_leaf, const U, const U&> u;
			typename std::conditional_t<V::is_leaf, const V, const V&> v;
			__ndArray_shape shp;
			size_t row;
			size_t col;
			size_t share;
			size_t* temp;
		private:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>(__ndArray_inner_base::* op_br)(size_t i) const;
			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>(__ndArray_inner_base::* func_at)(size_t i) const;

		public:
			static constexpr bool is_leaf = false;
			__ndArray_inner_base(const U& arr1, const V& arr2) : u(arr1), v(arr2), row(0), col(0), share(0), op_br(nullptr), func_at(nullptr), temp(nullptr)
			{
				LINALGTYPE arr1_type;
				LINALGTYPE arr2_type;

				switch ((arr1_type = type(arr1))) {
				case 4:
					share = arr1.raw_shape()[2] / arr1.raw_shape()[1];
					row = arr1.raw_shape()[1];
					break;
				case 5:
					throw bad_inner_product("left array is tensor");
				case 6:
					throw bad_inner_product("left array is scalar");
				case 1:
					share = arr1.raw_shape().back();
					row = 1;
					break;
				case 2:
					share = 1;
					row = arr1.raw_shape().back();
					break;
				case 3:
					share = arr1.raw_shape().back();
					row = 1;
					break;
				}

				switch ((arr2_type = linalg::type(arr2))) {
				case 4:
					assert(arr2.raw_shape()[2] / arr2.raw_shape()[1] == share);
					col = arr2.raw_shape()[1];
					break;
				case 5:
					throw bad_inner_product("right array is tensor");
				case 6:
					throw bad_inner_product("right array is scalar");
				case 1:
					assert(share == arr2.raw_shape().back());
					col = 1;
					break;
				case 2:
					assert(1 == share);
					col = arr2.raw_shape().back();
					break;
				case 3:
					assert(share == arr2.raw_shape().back());
					col = 1;
					break;
				}

				shp.init({ row, col }).shrink_dim_to_fit();

				if (arr1_type < linalg::LINARGTYPE_MATRIX && arr2_type < linalg::LINARGTYPE_MATRIX) {
					if (col == 1) {
						op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_inner_base::*)(size_t i) const>(
							&__ndArray_inner_lv1_s<U, V>::operator[]);
						func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_inner_base::*)(size_t i) const>(
							&__ndArray_inner_lv1_s<U, V>::at);
					}
					else {
						op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_inner_base::*)(size_t i) const>(
							&__ndArray_inner_lv1_m<U, V>::operator[]);
						func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_inner_base::*)(size_t i) const>(
							&__ndArray_inner_lv1_m<U, V>::at);
						temp = (size_t*)__ndArray_allocator_instance.allocate(row * sizeof(size_t));
						temp[0] = 0;

						for (size_t i = 1; i < row; ++i) {
							temp[i] = temp[i - 1] + col;
						}
					}
				}
				else if (arr1_type < linalg::LINARGTYPE_MATRIX && arr2_type == linalg::LINARGTYPE_MATRIX) {
					op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_inner_base::*)(size_t i) const>(
						&__ndArray_inner_lv2_l<U, V>::operator[]);
					func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_inner_base::*)(size_t i) const>(
						&__ndArray_inner_lv2_l<U, V>::at);
				}
				else if (arr1_type == linalg::LINARGTYPE_MATRIX && arr2_type < linalg::LINARGTYPE_MATRIX) {
					op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_inner_base::*)(size_t i) const>(
						&__ndArray_inner_lv2_r<U, V>::operator[]);
					func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_inner_base::*)(size_t i) const>(
						&__ndArray_inner_lv2_r<U, V>::at);
				}
				else if (arr1_type == linalg::LINARGTYPE_MATRIX && arr2_type == linalg::LINARGTYPE_MATRIX) {
					op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_inner_base::*)(size_t i) const>(
						&__ndArray_inner_lv3<U, V>::operator[]);
					func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_inner_base::*)(size_t i) const>(
						&__ndArray_inner_lv3<U, V>::at);
					temp = (size_t*)__ndArray_allocator_instance.allocate(row * sizeof(size_t));
					temp[0] = 0;

					for (size_t i = 1; i < row; ++i) {
						temp[i] = temp[i - 1] + share;
					}
				}
			}

			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
				operator[](size_t i) const { return (this->*op_br)(i); }

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
				at(size_t i) const { return (this->*func_at)(i); }

			const auto& raw_shape() const { return shp; }

			~__ndArray_inner_base() noexcept { if (temp) __ndArray_allocator_instance.deallocate(temp, row * sizeof(size_t)); }
		};
		#pragma endregion

		template <typename U, typename V>
		__ndArray_inner_base<U, V>
			dot(const ndArrayExpression<U>& arr1, const ndArrayExpression<V>& arr2) {
			return __ndArray_inner_base<U, V>(*static_cast<const U*>(&arr1), *static_cast<const V*>(&arr2));
		}

		
		#pragma region support_outer_product
		struct bad_outer_product {
			std::string err_msg;
			bad_outer_product(const char* msg) : err_msg(msg) {}
		};

		template <typename U, typename V>
		class __ndArray_outer_base;

		template <typename U, typename V>
		class __ndArray_outer_lv1_m : public __ndArray_outer_base<U, V> {
		public:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
				operator[](size_t i) const
			{
				ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
				res.alloc({ this->col });
				res.at(0) = this->u.at(i) * this->v.at(0);
				for (size_t j = this->col - 1; j != 0; --j) {
					res.at(j) = this->u.at(0) * this->v.at(j);
				}

				return res;
			}

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
				at(size_t i) const
			{
				size_t r = i / this->col;
				size_t c = i - this->temp[r];
				return this->u.at(r) * this->v.at(c);
			}
		};

		template <typename U, typename V>
		class __ndArray_outer_lv1_s : public __ndArray_outer_base<U, V> {
		public:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
				operator[](size_t i) const
			{
				ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
				res.alloc({ this->col });
				assert(i == 0);

				res.at(0) = this->u.at(0) * this->v.at(0);
				for (size_t j = this->share - 1; j != 0; --j) {
					res.at(j) += this->u.at(j) * this->v.at(j);
				}

				return res;
			}

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
				at(size_t i) const
			{
				assert(i == 0);

				std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))> sum = this->u.at(0) * this->v.at(0);

				for (size_t j = this->share - 1; j != 0; --j) {
					sum += this->u.at(j) * this->v.at(j);
				}

				return sum;
			}
		};

		template <typename U, typename V>
		class __ndArray_outer_lv2_r : public __ndArray_outer_base<U, V> {
		public:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
				operator[](size_t i) const
			{
				ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
				res.alloc({});

				size_t offset = this->temp[i];
				res.at(0) = this->u.at(offset) * this->v.at(0);
				for (size_t j = this->share - 1; j != 0; --j) {
					res.at(0) += this->u.at(offset + j) * this->v.at(j);
				}

				return res;
			}

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
				at(size_t i) const
			{
				size_t offset = this->temp[i];
				std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))> sum = this->u.at(offset) * this->v.at(0);
				for (size_t j = this->share - 1; j != 0; --j) {
					sum += this->u.at(offset + j) * this->v.at(j);
				}

				return sum;
			}
		};

		template <typename U, typename V>
		class __ndArray_outer_lv2_l : public __ndArray_outer_base<U, V> {
		public:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
				operator[](size_t i) const
			{
				assert(i == 0);
				ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
				res.alloc({ this->col });
				for (size_t j = this->col - 1, v_col = j * this->share;; --j, v_col -= this->share) {
					res.at(j) = this->u.at(0) * this->v.at(v_col);
					for (size_t k = this->share - 1, v_k = v_col + this->share - 1; k != 0; --k, --v_k) {
						res.at(j) += this->u.at(k) * this->v.at(v_k);
					}
					if (j == 0) break;
				}

				return res;
			}

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
				at(size_t i) const
			{
				size_t offset = this->temp[i];
				std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))> sum = this->u.at(0) * this->v.at(offset);
				for (size_t u_col = this->share - 1, v_col = offset + this->share - 1; u_col != 0; --u_col, --v_col) {
					sum += this->u.at(u_col) * this->v.at(v_col);
				}

				return sum;
			}
		};

		template <typename U, typename V>
		class __ndArray_outer_lv3 : public __ndArray_outer_base<U, V> {
		public:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
				operator[](size_t i) const
			{
				ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
				res.alloc({ this->col });
				size_t offset_u = i * this->share;
				for (size_t j = this->col; j != 0; --j) {
					size_t offset_v = j * this->share;
					res.at(j) = this->u.at(offset_u) * this->v.at(offset_v);
					for (size_t v_col = offset_v + this->share - 1, u_col = offset_u + this->share - 1; v_col != offset_v; --v_col, --u_col) {
						res.at(j) += this->u.at(u_col) * this->v.at(v_col);
					}
					if (j == 0) break;
				}

				return res;
			}

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
				at(size_t i) const
			{
				size_t r = i / this->col;
				size_t c = i - this->temp[r];
				size_t offset_u = r * this->share;
				size_t offset_v = c * this->share;
				std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))> sum = this->u.at(offset_u) * this->v.at(offset_v);
				for (size_t v_col = offset_v + this->share - 1, u_col = offset_u + this->share - 1; v_col != offset_v; --v_col, --u_col) {
					sum += this->u.at(u_col) * this->v.at(v_col);
				}

				return sum;
			}
		};

		template <typename U, typename V>
		class __ndArray_outer_base : public ndArrayExpression<__ndArray_outer_base<U, V>> {
		protected:
			typename std::conditional_t<U::is_leaf, const U, const U&> u;
			typename std::conditional_t<V::is_leaf, const V, const V&> v;
			__ndArray_shape shp;
			size_t row;
			size_t col;
			size_t share;
			size_t* temp;
			size_t temp_size;
		private:
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>(__ndArray_outer_base::* op_br)(size_t i) const;
			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>(__ndArray_outer_base::* func_at)(size_t i) const;

		public:
			static constexpr bool is_leaf = false;
			__ndArray_outer_base(const U& arr1, const V& arr2) : u(arr1), v(arr2), row(0), col(0), share(0), op_br(nullptr), func_at(nullptr), temp(nullptr), temp_size(0)
			{
				LINALGTYPE arr1_type;
				LINALGTYPE arr2_type;

				switch ((arr1_type = type(arr1))) {
				case 4:
					row = arr1.raw_shape()[2] / arr1.raw_shape()[1];
					share = arr1.raw_shape()[1];
					break;
				case 5:
					throw bad_outer_product("left array is tensor");
				case 6:
					throw bad_outer_product("left array is scalar");
				case 1:
					row = arr1.raw_shape().back();
					share = 1;
					break;
				case 2:
					row = 1;
					share = arr1.raw_shape().back();
					break;
				case 3:
					row = arr1.raw_shape().back();
					share = 1;
					break;
				}

				switch ((arr2_type = linalg::type(arr2))) {
				case 4:
					assert(arr2.raw_shape()[1] == share);
					col = arr2.raw_shape()[2] / arr2.raw_shape()[1];
					break;
				case 5:
					throw bad_outer_product("right array is tensor");
				case 6:
					throw bad_outer_product("right array is scalar");
				case 1:
					assert(share == 1);
					col = arr2.raw_shape().back();
					break;
				case 2:
					assert(arr2.raw_shape().back() == share);
					col = 1;
					break;
				case 3:
					assert(share == 1);
					col = arr2.raw_shape().back();
					break;
				}

				shp.init({ row, col }).shrink_dim_to_fit();

				if (arr1_type < linalg::LINARGTYPE_MATRIX && arr2_type < linalg::LINARGTYPE_MATRIX) {
					if (col == 1) {
						op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_outer_base::*)(size_t i) const>(
							&__ndArray_outer_lv1_s<U, V>::operator[]);
						func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_outer_base::*)(size_t i) const>(
							&__ndArray_outer_lv1_s<U, V>::at);
					}
					else {
						op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_outer_base::*)(size_t i) const>(
							&__ndArray_outer_lv1_m<U, V>::operator[]);
						func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_outer_base::*)(size_t i) const>(
							&__ndArray_outer_lv1_m<U, V>::at);
						temp = (size_t*)__ndArray_allocator_instance.allocate(row * sizeof(size_t));
						temp[0] = 0;

						for (size_t i = 1; i < row; ++i) {
							temp[i] = temp[i - 1] + col;
						}
						temp_size = row;
					}
				}
				else if (arr1_type < linalg::LINARGTYPE_MATRIX && arr2_type == linalg::LINARGTYPE_MATRIX) {
					op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_outer_base::*)(size_t i) const>(
						&__ndArray_outer_lv2_l<U, V>::operator[]);
					func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_outer_base::*)(size_t i) const>(
						&__ndArray_outer_lv2_l<U, V>::at);
					temp = (size_t*)__ndArray_allocator_instance.allocate(col * sizeof(size_t));
					temp[0] = 0;

					for (size_t i = 1; i < col; ++i) {
						temp[i] = temp[i - 1] + share;
					}
					temp_size = col;
				}
				else if (arr1_type == linalg::LINARGTYPE_MATRIX && arr2_type < linalg::LINARGTYPE_MATRIX) {
					op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_outer_base::*)(size_t i) const>(
						&__ndArray_outer_lv2_r<U, V>::operator[]);
					func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_outer_base::*)(size_t i) const>(
						&__ndArray_outer_lv2_r<U, V>::at);
					temp = (size_t*)__ndArray_allocator_instance.allocate(row * sizeof(size_t));
					temp[0] = 0;

					for (size_t i = 1; i < row; ++i) {
						temp[i] = temp[i - 1] + share;
					}
					temp_size = row;
				}
				else if (arr1_type == linalg::LINARGTYPE_MATRIX && arr2_type == linalg::LINARGTYPE_MATRIX) {
					op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_outer_base::*)(size_t i) const>(
						&__ndArray_outer_lv3<U, V>::operator[]);
					func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_outer_base::*)(size_t i) const>(
						&__ndArray_outer_lv3<U, V>::at);
					temp = (size_t*)__ndArray_allocator_instance.allocate(row * sizeof(size_t));
					temp[0] = 0;

					for (size_t i = 1; i < row; ++i) {
						temp[i] = temp[i - 1] + col;
					}
					temp_size = row;
				}
			}

			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
				operator[](size_t i) const { return (this->*op_br)(i); }

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
				at(size_t i) const { return (this->*func_at)(i); }

			const auto& raw_shape() const { return shp; }

			~__ndArray_outer_base() noexcept { if (temp) __ndArray_allocator_instance.deallocate(temp, temp_size * sizeof(size_t)); }
		};
		#pragma endregion

		template <typename U, typename V>
		__ndArray_outer_base<U, V>
			outer(const ndArrayExpression<U>& arr1, const ndArrayExpression<V>& arr2) {
			return __ndArray_outer_base<U, V>(*static_cast<const U*>(&arr1), *static_cast<const V*>(&arr2));
		}
		
	}

	/////////////////////////////////////////////////////////////////////		operator overloading		///////////////////////////////////////////////////////////////////
	
	#pragma region support_multiplication
	struct bad_matrix_multiplication{
		std::string err_msg;
		bad_matrix_multiplication(const char* msg) : err_msg(msg) {}
	};

	template <typename U, typename V>
	class __ndArray_matmul_base;
	
	//result => (n,m)
	template <typename U, typename V>
	class __ndArray_matmul_lv1_m : public __ndArray_matmul_base<U, V> {
	public:
		ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
			operator[](size_t i) const
		{
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
			res.alloc({ this->col });
			res.at(0) = this->u.at(i) * this->v.at(0);
			for (size_t j = this->col - 1; j != 0; --j) {
				res.at(j) = this->u.at(0) * this->v.at(j);
			}

			return res;
		}

		std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
			at(size_t i) const
		{
			size_t r = i / this->col;
			size_t c = i - this->temp[r];
			return this->u.at(r) * this->v.at(c);
		}
	};

	//result => (1,1)
	template <typename U, typename V>
	class __ndArray_matmul_lv1_s : public __ndArray_matmul_base<U, V> {
	public:
		ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
			operator[](size_t i) const
		{
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
			res.alloc({ this->col });
			assert(i == 0);

			res.at(0) = this->u.at(0) * this->v.at(0);
			for (size_t j = this->share - 1; j != 0; --j) {
				res.at(j) += this->u.at(j) * this->v.at(j);
			}

			return res;
		}

		std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
			at(size_t i) const
		{
			assert(i == 0);

			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))> sum = this->u.at(0) * this->v.at(0);

			for (size_t j = this->share - 1; j != 0; --j) {
				sum += this->u.at(j) * this->v.at(j);
			}

			return sum;
		}
	};

	//matrix * vector 에서 vector 가 right side 에 있으려면 반드시 column vector
	//result => (n, 1)
	template <typename U, typename V>
	class __ndArray_matmul_lv2_r : public __ndArray_matmul_base<U, V> {
	public:
		ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
			operator[](size_t i) const
		{
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
			res.alloc({});

			size_t offset = this->temp[i];
			res.at(0) = this->u.at(offset) * this->v.at(0);
			for (size_t j = this->share - 1; j != 0; --j) {
				res.at(0) += this->u.at(offset + j) * this->v.at(j);
			}

			return res;
		}

		std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
			at(size_t i) const
		{
			size_t offset = this->temp[i];
			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))> sum = this->u.at(offset) * this->v.at(0);
			for (size_t j = this->share - 1; j != 0; --j) {
				sum += this->u.at(offset + j) * this->v.at(j);
			}

			return sum;
		}
	};

	//vector * matrix 에서 vector 가 left side 에 있으려면 반드시 row vector
	//result => (1, n)
	template <typename U, typename V>
	class __ndArray_matmul_lv2_l : public __ndArray_matmul_base<U, V> {
	public:
		ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
			operator[](size_t i) const
		{
			assert(i == 0);
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
			res.alloc({ this->col });
			for (size_t j = this->col - 1;; --j) {
				res.at(j) = this->u.at(0) * this->v.at(j);
				for (size_t k = this->share - 1, v_k = j + k * this->col; k != 0; --k, v_k -= this->col) {
					res.at(j) += this->u.at(k) * this->v.at(v_k);
				}
				if (j == 0) break;
			}

			return res;
		}

		std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
			at(size_t i) const
		{
			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))> sum = this->u.at(0) * this->v.at(i);
			for (size_t j = this->share - 1, v_j = i + j * this->col; j != 0; --j, v_j -= this->col) {
				sum += this->u.at(j) * this->v.at(v_j);
			}

			return sum;
		}
	};

	template <typename U, typename V>
	class __ndArray_matmul_lv3 : public __ndArray_matmul_base<U, V> {
	public:
		ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
			operator[](size_t i) const
		{
			ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>> res;
			res.alloc({ this->col });
			for (size_t j = 0; j < this->col; ++j) {
				res.at(j) = 0;
				for (size_t k = this->temp[i], k_col = j; k_col < this->v.raw_shape().back(); ++k, k_col += this->col) {
					res.at(j) += this->u.at(k) * this->v.at(k_col);
				}
			}

			return res;
		}

		std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
			at(size_t i) const
		{
			size_t r = i / this->col;
			size_t c = i - r * this->col;
			std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))> sum = 0;
			for (size_t j = this->temp[r], j_col = c; j_col < this->v.raw_shape().back(); ++j, j_col += this->col) {
				sum += this->u.at(j) * this->v.at(j_col);
			}

			return sum;
		}
	};

	template <typename U, typename V>
	class __ndArray_matmul_base : public ndArrayExpression<__ndArray_matmul_base<U, V>> {
	protected:
		typename std::conditional_t<U::is_leaf, const U, const U&> u;
		typename std::conditional_t<V::is_leaf, const V, const V&> v;
		__ndArray_shape shp;
		size_t row;
		size_t col;
		size_t share;
		size_t* temp;
	private:
		ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>(__ndArray_matmul_base::* op_br)(size_t i) const;
		std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>(__ndArray_matmul_base::* func_at)(size_t i) const;

	public:
		static constexpr bool is_leaf = false;
		__ndArray_matmul_base(const U& arr1, const V& arr2) : u(arr1), v(arr2), row(0), col(0), share(0), op_br(nullptr), func_at(nullptr), temp(nullptr)
		{
			linalg::LINALGTYPE arr1_type;
			linalg::LINALGTYPE arr2_type;

			switch ((arr1_type = linalg::type(arr1))) {
			case 4:
				row = arr1.raw_shape()[2] / arr1.raw_shape()[1];
				share = arr1.raw_shape()[1];
				break;
			case 5:
				throw bad_matrix_multiplication("left array is tensor");
			case 6:
				throw bad_matrix_multiplication("left array is scalar");
			case 1:
				row = arr1.raw_shape().back();
				share = 1;
				break;
			case 2:
				row = 1;
				share = arr1.raw_shape().back();
				break;
			case 3:
				row = arr1.raw_shape().back();
				share = 1;
				break;
			}

			switch ((arr2_type = linalg::type(arr2))) {
			case 4:
				assert(arr2.raw_shape()[2] / arr2.raw_shape()[1] == share);
				col = arr2.raw_shape()[1];
				break;
			case 5:
				throw bad_matrix_multiplication("right array is tensor");
			case 6:
				throw bad_matrix_multiplication("right array is scalar");
			case 1:
				assert(share == arr2.raw_shape().back());
				col = 1;
				break;
			case 2:
				assert(1 == share);
				col = arr2.raw_shape().back();
				break;
			case 3:
				assert(share == arr2.raw_shape().back());
				col = 1;
				break;
			}

			shp.init({ row, col }).shrink_dim_to_fit();

			if (arr1_type < linalg::LINARGTYPE_MATRIX && arr2_type < linalg::LINARGTYPE_MATRIX) {
				if (col == 1) {
					op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_matmul_base::*)(size_t i) const>(
						&__ndArray_matmul_lv1_s<U, V>::operator[]);
					func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_matmul_base::*)(size_t i) const>(
						&__ndArray_matmul_lv1_s<U, V>::at);
				}
				else {
					op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_matmul_base::*)(size_t i) const>(
						&__ndArray_matmul_lv1_m<U, V>::operator[]);
					func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_matmul_base::*)(size_t i) const>(
						&__ndArray_matmul_lv1_m<U, V>::at);
					temp = (size_t*)__ndArray_allocator_instance.allocate(row * sizeof(size_t));
					temp[0] = 0;

					for (size_t i = 1; i < row; ++i) {
						temp[i] = temp[i - 1] + col;
					}
				}
			}
			else if (arr1_type < linalg::LINARGTYPE_MATRIX && arr2_type == linalg::LINARGTYPE_MATRIX) {
				op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_matmul_base::*)(size_t i) const>(
					&__ndArray_matmul_lv2_l<U, V>::operator[]);
				func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_matmul_base::*)(size_t i) const>(
					&__ndArray_matmul_lv2_l<U, V>::at);
			}
			else if (arr1_type == linalg::LINARGTYPE_MATRIX && arr2_type < linalg::LINARGTYPE_MATRIX) {
				op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_matmul_base::*)(size_t i) const>(
					&__ndArray_matmul_lv2_r<U, V>::operator[]);
				func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_matmul_base::*)(size_t i) const>(
					&__ndArray_matmul_lv2_r<U, V>::at);
				temp = (size_t*)__ndArray_allocator_instance.allocate(row * sizeof(size_t));
				temp[0] = 0;

				for (size_t i = 1; i < row; ++i) {
					temp[i] = temp[i - 1] + share;
				}
			}
			else if (arr1_type == linalg::LINARGTYPE_MATRIX && arr2_type == linalg::LINARGTYPE_MATRIX) {
				op_br = static_cast<ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>>(__ndArray_matmul_base::*)(size_t i) const>(
					&__ndArray_matmul_lv3<U, V>::operator[]);
				func_at = static_cast<std::_Remove_cvref_t<decltype(std::declval<U>().at(0) * std::declval<V>().at(0))>(__ndArray_matmul_base::*)(size_t i) const>(
					&__ndArray_matmul_lv3<U, V>::at);
				temp = (size_t*)__ndArray_allocator_instance.allocate(row * sizeof(size_t));
				temp[0] = 0;

				for (size_t i = 1; i < row; ++i) {
					temp[i] = temp[i - 1] + share;
				}
			}
		}

		ndArray<std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>>
			operator[](size_t i) const { return (this->*op_br)(i); }

		std::_Remove_cvref_t<decltype(std::declval<U>().at(0)* std::declval<V>().at(0))>
			at(size_t i) const { return (this->*func_at)(i); }

		const auto& raw_shape() const { return shp; }

		~__ndArray_matmul_base() noexcept { if (temp) __ndArray_allocator_instance.deallocate(temp, row * sizeof(size_t)); }
	};
	#pragma endregion
	
	//matrix multiplication
	//shrink dim to fit 적용
	//example: (1,n) * (n,1) => scalar, not 2d array with (1,1)
	template <typename U, typename V>
	__ndArray_matmul_base<U, V>
		operator^(ndArrayExpression<U> const& arr1, ndArrayExpression<V> const& arr2)
	{
		return __ndArray_matmul_base<U, V>(*static_cast<const U*>(&arr1), *static_cast<const V*>(&arr2));
	}


	/////////////////////////////////////////////////////////////////////		ndArray member function		///////////////////////////////////////////////////////////////////

	template <typename T>
	ndArray<T>& ndArray<T>::transpose()
	{
		constexpr size_t blocksize = 4;
		
		switch (linalg::type(*this))
		{
		case 1:
			_shape.extend_1d();
			break;
		case 2:
			_shape.shrink_dim_to_fit();
			break;
		case 3:
			_shape[1] = _shape[2];
			_shape.decrease_dim(2);
			break;
		case 4:
			if (_shape[2] == _shape[1] * _shape[1]) {
				T temp;
				size_t offset = 1;
				size_t origin_iter = 0;
				size_t res_iter = 0;
				for (; origin_iter < _shape[2]; ++offset, ++res_iter) {
					origin_iter += offset;

					for (size_t j = res_iter + offset * _shape[1]; j < _shape[2]; j += _shape[1], ++origin_iter) {
						temp = item[origin_iter];
						item[origin_iter] = item[j];
						item[j] = temp;
					}
				}
			}
			else {
				size_t row = _shape[2] / _shape[1];
				size_t col = _shape[1];
				size_t bs_col = blocksize * col;
				size_t bs_row = blocksize * row;
				T* temp = (T*)linalg::__ndArray_allocator_linalg.allocate(_shape.back() * sizeof(T));
				for (size_t i = 0, i_col = 0; i < row; i += blocksize, i_col += bs_col) {
					for (size_t j = 0, j_row_i = i; j < col; j += blocksize, j_row_i += bs_row) {
						//size_t max_i2 = i + blocksize < row ? i + blocksize : row;
						//size_t max_j2 = j + blocksize < col ? j + blocksize : col;
						size_t max_i2 = i + blocksize < row ? blocksize : row - i;
						size_t max_j2 = j + blocksize < col ? blocksize : col - j;
						linalg::transpose_matrix_block(temp + j_row_i, item + i_col + j, row, col, max_i2, max_j2);
					}
				}
				this->copy(temp);
				linalg::__ndArray_allocator_linalg.deallocate(temp, _shape.back() * sizeof(T));
				_shape[1] = row;
				_shape[2] = col * row;
			}
			break;
		case 5:
			throw linalg::bad_matrix_transpose("array is tensor");
			break;
		case 6:
			throw linalg::bad_matrix_transpose("array is scalar");
		}

		return *this;
	}

	template <typename T>
	template <typename E>
	linalg::__ndArray_inner_base<ndArray<T>, E> ndArray<T>::dot(const ndArrayExpression<E>& other)
	{
		return linalg::__ndArray_inner_base < ndArray<T>, E>(*this, *static_cast<const E*>(&other));
	}


	template <typename T>
	template<typename E>
	linalg::__ndArray_outer_base<ndArray<T>, E> ndArray<T>::outer(const ndArrayExpression<E>& other)
	{
		return linalg::__ndArray_outer_base < ndArray<T>, E>(*this, *static_cast<const E*>(&other));
	}
}

#endif