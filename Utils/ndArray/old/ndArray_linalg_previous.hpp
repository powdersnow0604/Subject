/////////////////////////////////////////////////////////////////////		member function example code		///////////////////////////////////////////////////////////////////

#include <iostream>
#include <typeinfo>

struct base {
	int i;
	void (base::* bf)(int) const;
	base(void (base::* _bf)(int) const) : bf(_bf), i(0) {}
	void func(int n) {
		std::cout << "this is base = " << 0 << '\n';
		(this->*bf)(n);
	}
};

struct derived : public base {
	derived() : base(static_cast<void (base::*)(int) const>(&derived::func)) {}
	void func(int n) const {
		std::cout << "this is derived = " << this->i << '\n';
	}
};

base ret_derived() {
	return derived();
}


int main() {

	auto var = ret_derived();
	var.func(2);
	std::cout << typeid(var).name() << '\n';
	return 0;
}

/////////////////////////////////////////////////////////////////////		transpose		///////////////////////////////////////////////////////////////////


//expression template first version
/*
		template <typename T>
		class __ndArray_linarg_transpose_matrix : public ndArrayExpression<__ndArray_linarg_transpose_matrix<T>> {
			struct bad_matrix_transpose {
				std::string err_message;
				bad_matrix_transpose(const char* str) : err_message(str) {}
			};

			typename std::conditional_t<T::is_leaf, const T, const T&> _u;
			__ndArray_shape shp;
			size_t col;
			size_t row;
			size_t* col_arr;
			size_t* row_arr;
			unsigned char type;
		public:
			static constexpr bool is_leaf = false;

			__ndArray_linarg_transpose_matrix(T const& u) : _u(u)
			{
				if (u.raw_shape().size() == 2) {
					shp.init({ 1, u.raw_shape().back() });
					col_arr = row_arr = 0;
					col = row = 0;
					type = 0;
				}
				else if (u.raw_shape().size() == 3) {
					if (u.raw_shape()[1] == u.raw_shape()[2]) {
						shp.init({ u.raw_shape().back() });
						col_arr = row_arr = 0;
						col = row = 0;
						type = 1;
					}
					else if (u.raw_shape()[1] == 1) {
						shp.init({ 1, u.raw_shape().back() });
						col_arr = row_arr = 0;
						col = row = 0;
						type = 2;
					}
					else {
						row = u.raw_shape()[2] / u.raw_shape()[1];
						col = u.raw_shape()[1];
						shp.init({ col, row });

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
						type = 3;
					}
				}
				else {
					throw bad_matrix_transpose("its not vector");
				}
			}
			decltype(auto) operator[](size_t i) const
			{
				if (type == 0 || type == 2) {
					assert(i == 0);
					ndArray<decltype(_u.at(0))> res = _u;
					res.reshape({ shp.back() });
					return res;
				}
				else if (type == 1) {
					return _u[0][i];
				}
				else if (type == 3) {
					ndArray<decltype(_u.at(0))> res;
					res.alloc({ row });
					for (size_t j = row - 1; j != 0; --j) {
						res.at(j) = _u.at(row_arr[j] + i);
					}
					res.at(0) = _u.at(row_arr[0] + i);

					return res;
				}
			}
			decltype(auto) at(size_t i) const
			{
				if (type == 3) {
					size_t r = i / row;
					size_t c = i - col_arr[r];
					return _u.at(row_arr[c] + r);
				}
				else {
					return _u.at(i);
				}
			}
			const auto& raw_shape() const { return shp; }
			~__ndArray_linarg_transpose_matrix() noexcept
			{
				if (type == 3) {
					__ndArray_allocator_instance.deallocate(col_arr, col);
					__ndArray_allocator_instance.deallocate(row_arr, row);
				}
			}
		};
		*/

template <typename T>
__ndArray_linarg_transpose_base<T> transpose(const ndArrayExpression<T>& arr) {
	const auto& shp = arr.raw_shape();
	assert(shp.size() <= 3 && shp.size() > 1);

	//non expression template version
	if (arr.dim() == 1) {
		ndArray<T> res = arr.copy();
		res.reshape({ 1, arr.total_size() });
		return res;
	}
	else {
		const __ndArray_shape& shp = arr.raw_shape();
		if (shp[1] * shp[1] == shp[2]) {
			ndArray<T> res;
			res.alloc(shp);

			for (size_t i = 0, row_i = 0; i < shp[1]; ++i, row_i += shp[1]) {
				for (size_t j = 0, col_j = 0; j < shp[1]; ++j, col_j += shp[1]) {
					res.at(i + col_j) = arr.at(row_i + j);
				}
			}

			return res;
		}
		else {
			ndArray<T> res;
			size_t row = shp[2] / shp[1];
			size_t col = shp[1];
			res.alloc({ col, row });

			for (size_t i = 0, row_i = 0; i < row; ++i, row_i += col) {
				for (size_t j = 0, col_j = 0; j < col; ++j, col_j += row) {
					res.at(i + col_j) = arr.at(row_i + j);
				}
			}

			return res;
		}
	}

	//expression template first version
	//return __ndArray_linarg_transpose_matrix<ndArrayExpression<T>>(arr);

}