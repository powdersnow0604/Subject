#ifndef  __COMPLEX_NUM_H__
#define  __COMPLEX_NUM_H__

#define _USE_MATH_DEFINES
#include <math.h>

typedef struct {
	double Re;
	double Im;
}complex_num;

inline complex_num complexAdd(complex_num A, complex_num B)
{
	complex_num res = { A.Re + B.Re, A.Im + B.Im };
	return res;
}

inline complex_num complexMul(complex_num A, complex_num B)
{
	complex_num res = { A.Re * B.Re - B.Im * A.Im, A.Re * B.Im + A.Im * B.Re };
	return res;
}

inline complex_num complexSub(complex_num A, complex_num B)
{
	complex_num res = { A.Re - B.Re, A.Im - B.Im };
	return res;
}

inline complex_num complexDiv(complex_num A, complex_num B)
{
	const double denom = B.Re * B.Re + B.Im * B.Im;
	complex_num temp = { (A.Re * B.Re + A.Im + B.Im)/denom, (A.Im * B.Re - A.Re * B.Im) / denom };
	return temp;
}

inline double complexAbs(complex_num A)
{
	//return sqrt(A.Im * A.Im + A.Re * A.Re);
	return hypot(A.Re, A.Im);
}

inline complex_num conjugate(complex_num A)
{
	complex_num temp = { A.Re,A.Im * -1 };
	return temp;
}

inline double phaseAngle(complex_num A)
{
	return atan2(A.Im, A.Re);
}

inline complex_num twiddle_factor(int N, int exp)
{
	complex_num t = { cos(2 * M_PI / N * exp), -1 * sin(2 * M_PI / N * exp) };

	return t;
}

extern complex_num* data2complex(void* data, size_t size, char datatype[]);

extern complex_num* mkCdata(size_t size, double (*func)(int));

#pragma region checkType
#define checkType(x) _Generic((x),	char : "char", short : "short", \
									int : "int", long : "long", \
									float : "float", long long : "long long", \
									double : "double", default : "Nan")
#pragma endregion

extern const complex_num I;

#endif