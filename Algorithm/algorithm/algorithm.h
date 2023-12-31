#ifndef __ALGORITHM__
#define __ALGORITHM__

typedef unsigned long long size_t;
typedef long long lsize_t;


#pragma region sorting
//compare -> '>' is ascending order
void Qsort(void* arr, lsize_t s, lsize_t e, size_t element_size, int(*compare)(const void*, const void*));

void mergeSort(void* arr, size_t s, size_t e, size_t element_size, int(*compare)(const void*, const void*));

//use without equal
int isSorted(void* arr, size_t s, size_t end, size_t element_size, int(*compare)(const void*, const void*));
#pragma endregion

#pragma region search
lsize_t binarySearch(void* arr, void* element, size_t s, size_t e, size_t size, int(*compare)(const void*, const void*));
#pragma endregion


#pragma region compare function
int greater_i(const void* a, const void* b);
int greater_d(const void* a, const void* b);
int less_i(const void* a, const void* b);
int less_d(const void* a, const void* b);

int greaterEq_i(const void* a, const void* b);
int greaterEq_d(const void* a, const void* b);
int lessEq_i(const void* a, const void* b);
int lessEq_d(const void* a, const void* b);

int isEqual(void* opr1, void* opr2, size_t size);
#pragma endregion


#pragma region etc
void copy(void* dst, void* src, size_t size);
void median(void* arr, lsize_t k, lsize_t s, lsize_t e, size_t size, int(*compare)(const void*, const void*));
#pragma endregion

#endif