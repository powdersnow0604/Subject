#include "algorithm.h"
#include <stdlib.h>
#include <stdio.h>

//macro
#define AT(arr, index, size) ((char*)arr + ((index) * (size)))


//declaration
static lsize_t partition(void* arr, lsize_t start, lsize_t end, size_t size, int(*compare)(const void*, const void*));
static void merge(void* arr, size_t ls, size_t le, size_t rs, size_t re, size_t size, int(*compare)(const void*, const void*));
static void swap(void*, void*, size_t);

//definition
#pragma region sorting
void Qsort(void* arr, lsize_t s, lsize_t e, size_t size, int(*compare)(const void*, const void*))
{
	if (s >= e) return;

	long long p = partition(arr, s, e, size, compare);

	Qsort(arr, s, p - 1, size, compare);
	Qsort(arr, p + 1, e, size, compare);
}

void mergeSort(void* arr, size_t s, size_t e, size_t size, int(*compare)(const void*, const void*))
{
	if (s == e) return;
	size_t mid = (s + e) / 2;

	mergeSort(arr, s, mid, size, compare);
	mergeSort(arr, mid+1, e, size, compare);

	merge(arr, s, mid, mid + 1, e, size, compare);
}

int isSorted(void* arr, size_t s, size_t end, size_t size, int(*compare)(const void*, const void*))
{
	size_t i = s + 1;
	for (; i <= end; ++i) {
		//if (compare((char*)arr + (i - 1) * size, (char*)arr + i * size)) return 0;
		if (compare(AT(arr, (i - 1), size), AT(arr, i, size))) return 0;
	}

	return 1;
}
#pragma endregion


#pragma region search
lsize_t binarySearch(void* arr, void* element, size_t s, size_t e, size_t size, int(*compare)(const void*, const void*))
{
	if (s == e) return isEqual(element, AT(arr, s, size),size) ? s : -1;

	lsize_t mid = (s + e) / 2;

	if (isEqual(element, AT(arr, mid, size), size)) return mid;
	else if (compare(element, AT(arr, mid, size))) return binarySearch(arr, element, mid + 1, e, size, compare);
	else return binarySearch(arr, element, s, mid - 1, size, compare);
}
#pragma endregion


#pragma region compare function
int greater_i(const void* a, const void* b)
{
	return *((int*)a) > *((int*)b);
}
int greater_d(const void* a, const void* b)
{
	return *((double*)a) > * ((double*)b);
}
int less_i(const void* a, const void* b)
{
	return *((int*)a) < * ((int*)b);
}
int less_d(const void* a, const void* b)
{
	return *((double*)a) < *((double*)b);
}

int greaterEq_i(const void* a, const void* b)
{
	return *((int*)a) >= *((int*)b);
}
int greaterEq_d(const void* a, const void* b)
{
	return *((double*)a) >= *((double*)b);
}
int lessEq_i(const void* a, const void* b)
{
	return *((int*)a) <= *((int*)b);
}
int lessEq_d(const void* a, const void* b)
{
	return *((double*)a) <= *((double*)b);
}


int isEqual(void* opr1, void* opr2, size_t size)
{
	size_t i = 0;
	for (; i < size >> 3; ++i) {
		if (((size_t*)opr1)[i] != ((size_t*)opr2)[i]) return 0;
	}

	opr1 = (size_t*)opr1 + i;
	opr2 = (size_t*)opr2 + i;


	for (i = 0; i < (size & 4) >> 2; ++i) {
		if (((unsigned int*)opr1)[i] != ((unsigned int*)opr2)[i]) return 0;
	}

	opr1 = (unsigned int*)opr1 + i;
	opr2 = (unsigned int*)opr2 + i;

	for (; i < (size & 3); ++i) {
		if (((unsigned char*)opr1)[i] != ((unsigned char*)opr2)[i]) return 0;
	}

	return 1;
}

#pragma endregion


#pragma region etc
void swap(void* opr1, void* opr2, size_t size)
{
	if (size <= 8) {
		size_t tmp;
		copy(&tmp, opr1, size);
		copy(opr1, opr2, size);
		copy(opr2, &tmp, size);
	}
	else if (size <= 64)
	{
		struct temp {
			size_t a, b, c, d, e, f, g, h;
		}tmp;
		copy(&tmp, opr1, size);
		copy(opr1, opr2, size);
		copy(opr2, &tmp, size);
	}
	else {
		void* temp = malloc(size);

		copy(temp, opr1, size);
		copy(opr1, opr2, size);
		copy(opr2, temp, size);

		free(temp);
	}
}


void copy(void* dst, void* src, size_t size)
{
	size_t i = 0; 
	for (; i < size >> 3; ++i) {
		((size_t*)dst)[i] = ((size_t*)src)[i];
	}

	dst = (size_t*)dst + i;
	src = (size_t*)src + i;


	for (i = 0; i < (size & 4) >> 2; ++i) {
		((unsigned int*)dst)[i] = ((unsigned int*)src)[i];
	}

	dst = (unsigned int*)dst + i;
	src = (unsigned int*)src + i;

	for (; i < (size & 3); ++i) {
		((unsigned char*)dst)[i] = ((unsigned char*)src)[i];
	}
}


static lsize_t partition(void* arr, lsize_t s, lsize_t e, size_t size, int(*compare)(const void*, const void*))
{
	lsize_t lptr = s + 1, rptr = e;
	while (lptr <= rptr) {
		while (compare(AT(arr, rptr, size), AT(arr, s, size)) && (lptr <= rptr)) { --rptr; }

		while (compare(AT(arr, s, size), AT(arr, lptr, size)) && (lptr <= rptr)) {
			++lptr;
		}
		
		if (lptr <= rptr) { 
			swap(AT(arr, rptr--, size), AT(arr, lptr++, size), size); 
		}
	}

	swap(AT(arr, rptr, size), AT(arr, s, size), size);
	return rptr;
}


static void merge(void* arr, size_t ls, size_t le, size_t rs, size_t re, size_t size, int(*compare)(const void*, const void*))
{
	size_t lptr = ls, rptr = rs, bptr = 0;
	size_t len = ((le - ls) + (re - rs) + 2);
	void* B = malloc(len * size);

	while (lptr <= le && rptr <= re)
	{
		if (compare(AT(arr, lptr, size), AT(arr, rptr, size)))  copy(AT(B, bptr++, size), AT(arr, rptr++, size), size);
		else copy(AT(B, bptr++, size), AT(arr, lptr++, size), size);
	}

	if (lptr > le) {
		for (; rptr <= re; ++rptr) copy(AT(B, bptr++, size), AT(arr, rptr, size), size);
	}
	if (rptr > re) {
		for (; lptr <= le; ++lptr) copy(AT(B, bptr++, size), AT(arr, lptr, size), size);
	}

	copy(AT(arr, ls, size), AT(B, 0, size), len*size);

	free(B);
}


void median(void* arr, lsize_t k, lsize_t s, lsize_t e, size_t size, int(*compare)(const void*, const void*))
{
	if (s == e) return;

	lsize_t m = partition(arr, s, e, size, compare);

	if (m == k) return;

	if (m < k) median(arr, k, m + 1, e, size, compare);
	else if (k < m) median(arr, k, s, m - 1, size, compare);
}

#pragma endregion