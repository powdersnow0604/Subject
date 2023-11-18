#ifndef __MPS_VECTOR_H__
#define __MPS_VECTOR_H__


#ifdef VECTORTYPE 
typedef unsigned long long size_t;


#define VECTOR_INIT vector_init_ ## voidp
#define VECTOR_PUSH_BACK vector_push_back_ ## voidp
#define VECTOR_RESIZE vector_resize_ ## voidp

typedef struct vector_ {
	size_t num;
	size_t capacity;
}vec_manager;

VECTORTYPE* VECTOR_INIT(vec_manager* q, size_t size);

void VECTOR_PUSH_BACK(VECTORTYPE* vec, vec_manager* q, VECTORTYPE item);

void VECTOR_RESIZE(VECTORTYPE* vec, vec_manager* q, size_t nsize);
#endif

#endif
