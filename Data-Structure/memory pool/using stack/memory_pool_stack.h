#ifndef __MEOMRY_POOL_STACK_H__
#define __MEOMRY_POOL_STACK_H__
#define STACKTYPE void*
#include "mps_stack.h"
#define VECTORTYPE void*
#include "mps_vector.h"


typedef unsigned long long size_t;

typedef struct mps_struct_ {
	size_t num; //in use
	size_t capacity;
	size_t item_size;
	vec_manager vec_m;
	void** first_addr;
	stack_voidp stack;
}mps_struct;

void mps_init(mps_struct* mm, size_t size, size_t count);
void* mps_alloc_one(mps_struct* mm);
void mps_free_one(mps_struct* mm, void* ptr);
void mps_destroy(mps_struct* mm);

#endif
