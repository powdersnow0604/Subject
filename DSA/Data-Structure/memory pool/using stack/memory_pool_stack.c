#include "memory_pool_stack.h"
#include <stdlib.h>
#include <stdio.h>


void mps_init(mps_struct* mm, size_t size, size_t count)
{
	mm->num = 0;
	mm->capacity = count;
	mm->item_size = size;
	mm->first_addr = vector_init_voidp(&mm->vec_m, 5);
	
	mm->first_addr[0] = malloc(size * count);
	stack_init_voidp(&mm->stack, count);
	
	for (size_t i = 0; i < count; ++i) {
		stack_push_voidp(&mm->stack, (void*)((char*)(mm->first_addr[0]) + i * size));
	}
	
}

void* mps_alloc_one(mps_struct* mm)
{
	if (mm->num == mm->capacity) {
		vector_push_back_voidp(mm->first_addr, &mm->vec_m, malloc(mm->item_size * mm->capacity));
		size_t curr = mm->vec_m.num - 1;
		for (int i = 0; i < mm->capacity; ++i) {
			stack_push_voidp(&mm->stack, (char*)(mm->first_addr[curr]) + i * mm->item_size);
		}
		(mm->capacity) <<= 1;
	}
	++(mm->num);
	return stack_pop_voidp(&mm->stack);
}


void mps_free_one(mps_struct* mm, void* ptr)
{
	--(mm->num);
	stack_push_voidp(&mm->stack, ptr);
}


void mps_destroy(mps_struct* mm)
{
	stack_free_voidp(&mm->stack);

	for (size_t i = 0; i < mm->vec_m.num; ++i) {
		free(mm->first_addr[i]);
	}
	free(mm->first_addr);

	mm->vec_m.num = 0;
	mm->vec_m.capacity = 0;
	mm->item_size = 0;
	mm->capacity = 0;
	mm->num = 0;
}
