#define STACKTYPE void*
#include "mps_stack.h"
#include <stdlib.h>


void STACK_INIT(STACK_NAME* q, size_t size)
{
	q->num = 0;
	q->capacity = size;
	q->top = 0;
	q->elements = (STACKTYPE*)malloc(sizeof(STACKTYPE) * size);
}

void STACK_PUSH(STACK_NAME* q, STACKTYPE item)
{
	if (q->num == q->capacity) {
		void* tmp = q->elements;
		(q->capacity) <<= 1;
		q->elements = realloc(q->elements, q->capacity);
		if (q->elements == NULL) {
			free(q->elements);
			return;
		}
	}

	q->elements[(q->num)++] = item;
}

STACKTYPE STACK_POP(STACK_NAME* q)
{
	if (STACK_IS_EMPTY(q)) return (STACKTYPE)0;
	return 	q->elements[--(q->num)];
}

void STACK_FREE(STACK_NAME* q)
{
	free(q->elements);
}

int STACK_IS_EMPTY(STACK_NAME* q)
{
	return q->num == 0;
}

void STACK_CLEAR(STACK_NAME* q)
{
	q->num = 0;
}