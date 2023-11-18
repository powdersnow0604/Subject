#ifndef __MM_STACK_H__
#define __MM_STACK_H__


#ifdef STACKTYPE 
typedef unsigned long long size_t;


#define STACK_INIT stack_init_ ## voidp
#define STACK_PUSH stack_push_ ## voidp
#define STACK_POP stack_pop_ ## voidp
#define STACK_FREE stack_free_ ## voidp
#define STACK_IS_EMPTY stack_is_empty_ ## voidp
#define STACK_CLEAR stack_clear_ ## voidp
#define STACK_NAME stack_ ## voidp


typedef struct stack_ {
	STACKTYPE* elements;
	size_t num;
	size_t capacity;
	size_t top;
}STACK_NAME;

void STACK_INIT(STACK_NAME* q, size_t size);

void STACK_PUSH(STACK_NAME* q, STACKTYPE item);

STACKTYPE STACK_POP(STACK_NAME* q);

void STACK_FREE(STACK_NAME* q);

int STACK_IS_EMPTY(STACK_NAME* q);

void STACK_CLEAR(STACK_NAME* q);
#endif

#endif