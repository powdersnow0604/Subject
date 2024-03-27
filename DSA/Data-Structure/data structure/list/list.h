#ifndef __LIST_H__
#define __LIST_H__


#ifdef LISTTYPE 

#include "memory_pool_stack.h"

#define LIST_INIT list_init_ ## int
#define LIST_PUSH_BACK list_push_back_ ## int
#define LIST_POP_FRONT list_pop_front_ ## int
#define LIST_POP_BACK list_pop_back_ ## int
#define LIST_PUSH_FRONT list_push_front_ ## int
#define LIST_FREE list_free_ ## int
#define LIST_IS_EMPTY list_is_empty_ ## int
#define LIST_CLEAR list_clear_ ## int
#define LIST_NAME list_ ## int
#define NODE_NAME_ORIGIN node_ ## int ## _
#define NODE_NAME node_ ## int

typedef struct NODE_NAME_ORIGIN NODE_NAME;

struct NODE_NAME_ORIGIN {
	LISTTYPE element;
	NODE_NAME* next;
	NODE_NAME* prev;
};


typedef struct list_{
	NODE_NAME* front;
	NODE_NAME* back;
	int num;
	mps_struct mm;
}LIST_NAME;

void LIST_INIT(LIST_NAME* list, size_t size);

void LIST_PUSH_BACK(LIST_NAME* list, LISTTYPE item);

LISTTYPE LIST_POP_FRONT(LIST_NAME* list);

void LIST_PUSH_FRONT(LIST_NAME* list, LISTTYPE item);

LISTTYPE LIST_POP_BACK(LIST_NAME* list);

void LIST_FREE(LIST_NAME* list);

int LIST_IS_EMPTY(LIST_NAME* list);

void LIST_CLEAR(LIST_NAME* q);

#endif

#endif