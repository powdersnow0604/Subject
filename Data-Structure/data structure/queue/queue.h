#ifndef __QUEUE_H__
#define __QUEUE_H__


#ifdef QUEUETYPE 

#define LISTTYPE QUEUETYPE
#include "list.h"

#define Q_INIT q_init_ ## int
#define Q_PUSH_BACK q_push_back_ ## int
#define Q_POP_FRONT q_pop_front_ ## int
#define Q_FREE q_free_ ## int
#define Q_IS_EMPTY q_is_empty_ ## int
#define Q_CLEAR q_clear_ ## int
#define Q_NAME queue_ ## int


typedef struct queue_ {
	LIST_NAME elements;
	size_t num;
}Q_NAME;

void Q_INIT(Q_NAME* q, size_t size);

void Q_PUSH_BACK(Q_NAME* q, QUEUETYPE item);

QUEUETYPE Q_POP_FRONT(Q_NAME* q);

void Q_FREE(Q_NAME* q);

int Q_IS_EMPTY(Q_NAME* q);

void Q_CLEAR(Q_NAME* q);
#endif

#endif