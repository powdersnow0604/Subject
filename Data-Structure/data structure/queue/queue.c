#define QUEUETYPE int
#include "queue.h"
#include <stdlib.h>


void Q_INIT(Q_NAME* q, size_t size)
{
	q->num = 0;
	LIST_INIT(&q->elements, size);
}

void Q_PUSH_BACK(Q_NAME* q, QUEUETYPE item)
{
	LIST_PUSH_BACK(&q->elements, item);
	++(q->num);
}

QUEUETYPE Q_POP_FRONT(Q_NAME* q)
{
	if (q->num == 0) return (QUEUETYPE)NULL;
	--(q->num);
	return LIST_POP_FRONT(&q->elements);
}

void Q_FREE(Q_NAME* q)
{
	q->num = 0;
	LIST_FREE(&q->elements);
}


int Q_IS_EMPTY(Q_NAME* q)
{
	return q->num == 0;
}


void Q_CLEAR(Q_NAME* q)
{
	q->num = 0;
	LIST_CLEAR(&q->elements);
}