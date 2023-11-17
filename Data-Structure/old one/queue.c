#include "queue.h"
#include <stdlib.h>

void q_init(queue* q, int size)
{
	q->num = 0;
	q->capacity = size;
	q->elements = (QUEUETYPE*)malloc(sizeof(QUEUETYPE) * q->capacity);
	q->front = -1;
	q->back = -1;
}

void q_push_back(queue* q, QUEUETYPE item)
{
	if (q->num == q->capacity) return;
	q->elements[++(q->back)] = item;
	++(q->num);
}

QUEUETYPE q_pop_front(queue* q)
{
	if (q->num == 0) return;
	--(q->num);
	return q->elements[++(q->front)];
}

void q_free(queue* q)
{
	free(q->elements);
}


int q_is_empty(queue* q)
{
	return q->num == 0;
}
