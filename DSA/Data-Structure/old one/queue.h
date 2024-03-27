#ifndef __QUEUE_H__
#define __QUEUE_H__

typedef struct pos_ {
	int row;
	int col;
}pos;

typedef pos QUEUETYPE;

typedef struct queue_ {
	QUEUETYPE* elements;
	int num;
	int capacity;
	int front;
	int back;
}queue;

void q_init(queue* q, int size);

void q_push_back(queue* q, QUEUETYPE item);

QUEUETYPE q_pop_front(queue* q);

void q_free(queue* q);

int q_is_empty(queue* q);

#endif