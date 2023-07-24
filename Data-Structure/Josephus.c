#include <stdio.h>
#include <stdlib.h>

typedef struct {
	int size;
	int count;
	int* item;
	int rear; 
	int front;
}Cqueue;

void createQueue(Cqueue* q, size_t _size);
int isFull(Cqueue* q) { return q->count == q->size; }
int isEmpty(Cqueue* q) { return q->count == 0; }
void Enqueue(Cqueue* q, int x);
int Dequeue(Cqueue* q);

int main_Josephus(void)
{
	int n;
	int k;
	printf("How many people are there?: ");
	scanf_s("%d", &n);
	printf("How much period does it take to kill one person?: ");
	scanf_s("%d", &k);

	Cqueue tq;
	createQueue(&tq, n);

	//1. Add n people in the queue
	for (int i = 1; i <= n; i++) Enqueue(&tq, i);

	//2. Execute <kill every k-th perdon> algorithm until (k-1) people remain in the queue
	int temp;
	int turn = 0;
	while (tq.count >= k)
	{
		turn++;
		temp = Dequeue(&tq);
		if (turn % k != 0) Enqueue(&tq, temp);
	}

	//3.report the (k-1) people
	printf("\nsurvived people are: ");
	for (int i = 0; i < k-1; i++)
	{
		printf("%d ", tq.item[++tq.front]);
	}
	puts("");

	return EXIT_SUCCESS;
}


/////////////////////////////////////////////////////////////////
void createQueue(Cqueue* q, size_t _size)
{
	q->size = _size;
	q->item = (int*)malloc(sizeof(int) * q->size);
	q->count = 0;
	q->rear = q->front = -1;
}

void Enqueue(Cqueue* q, int x)
{
	if (isFull(q))
	{
		q->size *= 2;
		q->size = (int*)realloc(q->size, sizeof(int) * q->size);
	}
	q->rear = (q->rear + 1) % q->size;
	q->item[q->rear] = x;
	q->count++;
}

int Dequeue(Cqueue* q)
{
	if (isEmpty(q)) return 0;
	q->front = (q->front + 1) % q->size;
	q->count--;
	return q->item[q->front];
}
