#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <conio.h>

typedef struct {
	size_t size;
	int count;
	char (*item)[256];
	int rear;
	int front;
}Cqueue;

typedef struct {
	size_t size;
	int count;
	double* item;
	int rear;
	int front;
}Dqueue;

typedef struct {
	double* item;
	int top;
	size_t size;
}doubleStack;

double reg[32];
double mem[32];

static void createQueue(Cqueue* q, size_t _size);
static int isFull(Cqueue* q) { return q->count == q->size; }
static int isEmpty(Cqueue* q) { return q->count == 0; }
static void Enqueue(Cqueue* q, char (*x)[256]);
static char (*Dequeue(Cqueue* q))[256];
static void deleteQueue(Cqueue* q);

static void d_createQueue(Dqueue* q, size_t _size);
static int d_isFull(Dqueue* q) { return q->count == q->size; }
static int d_isEmpty(Dqueue* q) { return q->count == 0; }
static void d_Enqueue(Dqueue* q, double x);
static double d_Dequeue(Dqueue* q);
static void d_deleteQueue(Dqueue* q);

static void double_createStack(doubleStack* s, size_t _size);
static int double_isEmpty(doubleStack* s) { return s->top == 0; }
static int double_isFull(doubleStack* s) { return s->top == s->size; }
static void double_push(doubleStack* s, double x);
static double double_pop(doubleStack* s);

static void init_mem(double* mem);
static void load_process(Cqueue* q, char* fname);
static int processor(char(* cpu)[256]);


doubleStack stack;
Dqueue devTemp;
Cqueue jobQ;
Cqueue readyQ;
Cqueue deviceQ;


int main_process_scheduler(void)
{
	//inititalize
	init_mem((double*)reg);
	init_mem((double*)mem);
	d_createQueue(&devTemp,40);
	char* fname = "process_file.txt";
	createQueue(&jobQ, 10);
	createQueue(&readyQ, 10);
	createQueue(&deviceQ, 10);
	double_createStack(&stack, 10);
	

	//load processs
	load_process(&jobQ, fname);

	//mov jobQ -> readyQ
	for (int i = 0, max = jobQ.count; i < max; i++)
	{
		Enqueue(&readyQ, Dequeue(&jobQ));
	}

	//exe
	char(*cpu)[256];
	puts("=========process start===========\n\n");
	while (!isEmpty(&readyQ) || !isEmpty(&deviceQ)) 
	{
		cpu = Dequeue(&readyQ);
		puts("==process executing==\n");
		if (processor(cpu))
		{
			printf("\nresult is %lf\n", double_pop(&stack));
		}
		else
		{
			puts("\niorequest");
		}
		/*if (cpu != NULL)                                 //prototype
		{
			sscanf(cpu, "%s %c",inst, &io_req);
			if (io_req == 'Y')
			{
				printf("[%s %c] io request\n", inst,io_req);
				*strchr(cpu, 'Y') = 'N';
				Enqueue(&deviceQ, cpu);
			}
			else
			{
				printf("[%s %c] done\n", inst,io_req);
			}
		}*/
		if (!isEmpty(&deviceQ))
		{
			Enqueue(&readyQ, Dequeue(&deviceQ));
		}
		puts("\n[다음 프로세스를 위해 아무 키나 입력]\n");
		_getch();
	}
	
	puts("\n================All process has done=================");
	
	deleteQueue(&jobQ);
	deleteQueue(&readyQ);
	deleteQueue(&deviceQ);
	free(stack.item);

	return EXIT_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////////////////////
static void createQueue(Cqueue* q, size_t _size)
{
	q->size = _size;
	q->item = (char(*)[256])malloc(sizeof(char[256]) * q->size);
	q->count = 0;
	q->rear = q->front = -1;
}

static void Enqueue(Cqueue* q, char(* x)[256])
{
	if (isFull(q))
	{
		q->size *= 2;
		q->item = (char(*)[256])realloc(q->item, sizeof(char[256]) * q->size);
		puts("Enque: isFull");
	}
	q->rear = (q->rear + 1) % q->size;
	strcpy(q->item[q->rear], (char*)x);
	q->count++;
}

static char (*Dequeue(Cqueue* q))[256]
{
	if (isEmpty(q)) return NULL;
	q->front = (q->front + 1) % q->size;
	q->count--;
	return (char(*)[256])q->item[q->front];
}

static void deleteQueue(Cqueue* q)
{
	free(q->item);
}

static void d_createQueue(Dqueue* q, size_t _size)
{
	q->size = _size;
	q->item = (double*)malloc(sizeof(double) * q->size);
	q->count = 0;
	q->rear = q->front = -1;
}

static void d_Enqueue(Dqueue* q, double x)
{
	if (d_isFull(q))
	{
		q->size *= 2;
		q->item = (double*)realloc(q->item, sizeof(double) * q->size);
		puts("Enque: isFull");
	}
	q->rear = (q->rear + 1) % q->size;
	q->item[q->rear] = x;
	q->count++;
}

static double d_Dequeue(Dqueue* q)
{
	if (d_isEmpty(q)) return 0;
	q->front = (q->front + 1) % q->size;
	q->count--;
	return q->item[q->front];
}

static void d_deleteQueue(Dqueue* q)
{
	free(q->item);
}

static void init_mem(double* mem)
{
	for (int i = 0; i < 32; i++)
	{
		mem[i] = i;
	}
}

static void load_process(Cqueue* q, char* fname)
{
	FILE* file = fopen(fname, "r+t");
	if (file == NULL) perror("file open err: ");

	char buf[256];
	
	int i = 0;
	while (fgets(buf, sizeof(buf), file) > 1)
	{

		if(strchr(buf, '\n')) *strchr(buf, '\n') = '\0';

		Enqueue(q, (char(*)[256])buf);

	}

	fclose(file);
}

void double_createStack(doubleStack* s, size_t _size)
{
	s->size = _size;
	s->item = (double*)malloc(_size * sizeof(double));
	s->top = 0;
}

void double_push(doubleStack* s, double x)
{
	if (s->item == NULL) return;
	if (double_isFull(s))
	{
		s->size *= 2;
		s->item = (double*)realloc(s->item, s->size * sizeof(double));
	}

	s->item[s->top++] = x;
}

double double_pop(doubleStack* s)
{
	if (s->item == NULL) return 0;
	if (double_isEmpty(s))
	{
		return 0;
	}
	return s->item[--s->top];
}

static int processor(char(* cpu)[256])
{
	char hardw[8];
	int num;
	double oprd1;
	double oprd2;
	char* ptrsave = (char*)cpu;
	char* cpuptr;
	while (cpuptr = strtok_s(NULL," ",&ptrsave))
	{
		if (strcmp(cpuptr, "push") == 0)
		{

			puts("push");

			strcpy(hardw, strtok_s(NULL, " ", &ptrsave));
			num = atoi(strtok_s(NULL, " ", &ptrsave));

			if (strcmp(hardw, "reg") == 0)
			{
				double_push(&stack, reg[num]);

			}
			else if (strcmp(hardw, "Q") == 0)
			{
				double_push(&stack, d_Dequeue(&devTemp));
			}
			else
			{
\
				double_push(&stack, mem[num]);
			}
		}
		else if (strcmp(cpuptr, "pop") == 0)
		{
			puts("pop");

			strcpy(hardw, strtok_s(NULL, " ", &ptrsave));
			num = atoi(strtok_s(NULL, " ", &ptrsave));
			
			if (strcmp(hardw, "reg") == 0)
			{
				reg[num] = double_pop(&stack);
			}
			else
			{
				mem[num] = double_pop(&stack);
			}

		}
		else if (strcmp(cpuptr, "add") == 0)
		{
			puts("add");

			oprd1 = double_pop(&stack);
			oprd2 = double_pop(&stack);
			double_push(&stack, oprd2 + oprd1);

		}
		else if (strcmp(cpuptr, "sub") == 0)
		{
			puts("sub");

			oprd1 = double_pop(&stack);
			oprd2 = double_pop(&stack);
			double_push(&stack, oprd2 - oprd1);

		}
		else if (strcmp(cpuptr, "mul") == 0)
		{
			puts("mul");

			oprd1 = double_pop(&stack);
			oprd2 = double_pop(&stack);
			double_push(&stack, oprd2 * oprd1);

		}
		else if (strcmp(cpuptr, "div") == 0)
		{
			puts("div");

			oprd1 = double_pop(&stack);
			oprd2 = double_pop(&stack);
			double_push(&stack, oprd2 / oprd1);

		}
		else if (strcmp(cpuptr, "free") == 0)
		{
			char* delptr = 0;
			puts("free");
			delptr = (char*)atoll(strtok_s(NULL, " ", &ptrsave));
			free(delptr);
		}
		else
		{
			char* temp = (char*)malloc(sizeof(char) * 256);
			strcpy(temp, "");
			while (!double_isEmpty(&stack))
			{
				oprd1 = double_pop(&stack);
				d_Enqueue(&devTemp,oprd1);
				strcat(temp, "push Q 0 ");
			}

			strcat(temp, ptrsave);

			if (strstr(temp, "free") == NULL)
			{
				char buf[16];
				strcat(temp, " free ");
				sprintf(buf, "%lld", (long long)temp);
				strcat(temp, buf);
			}

			Enqueue(&deviceQ, (char(*)[256])temp);

			return 0;
		}
	}
	return 1;
}

