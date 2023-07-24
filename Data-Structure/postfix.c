#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

typedef struct {
	char* item;
	int top;
	size_t size;
}stack;

typedef struct {
	double* item;
	int top;
	size_t size;
}doubleStack;


char priority[2][3] = { '*','/','%' , '+','-'};
char parenthesis[6] = { '(',')','{','}','[',']' };

static void createStack(stack* s, size_t _size);
static bool isEmpty(stack* s) { return s->top == 0; }
static bool isFull(stack* s) { return s->top == s-> size; }
static void push(stack* s,char x);
static char pop(stack* s);

bool checkParenthesis(char* str);
char* mid2postfix(char* str);
int findPriority(char x);
double calcPostfix(char* str);

static void double_createStack(doubleStack* s, size_t _size);
static bool double_isEmpty(doubleStack* s) { return s->top == 0; }
static bool double_isFull(doubleStack* s) { return s->top == s->size; }
static void double_push(doubleStack* s, double x);
static double double_pop(doubleStack* s);

int main_postfix(void)
{
	char* str = mid2postfix("(2+5)*3*(2+1)");
	printf("postfix: ");
	for (int i = 0; i < strlen(str); i++)
	{
		putc(str[i],stdout);
	}
	printf("\n\nresult of claculation: %lf\n", calcPostfix(str));
	free(str);
	puts("\nfinish");
	return 0;
}


///////////////////////////////////////////////////////////////////////////////////
void createStack(stack* s, size_t _size)
{
	s->size = _size;
	s->item = (char*)malloc(_size * sizeof(char));
	s->top = 0;
}

void push(stack* s,char x)
{
	if (s->item == NULL) return;
	if (isFull(s))
	{
		s->size *= 2;
		s->item = (char*)realloc(s->item, s->size * sizeof(char));
	}

	s->item[s->top++] = x;
}

char pop(stack* s)
{
	if (s->item == NULL) return 0;
	if (isEmpty(s))
	{
		return 0;
	}
	return s->item[--s->top];
}

bool checkParenthesis(char* str)
{
	//문제 있으면 1
	//문제 없으면 0
	stack tstack;
	createStack(&tstack, strlen(str));
	int i = 0;
	int error = false;
	while (i < strlen(str))
	{
		switch (str[i])
		{
		case '(':
		case '{':
		case '[': 
			push(&tstack, str[i]); 
			break;
		case ')':
		case '}':
		case ']':
			if (isEmpty(&tstack))
			{
				error = true;
			}
			else
			{
				char* p = memchr(parenthesis, str[i], 6);
				if (*(p-1) == tstack.item[tstack.top - 1])
				{
					pop(&tstack);
				}
				else
				{
					error = true;
				}
			}
			
		}
		i++;
	}
	if (isEmpty(&tstack))
	{
		free(tstack.item);
		return error;
	}
	else
		free(tstack.item);
		return true;
}


char* mid2postfix(char* str)
{
	char* postfix = (char*)malloc((strlen(str) + 1)*sizeof(str));
	stack tstack;
	createStack(&tstack, strlen(str)*2);

	if (checkParenthesis(str)) return NULL;

	int i = 0;   //str index
	int j = 0;   //postfix index
	int state=0; //continuous number state
	int statePrevNum = 0; //whether previous token was number
	int p;       //temporary variable for non parenthesis operator
	char* pt;    //temporary variable for parenthesis
	while (str[i] != '\0')
	{
		if (isdigit(str[i]))
		{
			postfix[j++] = str[i];
			state = 1;
		}
		else
		{
			if (state == 1)
			{
				postfix[j++] = 32;
				state = 0;
				statePrevNum = 1;
			}

			if ((pt = memchr(parenthesis, str[i], 6)) != NULL)
			{
				if ((pt - parenthesis) % 2)
				{
					while (tstack.item[tstack.top - 1] != *(pt - 1))
					{
						postfix[j++] = pop(&tstack);
					}
					pop(&tstack);
				}
				else
				{
					push(&tstack, str[i]);
					statePrevNum = 0;
				}
			}

			else if ((p= findPriority(str[i])) != 10)
			{
				if (str[i] == '-' && statePrevNum == 0)
				{
					postfix[j++] = '-';
					postfix[j++] = 32;
				}
				else
				{
					while ((findPriority(tstack.item[tstack.top - 1]) <= p) && !isEmpty(&tstack))
					{
						postfix[j++] = pop(&tstack);
					}
					push(&tstack, str[i]);
					statePrevNum = 0;
				}
			}
		}
		i++;
	}

	if (state == 1)
	{
		postfix[j++] = 32;
		state = 0;
	}
	while (!isEmpty(&tstack))
	{
		postfix[j++] = pop(&tstack);
	}
	postfix[j] = '\0';
	free(tstack.item);
	return postfix;
}

int findPriority(char x)
{
	for (int i = 0; i < 2; i++)
	{
		if (memchr(priority[i], x, 3) != NULL) return i;
	}
	return 10;
}

double calcPostfix(char* str)
{
	doubleStack tstack;
	double_createStack(&tstack, strlen(str));

	int i = 0;
	double temp = 0;
	double high;
	double low;
	int stateMinus = 0;
	while (str[i] != '\0')
	{
		if (isdigit(str[i]))
		{
			while (str[i] != 32)
			{
				temp = temp * 10 + (str[i++] - '0');
			}
			if (stateMinus == 1)
			{
				double_push(&tstack, -1*temp);
				stateMinus = 0;
			}
			else
			{
				double_push(&tstack, temp);
			}
			temp = 0;
		}
		else
		{
			switch (str[i]) {
			case '*':
				high = double_pop(&tstack);
				low = double_pop(&tstack);
				double_push(&tstack, low * high);
				break;
			case '/':
				high = double_pop(&tstack);
				low = double_pop(&tstack);
				double_push(&tstack, low / high);
				break;
			case '%':
				high = double_pop(&tstack);
				low = double_pop(&tstack);
				double_push(&tstack, (int)low % (int)high);
				break;
			case '+':
				high = double_pop(&tstack);
				low = double_pop(&tstack);
				double_push(&tstack, low + high);
				break;
			case '-':
				if (str[i + 1] == 32)
				{
					stateMinus = 1;
					i++;
					break;
				}
				else
				{
					high = double_pop(&tstack);
					low = double_pop(&tstack);
					double_push(&tstack, low - high);
					break;
				}
			}
		}
		i++;
	}
	temp = tstack.item[0];
	free(tstack.item);
	return temp;
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