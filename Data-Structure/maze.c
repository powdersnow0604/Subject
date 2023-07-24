#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <conio.h>

char maze[8][8] = { '1','1','1','1','1','1','1','1',
				    '0','0','1','1','0','0','0','1',
					'1','0','1','1','0','1','1','1',
					'1','0','1','1','0','1','1','1',
					'1','0','0','0','0','0','0','1',
					'1','1','1','1','1','1','0','1',
					'1','1','1','1','0','0','0','0',
					'1','1','1','1','1','1','1','1' };

typedef struct {
	int row;
	int column;
}position;

typedef struct {
	position* item;
	int top;
	size_t size;
}stack;

static void createStack(stack* s, size_t _size);
static int isEmpty(stack* s) { return s->top == 0; }
static int isFull(stack* s) { return s->top == s->size; }
static void push(stack* s, position x);
static position pop(stack* s);

void showMaze();



int main_maze(void)
{
	stack tstack;
	createStack(&tstack,sizeof(maze));
	position tpos = { 1,0 };  //starting point
	push(&tstack, tpos);

	int fowardingState;
	while (tstack.item[tstack.top - 1].column != 7 || tstack.item[tstack.top - 1].row != 6)   //escape point
	{
		maze[tstack.item[tstack.top - 1].row][tstack.item[tstack.top - 1].column] = 'C';
		printf("%14s\n", "MAZE");
		showMaze();
		puts("\n//C is the current point");
		puts("\npush any button for next move");
		_getch();
		system("cls");

		fowardingState = 0; 
		for (int i = 0; i < 4; i++)
		{
			if (maze[tstack.item[tstack.top - 1].row + ((-2 + i) % 2)][tstack.item[tstack.top - 1].column + ((-1 + i) % 2)] == '0' &&
				0 <= tstack.item[tstack.top - 1].column + ((-1 + i) % 2)  && tstack.item[tstack.top - 1].column + ((-1 + i) % 2) <= 7 &&
				0 <= tstack.item[tstack.top - 1].row + ((-2 + i) % 2) && tstack.item[tstack.top - 1].row + ((-2 + i) % 2) <= 7)
			{
				tpos.column = tstack.item[tstack.top - 1].column + ((-1 + i) % 2);
				tpos.row = tstack.item[tstack.top - 1].row + ((-2 + i) % 2);
				maze[tstack.item[tstack.top - 1].row][tstack.item[tstack.top - 1].column] = 'V';
				push(&tstack, tpos);
				fowardingState = 1;
				break;
			}
		}

		if (fowardingState == 0)
		{
			tpos = pop(&tstack);
			maze[tpos.row][tpos.column] = 'X';

		}
	}

	maze[tstack.item[tstack.top - 1].row][tstack.item[tstack.top - 1].column] = 'C';
	printf("%14s\n", "MAZE");
	showMaze();

	puts("\narrived at the escape point");
	free(tstack.item);

	return EXIT_SUCCESS;
}


///////////////////////////////////////////////
void showMaze()
{
	for (int i = 0; i < 8; i++)
	{
		printf("    ");
		for (int j = 0; j < 8; j++)
		{
			putc(maze[i][j], stdout);
			putc(32, stdout);
		}
		puts("");
	}
}

static void createStack(stack* s, size_t _size)
{
	s->size = _size;
	s->item = (position*)malloc(_size * sizeof(position));
	s->top = 0;
}

static void push(stack* s, position x)
{
	if (s->item == NULL) return;
	if (isFull(s))
	{
		s->size *= 2;
		s->item = (position*)realloc(s->item, s->size * sizeof(position));
	}

	s->item[s->top++] = x;
}

static position pop(stack* s)
{
	position dummy = { 0,0 };
	if (s->item == NULL) return dummy;
	if (isEmpty(s))
	{
		return dummy;
	}
	return s->item[--s->top];
}