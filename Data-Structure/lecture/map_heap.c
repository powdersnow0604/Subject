#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


typedef struct {
	int cnt;
	char word[32];
}freq;

typedef struct node_ node;
struct node_{
	freq item;
	node* lchild;
	node* rchild;
};

typedef struct {
	int size;
	int count;
	freq* item;
}heap;

static void filter(char s[]);
static void swap(freq* a, freq* b);

static void init_tree(node* tree);
static void search(node* tree, char* x);
static void insert(node* tree, char* x);
static void postorder(node* tree, heap* heap);

static void create_heap(heap* heap, int size_);
static void heapify_topdown(heap* heap, int idx);
static void heapify_bottomup(heap* heap, int idx);
static void push(heap* heap, freq x);
static freq pop(heap* heap);

 
int main(void)
{
	node* tree = (node*)malloc(sizeof(node));
	init_tree(tree);
	heap heap;
	create_heap(&heap, 128);

	//bst에 단어 삽입
	FILE* file = fopen("obama.txt", "r+t");
	if (file == NULL)
	{
		perror("file error");
		return EXIT_FAILURE;
	}

	char tok[64];
	while (fscanf(file, "%s", tok) != EOF)
	{
		filter(tok);
		if (strlen(tok) != 0)
		{
			insert(tree, (char*)tok);
		}
	}
	fclose(file);

	//bst에서 heap으로 단어 옮김
	postorder(tree, &heap);

	//상위 10개 빈도의 단어 출력
	puts("===========빈도수 상위 10개의 단어들(빈도수, 단어)===========\n");
	for (int i = 0; i < 10; i++)
	{
		freq temp = pop(&heap);
		printf("%20c%2d 번째:  [%d] %s\n", ' ',i+1, temp.cnt, temp.word);
	}
	puts("\n=============================================================");

	free(heap.item);
	return EXIT_SUCCESS;
}

static void filter(char s[])
{
	for (int i = 0; s[i] != '\0'; i++)
	{
		if (isdigit(s[i]))
		{
			s[0] = '\0';
			break;
		}
		if (!isalpha(s[i]))
		{
			for (int j = i; j < strlen(s); j++)
			{
				s[j] = s[j + 1];
			}
			i--;
		}
	}
}

static void swap(freq* a, freq* b)
{
	freq temp = *a;
	*a = *b;
	*b = temp;
}

static void init_tree(node* tree)
{
	strcpy(tree->item.word,"none");
	tree->lchild = NULL;
	tree->rchild = NULL;
}

static void search(node* tree, char *x)
{
	if (strcmp(tree->item.word, x) == 0)
	{
		tree->item.cnt += 1;
	}
	else if (strcmp(tree->item.word,x) > 0)
	{
		if (tree->lchild != NULL)
			search(tree->lchild, x);
		else
			insert(tree->lchild, x);
	}
	else
	{
		if (tree->rchild != NULL)
			search(tree->rchild, x);
		else
			insert(tree->rchild, x);
	}
}

static void insert(node* tree, char* x)
{
	//degenerate case
	if (strcmp(tree->item.word, "none") == 0)
	{
		strcpy(tree->item.word, x);
		tree->item.cnt = 1;
		return;
	}

	// x == key
	if (strcmp(tree->item.word, x) == 0)
	{
		tree->item.cnt += 1;
	}

	// x < key
	else if (strcmp(tree->item.word, x) > 0)
	{
		if (tree->lchild != NULL)
			insert(tree->lchild, x);
		else
		{
			tree->lchild = (node*)malloc(sizeof(node));
			strcpy(tree->lchild->item.word, x);
			tree->lchild->item.cnt = 1;
			tree->lchild->lchild = NULL;
			tree->lchild->rchild = NULL;
		}
	}

	// x > key
	else
	{
		if (tree->rchild != NULL)
			insert(tree->rchild, x);
		else
		{
			tree->rchild = (node*)malloc(sizeof(node));
			strcpy(tree->rchild->item.word, x);
			tree->rchild->item.cnt = 1;
			tree->rchild->lchild = NULL;
			tree->rchild->rchild = NULL;
		}
	}
}

static void postorder(node* tree, heap* heap)
{
	if (tree)
	{
		postorder(tree->lchild, heap);
		postorder(tree->rchild, heap);
		push(heap, tree->item);
		free(tree);
	}
}

static void create_heap(heap* heap, int size_)
{
	heap->size = size_;
	heap->item = (freq*)malloc(sizeof(freq) * heap->size);
	heap->count = 0;
}

static void heapify_topdown(heap* heap, int idx)
{
	if (2 * idx > heap->count) return;
	
	if (2 * idx == heap->count)
	{
		if (heap->item[idx].cnt < heap->item[2*idx].cnt)
			swap(&heap->item[idx], &heap->item[2 * idx]);
		return;
	}

	
	if (heap->item[2 * idx].cnt >= heap->item[2 * idx + 1].cnt && heap->item[2 * idx].cnt > heap->item[idx].cnt)
	{
		swap(&heap->item[idx], &heap->item[2 * idx]);
		heapify_topdown(heap,2*idx);
	}
	
	else if (heap->item[2 * idx +1].cnt >= heap->item[2 * idx].cnt && heap->item[2 * idx+1].cnt > heap->item[idx].cnt)
	{
		swap(&heap->item[idx], &heap->item[2 * idx + 1]);
		heapify_topdown(heap,2 * idx + 1);
	}
	
} 

static void heapify_bottomup(heap* heap, int idx)
{
	while (idx > 1)
	{
		if (heap->item[idx / 2].cnt < heap->item[idx].cnt)
		{
			swap(&heap->item[idx / 2], &heap->item[idx]);
			idx /= 2;
		}
		else
			break;
	}
}

static void push(heap* heap, freq x)
{
	if (heap->count == heap->size-1)
	{
		heap->size *= 2;
		heap->item = (freq*)realloc(heap->item, sizeof(freq) * heap->size);
	}
	heap->item[++(heap->count)] = x;
	heapify_bottomup(heap, heap->count);
}

static freq pop(heap* heap)
{
	freq temp = heap->item[1];
	heap->item[1] = heap->item[(heap->count)--];

	heapify_topdown(heap, 1);
	return temp;
}
