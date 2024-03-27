#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#define method1


typedef struct {
	int cnt;
	char word[32];
}freq;

typedef struct {
	freq* item;
	int count;
	int size;
}array;

static void filter(char s[]);
static void insert(array* arr,char* str);
static void init(array* arr);
static void insert2(array* arr, char* str);

static void quicksort_(array* arr);
static void sort_(int low, int high, array* arr);
static int partition_(int low, int high, array* arr);
static void swap_(int low, int high, array* arr);

int main_map_array(void)
{
	array arr;
	init(&arr);
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
			#ifdef method1
				insert(&arr, tok);
			#else
				insert2(&arr, tok);
			#endif		
		}
	}
	#ifdef method1

	quicksort_(&arr);
	#endif		

	for (int i = 0; i < 10; i++)
	{
		printf("[%d] %s\n", arr.item[i].cnt, arr.item[i].word);
	}
	
	fclose(file);
	free(arr.item);

	return EXIT_SUCCESS;
}


////////////////////////////////////////////////////////////////////////////
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

static void insert(array* arr, char* str)
{
	if (arr->count == arr->size)
	{
		arr->size *= 2;
		arr->item = (freq*)realloc(arr->item, sizeof(freq) * arr->size);
	}

	int i = 0;
	for (i = 0; i < arr->count; i++)
	{
		if (strcmp(arr->item[i].word, str) == 0)
		{
			arr->item[i].cnt++;
			break;
		}
	}
	if (i == arr->count)
	{
		strcpy(arr->item[arr->count].word, str);
		arr->item[arr->count++].cnt = 1;
	}
}

static void init(array* arr)
{
	arr->size = 100;
	arr->item = (freq*)malloc(sizeof(freq) * arr->size);
	arr->count = 0;
}

static void quicksort_(array* arr)
{
	sort_(0, arr->count-1, arr);
}

static void sort_(int low, int high, array* arr)
{
	if (high <= low)
	{
		return;
	}

	int mid = partition_(low, high, arr);

	sort_(low, mid - 1, arr);
	sort_(mid, high, arr);
}

static int partition_(int low, int high, array* arr)
{
	int pivot = arr->item[(low + high) / 2].cnt;
	while (low <= high)
	{
		while (arr->item[low].cnt > pivot)
		{
			low++;
		}
		while (arr->item[high].cnt< pivot)
		{
			high--;
		}
		if (low <= high)
		{
			swap_(low++, high--, arr);
		}
	}
	return low;
}

static void swap_(int low, int high, array* arr)
{
	freq temp = arr->item[low];
	arr->item[low] = arr->item[high];
	arr->item[high] = temp;
}

static void insert2(array* arr, char* str)
{
	if (arr->count == arr->size)
	{
		arr->size *= 2;
		arr->item = (freq*)realloc(arr->item, sizeof(freq) * arr->size);
	}

	int i = 0;
	for (i = 0; i < arr->count; i++)
	{
		if (strcmp(arr->item[i].word, str) == 0)
		{
			arr->item[i].cnt++;
			break;
		}
	}
	if (i == arr->count)
	{
		strcpy(arr->item[arr->count].word, str);
		arr->item[arr->count++].cnt = 1;
	}
	else
	{
		int j;
		for (j = i-1; j >= 0; j--)
		{
			if (arr->item[j].cnt >= arr->item[i].cnt) break;
		}
		freq temp = arr->item[i];
		for (int k = i; k > j+1; k--)
		{
			arr->item[k] = arr->item[k-1];
		}
		arr->item[j + 1] = temp;
	}
}
