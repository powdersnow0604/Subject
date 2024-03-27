#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

int total;

static struct student {
	char name[7];
	char gender[2];
	char district[7];
	char major[13];
	float score;
	int height;
	int weight;
}*std;

void quicksort(struct student* arr);
int sort(int low, int high, struct student* arr);
int partition(int low, int high, struct student* arr);
void swap(int low, int high, struct student* arr);

int count_lines(char filenm[]);
static void process_create();
static void process_load(char filenm[]);
static void process_print();
static void process_insert(char tok2[], char tok3[], char tok4[], char tok5[], float tok6, int tok7, int tok8);
static void process_delete(char tok2[]);
static void process_search(char tok2[]);

void main_arr()
{
	// 0. 명령어 line 수 세기

	int nlines = count_lines("input.txt");
	printf("명령어 수는 %d개 입니다\n", nlines);

	FILE* fcin = fopen("input.txt", "r"); //input open2


	// 토큰 선언
	char tok1[32], tok2[32], tok3[32], tok4[32], tok5[32]; float tok6; int tok7, tok8;

	for (int i = 0; i < nlines; i++) {  //nlines로 바꾸기

		fscanf(fcin, "%s", tok1);

		if (strcmp(tok1, "CREATE") == 0) {
			process_create();
		}
		else if (strcmp(tok1, "LOAD") == 0) {
			fscanf(fcin, "%s", tok2);
			process_load(tok2);
		}
		else if (strcmp(tok1, "PRINT") == 0) {
			process_print();
		}
		else if (strcmp(tok1, "INSERT") == 0) {
			fscanf(fcin, "%s %s %s %s %f %d %d", tok2, tok3, tok4, tok5, &tok6, &tok7, &tok8);
			process_insert(tok2, tok3, tok4, tok5, tok6, tok7, tok8);
		}
		else if (strcmp(tok1, "DELETE") == 0) {
			fscanf(fcin, "%s", tok2);
			process_delete(tok2);
		}
		else if (strcmp(tok1, "SEARCH") == 0) {
			fscanf(fcin, "%s", tok2);
			process_search(tok2);
		}
		else {
			printf("%s is not a keyword.\n", tok1);
		}
		printf("\n%s done ===============\n", tok1);
		printf("다음 명령어를 불러오기 위해서 아무키나 누십시오");
		_getch();
		system("cls");
	}
	fclose(fcin); //input close2
	free(std); // std free 
	printf("\n종료합니다.\n");
	return 0;
}


int count_lines(char filenm[])
{
	FILE* file = fopen("input.txt", "r"); //input open1

	if (file == NULL)
	{
		printf("파일 열기 실패\n");
		return 1;
	}

	int cnt = 1;
	char c;
	while ((c = fgetc(file)) != EOF)
	{
		if (c == '\n') cnt++;
	}

	fclose(file); //input close1
	return cnt;
}

void process_create()
{
	std = (struct student*)calloc(100, sizeof(struct student)); //std dynamic;
}


void process_load(char filenm[])
{
	FILE* file = fopen(filenm, "r"); //list open

	if (file == NULL)
	{
		printf("파일 열기 실패\n");
		return 1;
	}

	int cnt = 1;
	char c;
	while ((c = fgetc(file)) != EOF)
	{
		if (c == '\n') cnt++;
	}
	total = cnt;
	fseek(file, 0L, SEEK_SET);

	printf("총 %d명의 학생이 있습니다.\n", total);

	for (int j = 0; j < total; j++)
	{
		fscanf(file, "%s %s %s %s %f %d %d", std[j].name, &std[j].gender, std[j].district, std[j].major, &(std[j].score), &(std[j].height), &(std[j].weight));

	}

	fclose(file); //list close
	quicksort(std);

}


void process_print()
{
	for (int i = 0; i < total; i++)
	{
		printf("\n%2d번째 학생의 정보는: %-6s %s %-6s %-12s %0.1f %d %d\n", i + 1, std[i].name, std[i].gender, std[i].district, std[i].major, std[i].score, std[i].height, std[i].weight);

	}
}

void process_insert(char tok2[], char tok3[], char tok4[], char tok5[], float tok6, int tok7, int tok8)
{
	// 위치 찾기
	int i = 0;
	while (strcmp(std[i].name, tok2) == -1 && i < total)
	{
		i++;
	}
	// 원소 이동

	for (int j = total - 1; j >= i; j--)
	{
		memcpy(std[j + 1].name, std[j].name, sizeof(struct student));
	}
	// 키 값 대입
	strcpy(std[i].name, tok2);
	strcpy(std[i].gender, tok3);
	strcpy(std[i].district, tok4);
	strcpy(std[i].major, tok5);
	std[i].score = tok6;
	std[i].height = tok7;
	std[i].weight = tok8;


	//count 증가
	total++;

	printf("\n추가한 학생의 정보는: %-6s %s %-6s %-12s %0.1f %d %d\n",  std[i].name, std[i].gender, std[i].district, std[i].major, std[i].score, std[i].height, std[i].weight);

}


void quicksort(struct student* arr)
{
	sort(0, total - 1, arr);
}

int sort(int low, int high, struct student* arr)
{
	if (high <= low)
	{
		return 0;
	}

	int mid = partition(low, high, arr);

	sort(low, mid - 1, arr);
	sort(mid, high, arr);
}

int partition(int low, int high, struct student* arr)
{
	char pivot[7];
	strcpy(pivot, arr[(low + high) / 2].name);
	while (low <= high)
	{
		while (strcmp(arr[low].name, pivot) == -1)
		{
			low++;
		}
		while (strcmp(arr[high].name, pivot) == 1)
		{
			high--;
		}
		if (low <= high)
		{
			swap(low, high, arr);
			low++;
			high--;
		}
	}
	return low;
}

void swap(int low, int high, struct student* arr)
{
	void* temp = malloc(sizeof(struct student)); //temp dynamic
	memcpy(temp, &arr[low], sizeof(struct student));
	memcpy(&arr[low], &arr[high], sizeof(struct student));
	memcpy(&arr[high], temp, sizeof(struct student));
	free(temp); //free temp
}

void process_delete(char tok2[])
{
	// 위치 찾기
	int i = 0;
	while (strcmp(std[i].name, tok2) != 0 && i < total)
	{
		i++;
	}
	if (i == total)
	{
		printf("제거할 학생이 명단에 존재하지 않습니다.");
		return;
	}
	// 원소 이동
	for (int j = i; j < total - 1; j++)
	{
		memcpy(&std[j], &std[j + 1], sizeof(struct student));
	}
	//count 감소
	total--;

	printf("\n제거한 학생의 이름은: %s\n", tok2);

}

void process_search(char tok2[])
{
	int start = 0;
	int end = total - 1;

	while (start <= end)
	{
		int mid = (start + end) / 2;
		if (strcmp(std[mid].name, tok2) == 0)
		{
			printf("\n%2d번째 학생의 정보는: %-6s %s %-6s %-12s %0.1f %d %d\n", mid + 1, std[mid].name, std[mid].gender, std[mid].district, std[mid].major, std[mid].score, std[mid].height, std[mid].weight);
			return;
		}
		else if (strcmp(std[mid].name, tok2) == -1)
		{
			start = mid + 1;
		}
		else
		{
			end = mid - 1;
		}
	}
	printf("\n찾는 학생이 없습니다\n");
	return -1;
}