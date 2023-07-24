#include <stdio.h>
#include <conio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#define input2              //input2를 열고 싶다면 주석 해제

typedef struct {
	char name[10];
	char gender;
	char district[10];
	char major[20];
	float score;
	int height;
	int weight;
}student;

typedef struct node_ node;
struct node_ {
	student std;
	node* link;
};

typedef struct{
	node* link;
} hnode;

void process_insert(hnode *first, char *tok2, char *tok3, char *tok4, char *tok5, char *tok6, char *tok7, char *tok8);
void process_create(hnode *first);
void insert(hnode *first,student temp);
void process_load(hnode *first,char *tok2);
void deleteAll(hnode *first);
void process_print(hnode *first);
void process_delete(hnode *first, char *tok2);
node* process_search(hnode *first, char* tok2);



int main_linkedlist(void)
{
	// 0. 명령어 line 수 세기 함수 필요 없는 버전
	FILE* fp;
#ifdef input2
	fopen_s(&fp,"input2.txt", "r+t");
#else
	fopen_s(&fp, "input.txt", "r+t");
#endif
	char input[512];
	char tok1[32], tok2[32], tok3[32], tok4[32], tok5[32], tok6[32], tok7[32], tok8[32];  // 토큰 선언

	hnode first; 
	while (fgets(input, 512, fp) != NULL) {
		printf("input: %d %d %d %d %d %d %d %d",input[0],input[1],input[2], input[3], input[4], input[5], input[6], input[7]);
		sscanf_s(input, "%s%s%s%s%s%s%s%s", tok1,(unsigned)sizeof(tok1),tok2, (unsigned)sizeof(tok2), tok3, (unsigned)sizeof(tok3), tok4, (unsigned)sizeof(tok4), tok5, (unsigned)sizeof(tok5), tok6, (unsigned)sizeof(tok6), tok7, (unsigned)sizeof(tok7), tok8, (unsigned)sizeof(tok8));

		if (strcmp(tok1, "CREATE") == 0) {
			process_create(&first);
			printf(" \n%s done ==================================================\n\n", tok1);
		}
		else if (strcmp(tok1, "LOAD") == 0) {
			process_load(&first,tok2);
			printf(" \n%s done ==================================================\n\n", tok1);
		}
		else if (strcmp(tok1, "PRINT") == 0) {
			process_print(&first);
			printf(" \n%s done ==================================================\n\n", tok1);
		}
		else if (strcmp(tok1, "INSERT") == 0) {
			process_insert(&first, tok2, tok3, tok4, tok5, tok6, tok7, tok8);
			printf(" \n%s done ==================================================\n\n", tok1);
		}
		else if (strcmp(tok1, "DELETE") == 0) {
			process_delete(&first,tok2);
			printf(" \n%s done ==================================================\n\n", tok1);
		}
		else if (strcmp(tok1, "SEARCH") == 0) {
			process_search(&first,tok2);
			printf(" \n%s done ==================================================\n\n", tok1);
		}
		else {
			printf("\n%s is not a keyword.\n", tok1);
		}
		printf("다음 명령어를 불러오기 위해서 아무 키나 누르십시오");
		_getch();
		system("cls");
	}
	deleteAll(&first);
	fclose(fp);
	
	

	return 0;
}

void process_create(hnode *first) {
	first->link = NULL;
}


void process_insert(hnode *first,char *tok2, char *tok3, char *tok4, char *tok5, char *tok6, char *tok7, char *tok8)
{
	student temp;
	strcpy_s(temp.name, sizeof(temp.name), tok2);
	temp.gender = tok3[0];
	strcpy_s(temp.district, sizeof(temp.district), tok4);
	strcpy_s(temp.major, sizeof(temp.major),tok5);
	temp.score = atof(tok6);
	temp.height = atoi(tok7);
	temp.weight = atoi(tok8);
	insert(first, temp);
	printf("\n추가한 학생의 이름은: %s\n", temp.name);

}

void insert(hnode *first, student temp)
{
	if (first->link == NULL) {
		node* nnode = calloc(1, sizeof(node));
		nnode->std = temp;
		nnode->link = NULL;
		first->link = nnode;
		//first.link->std = temp;
		//first.link->link = NULL;
		return;
	}
	if (first->link->link == NULL)
	{
		node* nnode = calloc(1, sizeof(node));
		nnode->std = temp;
		if (strcmp(first->link->std.name, temp.name) >= 0)
		{
			nnode->link = first->link;
			first->link = nnode;
		}
		else
		{
			first->link->link = nnode;
		}
		return;
	}
	
	node* curr = first->link;
	while (curr->link != NULL)
	{
		if (strcmp(curr->link->std.name, temp.name) >= 0) break;
		curr = curr->link;
	}
	node* nnode = calloc(1, sizeof(node));
	nnode->std = temp;


	nnode->link = curr->link;
	curr->link = nnode;

}

void process_load(hnode *first, char *tok2)
{
	FILE* file;
	fopen_s(&file, tok2, "r");	//file open
	if (file == NULL)
	{
		printf("파일 열기 실패\n");
		return 1;
	}
	student temp;
	char lines[100];
	while (fgets(lines, 100, file) != NULL)
	{
		sscanf_s(lines,"%s %c %s %s %f %d %d", temp.name, (unsigned)sizeof(temp.name) ,&temp.gender,1, temp.district, (unsigned)sizeof(temp.district), temp.major, (unsigned)sizeof(temp.major), &temp.score, &temp.height, &temp.weight);
		insert(first, temp);
	}
	fclose(file);

}

void deleteAll(hnode *first)
{
	while (first->link != NULL)
	{
		node* dnode = first->link;
		first->link = dnode->link;
		free(dnode);
	}
}

void process_print(hnode* first)
{
	node* curr = first->link;
	int number = 1;
	while (curr != NULL)
	{
		printf("\n%02d번째 학생의 정보는: %-6s %c %-6s %-12s %0.1f %d %d\n", number++, curr->std.name, curr->std.gender, curr->std.district, curr->std.major, curr->std.score, curr->std.height, curr->std.weight);
		curr = curr->link;
	}

}

void process_delete(hnode *first, char* tok2)
{
	if (first->link == NULL) {
		printf("\n빈 리스트입니다.\n");
		return;
	}

	node* curr = first->link;
	if (strcmp(curr->std.name, tok2) == 0)
	{
		node* dnode = curr;
		first->link = curr->link;
		printf("\n제거한 학생의 이름은: %s\n", dnode->std.name);
		free(dnode);
	}
	else
	{
		while (curr->link != NULL)
		{
			if (strcmp(curr->link->std.name, tok2) == 0) break;
			curr = curr->link;
		}

		if (curr->link == NULL) {
			printf("\n%s 학생이 리스트에 존재하지 않습니다.\n",tok2);
			return;
		}
		node* dnode = curr->link;
		curr->link = dnode->link;
		printf("\n제거한 학생의 이름은: %s\n", dnode->std.name);

		free(dnode);
	}

}

node* process_search(hnode *first, char* tok2)
{
	node* curr = first->link;
	int cnt = 1;
	while (curr != NULL) {
		if (strcmp(curr->std.name, tok2) == 0)
		{
			printf("\n%s 학생은 %d번에 등재되어 있습니다.\n",tok2, cnt);
			return curr;
		}
		curr = curr->link;
		cnt++;
	}
	printf("\n%s 학생이 리스트에 등재되어있지 않습니다.\n",tok2);
	return NULL;
}
