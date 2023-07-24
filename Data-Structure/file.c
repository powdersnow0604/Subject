#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#define MAX 100

char line[MAX];
int total = 0;

static struct student {
	char name[7];
	char gender;
	char district[7];
	char major[13];
	float score;
	int height;
	int weight;
};

int main_file(void)
{

	FILE* file = fopen("list.txt", "r");

	if (file == NULL)
	{
		printf("파일 열기 실패\n");
		return 1;
	}


	while (fgets(line, MAX, file) != NULL)
	{
		total++;
	}

	fseek(file, 0, SEEK_SET);

	printf("총 %d명의 학생이 있습니다.\n", total);

	struct student* std = (struct student*)calloc(total, sizeof(struct student));
	int j = 0;

	while (fgets(line, MAX, file) != NULL)
	{
		
		sscanf(line, "%s %c %s %s %f %d %d", std[j].name, &std[j].gender, std[j].district, std[j].major, &(std[j].score), &(std[j].height), &(std[j].weight));


		j++;

	}

	fclose(file);
	for (int i = 0; i < total; i++)
	{
		printf("\n%02d번째 학생의 정보는: %-6s %c %-6s %-12s %0.1f %d %d\n", i + 1, std[i].name, std[i].gender, std[i].district, std[i].major, std[i].score, std[i].height, std[i].weight);

	}
	free(std);


	return 0;
}