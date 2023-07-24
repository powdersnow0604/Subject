#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>

static char* int2str(int x);

int main_mkprocess(void)
{
	srand(time(NULL));

	FILE* file = fopen("process_file.txt", "wt");

	char string[512] = "";
	int randn;
	for (int i = 0; i < 10; i++)
	{
		strcpy(string, "push reg \0");
		randn = rand() % 32;
		strcat(string, int2str(randn));
		int cnt = 4;
		for (int j = 0; j < 3; j++)
		{
			randn = rand() % 2;
			if (randn)
			{
				strcat(string, " push reg ");
			}
			else
			{
				strcat(string, " push mem ");
			}
			randn = rand() % 32;
			strcat(string, int2str(randn));
		}
		while (cnt > 1)
		{
			randn = rand() % 6 + 1;
			switch (randn) {
			case 1:
			{
				randn = rand() % 2;
				if (randn)
				{
					strcat(string, " pop reg ");
				}
				else
				{
					strcat(string, " pop mem ");
				}
				randn = rand() % 32;
				strcat(string, int2str(randn));
				cnt--;
				break;
			}
			case 2:
			{
				strcat(string, " iorequest");
				break;
			}
			case 3:
			{
				strcat(string, " add");
				cnt--;
				break;
			}
			case 4:
			{
				strcat(string, " sub");
				cnt--;
				break;
			}
			case 5:
			{
				strcat(string, " mul");
				cnt--;
				break;
			}
			case 6:
			{
				strcat(string, " sub");
				cnt--;
				break;
			}
			}
		}
		fputs(string,file);
		fputc('\n', file);
	}

	fclose(file);
	return 0;
}

static char* int2str(int x)
{
	static char temp[3];
	if (x / 10)
	{
		temp[0] = x / 10 + '0';
		temp[1] = x % 10 + '0';
		temp[2] = '\0';
	}
	else
	{
		temp[0] = x + '0';
		temp[1] = '\0';
	}
	return (char*)temp;
}
