#include "graph.h"
#include "queue.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>


//declaration
static void dfs_vertex(int vertex_num, graph* g, int* visit, void(*func)(int));

//definition
int init_graph(graph* g, const char* path)
{
	int u = 0, v = 0;
	char* check;
	char buf[64];
	FILE* file;
	fopen_s(&file, path, "rt");

	if (file == NULL) return -1;


	//check start point
	while ((check = fgets(buf, sizeof(buf), file)) != NULL)
	{
		if ((check = strstr(buf, "Input:")) != NULL) break;
	}

	if (check == NULL) return -1;
	if (strncmp(check, "Input:", 6)) return -1;


	//read vertex num, edge num
	fscanf_s(file, "%d %d", &(g->vertex_num), &(g->edge_num));


	//initialize adjacency list
	g->adj_list = (list_i*)malloc((g->vertex_num + 1) * sizeof(list_i));
	if (g->adj_list == NULL) return -1;

	for (int i = 0; i <= g->vertex_num; ++i) {
		init_list(&(g->adj_list[i]));
	}


	//read edge
	for (int i = 0; i < g->edge_num; ++i) {
		fscanf_s(file, "%d %d", &u, &v);
		list_insert_i(&(g->adj_list[u]), v, 1);
		list_insert_i(&(g->adj_list[v]), u, 1);
	}

	fclose(file);

	return 0;
}

void free_graph(graph* g)
{
	for (int i = 0; i <= g->vertex_num; ++i) {
		free_list(&(g->adj_list[i]));
	}

	free(g->adj_list);
	g->edge_num = 0;
	g->vertex_num = 0;
}

void dfs(graph* g, void (*func)(int))
{
	int* visit = (int*)calloc((g->vertex_num + 1), sizeof(int));
	if (visit == NULL) return;

	for (int i = 1; i <= g->vertex_num; ++i) {
		if (visit[i] == 0) dfs_vertex(i,g, visit, func);
	}

	free(visit);
}

void dfs_vertex(int vertex_num, graph* g, int* visit, void(*func)(int))
{
	if(func != NULL) func(vertex_num);

	visit[vertex_num] = 1;

	node_i* curr = g->adj_list[vertex_num].first;
	for (; curr != NULL; curr = curr->next) {
		if (!visit[curr->element]) dfs_vertex(curr->element, g, visit, func);
	}
}

void bfs(graph* g, int s, void (*func)(int))
{
	int u;
	queue_i Q;
	int* dist = (int*)malloc(sizeof(int) * (g->vertex_num + 1));

	if (dist == NULL) return;

	for (int i = 0; i < g->vertex_num + 1; ++i) {
		dist[i] = -1;
	}

	init_queue(&Q, g->vertex_num);

	q_push_back(&Q, s);
	dist[s] = 0;

	while (!q_is_empty(&Q)) {
		u = q_pop_front(&Q);

		if (func != NULL) func(u);

		node_i* curr = g->adj_list[u].first;
		for (; curr != NULL; curr = curr->next) {
			if (dist[curr->element] == -1) {
				dist[curr->element] = dist[u] + 1;
				q_push_back(&Q, curr->element);
			}
		}
	}
}