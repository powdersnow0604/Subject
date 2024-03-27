#ifndef __GRAPH_H__
#define __GRAPH_H__

#include "list.h"

typedef struct graph_ {
	list_i* adj_list;
	int vertex_num;
	int edge_num;
}graph;

int init_graph(graph* g, const char* path);
void free_graph(graph* g);

void dfs(graph* g, void (*func)(int));
#endif