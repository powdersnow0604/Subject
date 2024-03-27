#ifndef __LIST_H__
#define __LIST_H__

typedef unsigned long long size_t;
typedef struct int_node node_i;
typedef struct list_i_ list_i;

struct int_node {
	int element;
	node_i* prev;
	node_i* next;
};

struct list_i_ {
	node_i *first;
	size_t size;
};

void init_list(list_i* list);
void free_list(list_i* list);
void list_foreach_i(list_i* list, void(*func)(int));

void list_insert_i(list_i* list, int x, int flag);
void list_delete_i(list_i* list, int x);
int* list_search_i(list_i* list, int x);



#endif