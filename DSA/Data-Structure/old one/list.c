#include "list.h"
#include <stdlib.h>


//declaration
static void delete_list_(node_i* node);


//definition
void init_list(list_i* list)
{
	list->first = NULL;
	list->size = 0;
}
void free_list(list_i* list)
{
	delete_list_(list->first);
	free(list->first);
	list->size = 0;
}
static void delete_list_(node_i* node)
{
	if (node->next == NULL) return;

	delete_list_(node->next);
	free(node->next);
}
void list_foreach(list_i* list, void(*func)(int))
{
	node_i* curr = list->first;
	for (; curr != NULL; curr = curr->next) {
		func(curr->element);
	}
}

int* list_search_i(list_i* list, int x)
{
	node_i* curr = list->first;
	for (; curr != NULL; curr = curr->next) {
		if (curr->element == x) return &(curr->element);
	}

	return NULL;
}

void list_insert_i(list_i* list, int x, int flag)
{
	//create nnode
	node_i* nnode;
	nnode = (node_i*)malloc(sizeof(node_i));
	if (nnode == NULL) return;
	nnode->element = x;
	
	//increase size
	++(list->size);

	node_i* curr = list->first;

	//degenerate case: insert front
	if (flag) {
		if (curr->element > x) goto insert_as_first;
	}
	else {
		if (curr->element < x) goto insert_as_first;
	}


	for (; curr->next != NULL; curr = curr->next) {
		if (flag) {
			if (curr->next->element > x) break;
		}
		else {
			if (curr->next->element < x) break;
		}
	}

	//degenerate case: insert end
	if (curr->next != NULL) goto insert_as_last;

	nnode->next = curr->next;
	nnode->prev = curr;
	nnode->next->prev = nnode;
	curr->next = nnode;

	return;

insert_as_first:

	nnode->next = curr;
	nnode->prev = NULL;
	list->first = nnode;
	curr->prev = nnode;
	
	return;

insert_as_last:

	nnode->next = NULL;
	nnode->prev = curr;
	curr->next = nnode;

	return;
}
void list_delete_i(list_i* list, int x)
{

}
