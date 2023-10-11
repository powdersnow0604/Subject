#include "list.h"
#include <stdlib.h>


//declaration
static void free_list_(node_i* node);


//definition
void init_list(list_i* list)
{
	list->first = NULL;
	list->size = 0;
}
void free_list(list_i* list)
{
	if (list->first == NULL) return;
	free_list_(list->first);
	free(list->first);
	list->size = 0;
}
static void free_list_(node_i* node)
{
	if (node->next == NULL) return;

	free_list_(node->next);
	free(node->next);
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

	//degenerate case: enpty list
	if (curr == NULL) goto insert_in_empty_list;


	//degenerate case: insert front
	if (flag) {
		if (curr->element > x) goto insert_as_first;
	}
	else {
		if (curr->element < x) goto insert_as_first;
	}


	//generate case
	for (; curr->next != NULL; curr = curr->next) {
		if (flag) {
			if (curr->next->element > x) break;
		}
		else {
			if (curr->next->element < x) break;
		}
	}

	//degenerate case: insert end
	if (curr->next == NULL) goto insert_as_last;


insert_as_generate_case:

	nnode->next = curr->next;
	nnode->prev = curr;
	if(nnode->next != NULL) nnode->next->prev = nnode;
	curr->next = nnode;

	return;

insert_in_empty_list:

	nnode->next = curr;
	nnode->prev = NULL;
	list->first = nnode;

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


void list_foreach_i(list_i* list, void(*func)(int))
{
	node_i* curr = list->first;
	for (; curr != NULL; curr = curr->next) {
		func(curr->element);
	}
}