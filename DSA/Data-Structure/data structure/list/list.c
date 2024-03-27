#define LISTTYPE int
#include "list.h"

#define NULL ((void*)0)

void LIST_INIT(LIST_NAME* list, size_t size)
{
	mps_init(&list->mm, sizeof(NODE_NAME), size);
	list->num = 0;
	list->front = NULL;
	list->back = NULL;
}

static void list_initial_push(LIST_NAME* list, LISTTYPE item)
{
	list->front = list->back = (NODE_NAME*)mps_alloc_one(&list->mm);
	list->front->element = item;
	list->front->prev = NULL;
	list->front->next = NULL;
}

void LIST_PUSH_BACK(LIST_NAME* list, LISTTYPE item)
{
	if (list->num == 0) {
		list_initial_push(list, item);
		++(list->num);
		return;
	}
	list->back->next = mps_alloc_one(&list->mm);
	list->back->next->prev = list->back;
	list->back = list->back->next;
	list->back->element = item;
	list->back->next = NULL;

	++(list->num);
}

LISTTYPE LIST_POP_FRONT(LIST_NAME* list)
{
	if (list->num == 0) return (LISTTYPE)0;
	NODE_NAME* tmp = list->front;
	//LISTTYPE item = tmp->element;

	list->front = list->front->next;
	if(list->num != 1) list->front->prev = NULL;

	mps_free_one(&list->mm, tmp);

	--(list->num);

	return tmp->element;
}

void LIST_PUSH_FRONT(LIST_NAME* list, LISTTYPE item)
{
	if (list->num == 0) {
		list_initial_push(list, item);
		++(list->num);
		return;
	}

	list->front->prev = mps_alloc_one(&list->mm);
	list->front->prev->next = list->front;
	list->front = list->front->prev;
	list->front->element = item;
	list->front->prev = NULL;

	++(list->num);
}

LISTTYPE LIST_POP_BACK(LIST_NAME* list)
{
	if (list->num == 0) return (LISTTYPE)0;
	NODE_NAME* tmp = list->back;
	//LISTTYPE item = tmp->element;

	list->back = list->back->prev;
	if(list->num != 1) list->back->next = NULL;

	mps_free_one(&list->mm, tmp);

	--(list->num);

	return tmp->element;
}

void LIST_FREE(LIST_NAME* list)
{
	list->num = 0;
	list->front = NULL;
	list->back = NULL;
	mps_destroy(&list->mm);
}

int LIST_IS_EMPTY(LIST_NAME* list)
{
	return list->num == 0;
}

void LIST_CLEAR(LIST_NAME* list)
{
	NODE_NAME* curr = list->front;
	//void* tmp;
	for (size_t i = list->num; i > 0; --i) {
		//tmp = curr->next;
		mps_free_one(&list->mm, curr);
		//curr = tmp;

		curr = curr->next;
	}

	list->num = 0;
}