#define FIBOTYPE double
#include "fibonacci_heap.h"
#include <stdlib.h>
#include <math.h>

static void FIBO_CONSOLIDATION(FIBONAME* fibo);

static void FIBO_LINK2NODE(FIBONODE_NAME* n1, FIBONODE_NAME* n2);

static void FIBO_ADDNODE(FIBONODE_NAME* dest, FIBONODE_NAME* source);

static void FIBO_PRUNNING(FIBONAME* fibo, FIBONODE_NAME* node_);

static void FIBO_DELETE_HELPER(FIBONODE_NAME* node_);

const double GOLDENRATIO = 1.61803398875;
const double LOGGR = 0.481212; //std::log(GOLDENRATIO)


//////////////////////////////////////////////////////////////////////////////////////////////////////////////

void FIBO_SETVALUE(FIBONODE_NAME* node, FIBONODE_NAME* const parent_, FIBONODE_NAME* const child_, FIBONODE_NAME* const prev_, FIBONODE_NAME* const next_,
	const FIBOTYPE item_, const size_t degree_, const char marked_)
{
	node->parent = parent_;
	node->child = child_;
	node->next = next_;
	node->prev = prev_;
	node->parent = parent_;
	node->degree = degree_;
	node->item = item_;
	node->marked = marked_;
}

void FIBO_INIT(FIBONAME* fibo, int (*comp_)(FIBOTYPE, FIBOTYPE))
{
	fibo->min = NULL;
	fibo->node_num = 0;
	fibo->comp = comp_;

}

FIBONODE_NAME* FIBO_INSERT(FIBONAME* fibo, const FIBOTYPE item)
{
	++(fibo->node_num);

	if (fibo->min == NULL) {
		fibo->min = (FIBONODE_NAME*)malloc(sizeof(FIBONODE_NAME));
		FIBO_SETVALUE(fibo->min, NULL, NULL, fibo->min, fibo->min, item, 0, 0);

		return fibo->min;
	}

	FIBONODE_NAME* nnode = (FIBONODE_NAME*)malloc(sizeof(FIBONODE_NAME));
	FIBO_SETVALUE(nnode, NULL, NULL, fibo->min->prev, fibo->min, item, 0, 0);

	fibo->min->prev->next = nnode;
	fibo->min->prev = nnode;

	if (fibo->comp(item, fibo->min->item)) {
		fibo->min = nnode;
	}

	return nnode;
}

FIBOTYPE FIBO_GETMIN(FIBONAME* fibo)
{
	return fibo->min->item;
}

FIBOTYPE FIBO_EXTRACTMIN(FIBONAME* fibo)
{
	if (fibo->min == NULL) return (FIBOTYPE)0;

	if (fibo->node_num == 1) {
		fibo->node_num = 0;
		FIBOTYPE temp = fibo->min->item;
		free(fibo->min);
		fibo->min = NULL;
		return temp;
	}

	FIBONODE_NAME* prev_min = fibo->min;
	FIBOTYPE min_value = fibo->min->item;

	if (fibo->min->child != NULL) {
		FIBO_LINK2NODE(fibo->min, fibo->min->child);
		fibo->min->child = NULL;
	}

	prev_min->prev->next = prev_min->next;
	prev_min->next->prev = prev_min->prev;

	fibo->min = prev_min->next;

	free(prev_min);

	--(fibo->node_num);

	FIBO_CONSOLIDATION(fibo);

	return min_value;
}

void FIBO_DECREASEKEY(FIBONAME* fibo, FIBONODE_NAME* key, const FIBOTYPE value)
{
	if (fibo->comp(key->item, value)) return;

	key->item = value;

	if (key->parent != NULL) {

		if (key->next != key) {
			key->next->prev = key->prev;
			key->prev->next = key->next;
			if (key->parent->child == key) {
				key->parent->child = key->next;
			}
		}
		else {
			key->parent->child = NULL;
		}

		FIBO_ADDNODE(fibo->min, key);

		if (fibo->comp(key->item, fibo->min->item)) {
			fibo->min = key;
		}

		FIBO_PRUNNING(fibo, key->parent);
		key->parent = NULL;
	}
	else {
		if (fibo->comp(key->item, fibo->min->item)) {
			fibo->min = key;
		}
	}
}

void FIBO_CONSOLIDATION(FIBONAME* fibo)
{
	size_t degree, max_degree = (size_t)(log((double)fibo->node_num) / LOGGR) + 1;
	FIBONODE_NAME* curr = fibo->min, * bigger, * smaller, * candidate, * last = fibo->min->prev;
	char running = 1;
	FIBONODE_NAME** vec = (FIBONODE_NAME**)calloc(max_degree + 1, sizeof(FIBONODE_NAME*));
	if (vec == NULL) return;

	do {
		if (fibo->comp(curr->item, fibo->min->item)) {
			fibo->min = curr;
		}

		if (curr == last) running = 0;

		curr->parent = NULL;
		degree = curr->degree;
		candidate = curr;
		curr = curr->next;

		while (degree <= max_degree) {
			if (vec[degree] == NULL) {
				vec[degree] = candidate;
				break;
			}

			if (fibo->comp(vec[degree]->item, candidate->item)) {
				smaller = vec[degree];
				bigger = candidate;
			}
			else {
				smaller = candidate;
				bigger = vec[degree];
			}

			if (smaller->child == NULL) {
				bigger->next->prev = bigger->prev;
				bigger->prev->next = bigger->next;

				smaller->child = bigger;
				bigger->parent = smaller;
				bigger->next = bigger->prev = bigger;
			}
			else {
				bigger->next->prev = bigger->prev;
				bigger->prev->next = bigger->next;

				bigger->parent = smaller;

				FIBO_ADDNODE(smaller->child, bigger);
			}

			vec[degree] = NULL;
			++(smaller->degree);
			++degree;
			candidate = smaller;
		}
	} while (running);

	free(vec);
}

void FIBO_PRUNNING(FIBONAME* fibo, FIBONODE_NAME* node_)
{
	if (node_->marked) {
		node_->marked = 0;

		if (node_->parent != NULL) {

			if (node_->next != node_) {
				node_->next->prev = node_->prev;
				node_->prev->next = node_->prev;
			}

			FIBO_ADDNODE(fibo->min, node_);

			if (fibo->comp(node_->item, fibo->min->item)) {
				fibo->min = node_;
			}

			FIBO_PRUNNING(fibo, node_->parent);
			node_->parent = NULL;
		}
		else {
			if (fibo->comp(node_->item, fibo->min->item)) {
				fibo->min = node_;
			}
		}
	}
	else {
		node_->marked = 1;
	}
}

void FIBO_LINK2NODE(FIBONODE_NAME* n1, FIBONODE_NAME* n2)
{
	if (n1 == NULL) return;
	if (n2 == NULL) return;

	FIBONODE_NAME* temp = n1->prev;
	n1->prev->next = n2;
	n2->prev->next = n1;
	n1->prev = n2->prev;
	n2->prev = temp;
}

void FIBO_ADDNODE(FIBONODE_NAME* dest, FIBONODE_NAME* source)
{
	if (dest == NULL) return;
	if (source == NULL) return;

	source->next = dest;
	source->prev = dest->prev;
	dest->prev->next = source;
	dest->prev = source;
}

void FIBO_DELETE(FIBONAME* fibo)
{
	if (fibo->min == NULL) return;
	FIBONODE_NAME* last = fibo->min->prev->next;
	FIBONODE_NAME* curr = fibo->min;

	do {
		FIBO_DELETE_HELPER(curr->child);
		curr = curr->next;
		free(curr->prev);
	} while (last == curr);
}

void FIBO_DELETE_HELPER(FIBONODE_NAME* node_)
{
	if (node_ == NULL) return;

	FIBONODE_NAME* last = node_->prev->next;
	FIBONODE_NAME* curr = node_;

	do {
		FIBO_DELETE_HELPER(curr->child);
		curr = curr->next;
		free(curr->prev);
	} while (last == curr);
}

int FIBO_DEFAULTLESS(FIBOTYPE arg1, FIBOTYPE arg2)
{
	return arg1 < arg2;
}

int FIBO_DEFAULTGREATER(FIBOTYPE arg1, FIBOTYPE arg2)
{
	return arg1 > arg2;
}