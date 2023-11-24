#ifndef __FIBONACCI_HEAP_H__
#define __FIBONACCI_HEAP_H__


typedef struct omp_node_ {
	struct omp_node_* left;
	struct omp_node_* right;
	int weight;
}omp_node;

#define FIBOTYPE omp_node*

#ifdef FIBOTYPE


#define FIBO_SETVALUE fibo_set_value_ ## omp
#define FIBO_INIT fibo_heap_init_ ## omp
#define FIBO_INSERT fibo_heap_insert_ ## omp
#define FIBO_GETMIN fibo_heap_getmin_ ## omp
#define FIBO_DECREASEKEY fibo_heap_decrease_key_ ## omp
#define FIBO_EXTRACTMIN fibo_heap_extract_min_ ## omp
#define FIBO_CONSOLIDATION fibo_heap_consolidation_ ## omp
#define FIBO_LINK2NODE fibo_heap_link_2_node_ ## omp
#define FIBO_ADDNODE fibo_heap_add_node_ ## omp
#define FIBO_PRUNNING fibo_heap_prunning_ ## omp
#define FIBO_ISEMPTY fibo_heap_is_empty_ ## omp
#define FIBO_DELETE fibo_heap_delete_ ## omp 
#define FIBO_DELETE_HELPER fibo_delete_helper_ ## omp
#define FIBO_DEFAULTLESS fibo_default_less_ ## omp
#define FIBO_DEFAULTGREATER fibo_default_greater_ ## omp
#define FIBONAME_ORIGIN fibo_heap_ ## omp ## _
#define FIBONAME fibo_heap_ ## omp
#define FIBONODE_NAME_ORIGIN fibo_node_ ## omp ## _
#define FIBONODE_NAME fibo_node_ ## omp

typedef struct FIBONODE_NAME_ORIGIN FIBONODE_NAME;
typedef unsigned long long size_t;


struct FIBONODE_NAME_ORIGIN {
	FIBONODE_NAME* parent;
	FIBONODE_NAME* child;
	FIBONODE_NAME* prev;
	FIBONODE_NAME* next;
	FIBOTYPE item;
	size_t degree;
	char marked;
};


typedef struct FIBONAME_ORIGIN {
	FIBONODE_NAME* min;
	int (*comp)(FIBOTYPE, FIBOTYPE);
	size_t node_num;
}FIBONAME;

void FIBO_SETVALUE(FIBONODE_NAME* node, FIBONODE_NAME* const parent_, FIBONODE_NAME* const child_, FIBONODE_NAME* const prev_, FIBONODE_NAME* const next_,
	FIBOTYPE item_, size_t degree_, char marked_);

void FIBO_INIT(FIBONAME* fibo, int (*comp_)(FIBOTYPE, FIBOTYPE));

FIBONODE_NAME* FIBO_INSERT(FIBONAME* fibo, FIBOTYPE item);

FIBOTYPE FIBO_GETMIN(FIBONAME* fibo);

FIBOTYPE FIBO_EXTRACTMIN(FIBONAME* fibo);

void FIBO_DECREASEKEY(FIBONAME* fibo, FIBONODE_NAME* key, FIBOTYPE value);

int FIBO_ISEMPTY(FIBONAME* fibo);

void FIBO_DELETE(FIBONAME* fibo);

int FIBO_DEFAULTLESS(FIBOTYPE arg1, FIBOTYPE arg2);

int FIBO_DEFAULTGREATER(FIBOTYPE arg1, FIBOTYPE arg2);
#endif


#endif