#ifndef __FIBONACCI_HEAP_H__
#define __FIBONACCI_HEAP_H__

#ifdef FIBOTYPE


#define FIBO_SETVALUE fibo_set_value_ ## int
#define FIBO_INIT fibo_heap_init_ ## int
#define FIBO_INSERT fibo_heap_insert_ ## int
#define FIBO_GETMIN fibo_heap_getmin_ ## int
#define FIBO_DECREASEKEY fibo_heap_decrease_key_ ## int
#define FIBO_EXTRACTMIN fibo_heap_extract_min_ ## int
#define FIBO_CONSOLIDATION fibo_heap_consolidation_ ## int
#define FIBO_LINK2NODE fibo_heap_link_2_node_ ## int
#define FIBO_ADDNODE fibo_heap_add_node_ ## int
#define FIBO_PRUNNING fibo_prunning_ ## int
#define FIBO_DELETE fibo_delete_ ## int 
#define FIBO_DELETE_HELPER fibo_delete_helper_ ## int 
#define FIBONAME_ORIGIN fibo_heap_ ## int ## _
#define FIBONAME fibo_heap_ ## int
#define FIBONODE_NAME_ORIGIN fibo_node_ ## int ## _
#define FIBONODE_NAME fibo_node_ ## int

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
	const FIBOTYPE item_, const size_t degree_, const char marked_);

void FIBO_INIT(FIBONAME* fibo, int (*comp_)(FIBOTYPE, FIBOTYPE));

FIBONODE_NAME* FIBO_INSERT(FIBONAME* fibo, const FIBOTYPE item);

FIBOTYPE FIBO_GETMIN(FIBONAME* fibo);

FIBOTYPE FIBO_EXTRACTMIN(FIBONAME* fibo);

void FIBO_DECREASEKEY(FIBONAME* fibo, FIBONODE_NAME* key, const FIBOTYPE value);

void FIBO_DELETE(FIBONAME* fibo);

#endif


#endif