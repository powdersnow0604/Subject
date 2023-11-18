#define VECTORTYPE int
#include "vector.h"
#include  <stdlib.h>


VECTORTYPE* VECTOR_INIT(vec_manager* q, size_t size)
{
	q->capacity = size;
	q->num = 0;
	return (VECTORTYPE*)malloc(sizeof(VECTORTYPE) * size);
}


void VECTOR_PUSH_BACK(VECTORTYPE* vec, vec_manager* q, VECTORTYPE item)
{
	if (q->capacity == q->num) {
		void* tmp = vec;
		(q->capacity) <<= 1;
		vec = realloc(vec, q->capacity);
		if (vec == NULL) {
			free(tmp);
			return;
		}
	}

	vec[(q->num)++] = item;
}


void VECTOR_RESIZE(VECTORTYPE* vec, vec_manager* q, size_t nsize)
{
	if (q->num >= nsize) {
		q->num = nsize;
		return;
	}
	else if (q->num < nsize) {
		if (nsize < q->capacity) {
			for (q->num; q->num != nsize; ++(q->num)) {
				vec[(q->num)++] = (VECTORTYPE)0;
			}
		}
		else {
			void* tmp = vec;
			size_t temp_size = (q->capacity) <<= 1;
			q->capacity = temp_size > nsize ? temp_size : nsize * 2;
			vec = realloc(vec, q->capacity);
			if (vec  == NULL) {
				free(tmp);
				return;
			}

			for (q->num; q->num != nsize; ++(q->num)) {
				vec[(q->num)++] = (VECTORTYPE)0;
			}
		}
	}
}
