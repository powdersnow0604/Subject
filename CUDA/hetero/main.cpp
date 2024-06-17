#include "kernelCall.h"
#include "unistd.h"

int main() {
	kernelCall();
	sleep(3);
	printf("Host code running on CPU\n");

	return 0;
}