#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdbool.h>

#define NUM_BYTES(n) ((n) * (sizeof(int32_t)))

bool fileExists (char* name) {
   FILE* tmp   = fopen (name, "rb");
   bool exists = (tmp != NULL);
   if (tmp != NULL) fclose (tmp);
   return exists;
}

int main(int argc, char* argv[]) {

	if (argc != 2) {
		fprintf(stderr, "Usage:\n%s filename", argv[0]);
		exit(1);
	}
	else if (!fileExists(argv[1])) {
		fprintf(stderr, "File %s does not exist.\n", argv[2]);
		exit(1);
	}

	FILE* in = fopen(argv[1], "r");

	int32_t height;
	int32_t width;

	int32_t result = fread((void*)&height, NUM_BYTES(1), 1, in);
	if (result != 1) {
		fprintf(stderr, "Reading Error\n");
		exit(1);
	}
	result = fread((void*)&width, NUM_BYTES(1), 1, in);
	if (result != 1) {
		fprintf(stderr, "Reading Error\n");
		exit(1);
	}

	int i;
	int32_t num;
	for (i = 0 ; i < height * width; i++) {
		result = fread((void*)&num, NUM_BYTES(1), 1, in);
		if (result != 1) {
			fprintf(stderr, "Reading Error\n");
			exit(1);
		}
		printf("%i, ", num);
		if ((i + 1) % height == 0)
			printf("\n");
	}
	exit(0);
}