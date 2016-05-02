#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned long size;

int main(int argc, char* argv[]) {
	if (argc != 4) {
		fprintf(stderr, "Usage: %s <height> <width> <%c sparsity>\n", argv[0], '%');
		return EXIT_FAILURE;
	}
	/////////////////////////////
	char* endptr_height;
	char* endptr_width;
	char* endptr_sparsity;
	int height = strtol(argv[1], &endptr_height, 10);
	int width = strtol(argv[2], &endptr_width, 10);
	int sparsity = strtol(argv[3], &endptr_sparsity, 10);
	if (height <= 0 || width <= 0 || sparsity < 0 || sparsity > 100) {
		fprintf(stderr, "width and height must be greater than 0, sparsity must be at least 0 and no greater than 100\n");
		return EXIT_FAILURE;
	}

	printf("sparsity: %i%c are non-zero\n", 100 - sparsity, '%');
	//////////////////////////////
	int newMatrix[height][width];
	srand(time(NULL));

	long i, j;
	int count = height * width;
	printf("Generating matrix..\n");
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			newMatrix[i][j] = (rand() % 10) + 1;
			int isZero = rand() % 100;
			if (isZero < sparsity) {
				newMatrix[i][j] = 0;
				count--;
			}
		}
	}
	printf("NonZero to Total (~100 - sparsity) ratio: %i/%i\n", count, width * height);
	///////////////////////////////
	char name[32];
	sprintf(name, "matrix_%i_%i_%s", width, height, argv[3]);
	FILE* file = fopen(name, "wb+");
	fwrite((void*)&height, sizeof(height), 1, file);
	fwrite((void*)&width, sizeof(width), 1, file);

	for (i = 0; i < height; i++) {
		fwrite((void*)&newMatrix[i][0], sizeof(newMatrix[i][0]), width, file);
	}

	fclose(file);
	printf("Success! File %s created\n", name);
	return 0;
}
