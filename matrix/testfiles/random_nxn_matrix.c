#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

unsigned long size;

int main(int argc, char* argv[]) {
	if (argc != 4) {
		fprintf(stderr, "Usage: %s <height> <width> <low|med|high>", argv[0]);
		return EXIT_FAILURE;
	}
	/////////////////////////////
	char* endptr_height;
	char* endptr_width;
	int height = strtol(argv[1], &endptr_height, 10);
	int width = strtol(argv[2], &endptr_width, 10);
	if (height <= 0 || width <= 0 || height > 32 || width > 32) {
		fprintf(stderr, "width and height must be greater than 0 and no greater than 32\n");
		return EXIT_FAILURE;
	}
	////////////////////////////
	int sparsity;
	if (!strcmp(argv[3], "low")) {
		sparsity = 2;
	}
	else if (!strcmp(argv[3], "med")) {
		sparsity = 4;
	}
	else if (!strcmp(argv[3], "high")) {
		sparsity = 5;
	}
	else {
		fprintf(stderr, "Usage: %s <height> <width> <low|med|high>", argv[0]);
		return EXIT_FAILURE;		
	}
	printf("sparsity: 1/%i are non-zero\n", sparsity);
	//////////////////////////////
	int newMatrix[height][width];
	srand(time(NULL));

	long i, j, count = 0;
	printf("Generating matrix:\n");
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			newMatrix[i][j] = rand() % 10;
			count++;
			if (rand() % sparsity != 0) {
				newMatrix[i][j] = 0;
				count--;
			}
			printf("%i,", newMatrix[i][j]);
		}
		printf("\n");
	}
	printf("Actual ratio: %li/%i\n", count, width * height);
	///////////////////////////////
	char name[32];
	sprintf(name, "matrix_%i_%i_%s", width, height, argv[3]);
	FILE* file = fopen(name, "ab+");
	fwrite((void*)&height, sizeof(height), 1, file);
	fwrite((void*)&width, sizeof(width), 1, file);

	for (i = 0; i < height; i++) {
		fwrite((void*)&newMatrix[i][0], sizeof(newMatrix[i][0]), width, file);
	}

	fclose(file);
	printf("Success! File %s created\n", name);
	return 0;
}
