#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

typedef int32_t Int;


#define NUM_BYTES(n) ((n) * (sizeof(Int)))

__global__ void compressedRow(Int* matrix, Int* rowIndex, Int* nums, Int* cols, Int* rows, size_t pitch, Int width) {
	Int* row = matrix + blockIdx.x * pitch;
	Int offset = *(rowIndex + blockIdx.x);

	Int i = 0;
	for (; i < width; i++) {
		if (*(row + i) != 0) {
			*(nums + offset) = *(row + i);
			*(cols + offset) = i;
			*(rows + offset) = blockIdx.x;
			offset++;
		}
	}
}

bool fileExists (char* name) {
   FILE* tmp   = fopen (name, "rb");
   bool exists = (tmp != NULL);
   if (tmp != NULL) fclose (tmp);
   return exists;
}

void compress(Int height, Int width, FILE* in, char* infile) {
	
	Int count = 0, rowIndex[height];
	Int matrix[height][width];
	Int i = 0, j = 0;
	Int result;

	for (; i < height; i++) {

		if (i == 0)
			rowIndex[i] = 0;
		else 
			rowIndex[i] = rowIndex[i-1] + count;

		for (; j < width; j++) {
			result = fread((void*)&matrix[i][j], NUM_BYTES(1), 1, in);
			if (result != 1) {
				fprintf(stderr, "Reading Error\n");
				exit(1);
			}
			matrix[i][j] != 0 ? count++ : count += 0;
		}
	}
	fclose(in);

	Int* cRowIndex;
	Int* cnums, *nums;
	Int* cRow, *row;
	Int* cCol, *col;
	Int* cMatrix;
	size_t pitch;

	cudaMalloc((void**)&cRowIndex, NUM_BYTES(height));
	cudaMalloc((void**)&cnums, NUM_BYTES(count));
	cudaMalloc((void**)&cRow, NUM_BYTES(count));
	cudaMalloc((void**)&cCol, NUM_BYTES(count));
	cudaMallocPitch((void**)&cMatrix, &pitch, (size_t)NUM_BYTES(width), (size_t)NUM_BYTES(height));

	cudaMemcpy(cRowIndex, rowIndex, NUM_BYTES(height), cudaMemcpyHostToDevice);
	cudaMemcpy2D((void*)cMatrix, pitch, &matrix[0][0], NUM_BYTES(width), NUM_BYTES(width), NUM_BYTES(height), cudaMemcpyHostToDevice);

	compressedRow<<<height, 1>>>(cMatrix, cRowIndex, cnums, cCol, cRow, pitch, width);

	nums = (Int*)malloc(NUM_BYTES(count));
	row = (Int*)malloc(NUM_BYTES(count));
	col = (Int*)malloc(NUM_BYTES(count));

	cudaMemcpy(nums, cnums, NUM_BYTES(count), cudaMemcpyDeviceToHost);
	cudaMemcpy(col, cCol, NUM_BYTES(count), cudaMemcpyDeviceToHost);
	cudaMemcpy(row, cRow, NUM_BYTES(count), cudaMemcpyDeviceToHost);

	cudaFree(cRowIndex);
	cudaFree(cnums);
	cudaFree(cRow);
	cudaFree(cCol);	
	cudaFree(cMatrix);

	char name[64];
	sprintf(name, "%s.crs", infile);
	FILE* file = fopen(name, "ab+");
	fwrite((void*)&height, sizeof(height), 1, file);
	fwrite((void*)&width, sizeof(width), 1, file);

	fwrite((void*)&nums[0], NUM_BYTES(1), count, file);
	fwrite((void*)&col[0], NUM_BYTES(1), count, file);
	fwrite((void*)&row[0], NUM_BYTES(1), count, file);

	fclose(file);

	free(nums);
	free(row);
	free(col);
}

void uncompress() {
	printf("uncompress to be done\n");
}


int main(int argc, char* argv[]) {

	if (argc != 3 || (strcmp(argv[1], "-c") != 0 && strcmp(argv[1], "-u") != 0)) {
		fprintf(stderr, "Usage:\n%s -c filename ....... to compress\n%s -u filename ....... to uncompress\n", argv[0], argv[0]);
		exit(1);
	}
	else if (!fileExists(argv[2])) {
		fprintf(stderr, "File %s does not exist.\n", argv[2]);
		exit(1);
	}

	FILE* in = fopen(argv[2], "r");

	Int height;
	Int width;

	Int result = fread((void*)&height, NUM_BYTES(1), 1, in);
	if (result != 1) {
		fprintf(stderr, "Reading Error\n");
		exit(1);
	}
	result = fread((void*)&width, NUM_BYTES(1), 1, in);
	if (result != 1) {
		fprintf(stderr, "Reading Error\n");
		exit(1);
	}

	if (strcmp(argv[1], "-c") == 0) {
		compress(height, width, in, argv[2]);
	}
	else {
		uncompress();
	}



	exit(0);
}

