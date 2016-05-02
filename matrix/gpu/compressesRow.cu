#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

typedef int32_t Int;


#define NUM_BYTES(n) ((n) * (sizeof(Int)))

__global__ void compressedRow(Int* matrix, Int* rowIndex, Int* nums, Int* cols, Int width) {
	Int* row = ((matrix + (blockIdx.x * width)));
	Int offset = *(rowIndex + blockIdx.x);

	Int i = 0;
	for (; i < width; i++) {
		if (*(row + i) != 0) {
			*(nums + offset) = *(row + i);
			*(cols + offset) = i;
			offset++;
		}
	}
}

__global__ void decompressRow(Int* matrix, Int* nums, Int* cols, Int* rows, Int width) {
	Int* row = ((matrix + (blockIdx.x * width)));
	Int i;

	Int offset = *(rows + blockIdx.x);
	Int num_in_row = *(rowIndex + blockIdx.x + 1) - *(rowIndex + blockIdx.x) //make sure row index is height + 1 to accomidate last row
	Int* val_offset = nums + offset;
	Int* col_offset = cols + offset

	*(row + threadIdx.x) = 0;
	for (i = 0; i < num_in_row; i++) {
		if (*(col_offset + i) == threadIdx.x) {
			*(row + threadIdx.x) = *(val_offset + i);
			break;
		}
	}

}

bool fileExists (char* name) {
   FILE* tmp   = fopen (name, "rb");
   bool exists = (tmp != NULL);
   if (tmp != NULL) fclose (tmp);
   return exists;
}

void compress(char* infile) {
	
	Int height, width;
	Int i, j;
	Int result;
	Int temp, tempC, tempR;

	FILE* in = fopen(infile, "r");

	result = fread((void*)&height, NUM_BYTES(1), 1, in);
	if (result != 1) {
		fprintf(stderr, "Reading Error\n");
		exit(1);
	}
	result = fread((void*)&width, NUM_BYTES(1), 1, in);
	if (result != 1) {
		fprintf(stderr, "Reading Error\n");
		exit(1);
	}
	Int count = 0, rowCount = 0, rowIndex[height + 1];
	for (i = 0; i < height + 1; i++) {
		rowIndex[i] = 0;
	}
	Int* matrix = (Int*)malloc(NUM_BYTES(height * width));

	for (i = 0; i < width * height; i++) {
		result = fread((void*)&tempR, NUM_BYTES(1), 1, in);
		if (result != 1) {
			fprintf(stderr, "Reading Error\n");
			exit(1);
		}
		result = fread((void*)&tempC, NUM_BYTES(1), 1, in);
		if (result != 1) {
			fprintf(stderr, "Reading Error\n");
			exit(1);
		}
		result = fread((void*)&temp, NUM_BYTES(1), 1, in);
		if (result != 1) {
			fprintf(stderr, "Reading Error\n");
			exit(1);
		}	
		*(matrix + (tempR * width) + tempC) = temp;
		if (temp != 0) {
			for (j = tempR + 1, j < height + 1; j++) {
				rowIndex[j]++;
			}
			count++;
		}
	}
	fclose(in);

	Int* cRowIndex;
	Int* cnums;
	Int* nums;
	Int* row;
	Int* cCol; 
	Int* col;
	Int* cMatrix;

	cudaMalloc((void**)&cRowIndex, NUM_BYTES(height));
	cudaMalloc((void**)&cnums, NUM_BYTES(count));
	cudaMalloc((void**)&cCol, NUM_BYTES(count));
	cudaMalloc((void**)&cMatrix, NUM_BYTES(width * height));

	cudaMemcpy(cRowIndex, rowIndex, NUM_BYTES(height+1), cudaMemcpyHostToDevice);
	cudaMemcpy(cMatrix, matrix, NUM_BYTES(width * height), cudaMemcpyHostToDevice);

	compressedRow<<<height, 1>>>(cMatrix, cRowIndex, cnums, cCol, width);

	nums = (Int*)malloc(NUM_BYTES(count));
	col = (Int*)malloc(NUM_BYTES(count));

	cudaMemcpy(nums, cnums, NUM_BYTES(count), cudaMemcpyDeviceToHost);
	cudaMemcpy(col, cCol, NUM_BYTES(count), cudaMemcpyDeviceToHost);

	cudaFree(cRowIndex);
	cudaFree(cnums);
	cudaFree(cCol);	
	cudaFree(cMatrix);

	char name[64];
	sprintf(name, "%s.crs", infile);
	FILE* file = fopen(name, "wb+");
	fwrite((void*)&height, NUM_BYTES(1), 1, file);
	fwrite((void*)&width, NUM_BYTES(1), 1, file);

	fwrite((void*)&rowIndex[0], NUM_BYTES(1), height + 1, file);
	fwrite((void*)&count, NUM_BYTES(1), 1, file);
	fwrite((void*)&nums[0], NUM_BYTES(1), count, file);
	fwrite((void*)&col[0], NUM_BYTES(1), count, file);

	fclose(file);

	free(nums);
	free(col);
	free(matrix);
}

void uncompress(char* infile) {

	Int height, width, count;
	Int i, j;
	Int result;
	Int temp;

	FILE* in = fopen(infile, "r");

	result = fread((void*)&height, NUM_BYTES(1), 1, in);
	if (result != 1) {
		fprintf(stderr, "Reading Error\n");
		exit(1);
	}
	result = fread((void*)&width, NUM_BYTES(1), 1, in);
	if (result != 1) {
		fprintf(stderr, "Reading Error\n");
		exit(1);
	}
	Int rowIndex[height + 1];
	result = fread((void*)&rowIndex[0], NUM_BYTES(1), height + 1, in);
	if (result != 1) {
		fprintf(stderr, "Reading Error\n");
		exit(1);
	}
	result = fread((void*)&count, NUM_BYTES(1), 1, in);
	if (result != 1) {
		fprintf(stderr, "Reading Error\n");
		exit(1);
	}
	Int nums[count];
	Int cols[count];

	for (i = 0; i < count; i++) {
		result = fread((void*)&temp, NUM_BYTES(1), 1, in);
		nums[i] = temp;
		if (result != 1) {
			fprintf(stderr, "Reading Error\n");
			exit(1);
		}
	}
	for (i = 0; i < count; i++) {
		result = fread((void*)&temp, NUM_BYTES(1), 1, in);
		cols[i] = temp;
		if (result != 1) {
			fprintf(stderr, "Reading Error\n");
			exit(1);
		}
	}
	fclose(in);

	Int* cNums;
	Int* cRowIndex;
	Int* cCols;
	Int* matrix = (Int*)malloc(NUM_BYTES(height * width));
	Int* cMatrix;

	cudaMalloc((void**)&cNums, NUM_BYTES(count));
	cudaMalloc((void**)&cRowIndex, NUM_BYTES(height + 1));
	cudaMalloc((void**)&cCols, NUM_BYTES(count));
	cudaMalloc((void**)&cMatrix, NUM_BYTES(width * height));

	cudaMemcpy(cNums, nums, NUM_BYTES(count), cudaMemcpyHostToDevice);
	cudaMemcpy(cCols, cols, NUM_BYTES(count), cudaMemcpyHostToDevice);
	cudaMemcpy(cRowIndex, rowIndex, NUM_BYTES(height + 1), cudaMemcpyHostToDevice);

	decompressRow<<<height, width>>>(cMatrix, cNums, cCols, cRowIndex, width);

	cudaMemcpy((void*)matrix, cMatrix, NUM_BYTES(width * height), cudaMemcpyDeviceToHost);
	
	cudaFree(cNums);
	cudaFree(cRowIndex);
	cudaFree(cCols);
	cudaFree(cMatrix);

	char name[64];
	sprintf(name, "%s.out", infile);
	FILE* newfile = fopen(name, "wb+");
	fwrite((void*)&height, NUM_BYTES(1), 1, newfile);
	fwrite((void*)&width, NUM_BYTES(1), 1, newfile);
	for (i = 0; i < height; i++) {
		for (j = 0; j < width, j++) {
			fwrite((void*)&i, NUM_BYTES(1), 1, newfile);
			fwrite((void*)&j, NUM_BYTES(1), 1, newfile);
			fwrite((void*)(matrix + (i * width) + j), NUM_BYTES(1), 1, newfile);
		}
	}

	fclose(newfile);
	free(matrix);
	return;
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

	if (strcmp(argv[1], "-c") == 0) {
		compress(argv[2]);
	}
	else {
		uncompress(argv[2]);
	}
	printf("HERE\n");


	exit(0);
}

