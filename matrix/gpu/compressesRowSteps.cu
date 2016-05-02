#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>

typedef int32_t Int;


#define NUM_BYTES(n) ((n) * (sizeof(Int)))

__global__ void compressedRow(Int* matrix, Int* rowIndex, Int* nums, Int* cols, Int* rows, Int width) {
	Int* row = ((matrix + (blockIdx.x * width)));
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

__global__ void decompressRow(Int* matrix, Int* nums, Int* cols, Int* rows, Int count, Int width) {
	Int* row = ((matrix + (blockIdx.x * width)));
	Int i;

    *(row + threadIdx.x) = 0;
	for (i = 0; i < count; i++) {
		if (*(rows + i) == blockIdx.x) {
			if (*(cols + i) == threadIdx.x) {
				*(row + threadIdx.x) = *(nums + i);
			}
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
	
	FILE* time_results = fopen("sectionResultsCompress.csv", "a+");
	struct timeval stop, start;

	gettimeofday(&start, NULL); //BEGIN READ-IN SECTION
	Int height, width;
	Int count = 0, rowCount = 0;
	Int i, j;
	Int temp;

	FILE* in = fopen(infile, "rb");

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
	Int rowIndex[height];
	Int* matrix = (Int*)malloc(NUM_BYTES(height * width));
	for (i = 0; i < height; i++) {

		if (i == 0)
			rowIndex[i] = 0;
		else 
			rowIndex[i] = rowIndex[i-1] + rowCount;
		rowCount = 0;

		for (j = 0; j < width; j++) {
			result = fread((void*)&temp, NUM_BYTES(1), 1, in);
			*(matrix + (i * width) + j) = temp;
			if (result != 1) {
				fprintf(stderr, "Reading Error\n");
				exit(1);
			}
			if (*(matrix + (i * width) + j) != 0) {
				count++;
				rowCount++;
			}
		}
	}
	fclose(in);
	
	gettimeofday(&stop, NULL); //END READ-IN SECTION
	long timeMillies = ((stop.tv_usec) + (stop.tv_sec * 1000000)) - ((start.tv_usec) + (start.tv_sec * 1000000));
	fprintf(time_results, "%li,", timeMillies);

	gettimeofday(&start, NULL); //BEGIN CPU-TO-GPU SECTION
	Int* cRowIndex;
	Int* cnums;
	Int* nums;
	Int* cRow; 
	Int* row;
	Int* cCol; 
	Int* col;
	Int* cMatrix;
	//size_t pitch;

	//cudaError_t cerr;
	cudaMalloc((void**)&cRowIndex, NUM_BYTES(height));
	cudaMalloc((void**)&cnums, NUM_BYTES(count));
	cudaMalloc((void**)&cRow, NUM_BYTES(count));
	cudaMalloc((void**)&cCol, NUM_BYTES(count));
	//if (cerr != cudaSuccess)
	//	fprintf(stderr, "Error with cCol malloc: %s", cudaGetErrorString(cerr));
	//cudaMallocPitch((void**)&cMatrix, &pitch, (size_t)NUM_BYTES(width), (size_t)NUM_BYTES(height));
	cudaMalloc((void**)&cMatrix, NUM_BYTES(width * height));
	//if (cerr != cudaSuccess)
	//	fprintf(stderr, "Error with cCol malloc: %s", cudaGetErrorString(cerr));

	cudaMemcpy(cRowIndex, rowIndex, NUM_BYTES(height), cudaMemcpyHostToDevice);
	//cudaMemcpy2D((void*)cMatrix, pitch, matrix, NUM_BYTES(width), NUM_BYTES(width), NUM_BYTES(height), cudaMemcpyHostToDevice);
	cudaMemcpy(cMatrix, matrix, NUM_BYTES(width * height), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	gettimeofday(&stop, NULL); //END CPU-TO-GPU SECTION
	timeMillies = ((stop.tv_usec) + (stop.tv_sec * 1000000)) - ((start.tv_usec) + (start.tv_sec * 1000000));
	fprintf(time_results, "%li,", timeMillies);\

	gettimeofday(&start, NULL); //BEGIN GPU-COMPRESSION SECTION
	compressedRow<<<height, 1>>>(cMatrix, cRowIndex, cnums, cCol, cRow, width);
	cudaDeviceSynchronize();
	gettimeofday(&stop, NULL); //END GPU-COMPRESSION SECTION
	timeMillies = ((stop.tv_usec) + (stop.tv_sec * 1000000)) - ((start.tv_usec) + (start.tv_sec * 1000000));
	fprintf(time_results, "%li,", timeMillies);

	gettimeofday(&start, NULL); //BEGIN GPU-TO-CPU SECTION
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
	cudaDeviceSynchronize();

	gettimeofday(&stop, NULL); //END GPU-TO-CPU SECTION
	timeMillies = ((stop.tv_usec) + (stop.tv_sec * 1000000)) - ((start.tv_usec) + (start.tv_sec * 1000000));
	fprintf(time_results, "%li,", timeMillies);

	gettimeofday(&start, NULL); //BEGIN WRITE-OUT SECTION
	char name[64];
	sprintf(name, "%s.crs", infile);
	FILE* file = fopen(name, "ab+");
	fwrite((void*)&height, NUM_BYTES(1), 1, file);
	fwrite((void*)&width, NUM_BYTES(1), 1, file);
	fwrite((void*)&count, NUM_BYTES(1), 1, file);

	fwrite((void*)&nums[0], NUM_BYTES(1), count, file);
	fwrite((void*)&col[0], NUM_BYTES(1), count, file);
	fwrite((void*)&row[0], NUM_BYTES(1), count, file);

	fclose(file);

	free(nums);
	free(row);
	free(col);
	free(matrix);
	gettimeofday(&stop, NULL); //END WRITE-OUT-SECTION
	timeMillies = ((stop.tv_usec) + (stop.tv_sec * 1000000)) - ((start.tv_usec) + (start.tv_sec * 1000000));
	fprintf(time_results, "%li\n", timeMillies);
	fclose(time_results);	
}

void uncompress(char* infile) {

	FILE* time_results = fopen("sectionResultsUncompress.csv", "a+");

	struct timeval start, stop;

	gettimeofday(&start, NULL); //BEGIN READ-IN SECTION
	Int height, width;
	Int count = 0;
	Int i;
	Int temp;

	FILE* in = fopen(infile, "r");

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

	result = fread((void*)&count, NUM_BYTES(1), 1, in);
	if (result != 1) {
		fprintf(stderr, "Reading Error\n");
		exit(1);
	}
	Int nums[count];
	Int rows[count];
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
	for (i = 0; i < count; i++) {
		result = fread((void*)&temp, NUM_BYTES(1), 1, in);
		rows[i] = temp;
		if (result != 1) {
			fprintf(stderr, "Reading Error\n");
			exit(1);
		}
	}
	fclose(in);

	gettimeofday(&stop, NULL); //END READ-IN SECTION
	long timeMillies = ((stop.tv_usec) + (stop.tv_sec * 1000000)) - ((start.tv_usec) + (start.tv_sec * 1000000));
	fprintf(time_results, "%li,", timeMillies);

	gettimeofday(&start, NULL); //BEGIN CPU-TO-GPU SECTION
	Int* cNums;
	Int* cRows;
	Int* cCols;
	Int* matrix = (Int*)malloc(NUM_BYTES(height * width));
	Int* cMatrix;
	//size_t pitch;

	//cudaError_t cerr;
	cudaMalloc((void**)&cNums, NUM_BYTES(count));
	cudaMalloc((void**)&cRows, NUM_BYTES(count));
	cudaMalloc((void**)&cCols, NUM_BYTES(count));
	//cudaMallocPitch((void**)&cMatrix, &pitch, NUM_BYTES(width), NUM_BYTES(height));
	cudaMalloc((void**)&cMatrix, NUM_BYTES(width * height));

	cudaMemcpy(cNums, nums, NUM_BYTES(count), cudaMemcpyHostToDevice);
	cudaMemcpy(cCols, cols, NUM_BYTES(count), cudaMemcpyHostToDevice);
	cudaMemcpy(cRows, rows, NUM_BYTES(count), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();

	gettimeofday(&stop, NULL); //END CPU-TO-GPU SECTION
	timeMillies = ((stop.tv_usec) + (stop.tv_sec * 1000000)) - ((start.tv_usec) + (start.tv_sec * 1000000));
	fprintf(time_results, "%li,", timeMillies);

	gettimeofday(&start, NULL); //BEGIN GPU-COMPRESSION SECTION

	decompressRow<<<height, width>>>(cMatrix, cNums, cCols, cRows, count, width);
	cudaDeviceSynchronize();

	gettimeofday(&stop, NULL); //END GPU-COMPRESSION SECTION
	timeMillies = ((stop.tv_usec) + (stop.tv_sec * 1000000)) - ((start.tv_usec) + (start.tv_sec * 1000000));
	fprintf(time_results, "%li,", timeMillies);

	gettimeofday(&start, NULL); //BEGIN GPU-TO-CPU SECTION

	//cudaMemcpy2D((void*)matrix, NUM_BYTES(width), cMatrix, pitch, NUM_BYTES(width), NUM_BYTES(height), cudaMemcpyDeviceToHost);
	cudaMemcpy((void*)matrix, cMatrix, NUM_BYTES(width * height), cudaMemcpyDeviceToHost);
	//cudaMemcpy(row, cRow, NUM_BYTES(count), cudaMemcpyDeviceToHost);
	cudaFree(cNums);
	cudaFree(cRows);
	//if (cerr != cudaSuccess)
	//	fprintf(stderr, "Error with cRows free: %s\n", cudaGetErrorString(cerr));
	cudaFree(cCols);
	//if (cerr != cudaSuccess)
	//	fprintf(stderr, "Error with cCols free: %s\n", cudaGetErrorString(cerr));
	cudaFree(cMatrix);

	cudaDeviceSynchronize();

	gettimeofday(&stop, NULL); //END GPU-TO-CPU SECTION
	timeMillies = ((stop.tv_usec) + (stop.tv_sec * 1000000)) - ((start.tv_usec) + (start.tv_sec * 1000000));
	fprintf(time_results, "%li,", timeMillies);

	gettimeofday(&start, NULL); //BEGIN WRITE-OUT SECTION

	char name[64];
	sprintf(name, "%s.out", infile);
	FILE* newfile = fopen(name, "ab+");
	fwrite((void*)&height, NUM_BYTES(1), 1, newfile);
	fwrite((void*)&width, NUM_BYTES(1), 1, newfile);

	//this section could be done without the nested fors, but I already got test results
	//for the end-to-end time tests, and I have to keep it this way for research integrity
	Int j;
	for (i = 0; i < height; i++) {
		for (j = 0; j < width; j++) {
			fwrite((void*)(matrix + (i * width) + j), NUM_BYTES(1), 1, newfile);
		}
	}
	fclose(newfile);
	//free(matrix);

	gettimeofday(&stop, NULL); //END WRITE-OUT-SECTION
	timeMillies = ((stop.tv_usec) + (stop.tv_sec * 1000000)) - ((start.tv_usec) + (start.tv_sec * 1000000));
	fprintf(time_results, "%li\n", timeMillies);
	fclose(time_results);

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


	exit(0);
}

