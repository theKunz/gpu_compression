#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include <sys/resource.h>

//TODO: add support for when there's more sets than there are blocks
__global__ void gCompress(uint8_t* fours, uint8_t* threes, size_t pitchfour, size_t pitchthree, size_t setcount) {
	long a = blockIdx.x;

	uint8_t* row4;
	uint8_t* row3; 

	do {
		row4 = fours + a * pitchfour;
		row3 = threes + a * pitchthree;
		
		row3[threadIdx.x] = 0;
		row3[threadIdx.x] = (row4[threadIdx.x]) << (2 + (threadIdx.x * 2));
		row3[threadIdx.x] += (row4[threadIdx.x + 1]) >> (4 - (threadIdx.x * 2));

		a += 65535;
	} while(a < setcount);

}

__global__ void gUncompress(uint8_t* fours, uint8_t* threes, size_t pitchfour, size_t pitchthree, size_t setcount) {
	long a = blockIdx.x;

	uint8_t* row4;
	uint8_t* row3; 

	do {
		row4 = fours + a * pitchfour;
		row3 = threes + a * pitchthree;
		
		row4[threadIdx.x] = 0;
		int i, ander = 0;
		for (i = 0; i < threadIdx.x; i++)
			ander += (48 >> (i*2));
		if (threadIdx.x != 0)
			row4[threadIdx.x] = ((row3[threadIdx.x - 1]) << (4 - (2 * (threadIdx.x - 1)))) & ander;
		if (threadIdx.x != 3)
			row4[threadIdx.x] += ((row3[threadIdx.x]) >> (2 + (threadIdx.x * 2)));

		a += 65535;
	} while (a < setcount);

}

uint8_t getCharVal(char c) {
	if (c >= '0' && c <= '9')
		return c - '0';
	else if (c >= 'a' && c <= 'z')
		return 10 + (c - 'a');
	else if (c >= 'A' && c <= 'Z')
		return 36 + (c - 'A');
	else 
		return 62;
}

char getOriginalVal(uint8_t t) {
	if (t <= 9) 
		return '0' + t;
	else if (t >= 10 && t <= 35)
		return 'a' + t - 10;
	else if (t >= 36 && t <= 61)
		return 'A' + t - 36;
	else
		return '\n';
}

bool fileExists (char* name)
{
   FILE* tmp   = fopen (name, "rb");
   bool exists = (tmp != NULL);
   if (tmp != NULL) fclose (tmp);
   return exists;
}

void compress(char* argv[]);
void uncompress(char* argv[]);
size_t setCount;
size_t overflow;
int main(int argc, char* argv[]) {


	if (argc != 3 || (strcmp(argv[1], "-c") != 0 && strcmp(argv[1], "-u") != 0)) {
		fprintf(stderr, "Usage:\n%s -c filename ....... to compress\n%s -u filename ....... to uncompress\n", argv[0], argv[0]);
		exit(0);
	}
	else if (!fileExists(argv[2])) {
		fprintf(stderr, "File %s does not exist.\n", argv[2]);
		exit(0);
	}

    const rlim_t kStackSize = 64L * 1024L * 1024L;   // min stack size = 64 Mb
    struct rlimit rl;
    int result;
    result = getrlimit(RLIMIT_STACK, &rl);
    if (result == 0)
    {
        if (rl.rlim_cur < kStackSize)
        {
            rl.rlim_cur = kStackSize;
            result = setrlimit(RLIMIT_STACK, &rl);
            if (result != 0)
            {
                fprintf(stderr, "setrlimit returned result = %d\n", result);
                exit(0);
            }
        }
    }

	setCount = 0;
	if (strcmp(argv[1], "-c") == 0)
		compress(argv);
	else {
		uncompress(argv);
	}


	exit(0);
}


void compress(char* argv[]) {
	size_t i;

	char* filename = argv[2];
	char* outfilename = (char*)malloc(sizeof(char) * 64);
	sprintf(outfilename, "%s.bcg", filename);
	FILE* infile = fopen(filename, "r");
	FILE* outfile = fopen(outfilename, "w+");

	long filesize = 0;
	fseek(infile, 0, SEEK_END);
	filesize = ftell(infile);
	fseek(infile, 0, SEEK_SET);

	overflow = filesize % 4;
	setCount = filesize / 4;
	if (overflow > 0)
		setCount++;

	uint8_t threebytes[setCount][3];
	uint8_t fourbytes[setCount][4];

	i = 0;
	while (!feof(infile)) {
		fourbytes[i / 4][i % 4] = getCharVal(fgetc(infile));
		i++;
	}
	fclose(infile);

	size_t pitch3, pitch4;
	uint8_t* garr3;
	uint8_t* garr4;

	cudaMallocPitch((void**)&garr3, &pitch3, (size_t)(3 * sizeof(uint8_t)), setCount);
	cudaMallocPitch((void**)&garr4, &pitch4, (size_t)(4 * sizeof(uint8_t)), setCount);

	cudaMemcpy2D((void*)garr4, pitch4, fourbytes, 4 * sizeof(uint8_t), 4 * sizeof(uint8_t), setCount, cudaMemcpyHostToDevice);
	
	if (setCount <= 65535)
		gCompress<<<setCount, 3>>>(garr4, garr3, pitch4, pitch3, setCount);
	else
		gCompress<<<65535, 3>>>(garr4, garr3, pitch4, pitch3, setCount);

	cudaMemcpy2D(threebytes, 3 * sizeof(uint8_t), garr3, pitch3, 3 * sizeof(uint8_t), setCount, cudaMemcpyDeviceToHost);
	cudaFree(garr3);
	cudaFree(garr4);

	for (i = 0; i < setCount; i++) {
		fprintf(outfile, "%c%c%c", threebytes[i][0], threebytes[i][1], threebytes[i][2]);
	}
	fprintf(outfile, "%i", overflow);

	fclose(outfile);
	free(outfilename);	
}

void uncompress(char* argv[]) {
	size_t i;

	//acquire and handle file overhead
	char* filename = argv[2];
	char* outfilename = (char*)malloc(sizeof(char) * 64);
	sprintf(outfilename, "%s.out", filename);
	FILE* infile = fopen(filename, "r");
	FILE* outfile = fopen(outfilename, "w+");

	//determine file size and number of sets
	long filesize = 0;
	fseek(infile, 0, SEEK_END);
	filesize = ftell(infile) - 1; //don't count end delimiter
	fseek(infile, 0, SEEK_SET);

	setCount = filesize / 3;

	uint8_t threebytes[setCount][3];
	uint8_t fourbytes[setCount][4];

	//get file data
	i = 0;
	while (i < filesize) {
		threebytes[i / 3][i % 3] = (uint8_t)(fgetc(infile));
		i++;
	}
	uint8_t delim = fgetc(infile) - '0';
	fclose(infile);	

	//begin gpu section
	size_t pitch3, pitch4;
	uint8_t* garr3;
	uint8_t* garr4;

	cudaMallocPitch((void**)&garr3, &pitch3, (size_t)(3 * sizeof(uint8_t)), setCount);
	cudaMallocPitch((void**)&garr4, &pitch4, (size_t)(4 * sizeof(uint8_t)), setCount);

	cudaMemcpy2D((void*)garr3, pitch3, threebytes, 3 * sizeof(uint8_t), 3 * sizeof(uint8_t), setCount, cudaMemcpyHostToDevice);

	if (setCount <= 65535)
		gUncompress<<<setCount, 4>>>(garr4, garr3, pitch4, pitch3, setCount);
	else
		gUncompress<<<65535, 4>>>(garr4, garr3, pitch4, pitch3, setCount);	

	cudaMemcpy2D(fourbytes, 4 * sizeof(uint8_t), garr4, pitch4, 4 * sizeof(uint8_t), setCount, cudaMemcpyDeviceToHost);
	cudaFree(garr3);
	cudaFree(garr4);

	for (i = 0; i < setCount; i++) {
		if (delim == 0 || i != setCount - 1) 
			fprintf(outfile, "%c%c%c%c", 
				getOriginalVal(fourbytes[i][0]), 
				getOriginalVal(fourbytes[i][1]), 
				getOriginalVal(fourbytes[i][2]), 
				getOriginalVal(fourbytes[i][3]));
		else {
			int k;
			for (k = 0; k < delim; k++)
				fprintf(outfile, "%c", getOriginalVal(fourbytes[i][k]));
		}
	}

	fclose(outfile);
	free(outfilename);
}


