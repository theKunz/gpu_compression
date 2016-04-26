#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>

bool fileExists (char* name)
{
   FILE* tmp   = fopen (name, "rb");
   bool exists = (tmp != NULL);
   if (tmp != NULL) fclose (tmp);
   return exists;
}

void compress(char* argv[]);
void uncompress(char* argv[]);
//out must be length 3 and already initialized
//the bytes are no larger than 63 inclusive to ensure 6 bits
void combine6bytes(uint8_t* byte1, uint8_t* byte2, uint8_t* byte3, uint8_t* byte4, uint8_t* out) {
	uint8_t* out1 = out;
	uint8_t* out2 = out + 1;
	uint8_t* out3 = out + 2;

	*out1 = 0;
	*out1 = ((*byte1) << 2);
	*out1 += ((*byte2) >> 4);

	*out2 = 0;
	*out2 = ((*byte2) << 4);
	*out2 += ((*byte3) >> 2);

	*out3 = 0;
	*out3 = ((*byte3) << 6);
	*out3 += *byte4;
}

void separate6bytes(uint8_t* byte1, uint8_t* byte2, uint8_t* byte3, uint8_t* byte4, uint8_t* in) {
	uint8_t* in1 = in;
	uint8_t* in2 = in + 1;
	uint8_t* in3 = in + 2;

	*byte1 = 0;

	*byte1 = ((*in1) >> 2);


	*byte2 = 0;
	*byte2 = (((*in1) << 4) & 48); //0011 0000
	*byte2 += ((*in2) >> 4);

	*byte3 = 0;
	*byte3 = (((*in2) << 2) & 60); //0011 1100
	*byte3 += ((*in3) >> 6);

	*byte4 = 0;
	*byte4 = ((*in3) & 63); //0011 1111,

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
	if (t >= 0 && t <= 9) 
		return '0' + t;
	else if (t >= 10 && t <= 35)
		return 'a' + t - 10;
	else if (t >= 36 && t <= 61)
		return 'A' + t - 36;
	else
		return '\n';
}

int main(int argc, char* argv[]) {


	if (argc != 3 || (strcmp(argv[1], "-c") != 0 && strcmp(argv[1], "-u") != 0) || !fileExists(argv[2])) {
		printf("Usage:\n%s -c filename ....... to compress\n%s -u filename ....... to uncompress\n", argv[0], argv[0]);
		exit(0);
	}

	if (strcmp(argv[1], "-c") == 0)
		compress(argv);
	else {
		uncompress(argv);
	}


	exit(0);
}

void compress(char* argv[]) {
	uint8_t i;

	char* filename = argv[2];
	char* outfilename = malloc(sizeof(char) * 64);
	sprintf(outfilename, "%s.bc", filename);
	FILE* infile = fopen(filename, "r");
	FILE* outfile = fopen(outfilename, "w+");

	char store;
	uint8_t* threebytes = malloc(sizeof(uint8_t) * 3);
	uint8_t* fourbytes = malloc(sizeof(uint8_t) * 4);
	i = 0;
	int printcount = 0;
	while (!feof(infile)) {
		store = fgetc(infile);
		*(fourbytes + i) = getCharVal(store);
		if (!feof(infile)) {
			if (i == 3) {
				i = 0;
				printcount++;
				combine6bytes(fourbytes, fourbytes + 1, fourbytes + 2, fourbytes + 3, threebytes);
				fprintf(outfile, "%c%c%c", (char)*threebytes, (char)*(threebytes + 1), (char)*(threebytes + 2));			
			}
			else {
				i++;
			}
		}
		//printf("i: %i\n", i);
	}
	if (i != 0) {
		printcount++;
		combine6bytes(fourbytes, fourbytes + 1, fourbytes + 2, fourbytes + 3, threebytes);
		fprintf(outfile, "%c%c%c", (char)*threebytes, (char)*(threebytes + 1), (char)*(threebytes + 2));
	}
	fprintf(outfile, "%i", i);

	fclose(infile);

	fclose(outfile);
	printf("Sets printed: %i\n", printcount);
	free(threebytes);
	free(fourbytes);
	free(outfilename);	
}

void uncompress(char* argv[]) {
	int i;

	char* filename = argv[2];
	char* outfilename = malloc(sizeof(char) * 64);
	sprintf(outfilename, "%s.out", filename);
	FILE* infile = fopen(filename, "r");
	FILE* outfile = fopen(outfilename, "w+");

	char store = 0, prev, prev2;
	uint8_t* threebytes = malloc(sizeof(uint8_t) * 3);
	uint8_t* fourbytes = malloc(sizeof(uint8_t) * 4);


	bool breakout = false;
	prev = fgetc(infile);
	store = fgetc(infile);
	while(!breakout) {
		for (i = 0; i < 3; i++) {
			prev2 = prev;
			prev = store;
			store = fgetc(infile);
			*(threebytes + i) = (uint8_t)prev2;	
			if (feof(infile)) {
				breakout = true;
				uint8_t delim = (uint8_t)(prev - '0');
				separate6bytes(fourbytes, fourbytes + 1, fourbytes + 2, fourbytes + 3, threebytes);
				if (delim > 0) {
					int k;
					for (k = 0; k < delim; k++) 
						fprintf(outfile, "%c", getOriginalVal(*(fourbytes + i)));
				}
				else {
					fprintf(outfile, "%c%c%c%c", getOriginalVal(*fourbytes), getOriginalVal(*(fourbytes + 1)), getOriginalVal(*(fourbytes + 2)), getOriginalVal(*(fourbytes + 3)));				
				}
				break;
			}
			if (i == 2) {
				separate6bytes(fourbytes, fourbytes + 1, fourbytes + 2, fourbytes + 3, threebytes);
				fprintf(outfile, "%c%c%c%c", getOriginalVal(*fourbytes), getOriginalVal(*(fourbytes + 1)), getOriginalVal(*(fourbytes + 2)), getOriginalVal(*(fourbytes + 3)));				
			}
		}
	}

	fclose(outfile);
	fclose(infile);
	free(threebytes);
	free(fourbytes);
	free(outfilename);

}

