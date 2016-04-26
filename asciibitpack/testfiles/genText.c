#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>

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

int main() {

	 FILE* a = fopen("bible10.txt", "w+");
	 srand(time(NULL));
	 int i;
	 for (i = 0; i < 4947177; i++) {
	 	fprintf(a, "%c", getOriginalVal((uint8_t)(rand() % 63)));
	 }
	 fclose(a);
	 exit(0);
}