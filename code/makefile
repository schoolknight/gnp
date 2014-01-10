objects = gnp.o mylib.o simplex_downhill.o

gnp:$(objects)
	gcc -O2 -o ../try/gnp $(objects) -lm

gnp.o:gnp.c gnp.h simplex_downhill.h mylib.h
	gcc -O2 -c gnp.c -lm

simplex_downhill.o:simplex_downhill.c simplex_downhill.h
	gcc -O2 -c simplex_downhill.c -lm

mylib.o:mylib.h mylib.c
	gcc -O2 -c mylib.c -lm

clean:
	rm gnp $(objects)	