all:
	nvcc -o RTX.o -dc src/RTX.cu
	nvcc -o kernel.o -dc src/kernel.cu
	nvcc -o a src/main.cpp RTX.o kernel.o -lfreeglut -lglew32
	make clean

clean:
	rm -f *.o
	rm -f *.exp
	rm -f *.lib
	rm -f *.pdb