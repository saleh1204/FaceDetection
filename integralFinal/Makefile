# I am a comment, and I want to say that the variable CC will be
# the compiler to use.
NCC=nvcc
CC =gcc
# Hey!, I am comment number 2. I want to say that CFLAGS will be the
# options I'll pass to the compiler.
CFLAGS=

all: sequential parallel

sequential: sequential.c
	$(CC) -o sequential sequential.c
	
parallel: integralGlobal2D_S.cu
	$(NCC) -o parallel integralGlobal2D_S.cu
clean:
	rm -rf sequential parallel




