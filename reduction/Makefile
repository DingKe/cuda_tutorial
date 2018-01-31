NVCC = nvcc
CC = gcc

VERSION = -arch=sm_35 -gencode arch=compute_35,code=sm_35
CFLAGS = ${VERSION}
LDFLAGS = ${VERSION}
DEPS = utils.h

TARGET = main
SRC = benchmark.cu global_reduction.cu shared_reduction.cu warp_reduction.cu
OBJ = $(SRC:.cu=.o)

all:${TARGET}
	echo "done!"

${TARGET}:${OBJ}
	${NVCC} -o $@ $^ $(LDFLAGS) 

%.o:%.cu ${DEPS}
	${NVCC} -c -o $@ $< ${CFLAGS} 

clean:
	rm -f ${OBJ} ${TARGET}
