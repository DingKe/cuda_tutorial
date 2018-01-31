VERSION = -arch=sm_35 -gencode arch=compute_35,code=sm_35
CFLAGS = ${VERSION}
LDFLAGS = ${VERSION}

TARGET = main
SRC = kernel.cu main.cu
OBJ = $(SRC:.cu=.o)
DEPS = 

all:${TARGET}
	echo "Done!"

${TARGET}:${OBJ}
	nvcc -o $@ $^ $(LDFLAGS)

%.o:%.cu ${DEPS}
	nvcc -c -o $@ $< $(CFLAGS) 

clean:
	rm -f ${OBJ} ${TARGET}
