CC      = g++
CPPFLAGS = -O3 -c -std=c++11 -fopenmp -mavx -w -fopenmp-simd
#CPPFLAGS = -g -O0 -Wall -c -std=c++11 -fopenmp -mavx -w 
LDFLAGS = -fopenmp -m64 -lpthread  -lboost_timer -lboost_system -fopenmp-simd

SOURCES = main.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

all: $(SOURCES) pr

pr : $(OBJECTS)  

	$(CC) $(OBJECTS) $(LDFLAGS) -o $@

.cpp.o : 
	$(CC) $(CPPFLAGS) $(OMP) $< -o $@

clean:
	rm -f *.o pr dump*

