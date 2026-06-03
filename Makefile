CXX = g++
CXXFLAGS = -std=c++17 -O3 -mavx2 -Wall -Wextra -I./include

all: valloc_test

valloc_test: tests/main.cpp
	$(CXX) $(CXXFLAGS) -o valloc_test tests/main.cpp

clean:
	rm -f valloc_test