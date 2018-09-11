.PHONY: all clean

.DEFAULT: all


CXX = c++

CXXFLAGS = -ansi -pedantic -std=c++11  -O3

INCLUDES = -I ./eigen

TARGETS = 1d

all: $(TARGETS)

clean:
	rm -f $(TARGETS)

$(TARGETS) : %: main.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $<
