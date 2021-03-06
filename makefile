C_FILES := $(wildcard *.c) $(wildcard *.cpp) #nvwa/debug_new.cpp
GENBOT_C := $(wildcard genbot/*.c) $(wildcard genbot/*.cpp)
H_FILES := $(wildcard *.h)
GENBOT_H := $(wildcard genbot/*.h)
.PHONY: all
all: momstuff testbots readresults topbotsave perftest
momstuff: $(C_FILES) $(H_FILES) $(GENBOT_C) $(GENBOT_H)
	g++ -g -O3 -Wall -Wextra -o $@ $(C_FILES) $(GENBOT_C) -I. -lncurses -std=gnu++0x -lm #$(wildcard nvwa/*.cpp)
testbots: testc/testbots.c testc/testbots.h
	g++ -g -o $@ testc/testbots.c $(GENBOT_C) -I. -lncurses -std=gnu++0x -lm
readresults: testc/readresults.c
	g++ -g -o $@ testc/readresults.c -I. -std=gnu++0x
topbotsave: testc/topbotsave.c
	g++ -g -o $@ testc/topbotsave.c -I. -std=gnu++0x
perftest: testc/perftest.c
	g++ -g -o $@ testc/perftest.c $(GENBOT_C) -I. -lncurses -std=gnu++0x
