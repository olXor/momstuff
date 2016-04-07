C_FILES := $(wildcard *.c) $(wildcard *.cpp) #nvwa/debug_new.cpp
H_FILES := $(wildcard *.h)
.PHONY: all
all: momstuff testbots readresults topbotsave evaluatetopbots
momstuff: $(C_FILES) $(H_FILES) 
	g++ -g -O3 -Wall -Wextra -o $@ $(C_FILES) -I. -lncurses -std=gnu++0x -lm #$(wildcard nvwa/*.cpp)
	#g++ -g -shared -o $@.dll $(C_FILES) -I. -lncurses -std=gnu++0x
testbots: testc/testbots.c testc/testbots.h
	g++ -g -o $@ testc/testbots.c genome.c cluster.c genbot.c mt4pipegen.c -I. -lncurses -std=gnu++0x -lm
readresults: testc/readresults.c
	g++ -g -o $@ testc/readresults.c -I. -std=gnu++0x
topbotsave: testc/topbotsave.c
	g++ -g -o $@ testc/topbotsave.c -I. -std=gnu++0x
