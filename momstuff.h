#ifndef GENSTOCK_HEADER_FILE
#define GENSTOCK_HEADER_FILE


#include <ncurses.h>
#include <iostream>
#include "genbot/cluster.h"
#include "event.h"
#include <string>
#include <fstream>
#include "genbot/genbot.h"
#include "genbot/genome.h"
#include <list>
#include <dirent.h>

//---simulation parameters
#define NUMGENBOTS 4
#define NUMWINBOTS 2
#define NUMSTATICTOPBOTS 0
#define NUMPARENTBOTS 2
#define CHILD_INHERITS_PARENT_LEARNING 1

#define ROUND_NUM_INCREMENT 5
#define ROUND_GROUP_LENGTH 10
#define ROUND_MIN_REPLACEMENTS 1
#define ROUND_MAX_REPLACEMENTS 2

//#define STEPFACTOR 0.00005
#define STEPFACTOR 0.0001

#define NUMOUTPUTS 1
#define NUMINPUTS 50

#define SKIP_TRAIN_ON_ROUND_1 0
#define SKIP_TEST_ON_ROUND_1 0
#define TEST_TESTSET 1
#define TEST_TRAINSET 1
#define TEST_TEST2SET 1
#define TEST_SET "testset"
#define TRAIN_SET "trainset"
#define TEST2_SET "test2set"

#define CHECK_TEST 0
#define CHECK_TRAIN 1

#define INITIAL_OUTPUT_THRESHOLD -120

#define NUM_TRAIN_THREADS 4

#define NUM_ROUNDS_SAVED 100

static ConvolutionProperties defaultConvProp = {
    1, {1}, 0, NUMINPUTS-1, {NUMINPUTS}, 1, 1, 1, 1, 0
};
//--------------------------

WINDOW* mainwin;
WINDOW* networkwin;
WINDOW* errorwin;
WINDOW* statuswin;

EventLog* eventLog = new EventLog();
EventLog* networkLog = new EventLog();
EventLog* errorLog = new EventLog();
EventLog* statusLog = new EventLog();

struct ThreadInputs {
    Genbot* genbot;
    std::vector<double> correctoutputs;
    std::vector<double> inputs;
    double* individualerror; //pointer
    double* error;  //pointer
    bool train;
    int id;
};

pthread_t* threads;
std::list<ThreadInputs> threadList;
pthread_cond_t threadCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t threadMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t processingCond = PTHREAD_COND_INITIALIZER;
pthread_cond_t errorCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t errorMutex = PTHREAD_MUTEX_INITIALIZER;
int numRunsInQueue = 0;
int numThreadsProcessing = 0;

int currentThreadGenbotId[NUM_TRAIN_THREADS];   //the id of the genbot each thread is currently running

size_t numTrainCycles = ROUND_NUM_INCREMENT;
size_t numBotChanges = 0;

#endif
