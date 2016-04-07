#ifndef GENSTOCK_HEADER_FILE
#define GENSTOCK_HEADER_FILE


#include <ncurses.h>
#include <iostream>
#include "cluster.h"
#include "event.h"
#include <string>
#include <fstream>
#include "genbot.h"
#include "genome.h"
#include "mt4pipegen.h"

struct ThreadInputs {
    Genbot* genbot;
    std::vector<double> correctoutputs;
    std::vector<double> inputs;
    double* individualerror; //pointer
    double* error;  //pointer
    bool train;
};

pthread_t* threads;
std::queue<ThreadInputs> threadQueue;
pthread_cond_t threadCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t threadMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t processingCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t processingMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t errorCond = PTHREAD_COND_INITIALIZER;
pthread_mutex_t errorMutex = PTHREAD_MUTEX_INITIALIZER;
int numRunsInQueue = 0;
int numThreadsProcessing = 0;

#endif
