#ifndef MT4PIPE_HEADER_FILE
#define MT4PIPE_HEADER_FILE

#include <windows.h>
#include <iostream>
#include "genbot.h"
#include <string>
#include "genome.h"
#include <unistd.h>
#include <pthread.h>
#include <queue>

#define SINGLE_BOT 0
#define AGREE_BOT 1
#define SEPARATE_BOT 2

struct TrainThreadInputs {
    Genbot* genbot;
    double pp;
    bool longTrade;
    double* outputs;
    double* inputs;
};

struct RunThreadInputs {
    Genbot* genbot;
    int turnsToProgress;
    double* outputs;
    int numOutputs;
};

void *makeTrainThread(void* args);
void *makeRunThread(void* args);

class MT4PipeGen {
    private:
        int NUMINPUTS;
        int NUMOUTPUTS;
        int numGenbots;
        int numStaticTopBots;
        int numThreads;
        pthread_t* trainThreads;
        pthread_t* runThreads;
        std::queue<TrainThreadInputs> trainThreadQueue;
        std::queue<RunThreadInputs> runThreadQueue;
        pthread_cond_t trainCond;
        pthread_cond_t runCond;
        pthread_mutex_t trainMutex;
        pthread_mutex_t runMutex;
        pthread_cond_t trainProcessingCond;
        pthread_cond_t runProcessingCond;
        pthread_mutex_t trainProcessingMutex;
        pthread_mutex_t runProcessingMutex;
        int numTrainsInQueue;
        int numRunsInQueue;
        int numTrainsProcessing;
        int numRunsProcessing;

        HANDLE createPipe(std::string name);
        void connectPipe(HANDLE pipe);
        bool sendString(HANDLE pipe, const char* data);
        bool readString(HANDLE pipe, char* buffer, int size);
        bool runMT4Cycle(HANDLE pipe, Genbot** genbots, int botnum, bool train, int outputType, int agreeThreshold, int botsPolled);
        std::string getCollectiveOutput(Genbot** genbots, double** outputs, int outputType, int botnum, int agreeThreshold, int botsPolled);
        void teachGenbot(Genbot* genbot, double pp, bool longTrade, double* outputs);
        void perturbGenbot(Genbot* genbot, double pp, bool longTrade, double* inputs);

    public:
        void runSim(Genbot** genbot, int botnum, int barsBack, int tradeCandleLimit, int learnDivisor, bool train, int testSample, int timePeriod, std::string testPeriod, bool useTrailingStop, bool useTilt, const char* savename, int outputType = SINGLE_BOT, int agreeThreshold = 0, int botsPolled = 0);
        void trainGenbotThread();
        void runGenbotThread();
        
        MT4PipeGen(int numInputs, int numOutputs, int nGenbots, int nStaticTopBots, int threads) {
            NUMINPUTS = numInputs;
            NUMOUTPUTS = numOutputs;
            numGenbots = nGenbots;
            numStaticTopBots = nStaticTopBots;
            numThreads = threads;

            numTrainsInQueue = 0;
            numRunsInQueue = 0;
            numTrainsProcessing = 0;
            numRunsProcessing = 0;

            trainCond = PTHREAD_COND_INITIALIZER;
            runCond = PTHREAD_COND_INITIALIZER;
            trainMutex = PTHREAD_MUTEX_INITIALIZER;
            runMutex = PTHREAD_MUTEX_INITIALIZER;
            trainProcessingCond = PTHREAD_COND_INITIALIZER;
            runProcessingCond = PTHREAD_COND_INITIALIZER;
            trainProcessingMutex = PTHREAD_MUTEX_INITIALIZER;
            runProcessingMutex = PTHREAD_MUTEX_INITIALIZER;

            trainThreads = new pthread_t[numThreads];
            for(int i=0; i<numThreads; i++) {
                pthread_create(&trainThreads[i], NULL, &makeTrainThread, this);
            }

            runThreads = new pthread_t[numThreads];
            for(int i=0; i<numThreads; i++) {
                pthread_create(&runThreads[i], NULL, &makeRunThread, this);
            }
        }

        ~MT4PipeGen() {
            delete trainThreads;
        }
};

#endif
