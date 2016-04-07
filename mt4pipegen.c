#include "mt4pipegen.h"

HANDLE MT4PipeGen::createPipe(std::string name) {
    std::string pname = "\\\\.\\pipe\\"+name;
    HANDLE pipe = CreateNamedPipe(
                    pname.c_str(),
                    PIPE_ACCESS_DUPLEX,
                    PIPE_TYPE_BYTE,
                    1, 0, 0, 0, NULL);

    if(pipe == NULL || pipe == INVALID_HANDLE_VALUE) {
        throw std::runtime_error("Failed to open pipe");
    }
    return pipe;
}

void MT4PipeGen::connectPipe(HANDLE pipe) {
    bool result = ConnectNamedPipe(pipe, NULL);
    if(!result) {
        throw std::runtime_error("Failed to make connection");
    }
}

bool MT4PipeGen::sendString(HANDLE pipe, const char* data) {
    DWORD numBytesWritten = 0;
    return WriteFile(pipe, data, strlen(data) * sizeof(char), &numBytesWritten, NULL);
}

bool MT4PipeGen::readString(HANDLE pipe, char* buffer, int size) {
    DWORD numBytesWritten = 0;
    return ReadFile(pipe, buffer, size*sizeof(char), &numBytesWritten, NULL);
}

void MT4PipeGen::teachGenbot(Genbot* genbot, double pp, bool longTrade, double* outputs) {
    if(pp == 0)
        return;

    if(longTrade) {
        if(outputs[0] > 0.5)
            genbot->learn(pp);
        else
            genbot->learn(-pp);
    }
    else if(!longTrade) {
        if(outputs[1] > 0.5)
            genbot->learn(pp);
        else
            genbot->learn(-pp);
    }
}

void MT4PipeGen::perturbGenbot(Genbot* genbot, double pp, bool longTrade, double* inputs) {
    if(pp == 0)
        return;

    double outputs[NUMOUTPUTS];
    double perturbedInputs[NUMINPUTS];

    for(int i=0; i<genbot->getNumPerturbRuns(); i++) {
        for(int j=0; j<NUMINPUTS; j++) {
            if(rand() % 10000 < 10000*genbot->getPerturbChance())
                perturbedInputs[j] = (rand()%10000)/10000.0;
            else
                perturbedInputs[j] = inputs[j];
        }

        int extraTurns;
        if(genbot->hasSideWeights())
            extraTurns = genbot->getExtraAnswerTurns();
        else
            extraTurns = 0;

        genbot->setInputs(perturbedInputs, NUMINPUTS);
        genbot->progressTurns(genbot->getMinDepth() + extraTurns, false);
        genbot->getOutputs(outputs, NUMOUTPUTS);

        teachGenbot(genbot, genbot->getPerturbFactor()*pp, longTrade, outputs);
    }
}

std::string MT4PipeGen::getCollectiveOutput(Genbot** genbots, double** outputs, int outputType, int botnum, int agreeThreshold, int botsPolled) {
    if(outputType == SINGLE_BOT) {
        if(botnum >= 0 && botnum < numGenbots) {
            std::ostringstream outputString;
            outputString << genbots[botnum]->getID() << " ";
            for(int i=0; i<NUMOUTPUTS; i++) {
                outputString << outputs[botnum][i];
                if(i < NUMOUTPUTS-1)
                    outputString << " ";
            }
            return outputString.str();
        }
        else
            return "dummy output";
    }
    else if(outputType == AGREE_BOT) {
        int numLong = 0;
        int numShort = 0;
        if((botnum >= 0 && botnum < numGenbots))
            throw std::runtime_error("Trying to get bot consensus with specific bot selected");
        if(botsPolled <= 0)
            throw std::runtime_error("Trying to get consensus with no bots polled");
        for(int i=0; i<botsPolled && i<numGenbots; i++) {
            if(outputs[i][0] > 0.5 && outputs[i][1] < 0.5)
                numLong++;
            else if(outputs[i][0] < 0.5 && outputs[i][1] > 0.5)
                numShort++;
        }
        if(numLong - numShort >= agreeThreshold)
            return "99999 1.0 0.0";
        else if(numShort - numLong >= agreeThreshold)
            return "99999 0.0 1.0";
        return "99999 0.0 0.0";
    }
    else if(outputType == SEPARATE_BOT) {
        std::ostringstream outputString;
        for(int i=0; i<numGenbots; i++) {
            outputString << genbots[i]->getID() << " ";
            for(int j=0; j<NUMOUTPUTS; j++) {
                outputString << outputs[i][j];
                if(j < NUMOUTPUTS-1)
                    outputString << " ";
            }
            if(i < numGenbots-1)
                outputString << " ";
        }
        return outputString.str();
    }

    return NULL;
}

void *makeTrainThread(void* pipegen) {
    MT4PipeGen* mt4pipegen = (MT4PipeGen*) pipegen;
    mt4pipegen->trainGenbotThread();
    pthread_exit(NULL);
}

void MT4PipeGen::trainGenbotThread() {
    while(true) {
        pthread_mutex_lock(&trainMutex);
        while(numTrainsInQueue <= 0)
            pthread_cond_wait(&trainCond, &trainMutex);
        pthread_mutex_lock(&trainProcessingMutex);
        TrainThreadInputs inputs = trainThreadQueue.front();
        trainThreadQueue.pop();
        numTrainsInQueue--;
        numTrainsProcessing++;
        pthread_mutex_unlock(&trainMutex);
        pthread_mutex_unlock(&trainProcessingMutex);

        teachGenbot(inputs.genbot, inputs.pp, inputs.longTrade, inputs.outputs);
        perturbGenbot(inputs.genbot, inputs.pp, inputs.longTrade, inputs.inputs);

        pthread_mutex_lock(&trainProcessingMutex);
        numTrainsProcessing--;
        pthread_mutex_unlock(&trainProcessingMutex);
        pthread_cond_broadcast(&trainProcessingCond);
    }
}

void *makeRunThread(void* pipegen) {
    MT4PipeGen* mt4pipegen = (MT4PipeGen*) pipegen;
    mt4pipegen->runGenbotThread();
    pthread_exit(NULL);
}

void MT4PipeGen::runGenbotThread() {
    while(true) {
        pthread_mutex_lock(&runMutex);
        while(numRunsInQueue <= 0)
            pthread_cond_wait(&runCond, &runMutex);
        pthread_mutex_lock(&runProcessingMutex);
        RunThreadInputs inputs = runThreadQueue.front();
        runThreadQueue.pop();
        numRunsInQueue--;
        numRunsProcessing++;
        pthread_mutex_unlock(&runMutex);
        pthread_mutex_unlock(&runProcessingMutex);

        inputs.genbot->progressTurns(inputs.turnsToProgress, false);
        inputs.genbot->getOutputs(inputs.outputs, inputs.numOutputs);

        pthread_mutex_lock(&runProcessingMutex);
        numRunsProcessing--;
        pthread_mutex_unlock(&runProcessingMutex);
        pthread_cond_broadcast(&runProcessingCond);
    }
}

bool MT4PipeGen::runMT4Cycle(HANDLE pipe, Genbot** genbots, int botnum, bool train, int outputType, int agreeThreshold, int botsPolled) {
    int size = 30*NUMINPUTS;
    char buffer[size];
    if(!readString(pipe, buffer, size)) return false;
    char* pch;
    double inputs[NUMINPUTS];
    pch = strtok(buffer, " ");
    for(int i=0; i<NUMINPUTS && pch != NULL; i++) {
        inputs[i] = strtod(pch, NULL);
        pch = strtok(NULL, " ");
    }

    double* outputs[numGenbots];
    for(int i=0; i<numGenbots; i++) {
        outputs[i] = new double[NUMOUTPUTS];
    }

    for(int i=0; i<numGenbots; i++) {
        if(botnum >= 0 && botnum < numGenbots && i != botnum) continue;
        if(train && i<numStaticTopBots) continue;
        genbots[i]->setInputs(inputs, NUMINPUTS);
        //ignore the extraAnswerTurns if the network is not recursive
        int extraTurns;
        if(genbots[i]->hasSideWeights())
            extraTurns = genbots[i]->getExtraAnswerTurns();
        else
            extraTurns = 0;

        RunThreadInputs runInputs;
        runInputs.genbot = genbots[i];
        runInputs.turnsToProgress = genbots[i]->getMinDepth() + extraTurns;
        runInputs.outputs = outputs[i];
        runInputs.numOutputs = NUMOUTPUTS;

        pthread_mutex_lock(&runMutex);
        runThreadQueue.push(runInputs);
        numRunsInQueue++;
        pthread_mutex_unlock(&runMutex);
        
        pthread_cond_broadcast(&runCond);
    }

    pthread_mutex_lock(&runProcessingMutex);
    while(numRunsProcessing > 0 || numRunsInQueue > 0)
        pthread_cond_wait(&runProcessingCond, &runProcessingMutex);
    pthread_mutex_unlock(&runProcessingMutex);

    std::string outputString = getCollectiveOutput(genbots, outputs, outputType, botnum, agreeThreshold, botsPolled);
    if(!sendString(pipe, outputString.c_str())) return false;

    if(train) {
        if(!readString(pipe, buffer, size)) return false;
        double pp;
        bool longTrade;
        pch = strtok(buffer, " ");
        if(strcmp(pch,"learn:") != 0)
            throw std::runtime_error("expected learn message, didn't get it");
        pch = strtok(NULL, " ");
        pp = strtod(pch, NULL);
        pch = strtok(NULL, " ");
        if(pch[0] == '0')
            longTrade = false;
        else
            longTrade = true;
        for(int i=0; i<numGenbots; i++) {
            if(botnum >= 0 && botnum < numGenbots && i != botnum) continue;
            if(i < numStaticTopBots) continue;

            TrainThreadInputs trainInputs;
            trainInputs.genbot = genbots[i];
            trainInputs.pp = pp;
            trainInputs.longTrade = longTrade;
            trainInputs.outputs = outputs[i];
            trainInputs.inputs = inputs;

            pthread_mutex_lock(&trainMutex);
            trainThreadQueue.push(trainInputs);
            numTrainsInQueue++;
            pthread_mutex_unlock(&trainMutex);

            pthread_cond_broadcast(&trainCond);
        }

        pthread_mutex_lock(&trainProcessingMutex);
        while(numTrainsProcessing > 0 || numTrainsInQueue > 0)
            pthread_cond_wait(&trainProcessingCond, &trainProcessingMutex);
        pthread_mutex_unlock(&trainProcessingMutex);
    }

    for(int i=0; i<numGenbots; i++) {
        delete [] outputs[i];
    }
    return true;
}

void MT4PipeGen::runSim(Genbot** genbots, int botnum, int barsBack, int tradeCandleLimit, int learnDivisor, bool train, int testSample, int timePeriod, std::string testPeriod, bool useTrailingStop, bool useTilt, const char* savename, int outputType, int agreeThreshold, int botsPolled) {
    std::ostringstream systemcall;
    systemcall << "./singletest.sh";
    systemcall << " -b " << barsBack;
    systemcall << " -c " << tradeCandleLimit;
    systemcall << " -l " << learnDivisor;
    if(train)
        systemcall << " -r";
    systemcall << " -t " << testSample;
    if(savename != NULL)
        systemcall << " -s " << savename;
    systemcall << " -p " << timePeriod;
    systemcall << " -T " << testPeriod;
    if(useTrailingStop)
        systemcall << " -S";
    if(useTilt)
        systemcall << " -u";
    systemcall << " &";

    HANDLE pipe = createPipe("mt4pipe");
    //sleep(60);
    system("wait");
    system("ps | grep wscript.exe | awk '{print $1}' | wait");
    sleep(5);
    system(systemcall.str().c_str());
    connectPipe(pipe);
    while(runMT4Cycle(pipe, genbots, botnum, train, outputType, agreeThreshold, botsPolled)) {}
    CloseHandle(pipe);
}
