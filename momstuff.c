#include "momstuff.h"
#include <iostream>
#include <windows.h>
#include <string>
//#include "nvwa/debug_new.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <math.h>
#include </usr/include/fenv.h>
//#pragma STDC FENV_ACCESS ON
#include <pthread.h>

#include <float.h>
#include <unistd.h>

//only for the test main function
#define PARENTLAYERS 2
#define CHILDLAYERS 2
#define PARENTNODESPERLAYER 5
#define CHILDNODESPERLAYER 5
//------

WINDOW* mainwin;
WINDOW* networkwin;
WINDOW* errorwin;
WINDOW* statuswin;

EventLog* eventLog = new EventLog();
EventLog* networkLog = new EventLog();
EventLog* errorLog = new EventLog();
EventLog* statusLog = new EventLog();

//---simulation parameters
#define NUMGENBOTS 3
#define NUMWINBOTS 2
#define NUMSTATICTOPBOTS 0
#define NUMPARENTBOTS 2
#define NUMTRAINCYCLES 10
#define CHILD_INHERITS_PARENT_LEARNING 1

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

#define NUM_TRAIN_THREADS 3

static ConvolutionProperties defaultConvProp = {
    1, {1}, 0, NUMINPUTS-1, {NUMINPUTS}, 1, 1, 1, 1
};
//--------------------------

const char* savestring = "savegenbot/";
const char* datastring = "rawdata/";

#define RESULT_TYPE_TRAIN 0
#define RESULT_TYPE_TEST 1
#define RESULT_TYPE_SECONDARY_TEST 2

bool stopAfterNextMT4Run = false;
bool stopAfterNextTestSet = false;
bool pauseAfterNextMT4Run = false;

int roundNum = 0;
int currentTrainRun = 0;
int currentTestRun = 0;

void *genbotThread(void* args) {
    (void)args;
    while(true) {
        pthread_mutex_lock(&threadMutex);
        while(numRunsInQueue <= 0)
            pthread_cond_wait(&threadCond, &threadMutex);
        pthread_mutex_lock(&processingMutex);
        ThreadInputs inputs = threadQueue.front();
        threadQueue.pop();
        numRunsInQueue--;
        numThreadsProcessing++;
        pthread_mutex_unlock(&processingMutex);
        pthread_mutex_unlock(&threadMutex);

        inputs.genbot->setInputs(&(inputs.inputs[0]), NUMINPUTS);
        inputs.genbot->progressTurns(inputs.genbot->getMinDepth(),false);
        double outputs[NUMOUTPUTS];
        inputs.genbot->getOutputs(outputs, NUMOUTPUTS);
        double boterror = pow((outputs[0] - inputs.correctoutputs[0]), 2);
        pthread_mutex_lock(&errorMutex);
        if(inputs.individualerror != NULL)
            *(inputs.individualerror) += boterror;
        *(inputs.error) += boterror;
        pthread_mutex_unlock(&errorMutex);

        if(inputs.train) {
            inputs.genbot->learnRawOutput(&(inputs.correctoutputs[0]), STEPFACTOR, NUMOUTPUTS);
        }

        pthread_mutex_lock(&processingMutex);
        numThreadsProcessing--;
        pthread_mutex_unlock(&processingMutex);
        pthread_cond_broadcast(&processingCond);
    }
}

double runSim(Genbot** genbots, std::string learnsetname, bool train, double* errors) {
    std::stringstream learnsetss;
    learnsetss << datastring << learnsetname;
    std::ifstream learnset(learnsetss.str().c_str());
    std::string line;
    double error = 0;
    for(int i=0; errors != NULL && i<NUMGENBOTS; i++) {
        errors[i] = 0;
    }

    int numerrors = 0;
    while(getline(learnset, line)) {
        std::stringstream lss(line);
        std::string fname;
        int column;
        int sstart;
        int send;
        std::vector<double> correctoutput(1);

        lss >> fname;
        lss >> column;
        lss >> sstart;
        lss >> send;
        lss >> correctoutput[0];


        std::stringstream fullfname;
        fullfname << datastring << fname;
        std::ifstream datafile(fullfname.str().c_str());
        if(!datafile.is_open()) {
            std::stringstream errormsg;
            errormsg << "Couldn't open file " << fullfname.str() << std::endl;
            throw std::runtime_error(errormsg.str().c_str()); 
        }
        std::string dline;

        int curinput = 0;
        std::vector<double> inputs(NUMINPUTS,0);
        int offset = rand() % NUMINPUTS;
        for(int i=1; i<=send && getline(datafile, dline); i++) {
            if(i < sstart)
                continue;
            else if(offset > 0) {
                offset--;
                continue;
            }

            std::string dum;
            std::stringstream dliness(dline);
            double in;
            for(int j=0; j<column-1; j++)
                dliness >> dum;

            dliness >> in;
            inputs[curinput] = in;
            curinput++;

            if(curinput >= NUMINPUTS) {
                numerrors++;
                double maxinput = -999999;
                double mininput = 999999;
                for(int j=0; j<NUMINPUTS; j++) {
                    if(inputs[j] > maxinput)
                        maxinput = inputs[j];
                    if(inputs[j] < mininput)
                        mininput = inputs[j];
                }
                for(int j=0; j<NUMINPUTS; j++) {
                    if(maxinput > mininput)
                        inputs[j] = 2*(inputs[j]-mininput)/(maxinput-mininput)-1;
                    else
                        inputs[j] = 0;
                }

                pthread_mutex_lock(&processingMutex);
                while(numThreadsProcessing > 0 || numRunsInQueue > 0) {
                    pthread_cond_wait(&processingCond, &processingMutex);
                }
                pthread_mutex_unlock(&processingMutex);

                for(int j=0; j<NUMGENBOTS; j++) {
                    ThreadInputs threadInputs;
                    threadInputs.genbot = genbots[j];
                    threadInputs.correctoutputs = correctoutput;
                    threadInputs.inputs = inputs;
                    if(errors != NULL)
                        threadInputs.individualerror = &errors[j];
                    else
                        threadInputs.individualerror = NULL;
                    threadInputs.error = &error;
                    threadInputs.train = train && (roundNum == 0 || j >= NUMSTATICTOPBOTS);

                    pthread_mutex_lock(&threadMutex);
                    threadQueue.push(threadInputs);
                    numRunsInQueue++;
                    pthread_mutex_unlock(&threadMutex);

                    pthread_cond_broadcast(&threadCond);
                }

                if(send-i < NUMINPUTS)
                    break;

                curinput = 0;
            }
        }
    }

    pthread_mutex_lock(&processingMutex);
    while(numThreadsProcessing > 0 || numRunsInQueue > 0) {
        pthread_cond_wait(&processingCond, &processingMutex);
    }
    pthread_mutex_unlock(&processingMutex);

    pthread_mutex_lock(&errorMutex);
    if(numerrors > 0) {
        error /= (1.0*NUMGENBOTS*numerrors);
        error = sqrt(error);
        for(int i=0; errors != NULL && i<NUMGENBOTS; i++) {
            errors[i] /= (1.0*numerrors);
            errors[i] = sqrt(errors[i]);
        }
    }
    pthread_mutex_unlock(&errorMutex);
    return error;
}

void printResults(const char* rstring) {
    std::ofstream outfile("savegenbot/results", std::ios::app);
    outfile << rstring << std::endl;
    outfile.close();
}

void addHistoryEntry(int id, const char* entry) {
    std::ostringstream fname;
    fname << "savegenbot/" << id << "/history";
    std::ofstream outfile(fname.str().c_str(), std::ios::app);
    outfile << entry << std::endl;
    outfile.close();
}

void saveStartInfo() {
    std::ofstream outfile("savegenbot/startinfo");
    outfile << "roundNum " << roundNum << std::endl;
    outfile << "currentTrainRun " << currentTrainRun << std::endl;
    outfile << "currentTestRun " << currentTestRun << std::endl;
    outfile.close();
}

void loadStartInfo() {
    roundNum = 1;
    currentTrainRun = 1;
    currentTestRun = 1;
    std::ifstream infile("savegenbot/startinfo");
    std::string line;
    while(std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string token1;
        int token2;
        if(!(iss>>token1>>token2)) continue;

        if(token1=="roundNum")
            roundNum = token2;
        if(token1=="currentTrainRun")
            currentTrainRun = token2;
        if(token1=="currentTestRun")
            currentTestRun = token2;
    }
    infile.close();
}

void swap(Genbot** genbots, double** profits, int i, int j, int numArrays) {
    Genbot* gentmp;
    double proftmp;
    gentmp = genbots[i];
    genbots[i] = genbots[j];
    genbots[j] = gentmp;
    for(int k=0; k<numArrays; k++) {
        proftmp = profits[k][i];
        profits[k][i] = profits[k][j];
        profits[k][j] = proftmp;
    }
}

void sortByProfit(Genbot** genbots, double** profits, int numArrays) {
    for(int i=1; i<NUMGENBOTS; i++) {
        for(int j=i-1; j>=0; j--) {
            if(profits[0][j+1] < profits[0][j])
                swap(genbots, profits, j+1, j, numArrays);
            else
                break;
        }
    }
}

void saveCurrentIDs(Genbot** genbots) {
    std::ofstream file("savegenbot/currentbots");
    for(int i=0; i<NUMGENBOTS; i++) {
        file << genbots[i]->getID() << std::endl;
    }
}

int findNextID(int id) {
    struct stat info;
    std::ostringstream fname;
    fname << "savegenbot/" << id;
    while(stat(fname.str().c_str(), &info) == 0) {
        id++;
        fname.clear();
        fname.str("");
        fname << "savegenbot/" << id;
    }

    return id;
}

void loadCurrentGenbots(Genbot** genbots) {
    std::ifstream file("savegenbot/currentbots");
    if(file.is_open()) {
        std::string line;
        for(int i=0; i < NUMGENBOTS && getline(file, line); i++) {
            int id;
            if(!(std::istringstream(line) >> id))
                throw std::runtime_error("found something that wasn't an id in the current Genbots file");
            
            std::ostringstream genomefname;
            genomefname << savestring << id << "/genome";

            Genome* genome = new Genome(defaultConvProp);
            genome->loadGenome(genomefname.str().c_str());
            genome->pars[0]->useOutputTransfer = false;
            genbots[i] = new Genbot(genome, NUMINPUTS, NUMOUTPUTS, id);

            std::ostringstream fname;
            fname << savestring << id << "/bot";
            genbots[i]->loadBot(fname.str().c_str());
            genbots[i]->progressTurnsSaved();
        }
    }
    else {
        //create the first set of bots
        for(int i=0; i<NUMGENBOTS; i++) {
            Genome* genome = new Genome(defaultConvProp);
            genome->createRandomGenome();
            genome->pars[0]->useOutputTransfer = false;
            genbots[i] = new Genbot(genome, NUMINPUTS, NUMOUTPUTS, findNextID(1));
            genbots[i]->setOutputThreshold(INITIAL_OUTPUT_THRESHOLD, 0);

            std::ostringstream foldercall;
            foldercall << "mkdir " << savestring << genbots[i]->getID();
            system(foldercall.str().c_str());
            std::ostringstream fname;
            fname << savestring << genbots[i]->getID() << "/genome";
            genbots[i]->getGenome()->saveGenome(fname.str().c_str());

            fname.clear();
            fname.str("");
            fname << savestring << genbots[i]->getID() << "/start";
            genbots[i]->saveBot(fname.str().c_str());

            std::ostringstream entry;
            entry << "Round " << roundNum << ": Born in initial batch";
            addHistoryEntry(genbots[i]->getID(), entry.str().c_str());

            genbots[i]->progressTurnsSaved();
        }
    }
}

void *startKeyboardCheck(void *threadarg) {
    (void)threadarg;
    stopAfterNextMT4Run = false;
    stopAfterNextTestSet = false;
    statusLog->addEvent("Running.", statuswin);
    while(true) {
        int ch;
        while((ch = getch()) != ERR) {
            switch(ch) {
                case 'q':
                    pauseAfterNextMT4Run = false;
                    stopAfterNextMT4Run = true;
                    stopAfterNextTestSet = true;
                    statusLog->addEvent("Stopping immediately", statuswin);
                    break;
                case 's':
                    pauseAfterNextMT4Run = false;
                    stopAfterNextMT4Run = false;
                    stopAfterNextTestSet = true;
                    statusLog->addEvent("Stopping at the end of this round", statuswin);
                    break;
                case 'g':
                    pauseAfterNextMT4Run = false;
                    stopAfterNextMT4Run = false;
                    stopAfterNextTestSet = false;
                    statusLog->addEvent("Running.", statuswin);
                    break;
                case 'p':
                    pauseAfterNextMT4Run = true;
                    stopAfterNextMT4Run = false;
                    stopAfterNextTestSet = false;
                    statusLog->addEvent("Pausing...", statuswin);
            }
        }
        sleep(1);
    }
}

std::string processNumSeconds(double t) {
    int minutes;
    int seconds;
    minutes = t/60;
    seconds = ((int)t)%60;
    std::ostringstream oss;
    oss << minutes << " m, " << seconds << " s";
    return oss.str();
}

void combineParentInitialConditions(Genbot* child, Genbot* parent1, Genbot* parent2) {
    Genbot* originalParent1 = new Genbot(parent1->getGenome()->copy(), NUMINPUTS, NUMOUTPUTS, -1);
    Genbot* originalParent2 = new Genbot(parent2->getGenome()->copy(), NUMINPUTS, NUMOUTPUTS, -1);

    std::string fhandle;
    if(CHILD_INHERITS_PARENT_LEARNING)
        fhandle = "/bot";
    else
        fhandle = "/start";

    std::ostringstream fname;
    fname << savestring << parent1->getID() << fhandle;
    originalParent1->loadBot(fname.str().c_str());

    fname.clear();
    fname.str("");
    fname << savestring << parent2->getID() << fhandle;
    originalParent2->loadBot(fname.str().c_str());

    if(rand()%2) {
        child->copyWeights(originalParent1, 1);
        child->copyWeights(originalParent2, 0.5);
    }
    else {
        child->copyWeights(originalParent2, 1);
        child->copyWeights(originalParent1, 0.5);
    }
    child->mutateWeights();

    delete originalParent1;
    delete originalParent2;
}

int main() {
    feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    srand(time(NULL));

    //ncurses stuff
    initscr();
    raw();
    keypad(stdscr, TRUE);
    //cbreak();
    curs_set(0);
    noecho();
    nodelay(stdscr, TRUE);

    refresh();

    statuswin = newwin(2, 160, 0, 0);
    mainwin = newwin(50, 80, 2, 0);
    networkwin = newwin(50, 80, 2, 80);
    errorwin = newwin(10, 160, 52, 0);

    pthread_t statusthread;
    pthread_create(&statusthread, NULL, &startKeyboardCheck, NULL);

    threads = new pthread_t[NUM_TRAIN_THREADS];
    for(int i=0; i<NUM_TRAIN_THREADS; i++) {
        pthread_create(&threads[i], NULL, &genbotThread, NULL);
    }

    Genbot* genbots[NUMGENBOTS];

    std::ostringstream fname;

    loadCurrentGenbots(genbots);
    loadStartInfo();

    //save initial ids and networks
    saveCurrentIDs(genbots);
    for(int i=0; i<NUMGENBOTS; i++) {
        fname.clear();
        fname.str("");
        fname << savestring << genbots[i]->getID() << "/bot";
        genbots[i]->saveBot(fname.str().c_str());
    }

    bool done = false;
    while(!done) {
        std::ostringstream oss;
        oss << "Starting round " << roundNum;
        eventLog->addEvent(oss.str().c_str(), mainwin);
        if(!(roundNum == 1 && SKIP_TRAIN_ON_ROUND_1)) {
            for(int i=0; i<NUMTRAINCYCLES; i++) {
                if(i+1 < currentTrainRun) continue;
                char* buf;
                asprintf(&buf, "starting train %d", i+1);
                networkLog->addEvent(buf, networkwin);
                time_t starttime = time(NULL);

                double trainresult = runSim(genbots, TRAIN_SET, true, NULL);

                for(int j=0; j<NUMGENBOTS; j++) {
                    fname.clear();
                    fname.str("");
                    fname << savestring << genbots[j]->getID() << "/bot";
                    genbots[j]->saveBot(fname.str().c_str());
                }
                currentTrainRun++;
                saveStartInfo();
                time_t endtime = time(NULL);
                double secsElapsed = difftime(endtime, starttime);
                std::ostringstream ost;
                ost << "Done. Error = " << trainresult << " (" << processNumSeconds(secsElapsed) << ")";
                networkLog->addEvent(ost.str().c_str(), networkwin);
                while(pauseAfterNextMT4Run) {
                    statusLog->addEvent("Paused.", statuswin);
                    sleep(5);
                }
                if(stopAfterNextMT4Run) return 1;
            }
        }

        double testprofits[NUMGENBOTS] = {0};
        double trainprofits[NUMGENBOTS] = {0};
        double test2profits[NUMGENBOTS] = {0};

        if(!(roundNum == 1 && SKIP_TEST_ON_ROUND_1)) {
            if(TEST_TESTSET && 1 == currentTestRun) {
                char* buf;
                asprintf(&buf, "starting test t");
                networkLog->addEvent(buf, networkwin);
                time_t starttime = time(NULL);

                double result = runSim(genbots, TEST_SET, false, testprofits);

                time_t endtime = time(NULL);
                double secsElapsed = difftime(endtime, starttime);
                std::ostringstream ost;
                ost << "Done. Error = " << result << " (" << processNumSeconds(secsElapsed) << ")";
                networkLog->addEvent(ost.str().c_str(), networkwin);
            }
            currentTestRun++;
            saveStartInfo();
            while(pauseAfterNextMT4Run) {
                statusLog->addEvent("Paused.", statuswin);
                sleep(5);
            }
            if(stopAfterNextMT4Run) return 1;

            if(TEST_TRAINSET && 2 == currentTestRun) {
                char* buf;
                asprintf(&buf, "starting test r");
                networkLog->addEvent(buf, networkwin);

                time_t starttime = time(NULL);

                double result = runSim(genbots, TRAIN_SET, false, trainprofits);
                time_t endtime = time(NULL);
                double secsElapsed = difftime(endtime, starttime);
                std::ostringstream ost;
                ost << "Done. Error = " << result << " (" << processNumSeconds(secsElapsed) << ")";
                networkLog->addEvent(ost.str().c_str(), networkwin);
            }

            currentTestRun++;
            saveStartInfo();
            while(pauseAfterNextMT4Run) {
                statusLog->addEvent("Paused.", statuswin);
                sleep(5);
            }
            if(stopAfterNextMT4Run) return 1;

            if(TEST_TEST2SET && 3 == currentTestRun) {
                char* buf;
                asprintf(&buf, "starting test t2");
                networkLog->addEvent(buf, networkwin);

                time_t starttime = time(NULL);

                double result = runSim(genbots, TEST2_SET, false, test2profits);

                time_t endtime = time(NULL);
                double secsElapsed = difftime(endtime, starttime);
                std::ostringstream ost;
                ost << "Done. Error = " << result << " (" << processNumSeconds(secsElapsed) << ")";
                networkLog->addEvent(ost.str().c_str(), networkwin);
            }

            currentTestRun++;
            saveStartInfo();
            while(pauseAfterNextMT4Run) {
                statusLog->addEvent("Paused.", statuswin);
                sleep(5);
            }
            if(stopAfterNextMT4Run) return 1;
        }
        double* profits[4];
        profits[0] = new double[NUMGENBOTS];
        profits[1] = testprofits;
        profits[2] = trainprofits;
        profits[3] = test2profits;

        for(int i=0; i<NUMGENBOTS; i++) {
            profits[0][i] = CHECK_TEST*testprofits[i] + CHECK_TRAIN*trainprofits[i];
        }
        sortByProfit(genbots, profits, 4);

        for(int i=0; i<NUMGENBOTS; i++) {
            std::ostringstream entry;
            entry << "Round " << roundNum << ": rank " << i+1 << " of " << NUMGENBOTS << " with test=" << testprofits[i] << ", train=" << trainprofits[i] << ", test2=" << test2profits[i];
            if(i < NUMSTATICTOPBOTS)
                entry << " (static)";
            if(i >= NUMWINBOTS)
                entry << " (eliminated)";
            addHistoryEntry(genbots[i]->getID(), entry.str().c_str());
        }

        std::ostringstream pss;
        pss << "Errors for round " << roundNum << ": ";
        for(int i=0; i<NUMGENBOTS; i++) {
            pss << genbots[i]->getID() << "=" << testprofits[i] << "(" << trainprofits[i] << "," << test2profits[i] << ")";
            if(i != NUMGENBOTS-1)
                pss << ", ";
        }
        eventLog->addEvent(pss.str().c_str(), mainwin);
        printResults(pss.str().c_str());
        pss.clear();
        pss.str("");
        pss << "Winning bots: ";
        for(int i=0; i<NUMWINBOTS; i++) {
            pss << genbots[i]->getID();
            if(i != NUMWINBOTS-1)
                pss << ", ";
        }
        eventLog->addEvent(pss.str().c_str(), mainwin);
        //replace losing genbots
        for(int i=NUMWINBOTS; i<NUMGENBOTS; i++) {
            delete genbots[i];

            int parent1 = rand() % NUMPARENTBOTS;
            int parent2 = rand() % NUMPARENTBOTS;
            Genome* genome = genbots[parent1]->getGenome()->mate(genbots[parent2]->getGenome());
            genome->pars[0]->useOutputTransfer = false;
            genbots[i] = new Genbot(genome, NUMINPUTS, NUMOUTPUTS, findNextID(1));
            combineParentInitialConditions(genbots[i], genbots[parent1], genbots[parent2]);

            std::ostringstream foldercall;
            foldercall << "mkdir " << savestring << genbots[i]->getID();
            system(foldercall.str().c_str());

            fname.clear();
            fname.str("");
            fname << savestring << genbots[i]->getID() << "/genome";
            genbots[i]->getGenome()->saveGenome(fname.str().c_str());

            fname.clear();
            fname.str("");
            fname << savestring << genbots[i]->getID() << "/start";
            genbots[i]->saveBot(fname.str().c_str());

            std::ostringstream entry;
            entry << "Round " << roundNum << ": Born to parents " << genbots[parent1]->getID() << " and " << genbots[parent2]->getID();
            addHistoryEntry(genbots[i]->getID(), entry.str().c_str());

            genbots[i]->progressTurnsSaved();
        }

        saveCurrentIDs(genbots);
        for(int i=0; i<NUMGENBOTS; i++) {
            fname.clear();
            fname.str("");
            fname << savestring << genbots[i]->getID() << "/bot";
            genbots[i]->saveBot(fname.str().c_str());
        }

        char* buf;
        asprintf(&buf, "Losing bots replaced.");
        eventLog->addEvent(buf, mainwin);

        roundNum++;
        currentTrainRun = 1;
        currentTestRun = 1;
        saveStartInfo();
        if(stopAfterNextTestSet) break;
        while(pauseAfterNextMT4Run) {
            statusLog->addEvent("Paused.", statuswin);
            sleep(5);
        }
    }

    for(int i=0; i<NUMGENBOTS; i++) {
        delete genbots[i];
    }

    delete [] threads;

    delete eventLog;
    delete networkLog;
    delete errorLog;
    delete statusLog;

    return 0;
}
