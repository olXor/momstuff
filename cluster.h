#ifndef CLUSTER_HEADER_FILE
#define CLUSTER_HEADER_FILE

#include <iostream>
#include <stdio.h>
#include <sstream>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <stdarg.h>
#include <string>
#include <cstring>
#include <ncurses.h>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <cstdlib>
//#include "nvwa/debug_new.h"

#define LEARNSTYLENONE 0
#define LEARNSTYLEBP 1
#define LEARNSTYLEHB 2
#define LEARNSTYLEALT 3
#define LEARNSTYLEALTR 4

class ClusterParameters {
    public:

        int numInputs;
        int numOutputs;
        int numLayers;
        int nodesPerLayer;
        int randomWeights;
        int numTurnsSaved;
        double stepfactor;
        double memfactor;
        double memnorm;
        bool useBackWeights;
        bool backPropBackWeights;
        bool useSideMems;
        bool useBackMems;
        bool useForwardMems;
        double propThresh;
        int learnStyleSide; //LEARNSTYLEBP: backprop every level
                            //LEARNSTYLEHB: hebb learn every level
                            //LEARNSTYLEALT: alternate (hebb first)
                            //LEARNSTYLEALTR: alternate (bp first)
                            //LEARNSTYLENONE: nothing
        bool bpMemStrengths;
        bool copyInputsToFirstLevel;
        bool lockMaxMem;

        int tlevel;     //level of transfer function (default 0)
        double transferWidth;

        bool useOutputTransfer; //whether to apply the transfer function to the final output

        ClusterParameters() {
            numInputs = 1;
            numOutputs = 1;
            numLayers = 1;
            nodesPerLayer = 1;
            randomWeights = 1;
            numTurnsSaved = 5;
            stepfactor = 1;
            memfactor = 0.005;
            memnorm = 0.98;
            useBackWeights = 0;
            backPropBackWeights = 0;
            useBackMems = 0;
            useForwardMems = 0;
            propThresh = 0.0;
            bpMemStrengths = true;
            tlevel = 0;
            transferWidth = 1.0;
            copyInputsToFirstLevel = false;
            lockMaxMem = false;
            useOutputTransfer = true;
        }

        ClusterParameters* copy() {
            ClusterParameters* cp = new ClusterParameters();
            cp->numInputs = numInputs;
            cp->numOutputs = numOutputs;
            cp->numLayers = numLayers;
            cp->nodesPerLayer = nodesPerLayer;
            cp->randomWeights = randomWeights;
            cp->numTurnsSaved = numTurnsSaved;
            cp->stepfactor = stepfactor;
            cp->memfactor = memfactor;
            cp->memnorm = memnorm;
            cp->useBackWeights = useBackWeights;
            cp->backPropBackWeights = backPropBackWeights;
            cp->useSideMems = useSideMems;
            cp->useBackMems = useBackMems;
            cp->useForwardMems = useForwardMems;
            cp->propThresh = propThresh;
            cp->learnStyleSide = learnStyleSide;
            cp->bpMemStrengths = bpMemStrengths;
            cp->copyInputsToFirstLevel = copyInputsToFirstLevel;
            cp->lockMaxMem = lockMaxMem;
            cp->tlevel = tlevel;
            cp->transferWidth = transferWidth;
            cp->useOutputTransfer = useOutputTransfer;
            return cp;
        }
};

enum WeightType {
    FORWARDWEIGHT,
    FORWARDMEM,
    SIDEWEIGHT,
    SIDEMEM,
    BACKWEIGHT,
    BACKMEM,
    BLANKWEIGHT
};

class Cluster {
    ClusterParameters *pars;
    double **inputToNodes;   //[pars.nodesPerLayer][pars.numInputs];
    double ***nodesToNodes;   //[pars.numLayers-1][pars.nodesPerLayer][pars.nodesPerLayer];
    double **thresholds;     //[pars.numLayers][pars.nodesPerLayer];
    double **nodesToOutput;  //[pars.numOutputs][pars.nodesPerLayer];
    double *outputThresholds;   //[pars.numOutputs];
    double ***savedNodeStrengths; //[pars.numTurnsSaved][pars.numLayers][pars.nodesPerLayer] = {0};
    double **savedInputs;    //[pars.numTurnsSaved][pars.numInputs] = {0};
    double **savedOutputs;   //[pars.numTurnsSaved][pars.numOutputs] = {0};
    double ***sideWeights;       //[pars.numLayers][pars.nodesPerLayer][pars.nodesPerLayer] = {0};
    double ***backWeights;       //[pars.numLayers-1][pars.nodesPerLayer][pars.nodesPerLayer];
    double ***forwardMems;  //[pars.numLayers-1][pars.nodesPerLayer][pars.nodesPerLayer]
    double ***sideMems;     //[pars.numLayers][pars.nodesPerLayer][pars.nodesPerLayer]
    double ***backMems;     //[pars.numLayers-1][pars.nodesPerlayer][pars.nodesPerLayer]

    Cluster ***nodeClusters; //[pars.numLayers][pars.nodesPerLayer]

    double **memStrengths; //[pars.numLayers][pars.nodesPerLayer]

    double ****nodeError; //[pars->numTurnsSaved][pars->numLayers+1][pars->nodesPerLayer][childpars->numInputs]
    double ***preNodeError; //[pars->numTurnSaved][pars->numLayers][pars->nodesPerLayer]

    double ** backPropagateOutputError; //[pars->numTurnsSaved][pars->numOutputs]

    int clusterTurn;
    int currentInput;
    bool isConvChild = false;

    public:
    void calculate();
    void backPropagate(double realoutput[], double abspp, double** inputError);
    void backPropagateError(double** outputError, double abspp, double** inputError);
    void memLearn(double pp);
    void setInputs(double input[]);
    void setInput(double in, int t, int i);
    double getInput(int t, int i);
    void getOutputs(double output[]);
    double getOutput(int t, int i);
    double getRawOutput(int t, int i);
    void printWeights(WINDOW *win, int lineoffset);
    bool saveWeights(const char* file);
    bool saveChildCluster(const char* file, int* layer, int* node, int depth);
    bool loadWeights(const char* file);
    bool loadChildCluster(const char* file, int* layer, int* node, int depth);
    void printState(WINDOW *win, int lineoffset);
    void learn(double pp);
    void learn(double pp, double ** inputError);
    void learnRawOutput(double* correctoutput, double learnfactor, int size);
    void learnRawOutput(double* correctoutput, double learnfactor, int size, double** inputError);
    void sleep(int n);
    void resetInputs();
    void setNextInput(double in);
    void propagateError(double* outputError, double** inputError, int turnsBack);
    void addCluster(int layer, int node, Cluster* cluster);
    void updateWeights(double abspp);
    int getInputNumber(int layer, int node, WeightType weightType, int fromNode);
    int getMinDepth();
    void clearNodeError();
    void copyWeights(Cluster* source, double changeChance);
    Cluster* getChildCluster(int* layer, int* node, int depth);
    void mutateWeights(double mutateFactor);

    double getInputToNode(int node, int input);
    double getNodeToNode(int layer, int node1, int node2);
    double getThreshold(int layer, int node);
    double getNodeToOutput(int output, int node);
    double getOutputThreshold(int output);
    double getSideWeight(int layer, int node1, int node2);
    double getBackWeight(int layer, int node1, int node2);
    double getForwardMem(int layer, int node1, int node2);
    double getSideMem(int layer, int node1, int node2);
    double getBackMem(int layer, int node1, int node2);

    //returns the actual pointers
    double **getInputToNodes();
    double ***getNodesToNodes();
    double **getThresholds();
    double **getNodesToOutput();
    double *getOutputThresholds();
    double ***getSideWeights();
    double ***getBackWeights();
    double ***getForwardMems();
    double ***getSideMems();
    double ***getBackMems();

    void setOutputThreshold(double thresh, int num);    //for setting the initial ballpark output value when the output is non-binary
    void setConvolutionBase(Cluster* convBase);

    void setIsConvChild(bool c) {
        isConvChild = c;
    }

    bool getIsConvChild() {
        return isConvChild;
    }

    ClusterParameters* getPars() {return pars;}

    Cluster() {
        pars = new ClusterParameters();
        createCluster();
    }

    Cluster(ClusterParameters *cpars) {
        pars = cpars;
        createCluster();
    }

    ~Cluster() {
        for(int i=0; i<pars->numLayers; i++) {
            for(int j=0; j<pars->nodesPerLayer; j++) {
                if(nodeClusters[i][j] != NULL)
                    delete nodeClusters[i][j];
            }
        }
        deleteArrays();
        delete pars;
    }

    private:
    void zeroArrays();
    void createCluster();
    void defineArrays();
    void deleteArrays();
    void deleteConvolutionSharedArrays();
    void initializeWeights();
    //tlevel is the level of the transfer functions
    double transferFunction(double in, int layer);
    double transferDerivative(double in, int layer);
    int getTransferType(int layer, int level);
    void propagateNodeError(int layer, int node, int turnsBack, double** inputError);
    double getRandomErrorWeight();
    double getOutputErrorWeight(double output, double realoutput);
    double memCorrelationChange(double node1, double node2, double pp);

    void incrementTurn() {
        clusterTurn++;
    }

    int getTurn(int n) {
        if(clusterTurn + n < 0)
            return 0;
        return (clusterTurn + n) % pars->numTurnsSaved;
    }
};
#endif
