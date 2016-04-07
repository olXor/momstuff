#ifndef GENBOT_HEADER_FILE
#define GENBOT_HEADER_FILE

#define DllExport __declspec( dllexport )

#include "cluster.h"
#include "genome.h"
#include <fstream>
#include <queue>
//#include "nvwa/debug_new.h"

#define MUTATE_WEIGHT_FACTOR 0.05
#define GIVE_CLUSTER_RAWDATA false

class DllExport Genbot {
    private:
    Genome* genome;
    Cluster* cluster;
    int id;
    std::vector<std::vector<int>> numConvolutionInstances;   //size numConvolutionLayers, numConvolutionTypes
    std::vector<std::vector<Cluster*>> convolutions;
    std::vector<std::vector<int>> numConvolutionInputs;      //size numConvolutions
    std::vector<std::vector<ConvolutionProperties>> convolutionProperties;   //size numConvolutions
    std::vector<std::vector<std::vector<int>>> convolutionGlobalInputNums;
    int numClusterInputs;
    bool giveClusterRawData = GIVE_CLUSTER_RAWDATA;
    std::queue<double*> savedRawInputs;
    size_t convMinDepth;
    int numInputs;
    bool alreadySetInputs = false;
    std::vector<int> maxConvInputs;

    public:
    Genbot() {
        genome = new Genome();
        genome->createRandomGenome();
        createConvolutions();
        cluster = createCluster(genome, getNumInputsWithConvolutions(2), 1);
        numInputs = 2;
        id = -1;
    }

    Genbot(Genome* g, int nInputs, int numOutputs, int ID) {
        genome = g;
        createConvolutions();
        cluster = createCluster(genome, getNumInputsWithConvolutions(nInputs), numOutputs);
        numInputs = nInputs;
        id = ID;
    }

    Genbot(int nInputs, int numOutputs, int ID) {
        genome = new Genome();
        genome->createRandomGenome();
        createConvolutions();
        cluster = createCluster(genome, getNumInputsWithConvolutions(nInputs), numOutputs);
        numInputs = nInputs;
        id = ID;
    }

    int getChildDepth() {
        return genome->getChildDepth();
    }

    int getExtraAnswerTurns() {
        return genome->getExtraAnswerTurns();
    }

    int getNumPerturbRuns() {
        return genome->getNumPerturbRuns();
    }

    double getPerturbChance() {
        return genome->getPerturbChance();
    }

    double getPerturbFactor() {
        return genome->getPerturbFactor();
    }

    bool hasSideWeights() {
        return genome->hasSideWeights();
    }

    ClusterParameters** getPars() {
        return genome->pars;
    }

    ~Genbot() {
        delete genome;
        delete cluster;
        for(size_t i=0; i<convolutions.size(); i++) {
            for(size_t j=0; j<convolutions[i].size(); j++) {
                delete convolutions[i][j];
            }
        }
    }

    int getID() {
        return id;
    }

    int getConvMinDepth();
    int getMinDepth();

    void backPropagateConvolutions(double** inputError, double abspp);
    void learn(double pp);
    void learnRawOutput(double* correctoutput, double learnfactor, int size);

    void mutateWeights() {
        cluster->mutateWeights(MUTATE_WEIGHT_FACTOR);
    }

    Genome* getGenome() {
        return genome;
    }

    Cluster* getCluster() {
        return cluster;
    }

    void setOutputThreshold(double thresh, int num) {
        cluster->setOutputThreshold(thresh, num);
    }

    void copyWeights(Genbot* source, double changeChance);

    void loadBot(std::string loc);
    void saveBot(std::string loc);
    void setInputs(double input[], int size);
    void getOutputs(double output[], int size);
    void progressTurns(int turns, bool blankFirstTurn);
    void progressTurnsSaved();

    private:
    Cluster* createCluster(Genome* genome, int numInputs, int numOutputs);
    void fillClusters(Cluster* cluster, Genome* genome, int depth);
    void loadChildren(int* layer, int* node, int depth, std::string loc);
    void saveChildren(int* layer, int* node, int depth, std::string loc);
    void loadCluster(const char* file, int* layer, int* node, int depth);
    void saveCluster(const char* file, int* layer, int* node, int depth);
    void copyChildrenWeights(Genbot* source, double changeChance, int* layer, int* node, int depth);
    void createConvolutions();
    int getConvolutionInputNumber(ConvolutionProperties cp, int* location);
    int getNumConvolutionInputs(ConvolutionProperties cp, int layer);
    bool checkConvolutionRange(ConvolutionProperties cp);
    int getNumConvolutionInstances(ConvolutionProperties cp);
    ClusterParameters* getConvolutionClusterParameters(ConvolutionProperties cp, int layer);
    std::vector<int> getConvolutionGlobalInputNums(ConvolutionProperties cp, int layer, int n);
    void getConvolutionInputs(int layer, int n, double* input, double* cinputs);
    std::vector<int> convolutionLocalInputToLocation(ConvolutionProperties cp, int in);
    int convolutionGlobalLocationToInput(ConvolutionProperties cp, std::vector<int> loc);
    Cluster* getConvolutionCluster(size_t layer, size_t n);
    int getNumInputsWithConvolutions(int nInputs);
    int getInputNumberFromConvolutionNumber(int n);
    bool convolutionContainedIn(std::vector<int> smallLoc, std::vector<int> smallDim, std::vector<int> bigLoc, std::vector<int> bigDim);
    std::vector<int> getConvolutionLocation(ConvolutionProperties cp, int n);

};

#endif
