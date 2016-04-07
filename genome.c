#include "genome.h"

Genome* Genome::copy() {
    Genome* cp = new Genome();
    for(int i=0; i<CHILDDEPTH; i++) {
        if(pars[i] != NULL)
            cp->pars[i] = pars[i]->copy();
        else
            cp->pars[i] = NULL;
    }
    cp->convProperties = convProperties;
    cp->defaultConvProp = defaultConvProp;

    cp->setExtraAnswerTurns(extraAnswerTurns);
    cp->setNumPerturbRuns(numPerturbRuns);
    cp->setPerturbChance(perturbChance);
    cp->setPerturbFactor(perturbFactor);
    cp->setNumConvolutionLayers(numConvolutionLayers);
    cp->setNumConvolutionTypes(numConvolutionTypes);

    return cp;
}

Genome* Genome::mate(Genome* partner) {
    Genome* child = new Genome();

    //fedaultConvProp
    if(coinflip())
        child->defaultConvProp = defaultConvProp;
    else
        child->defaultConvProp = partner->defaultConvProp;

    //extraAnswerTurns
    if(coinflip())
        child->setExtraAnswerTurns(getExtraAnswerTurns());
    else
        child->setExtraAnswerTurns(partner->getExtraAnswerTurns());
    if(mutate()) {
        if(coinflip()) {
            if(child->getExtraAnswerTurns() > 0)
                child->setExtraAnswerTurns(child->getExtraAnswerTurns() - 1);
        }
        else
            child->setExtraAnswerTurns(child->getExtraAnswerTurns() + 1);
    }

    //numPerturbRuns
    if(coinflip())
        child->setNumPerturbRuns(getNumPerturbRuns());
    else
        child->setNumPerturbRuns(partner->getNumPerturbRuns());
    if(mutate()) {
        if(coinflip()) {
            if(child->getNumPerturbRuns() > 0)
                child->setNumPerturbRuns(child->getNumPerturbRuns() - 1);
        }
        else {
            child->setNumPerturbRuns(child->getNumPerturbRuns() + 1);
        }
    }

    //perturbChance
    if(coinflip())
        child->setPerturbChance(getPerturbChance());
    else
        child->setPerturbChance(partner->getPerturbChance());
    if(mutate()) {
        if(coinflip()) {
            if(child->getPerturbChance() >= 0.05)
                child->setPerturbChance(child->getPerturbChance() - 0.05);
        }
        else if(child->getPerturbChance() <= 0.95)
            child->setPerturbChance(child->getPerturbChance() + 0.05);
    }

    //perturbFactor
    if(coinflip())
        child->setPerturbFactor(getPerturbFactor());
    else
        child->setPerturbFactor(partner->getPerturbFactor());
    if(mutate()) {
        if(coinflip()) {
            if(child->getPerturbFactor() >= 0.1)
                child->setPerturbFactor(child->getPerturbFactor() - 0.1);
        }
        else if(child->getPerturbFactor() <= 0.9)
            child->setPerturbFactor(child->getPerturbFactor() + 0.1);
    }

    for(int i=0; i<CHILDDEPTH; i++) {
        if(raremutate()) {
            if(coinflip() && i != 0)  //50% chance to be null (except on the first level)
                continue;
            else if(mutate()) { //chance to make new random cluster
                child->pars[i] = createRandomClusterParameters();
            }
            else {  //else choose a cluster from the parents at random
                ClusterParameters** p;
                if(coinflip())
                    p = pars;
                else
                    p = partner->pars;

                while(child->pars[i] == NULL) {
                    int r = rand() % CHILDDEPTH;
                    if(p[r] != NULL)
                        child->pars[i] = p[r]->copy();
                }
            }
        }
        else if(pars[i] == NULL && partner->pars[i] == NULL) continue;
        else if(pars[i] == NULL) {
            if(coinflip())
                continue;
            else
                child->pars[i] = partner->pars[i]->copy();
        }
        else if(partner->pars[i] == NULL) {
            if(coinflip())
                continue;
            else
                child->pars[i] = pars[i]->copy();
        }
        else if(coinflip())
            child->pars[i] = pars[i]->copy();
        else
            child->pars[i] = partner->pars[i]->copy();

        //numLayers
        if(pars[i] != NULL && (coinflip() || partner->pars[i] == NULL))
            child->pars[i]->numLayers = pars[i]->numLayers;
        else if(partner->pars[i] != NULL)
            child->pars[i]->numLayers = partner->pars[i]->numLayers;
        else
            child->pars[i]->numLayers = getRandomNumLayers();

        if(mutate()) {
            child->pars[i]->numLayers += -1 + (rand()%3);
            if(child->pars[i]->numLayers <= 0)
                child->pars[i]->numLayers = 1;
            else if(MAX_LAYERS > 0 && child->pars[i]->numLayers > MAX_LAYERS)
                child->pars[i]->numLayers = MAX_LAYERS;
        }

        //nodesPerLayer
        if(pars[i] != NULL && (coinflip() || partner->pars[i] == NULL))
            child->pars[i]->nodesPerLayer = pars[i]->nodesPerLayer;
        else if(partner->pars[i] != NULL)
            child->pars[i]->nodesPerLayer = partner->pars[i]->nodesPerLayer;
        else
            child->pars[i]->nodesPerLayer = getRandomNodesPerLayer();

        if(mutate()) {
            int maxChange = ceil(0.1*child->pars[i]->nodesPerLayer);
            int change = -maxChange + rand() % (2*maxChange + 1);
            child->pars[i]->nodesPerLayer += change;
            if(child->pars[i]->nodesPerLayer <= 0)
                child->pars[i]->nodesPerLayer = 1;
        }

        //stepfactor
        if(pars[i] != NULL && (coinflip() || partner->pars[i] == NULL))
            child->pars[i]->stepfactor = pars[i]->stepfactor;
        else if(partner->pars[i] != NULL)
            child->pars[i]->stepfactor = partner->pars[i]->stepfactor;
        else
            child->pars[i]->stepfactor = getRandomStepFactor();

        if(mutate()) {
            double steplog = log(child->pars[i]->stepfactor);
            steplog = gaussianRandom(steplog, 1);
            child->pars[i]->stepfactor = exp(steplog);
        }

        //memfactor
        if(pars[i] != NULL && (coinflip() || partner->pars[i] == NULL))
            child->pars[i]->memfactor = pars[i]->memfactor;
        else if(partner->pars[i] != NULL)
            child->pars[i]->memfactor = partner->pars[i]->memfactor;
        else
            child->pars[i]->memfactor = getRandomMemFactor();

        if(mutate()) {
            double memlog = log(child->pars[i]->memfactor);
            memlog = gaussianRandom(memlog, 1);
            child->pars[i]->memfactor = exp(memlog);
        }

        //memnorm
        if(pars[i] != NULL && (coinflip() || partner->pars[i] == NULL))
            child->pars[i]->memnorm = pars[i]->memnorm;
        else if(partner->pars[i] != NULL)
            child->pars[i]->memnorm = partner->pars[i]->memnorm;
        else
            child->pars[i]->memnorm = getRandomMemNorm();

        if(mutate()) {
            double memlog = log(1-child->pars[i]->memnorm);
            memlog = gaussianRandom(memlog, 0.1);
            child->pars[i]->memnorm = 1-exp(memlog);
            if(child->pars[i]->memnorm < 0)
                child->pars[i]->memnorm = 0;
        }

        //learnStyleSide
        if(pars[i] != NULL && (coinflip() || partner->pars[i] == NULL))
            child->pars[i]->learnStyleSide = pars[i]->learnStyleSide;
        else if(partner->pars[i] != NULL)
            child->pars[i]->learnStyleSide = partner->pars[i]->learnStyleSide;
        else
            child->pars[i]->learnStyleSide = getRandomLearnStyleSide();

        if(mutate()) {
            child->pars[i]->learnStyleSide = getRandomLearnStyleSide();
        }

        //transferWidth
        if(pars[i] != NULL && (coinflip() || partner->pars[i] == NULL))
            child->pars[i]->transferWidth = pars[i]->transferWidth;
        else if(partner->pars[i] != NULL)
            child->pars[i]->transferWidth = partner->pars[i]->transferWidth;
        else
            child->pars[i]->transferWidth = getRandomTransferWidth();

        if(mutate()) {
            double tranlog = log(child->pars[i]->transferWidth);
            tranlog = gaussianRandom(tranlog, 1);
            child->pars[i]->transferWidth = exp(tranlog);
        }
    }

    //Convolutions
    child->convProperties.clear();
    size_t nChildLayers = (coinflip() ? convProperties.size() : partner->convProperties.size());
    child->convProperties.resize(nChildLayers);
    child->setNumConvolutionLayers(nChildLayers);
    child->resizeNumConvolutionTypes(nChildLayers);
    for(size_t layer=0; layer<nChildLayers; layer++) {
        bool p1hasLayer = layer < convProperties.size();
        bool p2hasLayer = layer < partner->convProperties.size();
        size_t nConv;
        if(p1hasLayer && (coinflip() || !p2hasLayer))
            nConv = convProperties[layer].size();
        else
            nConv = partner->convProperties[layer].size();

        child->convProperties[layer].resize(nConv);

        for(size_t i=0; i<nConv; i++) {
            ConvolutionProperties childCP;
            if((p1hasLayer && i<convProperties[layer].size()) && (coinflip() || (p2hasLayer && i>=partner->convProperties[layer].size())))
                childCP = convProperties[layer][i];
            else
                childCP = partner->convProperties[layer][i];
            childCP = mutateConvolutionProperties(childCP);
            child->convProperties[layer][i] = childCP;
        }
    }

    for(size_t layer=0; layer<child->convProperties.size(); layer++) {
        if(mutate() && child->convProperties[layer].size() < MAX_CONVOLUTIONS) {
            child->convProperties[layer].push_back(getRandomConvolutionProperties(layer));
        }
        if(mutate() && child->convProperties[layer].size() > MIN_CONVOLUTIONS) {
            child->convProperties[layer].erase(child->convProperties[layer].begin() + rand()%((int)child->convProperties[layer].size()));
        }
        child->setNumConvolutionTypes(layer, child->convProperties[layer].size());
    }

    if(raremutate() && child->convProperties.size() < MAX_CONVOLUTION_LEVELS) {
        std::vector<ConvolutionProperties> convProps = getRandomConvolutionPropertiesLayer(child->convProperties.size());
        child->convProperties.push_back(convProps);
        child->addNumConvolutionTypesLayer(convProps.size());
    }
    if(raremutate() && child->convProperties.size() > MIN_CONVOLUTION_LEVELS) {
        child->convProperties.pop_back();
        child->eraseNumConvolutionTypesLayer();
    }
    child->setNumConvolutionLayers(child->convProperties.size());
    return child;
}

int Genome::getRandomNumLayers() {
    return (rand()%MAX_LAYERS) + 1;
}

int Genome::getRandomNodesPerLayer() {
    return (rand()%MAX_NODESPERLAYER) + 1;
}

double Genome::getRandomStepFactor() {
    int x = rand()%7;
    return exp(-x+3);
}

double Genome::getRandomMemFactor() {
    return exp(-(rand()%24)/2.0+1);
}

double Genome::getRandomMemNorm() {
    return 1-exp(-(rand()%20)/4.0-1);
}

int Genome::getRandomLearnStyleSide() {
    if(ALLOW_SIDE_WEIGHTS && ALLOW_SIDE_MEMS)
        return rand()%5;
    else if(ALLOW_SIDE_WEIGHTS) {
        if(coinflip())
            return LEARNSTYLENONE;
        else
            return LEARNSTYLEBP;
    }
    else if(ALLOW_SIDE_MEMS) {
        if(coinflip())
            return LEARNSTYLENONE;
        else
            return LEARNSTYLEHB;
    }
    else
        return LEARNSTYLENONE;
}

int Genome::getRandomExtraAnswerTurns() {
    return (int)fabs(gaussianRandom(0, 2));
}

int Genome::getRandomNumPerturbRuns() {
    int np = (int)fabs(gaussianRandom(0, 3));
    if(np > MAX_NUMPERTURBS)
        np = MAX_NUMPERTURBS;
    return np;
}

double Genome::getRandomPerturbChance() {
    return (rand()%50)/100.0;
}

double Genome::getRandomPerturbFactor() {
    return (rand()%100)/100.0;
}

double Genome::getRandomTransferWidth() {
    return exp((rand()%10)/2);
}

ClusterParameters* Genome::createRandomClusterParameters() {
    ClusterParameters* rPars = new ClusterParameters();
    rPars->numLayers = getRandomNumLayers();
    rPars->nodesPerLayer = getRandomNodesPerLayer();
    rPars->stepfactor = getRandomStepFactor();
    rPars->memfactor = getRandomMemFactor();
    rPars->memnorm = getRandomMemNorm();
    rPars->learnStyleSide = getRandomLearnStyleSide();
    rPars->transferWidth = getRandomTransferWidth();

    rPars->numInputs = 2;
    rPars->numOutputs = 1;
    rPars->numTurnsSaved = NUM_TURNS_SAVED;
    rPars->randomWeights = 1;
    rPars->useBackWeights = 0;
    rPars->backPropBackWeights = 0;
    rPars->useBackMems = 0;
    rPars->useForwardMems = 0;
    rPars->propThresh = 0.0;
    rPars->bpMemStrengths = false;
    rPars->tlevel = 0;
    rPars->copyInputsToFirstLevel = false;
    rPars->lockMaxMem = false;
    return rPars;
}

void Genome::createRandomGenome() {
    for(int i=0; i<CHILDDEPTH; i++) {
        if(rand()%(i+1) == 0)
            pars[i] = createRandomClusterParameters();
        else
            pars[i] = NULL;
    }

    extraAnswerTurns = getRandomExtraAnswerTurns();
    numPerturbRuns = getRandomNumPerturbRuns();
    perturbChance = getRandomPerturbChance();
    perturbFactor = getRandomPerturbFactor();

    convProperties.clear();
    numConvolutionLayers = getRandomNumConvolutionLayers();
    numConvolutionTypes = getRandomNumConvolutionTypes(numConvolutionLayers);
    convProperties.resize(numConvolutionLayers);
    for(int i=0; i<numConvolutionLayers; i++) {
        for(int j=0; j<numConvolutionTypes[i]; j++) {
            convProperties[i].push_back(getRandomConvolutionProperties(i));
        }
    }
}

bool Genome::mutate() {
    return rand() % 10 == 0;
}

bool Genome::raremutate() {
    return rand() % 50 == 0;
}

bool Genome::coinflip() {
    return rand() % 2 == 0;
}

double Genome::gaussianRandom(double in, double stdev) {
    static std::random_device rd;
    //static std::ranlux64_base_01 rgen(rd);
    //static std::default_random_engine rgen(rd());
    static std::mt19937 rgen(rd());
    std::normal_distribution<double> dist(in, stdev);
    return dist(rgen);
}

void Genome::saveGenome(const char* file) {
    std::ofstream outfile(file);
    outfile << "-1 extraAnswerTurns: " << extraAnswerTurns << std::endl;
    outfile << "-1 numPerturbRuns: " << numPerturbRuns << std::endl;
    outfile << "-1 perturbChance: " << perturbChance << std::endl;
    outfile << "-1 perturbFactor: " << perturbFactor << std::endl;

    for(int i=0; i<CHILDDEPTH; i++) {
        if(pars[i] != NULL) {
            outfile << i << " numLayers: " << pars[i]->numLayers << std::endl;
            outfile << i << " nodesPerLayer: " << pars[i]->nodesPerLayer << std::endl;
            outfile << i << " stepfactor: " << pars[i]->stepfactor << std::endl;
            outfile << i << " memfactor: " << pars[i]->memfactor << std::endl;
            outfile << i << " memnorm: " << pars[i]->memnorm << std::endl;
            outfile << i << " learnStyleSide: " << pars[i]->learnStyleSide << std::endl;
            outfile << i << " transferWidth: " << pars[i]->transferWidth << std::endl;
        }
    }

    outfile << "-1 numConvolutionLayers: " << numConvolutionLayers << std::endl;
    outfile << "-1 numConvolutionTypes: ";
    if(numConvolutionLayers==0)
        outfile << "-1";
    else {
        for(int i=0; i<numConvolutionLayers; i++) {
            outfile << numConvolutionTypes[i] << " ";
        }
    }
    outfile << std::endl;

    for(int layer=0; layer<numConvolutionLayers; layer++) {
        for(int i=0; i<numConvolutionTypes[layer]; i++) {
            outfile << "-1 Convolution: " << layer;
            outfile << " rank: " << convProperties[layer][i].rank;
            outfile << " dimensions: ";
            for(int j=0; j<convProperties[layer][i].rank; j++) {
                outfile << convProperties[layer][i].dimensions[j] << " ";
            }
            outfile << " inputRangeBegin: " << convProperties[layer][i].inputRangeBegin;
            outfile << " inputRangeEnd: " << convProperties[layer][i].inputRangeEnd;
            outfile << " inputSpaceDimensions: ";
            for(int j=0; j<convProperties[layer][i].rank; j++) {
                outfile << convProperties[layer][i].inputSpaceDimensions[j] << " ";
            }
            outfile << " numLayers: " << convProperties[layer][i].numLayers;
            outfile << " nodesPerLayer: " << convProperties[layer][i].nodesPerLayer;
            outfile << " stepfactor: " << convProperties[layer][i].stepfactor;
            outfile << " transferWidth: " << convProperties[layer][i].transferWidth;
            outfile << std::endl;
        }
    }
}

void Genome::loadGenome(const char* file) {
    //clear previous pars
    for(int i=0; i<CHILDDEPTH; i++) {
        if(pars[i] != NULL) {
            delete pars[i];
            pars[i] = NULL;
        }
    }
    std::ifstream infile(file);
    if(infile.is_open()) {
        std::string line;
        while(getline(infile, line)) {
            std::istringstream iss(line);
            std::string token1, token2, token3;
            if(!(iss>>token1>>token2>>token3))
                continue;

            int p;
            if(!(std::istringstream(token1) >> p)) continue;

            if(p < 0) {
                if(token2=="extraAnswerTurns:") {
                    std::istringstream(token3) >> extraAnswerTurns;
                }
                else if(token2=="numPerturbRuns:") {
                    std::istringstream(token3) >> numPerturbRuns;
                }
                else if(token2=="perturbChance:") {
                    std::istringstream(token3) >> perturbChance;
                }
                else if(token2=="perturbFactor:") {
                    std::istringstream(token3) >> perturbFactor;
                }
                else if(token2=="numConvolutionLayers:") {
                    std::istringstream(token3) >> numConvolutionLayers;
                    convProperties.resize(numConvolutionLayers);
                }
                else if(token2=="numConvolutionTypes:") {
                    numConvolutionTypes.resize(numConvolutionLayers);
                    for(int i=0; i<numConvolutionLayers; i++) {
                        std::istringstream(token3) >> numConvolutionTypes[i];
                        if(i<numConvolutionLayers-1)
                            iss >> token3;
                    }
                }
                else if(token2=="Convolution:") {
                    int convLayer;
                    std::istringstream(token3) >> convLayer;
                    ConvolutionProperties cp;
                    std::string subtoken1, subtoken2;
                    while(iss>>subtoken1>>subtoken2) {
                        if(subtoken1=="rank:") {
                            std::istringstream(subtoken2) >> cp.rank;
                        }
                        else if(subtoken1=="dimensions:") {
                            cp.dimensions.clear();
                            int dim;
                            std::istringstream(subtoken2) >> dim;
                            cp.dimensions.push_back(dim);
                            for(int i=1; i<cp.rank; i++) {
                                iss>>subtoken2;
                                std::istringstream(subtoken2) >> dim;
                                cp.dimensions.push_back(dim);
                            }
                        }
                        else if(subtoken1=="inputRangeBegin:") {
                            std::istringstream(subtoken2) >> cp.inputRangeBegin;
                        }
                        else if(subtoken1=="inputRangeEnd:") {
                            std::istringstream(subtoken2) >> cp.inputRangeEnd;
                        }
                        else if(subtoken1=="inputSpaceDimensions:") {
                            cp.inputSpaceDimensions.clear();
                            int dim;
                            std::istringstream(subtoken2) >> dim;
                            cp.inputSpaceDimensions.push_back(dim);
                            for(int i=1; i<cp.rank; i++) {
                                iss>>subtoken2;
                                std::istringstream(subtoken2) >> dim;
                                cp.inputSpaceDimensions.push_back(dim);
                            }
                        }
                        else if(subtoken1=="numLayers:") {
                            std::istringstream(subtoken2) >> cp.numLayers;
                        }
                        else if(subtoken1=="nodesPerLayer:") {
                            std::istringstream(subtoken2) >> cp.nodesPerLayer;
                        }
                        else if(subtoken1=="stepfactor:") {
                            std::istringstream(subtoken2) >> cp.stepfactor;
                        }
                        else if(subtoken1=="transferWidth:") {
                            std::istringstream(subtoken2) >> cp.transferWidth;
                        }
                    }
                    convProperties[convLayer].push_back(cp);
                }
            }
            if(p >= CHILDDEPTH)
                continue;

            if(pars[p] == NULL)
                pars[p] = createRandomClusterParameters();

            if(token2=="numLayers:") {
                std::istringstream(token3) >> pars[p]->numLayers;
            }
            else if(token2=="nodesPerLayer:") {
                std::istringstream(token3) >> pars[p]->nodesPerLayer;
            }
            else if(token2=="stepfactor:") {
                std::istringstream(token3) >> pars[p]->stepfactor;
            }
            else if(token2=="memfactor:") {
                std::istringstream(token3) >> pars[p]->memfactor;
            }
            else if(token2=="memnorm:") {
                std::istringstream(token3) >> pars[p]->memnorm;
            }
            else if(token2=="learnStyleSide:") {
                std::istringstream(token3) >> pars[p]->learnStyleSide;
            }
            else if(token2=="transferWidth:") {
                std::istringstream(token3) >> pars[p]->transferWidth;
            }
        }
        infile.close();
    }
    else
        throw std::runtime_error("couldn't open genome file");

}

bool Genome::hasSideWeights() {
    bool sideWeights = false;
    for(int i=0; i<CHILDDEPTH; i++) {
        if(pars[i] == NULL)
            break;
        if(pars[i]->learnStyleSide != LEARNSTYLENONE) {
            sideWeights = true;
            break;
        }
    }
    return sideWeights;
}

ConvolutionProperties Genome::mutateConvolutionProperties(ConvolutionProperties cp) {
    //dimensions
    for(int i=0; i<cp.rank; i++) {
        if(mutate())
            cp.dimensions[i]++;
        if(mutate() && cp.dimensions[i] > 1)
            cp.dimensions[i]--;
    }

    //numLayers
    if(raremutate() && cp.numLayers < MAX_CONVOLUTION_NODE_LAYERS)
        cp.numLayers++;
    if(raremutate() && cp.numLayers > 1)
        cp.numLayers--;

    //nodesPerLayer
    if(mutate() && cp.nodesPerLayer < MAX_CONVOLUTION_NODESPERLAYER)
        cp.nodesPerLayer++;
    if(mutate() && cp.nodesPerLayer > 1)
        cp.nodesPerLayer--;

    //stepfactor
    if(mutate()) {
        double steplog = log(cp.stepfactor);
        steplog = gaussianRandom(steplog, 1);
        cp.stepfactor = exp(steplog);
    }

    //transferWidth
    if(mutate()) {
        double tranlog = log(cp.transferWidth);
        tranlog = gaussianRandom(tranlog, 1);
        cp.transferWidth = exp(tranlog);
    }
    return cp;
}

std::vector<ConvolutionProperties> Genome::getRandomConvolutionPropertiesLayer(int level) {
    std::vector<ConvolutionProperties> levelCPs;
    int numCPs = getRandomNumConvolutions();
    for(int i=0; i<numCPs; i++) {
        levelCPs.push_back(getRandomConvolutionProperties(level));
    }
    return levelCPs;
}

ConvolutionProperties Genome::getRandomConvolutionProperties(int level) {
    ConvolutionProperties cp;

    cp.rank = defaultConvProp.rank;
    cp.inputRangeBegin = defaultConvProp.inputRangeBegin;
    cp.inputRangeEnd = defaultConvProp.inputRangeEnd;
    cp.inputSpaceDimensions = defaultConvProp.inputSpaceDimensions;

    cp.dimensions = getRandomConvolutionDimensions(cp.rank, level);
    cp.numLayers = getRandomConvolutionNumLayers();
    cp.nodesPerLayer = getRandomConvolutionNodesPerLayer();
    cp.stepfactor = getRandomConvolutionStepFactor();
    cp.transferWidth = getRandomConvolutionTransferWidth();

    return cp;

}

std::vector<int> Genome::getRandomConvolutionDimensions(int rank, int level) {
    std::vector<int> dim;
    if(level == 0) {
        for(int i=0; i<rank; i++) {
            dim.push_back(rand()%(MAX_CONVOLUTION_DIMENSION-MIN_CONVOLUTION_DIMENSION+1) + MIN_CONVOLUTION_DIMENSION);
        }
    }
    else {
        int maxDimension = (int)MAX_CONVOLUTION_DIMENSION*pow(CONVOLUTION_DIMENSION_LAYER_MULTIPLIER, level);
        int minDimension = (int)MAX_CONVOLUTION_DIMENSION*pow(CONVOLUTION_DIMENSION_LAYER_MULTIPLIER, level-1);
        for(int i=0; i<rank; i++) {
            dim.push_back(rand()%(maxDimension-minDimension+1) + minDimension);
        }
    }
    return dim;
}

int Genome::getRandomNumConvolutionLayers() {
    return rand()%(MAX_CONVOLUTION_LEVELS-MIN_CONVOLUTION_LEVELS+1) + MIN_CONVOLUTION_LEVELS;
}

int Genome::getRandomNumConvolutions() {
    return rand()%(MAX_CONVOLUTIONS-MIN_CONVOLUTIONS+1) + MIN_CONVOLUTIONS;
}

std::vector<int> Genome::getRandomNumConvolutionTypes(int numLevels) {
    std::vector<int> nc;
    for(int i=0; i<numLevels; i++) {
        nc.push_back(getRandomNumConvolutions());
    }
    return nc;
}

int Genome::getRandomConvolutionNumLayers() {
    return rand()%MAX_CONVOLUTION_NODE_LAYERS + 1;
}

int Genome::getRandomConvolutionNodesPerLayer() {
    return rand()%MAX_CONVOLUTION_NODESPERLAYER +1;
}

double Genome::getRandomConvolutionStepFactor() {
    return getRandomStepFactor();
}

double Genome::getRandomConvolutionTransferWidth() {
    return getRandomTransferWidth();
}
