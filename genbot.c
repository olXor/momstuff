#include "genbot.h"

Cluster* Genbot::createCluster(Genome* genome, int numInputs, int numOutputs) {
    if(genome->pars[0] == NULL)
        return NULL;
    genome->pars[0]->numInputs = numInputs;
    genome->pars[0]->numOutputs = numOutputs;

    ClusterParameters* pars = genome->pars[0]->copy();
    Cluster* clust = new Cluster(pars);
    fillClusters(clust, genome, 1);
    return clust;
}

void Genbot::fillClusters(Cluster* cluster, Genome* genome, int depth) {
    if(depth >= genome->getChildDepth())
        return;

    for(int i=0; i<cluster->getPars()->numLayers; i++) {
        for(int j=0; j<cluster->getPars()->nodesPerLayer; j++) {
            if(genome->pars[depth] == NULL)
                break;
            genome->pars[depth]->numInputs = cluster->getInputNumber(i, j, BLANKWEIGHT, 0);
            genome->pars[depth]->numOutputs = 1;
            ClusterParameters* pars = genome->pars[depth]->copy();
            Cluster* child = new Cluster(pars);
            cluster->addCluster(i, j, child);
            fillClusters(child, genome, depth+1);
        }
    }
}

//layer and node are arrays of length depth (which must be <= CHILDDEPTH)
void Genbot::loadCluster(const char* file, int* layer, int* node, int depth) {
    cluster->loadChildCluster(file, layer, node, depth);
}

void Genbot::saveCluster(const char* file, int* layer, int* node, int depth) {
    if(!cluster->saveChildCluster(file, layer, node, depth))
        throw std::runtime_error("couldn't save cluster file");
}

void Genbot::loadBot(std::string loc) {
    int maxdepth = getChildDepth();
    int layer[maxdepth];
    int node[maxdepth];
    for(int i=0; i<maxdepth; i++) {
        layer[i] = 0;
        node[i] = 0;
    }
    loadChildren(layer, node, 0, loc);

    for(size_t layer=0; layer<genome->convProperties.size(); layer++) {
        size_t nConv = 0;
        for(size_t i=0; i<genome->convProperties[layer].size(); i++) {
            std::stringstream cname;
            cname << loc << "_conv" << layer << "-" << i;
            int numInstances = numConvolutionInstances[layer][i];
            if(numInstances > 0)
                convolutions[layer][nConv]->loadWeights(cname.str().c_str());
            nConv += numInstances;
        }
    }
}

void Genbot::loadChildren(int* layer, int* node, int depth, std::string loc) {
    if(depth >= getChildDepth() || getPars()[depth] == NULL) return;

    std::ostringstream filename;
    filename << loc << ".txt";
    loadCluster(filename.str().c_str(), layer, node, depth);

    for(int i=0; i<getPars()[depth]->numLayers; i++) {
        for(int j=0; j<getPars()[depth]->nodesPerLayer; j++) {
            int* nlayer = new int[getChildDepth()];
            int* nnode = new int[getChildDepth()];
            for(int k=0; k<depth; k++) {
                nlayer[k] = layer[k];
                nnode[k] = node[k];
            }

            nlayer[depth] = i;
            nnode[depth] = j;
            std::ostringstream nloc;
            nloc << loc << "-{" << i << "-" << j << "}";
            loadChildren(nlayer, nnode, depth+1, nloc.str());
            delete [] nlayer;
            delete [] nnode;
        }
    }
}

void Genbot::saveBot(std::string loc) {
    int maxdepth = getChildDepth();
    int layer[maxdepth];
    int node[maxdepth];
    for(int i=0; i<maxdepth; i++) {
        layer[i] = 0;
        node[i] = 0;
    }
    saveChildren(layer, node, 0, loc);
    for(size_t layer=0; layer<genome->convProperties.size(); layer++) {
        size_t nConv = 0;
        for(size_t i=0; i<genome->convProperties[layer].size(); i++) {
            std::stringstream cname;
            cname << loc << "_conv" << layer << "-" << i;
            int numInstances = numConvolutionInstances[layer][i];
            if(numInstances > 0)
                convolutions[layer][nConv]->saveWeights(cname.str().c_str());
            nConv += numInstances;
        }
    }
}

void Genbot::saveChildren(int* layer, int* node, int depth, std::string loc) {
    if(depth >= getChildDepth() || getPars()[depth] == NULL) return;

    std::ostringstream filename;
    filename << loc << ".txt";
    saveCluster(filename.str().c_str(), layer, node, depth);

    for(int i=0; i<getPars()[depth]->numLayers; i++) {
        for(int j=0; j<getPars()[depth]->nodesPerLayer; j++) {
            int* nlayer = new int[getChildDepth()];
            int* nnode = new int[getChildDepth()];
            for(int k=0; k<depth; k++) {
                nlayer[k] = layer[k];
                nnode[k] = node[k];
            }

            nlayer[depth] = i;
            nnode[depth] = j;
            std::ostringstream nloc;
            nloc << loc << "-{" << i << "-" << j << "}";
            saveChildren(nlayer, nnode, depth+1, nloc.str());
            delete [] nlayer;
            delete [] nnode;
        }
    }
}

void Genbot::setInputs(double input[], int size) {
    if(getNumInputsWithConvolutions(size) != getPars()[0]->numInputs)
        throw std::runtime_error("Gave Genbot the wrong number of inputs");

    if(giveClusterRawData) {
        double* rawSaveInputs = new double[size];
        for(int i=0; i<size; i++) {
            rawSaveInputs[i] = input[i];
        }

        if(!alreadySetInputs)
            savedRawInputs.push(rawSaveInputs);
        else {
            for(int i=0; i<size; i++)
                savedRawInputs.back()[i] = input[i];
        }

        if(!alreadySetInputs) {
            if(savedRawInputs.size() >= convMinDepth) {
                for(int i=0; i<size; i++)
                    cluster->setInput(savedRawInputs.front()[i], 0, i);
                savedRawInputs.pop();
            }
            else {
                for(int i=0; i<size; i++) {
                    cluster->setInput(0, 0, i);
                }
            }
        }
    }

    for(size_t i=0; i<convolutions[0].size(); i++) {
        double cinputs[numConvolutionInputs[0][i]];
        getConvolutionInputs(0, i, input, cinputs);
        convolutions[0][i]->setInputs(cinputs);
    }
    alreadySetInputs = true;
}

void Genbot::getOutputs(double output[], int size) {
    if(size != getPars()[0]->numOutputs)
        throw std::runtime_error("Gave Genbot the wrong number of outputs");
    cluster->getOutputs(output);
}

//blanks inputs on every turn but the first (and the first if blankFirstTurn is set)
void Genbot::progressTurns(int turns, bool blankFirstTurn) {
    double blankinputs[numInputs];
    for(int i=0; i<numInputs; i++) {
        blankinputs[i] = 0;
    }

    int clusterInputOffset = (giveClusterRawData ? getPars()[0]->numInputs : 0);

    int maxConvolutionLevel = convolutions.size()-1;
    int numConvolutionsOnLastLevel = convolutions[maxConvolutionLevel].size();
    for(int i=0; i<turns; i++) {
        if(i > 0 || blankFirstTurn) {
            setInputs(blankinputs, numInputs);
        }

        for(size_t layer=0; layer<convolutions.size(); layer++) {
            for(size_t j=0; j<convolutions[layer].size(); j++) {
                if(layer != 0) {
                    for(size_t k=0; k<convolutionGlobalInputNums[layer][j].size(); k++) {
                        convolutions[layer][j]->setInput(convolutions[layer-1][convolutionGlobalInputNums[layer][j][k]]->getOutput(0,0), 0, k);
                    }
                }
                convolutions[layer][j]->calculate();
            }
        }
        for(int i=0; i<numConvolutionsOnLastLevel; i++) {
            cluster->setInput(convolutions[maxConvolutionLevel][i]->getOutput(0,0),0, i + clusterInputOffset);
        }
        cluster->calculate();
        alreadySetInputs = false;
    }
}

void Genbot::progressTurnsSaved() {
    progressTurns(getPars()[0]->numTurnsSaved, true);
}

void Genbot::copyWeights(Genbot* source, double changeChance) {
    int maxdepth = getChildDepth();
    int layer[maxdepth];
    int node[maxdepth];
    for(int i=0; i<maxdepth; i++) {
        layer[i] = 0;
        node[i] = 0;
    }
    copyChildrenWeights(source, changeChance, layer, node, 0);

    for(size_t layer=0; layer<genome->convProperties.size(); layer++) {
        for(size_t i=0; i<genome->convProperties[layer].size(); i++) {
            Cluster* convSource = source->getConvolutionCluster(layer, i);
            if(convSource != NULL)
                getConvolutionCluster(layer, i)->copyWeights(convSource, changeChance);
        }
    }
}

void Genbot::copyChildrenWeights(Genbot* source, double changeChance, int* layer, int* node, int depth) {
    if(depth >= getChildDepth() || getPars()[depth] == NULL) return;

    Cluster* sourceCluster = source->getCluster()->getChildCluster(layer, node, depth);
    Cluster* destCluster = cluster->getChildCluster(layer, node, depth);

    destCluster->copyWeights(sourceCluster, changeChance);

    for(int i=0; i<getPars()[depth]->numLayers; i++) {
        for(int j=0; j<getPars()[depth]->nodesPerLayer; j++) {
            int* nlayer = new int[getChildDepth()];
            int* nnode = new int[getChildDepth()];
            for(int k=0; k<depth; k++) {
                nlayer[k] = layer[k];
                nnode[k] = node[k];
            }

            nlayer[depth] = i;
            nnode[depth] = j;
            copyChildrenWeights(source, changeChance, nlayer, nnode, depth+1);
            delete [] nlayer;
            delete [] nnode;
        }
    }
}

int Genbot::getNumConvolutionInputs(ConvolutionProperties cp, int layer) {
    int numInputs;
    if(layer == 0) {
        numInputs = 1;
        if(cp.rank <= 0)
            throw std::runtime_error("found convolution of invalid rank");
        for(int i=0; i<cp.rank; i++) {
            numInputs *= cp.dimensions[i];
        }
    }
    else {
        int convNum = 0;
        numInputs = 0;
        std::vector<int> cloc;
        cloc.resize(cp.rank);
        for(int i=0; i<cp.rank; i++)
            cloc[i] = 0;

        for(size_t i=0; i<numConvolutionInstances[layer-1].size(); i++) {
            for(int j=0; j<numConvolutionInstances[layer-1][i]; j++) {
                std::vector<int> smallLoc = getConvolutionLocation(genome->convProperties[layer-1][i],j);
                if(convolutionContainedIn(smallLoc, genome->convProperties[layer-1][i].dimensions, cloc, cp.dimensions)) {
                    numInputs++;
                }
                convNum++;
            }
        }
    }
    return numInputs;
}

bool Genbot::checkConvolutionRange(ConvolutionProperties cp) {
    int numInputs = 1;
    if(cp.rank <= 0)
        throw std::runtime_error("found convolution of invalid rank");
    for(int i=0; i<cp.rank; i++) {
        numInputs *= cp.inputSpaceDimensions[i];
    }
    return numInputs == cp.inputRangeEnd - cp.inputRangeBegin + 1;
}

int Genbot::getNumConvolutionInstances(ConvolutionProperties cp) {
    int numInstances = 1;
    for(int i=0; i<cp.rank; i++) {
        if(cp.dimensions[i] <= 0)
            throw std::runtime_error("found convolution of invalid dimension");
        int nFits = (cp.inputSpaceDimensions[i] - cp.dimensions[i] + 1);
        if(nFits <= 0)
            return 0;
        numInstances *= nFits;
    }
    return numInstances;
}

ClusterParameters* Genbot::getConvolutionClusterParameters(ConvolutionProperties cp, int layer) {
    ClusterParameters* pars = new ClusterParameters();
    pars->numLayers = cp.numLayers;
    pars->nodesPerLayer = cp.nodesPerLayer;
    pars->stepfactor = cp.stepfactor;
    pars->transferWidth = cp.transferWidth;
    pars->numInputs = getNumConvolutionInputs(cp, layer);
    pars->numOutputs = 1;

    pars->memfactor = 1;
    pars->memnorm = 1;
    pars->learnStyleSide = LEARNSTYLENONE;
    pars->numTurnsSaved = NUM_TURNS_SAVED;
    pars->randomWeights = 1;
    pars->useBackWeights = 0;
    pars->backPropBackWeights = 0;
    pars->useBackMems = 0;
    pars->useForwardMems = 0;
    pars->propThresh = 0.0;
    pars->bpMemStrengths = false;
    pars->tlevel = 0;
    pars->copyInputsToFirstLevel = false;
    pars->lockMaxMem = false;

    return pars;
}

void Genbot::createConvolutions() {
    convolutions.clear();
    convolutions.resize(genome->getNumConvolutionLayers());
    numConvolutionInstances.clear();
    numConvolutionInstances.resize(genome->getNumConvolutionLayers());
    convolutionProperties.clear();
    convolutionProperties.resize(genome->getNumConvolutionLayers());
    numConvolutionInputs.clear();
    numConvolutionInputs.resize(genome->getNumConvolutionLayers());
    convolutionGlobalInputNums.clear();
    convolutionGlobalInputNums.resize(genome->getNumConvolutionLayers());

    for(int layer=0; layer<genome->getNumConvolutionLayers(); layer++) {
        for(size_t i=0; i<genome->convProperties[layer].size(); i++) {
            if(layer==0 && !checkConvolutionRange(genome->convProperties[layer][i]))
                throw std::runtime_error("found convolution with invalid input range");
            ClusterParameters* pars = getConvolutionClusterParameters(genome->convProperties[layer][i], layer);
            int numConvolutionInstances = getNumConvolutionInstances(genome->convProperties[layer][i]);
            Cluster* convBase;
            for(int j=0; j<numConvolutionInstances; j++) {
                Cluster* convClust = new Cluster(pars->copy());
                convolutions[layer].push_back(convClust);
                if(j==0)
                    convBase = convClust;
                else {
                    convClust->setConvolutionBase(convBase);
                    convClust->setIsConvChild(true);
                }
            }
            delete pars;
        }

        numConvolutionInstances[layer].resize(genome->convProperties[layer].size());
        convolutionProperties[layer].resize(convolutions[layer].size());
        int convNum = 0;
        for(size_t i=0; i<genome->convProperties[layer].size(); i++) {
            numConvolutionInstances[layer][i] = getNumConvolutionInstances(genome->convProperties[layer][i]);
            for(int j=convNum; j<convNum + numConvolutionInstances[layer][i]; j++) {
                convolutionProperties[layer][j] = genome->convProperties[layer][i];
            }
            convNum += numConvolutionInstances[layer][i];
        }

        numConvolutionInputs[layer].resize(convolutions[layer].size());
        if(layer==0) {
            for(size_t i=0; i<convolutions[layer].size(); i++) {
                numConvolutionInputs[layer][i] = getNumConvolutionInputs(convolutionProperties[layer][i],layer);
            }
        }

        convolutionGlobalInputNums[layer].resize(convolutions[layer].size());
        convNum = 0;
        for(size_t i=0; i<genome->convProperties[layer].size(); i++) {
            for(int j=convNum; j<convNum + numConvolutionInstances[layer][i]; j++) {
                convolutionGlobalInputNums[layer][j] = getConvolutionGlobalInputNums(genome->convProperties[layer][i], layer, j-convNum);
                numConvolutionInputs[layer][j] = convolutionGlobalInputNums[layer][j].size();
            }
            convNum += numConvolutionInstances[layer][i];
        }
    }

    convMinDepth = getConvMinDepth();   //has to go at the end of this method because it uses some of the above arrays

    size_t maxConvsOnLayer = 0;
    for(size_t i=0; i<convolutions.size(); i++) {
        if(convolutions[i].size() > maxConvsOnLayer)
            maxConvsOnLayer = convolutions[i].size();
    }
    maxConvInputs.resize(maxConvsOnLayer);
    for(size_t i=0; i<maxConvsOnLayer; i++) {
        maxConvInputs[i] = 0;
        for(size_t j=0; j<convolutions.size(); j++) {
            if(i < convolutions[j].size() && convolutions[j][i]->getPars()->numInputs > maxConvInputs[i]) {
                maxConvInputs[i] = convolutions[j][i]->getPars()->numInputs;
            }
        }
    }
}

int Genbot::convolutionGlobalLocationToInput(ConvolutionProperties cp, std::vector<int> loc) {
    int in = 0;
    int mult = 1;
    for(int i=0; i<cp.rank; i++) {
        in += loc[i]*mult;
        mult *= cp.inputSpaceDimensions[i];
    }

    return in;
}

std::vector<int> Genbot::convolutionLocalInputToLocation(ConvolutionProperties cp, int in) {
    std::vector<int> loc;
    int mult = 1;
    for(int i=0; i<cp.rank; i++) {
        loc.push_back(in%(cp.dimensions[i]*mult));
        mult *= cp.dimensions[i];
    }
    return loc;
}

std::vector<int> Genbot::getConvolutionLocation(ConvolutionProperties cp, int n) {
    std::vector<int> cloc;
    int mult = 1;
    for(int i=0; i<cp.rank; i++) {
        int d = n%((cp.inputSpaceDimensions[i] - cp.dimensions[i] + 1)*mult);
        mult *= (cp.inputSpaceDimensions[i] - cp.dimensions[i] + 1);
        cloc.push_back(d);
    }
    return cloc;
}

std::vector<int> addLocations(std::vector<int> loc1, std::vector<int> loc2) {
    std::vector<int> loc;
    for(size_t i=0; i<std::min(loc1.size(), loc2.size()); i++) {
        loc.push_back(loc1[i]+loc2[i]);
    }
    return loc;
}

bool Genbot::convolutionContainedIn(std::vector<int> smallLoc, std::vector<int> smallDim, std::vector<int> bigLoc, std::vector<int> bigDim) {
    for(size_t i=0; i<smallLoc.size(); i++) {
        if(smallLoc[i] < bigLoc[i] || smallLoc[i] + smallDim[i] > bigLoc[i] + bigDim[i])
            return false;
    }
    return true;
}

std::vector<int> Genbot::getConvolutionGlobalInputNums(ConvolutionProperties cp, int layer, int n) {
    std::vector<int> globInputs;
    
    std::vector<int> cloc = getConvolutionLocation(cp, n);

    if(layer == 0) {
        int numInputs = getNumConvolutionInputs(cp, layer);
        globInputs.resize(numInputs);

        for(int i=0; i<numInputs; i++) {
            std::vector<int> relLoc = convolutionLocalInputToLocation(cp, i);
            std::vector<int> inLoc = addLocations(cloc, relLoc);
            globInputs[i] = convolutionGlobalLocationToInput(cp, inLoc);
            if(globInputs[i] < cp.inputRangeBegin || globInputs[i] > cp.inputRangeEnd)
                throw std::runtime_error("Convolution tried to look at an out-of-bounds input!");
        }
    }
    else {
        int convNum = 0;
        for(size_t i=0; i<numConvolutionInstances[layer-1].size(); i++) {
            for(int j=0; j<numConvolutionInstances[layer-1][i]; j++) {
                if(convolutionContainedIn(getConvolutionLocation(genome->convProperties[layer-1][i],j), genome->convProperties[layer-1][i].dimensions, cloc, cp.dimensions)) {
                    globInputs.push_back(convNum);
                }
                convNum++;
            }
        }
    }

    return globInputs;
}

void Genbot::getConvolutionInputs(int layer, int n, double* input, double* cinputs) {
    for(int i=0; i<numConvolutionInputs[layer][n]; i++) {
        cinputs[i] = input[convolutionGlobalInputNums[layer][n][i]];
    }
}

Cluster* Genbot::getConvolutionCluster(size_t layer, size_t n) {
    if(n >= genome->convProperties[layer].size())
        return NULL;

    int nConv = 0;
    for(size_t i=0; i<n; i++) {
        nConv += numConvolutionInstances[layer][i];
    }
    return convolutions[layer][nConv];
}

int Genbot::getNumInputsWithConvolutions(int nInputs) {
    int n = 0;
    if(giveClusterRawData)
        n += nInputs;
    n += convolutions[convolutions.size()-1].size();
    return n;
}

int Genbot::getInputNumberFromConvolutionNumber(int n) {
    if(giveClusterRawData)
        n += genome->pars[0]->numInputs - convolutions[convolutions.size()-1].size();
    return n;
}

void Genbot::backPropagateConvolutions(double** inputError, double abspp) {
    size_t numTurnsSaved = cluster->getPars()->numTurnsSaved;
    size_t numConvLayers = convolutions.size();
    size_t numConvsOnLayer = convolutions[numConvLayers-1].size();
    size_t maxConvsOnLayer = 0;
    for(size_t i=0; i<numConvLayers; i++) {
        if(convolutions[i].size() > maxConvsOnLayer)
            maxConvsOnLayer = convolutions[i].size();
    }
    double*** convInputError = new double**[maxConvsOnLayer];
    double*** convOutputError = new double**[maxConvsOnLayer];
    for(size_t i=0; i<maxConvsOnLayer; i++) {
        convInputError[i] = new double*[numTurnsSaved];
        convOutputError[i] = new double*[numTurnsSaved];
        for(size_t j=0; j<numTurnsSaved; j++) {
            convOutputError[i][j] = new double[1];
            convInputError[i][j] = new double[maxConvInputs[i]];
        }
    }

    for(size_t i=0; i<numConvsOnLayer; i++) {
        for(size_t j=0; j<numTurnsSaved; j++) {
            convOutputError[i][j][0] = inputError[j][getInputNumberFromConvolutionNumber(i)]; //gotta be very careful with the turn indices when moving between clusters
        }

        size_t numInputs = convolutions[numConvLayers-1][i]->getPars()->numInputs;
        for(size_t j=0; j<numTurnsSaved; j++) {
            for(size_t k=0; k<numInputs; k++) {
                convInputError[i][j][k] = 0;
            }
        }

        convolutions[numConvLayers-1][i]->backPropagateError(convOutputError[i], abspp, convInputError[i]);
    }

    for(int layer=numConvLayers-2; layer>=0; layer--) {
        size_t numConvsOnHigherLayer = convolutions[layer+1].size();
        numConvsOnLayer = convolutions[layer].size();
        for(size_t i=0; i<numConvsOnLayer; i++) {
            for(size_t j=0; j<numTurnsSaved; j++) {
                convOutputError[i][j][0] = 0;
            }
        }

        //propagate the error between convolution layers
        for(size_t i=0; i<numConvsOnHigherLayer; i++) {
            size_t higherConvNumInputs = convolutions[layer+1][i]->getPars()->numInputs;
            for(size_t j=0; j<higherConvNumInputs; j++) {
                int inputConv = convolutionGlobalInputNums[layer+1][i][j];
                for(size_t k=0; k<numTurnsSaved; k++) {
                    convOutputError[inputConv][k][0] += convInputError[i][k][j];
                }
            }
        }

        if(layer != 0) {
            for(size_t i=0; i<numConvsOnLayer; i++) {
                size_t convNumInputs = convolutions[layer][i]->getPars()->numInputs;
                for(size_t j=0; j<numTurnsSaved; j++) {
                    for(size_t k=0; k<convNumInputs; k++) {
                        convInputError[i][j][k] = 0;
                    }
                }
            }
        }
        else {
            for(size_t i=0; i<maxConvsOnLayer; i++) {
                for(size_t j=0; j<numTurnsSaved; j++) {
                    delete [] convInputError[i][j];
                }
                delete [] convInputError[i];
            }
            delete [] convInputError;
            convInputError = NULL;
        }

        for(size_t i=0; i<numConvsOnLayer; i++) {
            convolutions[layer][i]->backPropagateError(convOutputError[i], abspp, (convInputError != NULL ? convInputError[i] : NULL));
        }
    }
    
    if(convInputError != NULL) {
        for(size_t i=0; i<maxConvsOnLayer; i++) {
            for(size_t j=0; j<numTurnsSaved; j++) {
                delete [] convInputError[i][j];
            }
            delete [] convInputError[i];
        }
        delete [] convInputError;
    }

    for(size_t i=0; i<maxConvsOnLayer; i++) {
        for(size_t j=0; j<numTurnsSaved; j++) {
            delete [] convOutputError[i][j];
        }
        delete [] convOutputError[i];
    }
    delete [] convOutputError;
}

int Genbot::getConvMinDepth() {
    int minConvDepth = 0;
    for(size_t layer=0; layer<genome->convProperties.size(); layer++) {
        int layerMinDepth = 999999;
        for(size_t i=0; i<genome->convProperties[layer].size(); i++) { 
            int cDepth = getConvolutionCluster(layer, i)->getMinDepth() - 1;
            if(cDepth < layerMinDepth)
                layerMinDepth = cDepth;
        }
        minConvDepth += layerMinDepth;
    }
    return minConvDepth;
}

int Genbot::getMinDepth() {
    if(convolutions.size() > 0)
        return cluster->getMinDepth() + getConvMinDepth();
    else
        return cluster->getMinDepth();
}

void Genbot::learn(double pp) {
    double** inputError = new double*[cluster->getPars()->numTurnsSaved];
    for(int i=0; i<cluster->getPars()->numTurnsSaved; i++) {
        inputError[i] = new double[cluster->getPars()->numInputs];
        for(int j=0; j<cluster->getPars()->numInputs; j++)
            inputError[i][j] = 0;
    }

    cluster->learn(pp, inputError);

    backPropagateConvolutions(inputError, fabs(pp));

    for(int i=0; i<cluster->getPars()->numTurnsSaved; i++) {
        delete [] inputError[i];
    }
    delete [] inputError;
}

void Genbot::learnRawOutput(double* correctoutput, double learnfactor, int size) {
    double** inputError = new double*[cluster->getPars()->numTurnsSaved];
    for(int i=0; i<cluster->getPars()->numTurnsSaved; i++) {
        inputError[i] = new double[cluster->getPars()->numInputs];
        for(int j=0; j<cluster->getPars()->numInputs; j++)
            inputError[i][j] = 0;
    }

    cluster->learnRawOutput(correctoutput, learnfactor, size, inputError);

    backPropagateConvolutions(inputError, learnfactor);

    for(int i=0; i<cluster->getPars()->numTurnsSaved; i++) {
        delete [] inputError[i];
    }
    delete [] inputError;
}
