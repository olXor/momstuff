#include "cluster.h"
#include <Windows.h>
#include "fastonebigheader.h"

void doubleToString(double x, char* str);

void Cluster::createCluster() {
    clusterTurn = 0;
    defineArrays();
    zeroArrays();
    initializeWeights();
}

void Cluster::setInputs(double input[]) {
    for(int i=0; i<pars->numInputs; i++)
        savedInputs[getTurn(0)][i] = input[i];
}

void Cluster::resetInputs() {
    for(int i=0; i<pars->numInputs; i++) {
        savedInputs[getTurn(0)][i] = 0;
    }
    currentInput = 0;
}

void Cluster::setInput(double in, int t, int i) {
    if(i < pars->numInputs)
        savedInputs[getTurn(t)][i] = in;
}

void Cluster::setNextInput(double in) {
    if(in > pars->numInputs)
        throw std::out_of_range("Cluster: no more inputs");
    savedInputs[getTurn(0)][currentInput] = in;
    currentInput++;
}

double Cluster::getInput(int t, int i) {
    if(i < pars->numInputs)
        return savedInputs[getTurn(t)][i];
    return 0;
}

void Cluster::getOutputs(double output[]) {
    for(int i=0; i<pars->numOutputs; i++)
        output[i] = transferFunction(savedOutputs[getTurn(0)][i],pars->numLayers);
}

double Cluster::getOutput(int t, int i) {
    if(i < pars->numOutputs)
        return transferFunction(savedOutputs[getTurn(t)][i],pars->numLayers);
    return 0;
}

double Cluster::getRawOutput(int t, int i) {
    if(i < pars->numOutputs)
        return savedOutputs[getTurn(t)][i];
    return 0;
}

void Cluster::learn(double pp) {
    if(pp == 0) return;
    double realoutput[pars->numOutputs];
    for(int i=0; i<pars->numOutputs; i++) {
        if(pp >= 0) {
            if(round(transferFunction(savedOutputs[getTurn(0)][i],pars->numLayers)) == 0)
                realoutput[i] = 0;
            else
                realoutput[i] = 1;
        }
        else {
            if(round(transferFunction(savedOutputs[getTurn(0)][i],pars->numLayers)) == 0)
                realoutput[i] = 1;
            else
                realoutput[i] = 0;
        }
    }

    memLearn(pp);
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            if(nodeClusters[i][j] != NULL)
                nodeClusters[i][j]->memLearn(pp);
        }
    }
    backPropagate(realoutput, fabs(pp), NULL);
}

void Cluster::learn(double pp, double** inputError) {
    if(pp == 0) return;
    double realoutput[pars->numOutputs];
    for(int i=0; i<pars->numOutputs; i++) {
        if(pp >= 0) {
            if(round(transferFunction(savedOutputs[getTurn(0)][i],pars->numLayers)) == 0)
                realoutput[i] = 0;
            else
                realoutput[i] = 1;
        }
        else {
            if(round(transferFunction(savedOutputs[getTurn(0)][i],pars->numLayers)) == 0)
                realoutput[i] = 1;
            else
                realoutput[i] = 0;
        }
    }

    memLearn(pp);
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            if(nodeClusters[i][j] != NULL)
                nodeClusters[i][j]->memLearn(pp);
        }
    }
    backPropagate(realoutput, fabs(pp), inputError);
}

void Cluster::learnRawOutput(double* correctoutput, double learnfactor, int size) {
    if(size != pars->numOutputs)
        throw std::runtime_error("tried to learn with correctoutput of the wrong size!");

    backPropagate(correctoutput, learnfactor, NULL);
}

void Cluster::learnRawOutput(double* correctoutput, double learnfactor, int size, double** inputError) {
    if(size != pars->numOutputs)
        throw std::runtime_error("tried to learn with correctoutput of the wrong size!");

    backPropagate(correctoutput, learnfactor, inputError);
}

void Cluster::calculate() {
    incrementTurn();
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            if(nodeClusters[i][j] != NULL) {
                nodeClusters[i][j]->resetInputs();
            }
        }
    }
    //clear the current turn's weights
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            savedNodeStrengths[getTurn(0)][i][j] = 0;
        }
    }
    for(int i=0; i<pars->numOutputs; i++)
        savedOutputs[getTurn(0)][i] = 0;
    //go from inputs to first layer
    for(int i=0; i<pars->nodesPerLayer; i++) {
        for(int j=0; j<pars->numInputs; j++) {
            if(nodeClusters[0][i] == NULL)
                savedNodeStrengths[getTurn(0)][0][i] += inputToNodes[i][j]*savedInputs[getTurn(-1)][j];
            else {
                nodeClusters[0][i]->setInput(inputToNodes[i][j]*savedInputs[getTurn(-1)][j],0,getInputNumber(0, i, FORWARDWEIGHT, j));
            }
        }
    }

    //side weights
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++)
            for(int k=0; k<pars->nodesPerLayer; k++) {
                if(pars->learnStyleSide == LEARNSTYLEBP || (pars->learnStyleSide == LEARNSTYLEALT && i%2==1) || (pars->learnStyleSide == LEARNSTYLEALTR && i%2==0)) {
                    if(nodeClusters[i][j] == NULL) {
                        savedNodeStrengths[getTurn(0)][i][j] += sideWeights[i][j][k]*transferFunction(savedNodeStrengths[getTurn(-1)][i][k],i);
                    }
                    else {
                        nodeClusters[i][j]->setInput(sideWeights[i][j][k]*transferFunction(savedNodeStrengths[getTurn(-1)][i][k],i),0,getInputNumber(i, j, SIDEWEIGHT, k));
                    }
                }
                if(pars->learnStyleSide == LEARNSTYLEHB || (pars->learnStyleSide == LEARNSTYLEALT && i%2==0) || (pars->learnStyleSide == LEARNSTYLEALTR && i%2==1)) {
                    if(nodeClusters[i][j] == NULL) {
                        savedNodeStrengths[getTurn(0)][i][j] += sideMems[i][j][k]*memStrengths[i][k]*transferFunction(savedNodeStrengths[getTurn(-1)][i][k],i);
                    }
                    else {
                        nodeClusters[i][j]->setInput(sideMems[i][j][k]*memStrengths[i][k]*transferFunction(savedNodeStrengths[getTurn(-1)][i][k],i), 0, getInputNumber(i, j, SIDEMEM, k));
                    }
                }
            }
    }

    //back weights
    for(int i=0; i<pars->numLayers-1; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++)
            for(int k=0; k<pars->nodesPerLayer; k++) {
                if(pars->useBackWeights) {
                    if(nodeClusters[i][j] == NULL) {
                        savedNodeStrengths[getTurn(0)][i][j] += backWeights[i][j][k]*transferFunction(savedNodeStrengths[getTurn(-1)][i+1][k],i);
                    }
                    else {
                        nodeClusters[i][j]->setInput(backWeights[i][j][k]*transferFunction(savedNodeStrengths[getTurn(-1)][i+1][k],i), 0, getInputNumber(i, j, BACKWEIGHT, k));
                    }
                }
                if(pars->useBackMems) {
                    if(nodeClusters[i][j] == NULL) {
                        savedNodeStrengths[getTurn(0)][i][j] += backMems[i][j][k]*memStrengths[i+1][k]*transferFunction(savedNodeStrengths[getTurn(-1)][i+1][k],i);
                    }
                    else {
                        nodeClusters[i][j]->setInput(backMems[i][j][k]*memStrengths[i+1][k]*transferFunction(savedNodeStrengths[getTurn(-1)][i+1][k],i), 0, getInputNumber(i, j, BACKMEM, k));
                    }
                }
            }
    }

    //forward weights
    for(int i=1; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            for(int k=0; k<pars->nodesPerLayer; k++) {
                if(nodeClusters[i][j] == NULL) {
                    savedNodeStrengths[getTurn(0)][i][j] += nodesToNodes[i-1][j][k]*transferFunction(savedNodeStrengths[getTurn(-1)][i-1][k],i-1);
                }
                else {
                    nodeClusters[i][j]->setInput(nodesToNodes[i-1][j][k]*transferFunction(savedNodeStrengths[getTurn(-1)][i-1][k],i-1), 0, getInputNumber(i, j, FORWARDWEIGHT, k));
                }
                if(pars->useForwardMems) {
                    if(nodeClusters[i][j] == NULL) {
                        savedNodeStrengths[getTurn(0)][i][j] += forwardMems[i-1][j][k]*memStrengths[i-1][k]*transferFunction(savedNodeStrengths[getTurn(-1)][i-1][k],i-1);
                    }
                    else {
                        nodeClusters[i][j]->setInput(forwardMems[i-1][j][k]*memStrengths[i-1][k]*transferFunction(savedNodeStrengths[getTurn(-1)][i-1][k],i-1), 0, getInputNumber(i, j, FORWARDMEM, k));
                    }
                }
            }
        }
    }

    //calculate child clusters
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            if(nodeClusters[i][j] != NULL) {
                nodeClusters[i][j]->calculate();
                savedNodeStrengths[getTurn(0)][i][j] = nodeClusters[i][j]->getRawOutput(0,0);
            }
        }
    }
    //thresholds
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            savedNodeStrengths[getTurn(0)][i][j] -= thresholds[i][j];
        }
    }
    //go from final node layer to output neuron
    for(int j=0; j<pars->numOutputs; j++) {
        for(int i=0; i<pars->nodesPerLayer; i++) {
            savedOutputs[getTurn(0)][j] += nodesToOutput[j][i]*transferFunction(savedNodeStrengths[getTurn(-1)][pars->numLayers-1][i],pars->numLayers-1);
        }
        savedOutputs[getTurn(0)][j] -= outputThresholds[j];
    }
}

int Cluster::getTransferType(int layer, int level) {
    if(!pars->useOutputTransfer && layer==pars->numLayers)
        return 0; //identity
    if(level==1)
        return 3; //arcsinh
    //return 1; //sigmoid
    return 2;   //rectifier
}

inline double Cluster::transferFunction(double in, int layer) {
    int type = getTransferType(layer,pars->tlevel);

    switch(type) {
        case 0: //identity
            return in;
        case 1: //sigmoid
            if(in/pars->transferWidth > 300)
                return 1;
            if(in/pars->transferWidth < -300)
                return 0;
            return 1/(1+fastexp(-in/pars->transferWidth));
        case 2: //rectifier:
            if(in/pars->transferWidth > 300)
                return in/pars->transferWidth;
            if(in/pars->transferWidth < -300)
                return 0;
            return fastlog(1+fastexp(in/pars->transferWidth));
        case 3: //arcsinh
            return fastlog(in + sqrt(pow(in,2)+1));
    }
    return 0.5;
}

inline double Cluster::transferDerivative(double in, int layer) {
    int type = getTransferType(layer,pars->tlevel);

    switch(type) {
        case 0: //identity
            return 1;
        case 1: //derivative of sigmoid
            if(fabs(in/pars->transferWidth)>300)
                return 0;
            return (1/pars->transferWidth)*fastexp(in/pars->transferWidth)/fastpow2(1+fastexp(in/pars->transferWidth));
        case 2: //derivative of rectifier (a sigmoid)
            if(in/pars->transferWidth > 300)
                return 1/pars->transferWidth;
            if(in/pars->transferWidth < -300)
                return 0;
            return (1/pars->transferWidth)/(1+fastexp(-in/pars->transferWidth));
        case 3: //arcsinh
            return 1/sqrt(1+pow(in,2));
    }
    return 0;
}

void printWeightsHelper(WINDOW *win, int* ld, int lineoffset, int pluslines, const char* format, ...) {
    va_list argptr;
    va_start(argptr, format);
    if(*ld >= lineoffset)
        vwprintw(win, format, argptr);
    va_end(argptr);

    *ld += pluslines;
}

void Cluster::printWeights(WINDOW *win, int lineoffset) {
    int ld = 0;
    //input to first layer
    printWeightsHelper(win, &ld, lineoffset, 1, "in to 0\n");
    for(int i=0; i<pars->nodesPerLayer; i++) {
        printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f; ", thresholds[0][i]);
        for(int j=0; j<pars->numInputs; j++) {
            printWeightsHelper(win, &ld, lineoffset, 0, " % 6.2f", inputToNodes[i][j]);
        }
        printWeightsHelper(win, &ld, lineoffset, 1, "\n");
    }

    //between layers
    for(int k=1; k<pars->numLayers; k++) {
        printWeightsHelper(win, &ld, lineoffset, 1, "%d to %d\n",k-1, k);
        for(int i=0; i<pars->nodesPerLayer; i++) {
            printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f; ", thresholds[k][i]);
            for(int j=0; j<pars->nodesPerLayer; j++) {
                printWeightsHelper(win, &ld, lineoffset, 0, " % 6.2f", nodesToNodes[k-1][i][j]);
            }
            printWeightsHelper(win, &ld, lineoffset, 1, "\n");
        }
    }

        //mems
    for(int i=0; i<pars->numLayers; i++) {
        if(pars->learnStyleSide == LEARNSTYLEBP || (pars->learnStyleSide == LEARNSTYLEALT && i%2==1) || (pars->learnStyleSide == LEARNSTYLEALTR && i%2==0)) {
            printWeightsHelper(win, &ld, lineoffset, 1, "side weights %d\n",i);
            for(int j=0; j<pars->nodesPerLayer; j++) {
                for(int k=0; k<pars->nodesPerLayer; k++) {
                    printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f ", sideWeights[i][j][k]);
                }
                printWeightsHelper(win, &ld, lineoffset, 1, "\n");
            }
        }
        if(pars->learnStyleSide == LEARNSTYLEHB || (pars->learnStyleSide == LEARNSTYLEALT && i%2==0) || (pars->learnStyleSide == LEARNSTYLEALTR && i%2==1)) {
            printWeightsHelper(win, &ld, lineoffset, 1, "side mems %d\n",i);
            for(int j=0; j<pars->nodesPerLayer; j++) {
                printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f ", memStrengths[i][j]);
            }
            printWeightsHelper(win, &ld, lineoffset, 1, "\n");
            printWeightsHelper(win, &ld, lineoffset, 1, "\n");
            for(int j=0; j<pars->nodesPerLayer; j++) {
                for(int k=0; k<pars->nodesPerLayer; k++) {
                    printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f ", sideMems[i][j][k]);
                }
                printWeightsHelper(win, &ld, lineoffset, 1, "\n");
            }
        }
        if(i<pars->numLayers-1) {
            if(pars->useForwardMems) {
                printWeightsHelper(win, &ld, lineoffset, 1, "forward mems %d\n",i);
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f ", memStrengths[i][j]);
                }
                printWeightsHelper(win, &ld, lineoffset, 1, "\n");
                printWeightsHelper(win, &ld, lineoffset, 1, "\n");
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    for(int k=0; k<pars->nodesPerLayer; k++) {
                        printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f ", forwardMems[i][j][k]);
                    }
                    printWeightsHelper(win, &ld, lineoffset, 1, "\n");
                }
            }
            if(pars->useBackMems) {
                printWeightsHelper(win, &ld, lineoffset, 1, "back mems %d\n",i);
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f ", memStrengths[i+1][j]);
                }
                printWeightsHelper(win, &ld, lineoffset, 1, "\n");
                printWeightsHelper(win, &ld, lineoffset, 1, "\n");
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    for(int k=0; k<pars->nodesPerLayer; k++) {
                        printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f ", backMems[i][j][k]);
                    }
                    printWeightsHelper(win, &ld, lineoffset, 1, "\n");
                }
            }
        }
    }

    //last layer to output
    printWeightsHelper(win, &ld, lineoffset, 1, "%d to out\n", pars->numLayers-1);
    for(int i=0; i<pars->numOutputs; i++) {
        printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f; ", outputThresholds[i]);
        for(int j=0; j<pars->nodesPerLayer; j++) {
            printWeightsHelper(win, &ld, lineoffset, 0, " % 6.2f", nodesToOutput[i][j]);
        }
        printWeightsHelper(win, &ld, lineoffset, 1, "\n");
    }
    printWeightsHelper(win, &ld, lineoffset, 1, "\n");
}

void Cluster::printState(WINDOW* win, int lineoffset) {
    int ld = 0;
    printWeightsHelper(win, &ld, lineoffset, 1, "Inputs\n", pars->numLayers-1);
    for(int i=0; i<pars->numInputs; i++) {
        printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f ", savedInputs[getTurn(0)][i]);
    }
    printWeightsHelper(win, &ld, lineoffset, 1, "\n");

    printWeightsHelper(win, &ld, lineoffset, 1, "Node States\n", pars->numLayers-1);
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f ", savedNodeStrengths[getTurn(0)][i][j]);
        }
        printWeightsHelper(win, &ld, lineoffset, 1, "\n");
    }

    printWeightsHelper(win, &ld, lineoffset, 1, "Outputs\n", pars->numLayers-1);
    for(int i=0; i<pars->numOutputs; i++) {
        printWeightsHelper(win, &ld, lineoffset, 0, "% 6.2f ", savedOutputs[getTurn(0)][i]);
    }
    printWeightsHelper(win, &ld, lineoffset, 1, "\n");
}

int Cluster::getInputNumber(int layer, int node, WeightType weightType, int fromNode) {
    int n = 0;
    (void)node;
    if(weightType == FORWARDWEIGHT) {
        if((layer==0 && fromNode < pars->numInputs) || (layer!=0 && fromNode < pars->nodesPerLayer))
            return n + fromNode;
        throw std::out_of_range("invalid input number");
    }
    if(layer==0) n += pars->numInputs;
    else n += pars->nodesPerLayer;

    if(weightType == FORWARDMEM) {
        if(pars->useForwardMems && layer!=0 && fromNode < pars->nodesPerLayer)
            return n + fromNode;
        throw std::out_of_range("invalid input number");
    }
    if(pars->useForwardMems && layer > 0) n += pars->nodesPerLayer;

    if(weightType == SIDEWEIGHT) {
        if((pars->learnStyleSide == LEARNSTYLEBP || (pars->learnStyleSide == LEARNSTYLEALT && layer%2==1) || (pars->learnStyleSide == LEARNSTYLEALTR && layer%2==0)) && fromNode < pars->nodesPerLayer)
            return n + fromNode;
        throw std::out_of_range("invalid input number");
    }
    if((pars->learnStyleSide == LEARNSTYLEBP || (pars->learnStyleSide == LEARNSTYLEALT && layer%2==1) || (pars->learnStyleSide == LEARNSTYLEALTR && layer%2==0))) n += pars->nodesPerLayer;

    if(weightType == SIDEMEM) {
        if((pars->learnStyleSide == LEARNSTYLEHB || (pars->learnStyleSide == LEARNSTYLEALT && layer%2==0) || (pars->learnStyleSide == LEARNSTYLEALTR && layer%2==1)) && fromNode < pars->nodesPerLayer)
            return n + fromNode;
        throw std::out_of_range("invalid input number");
    }
    if((pars->learnStyleSide == LEARNSTYLEHB || (pars->learnStyleSide == LEARNSTYLEALT && layer%2==0) || (pars->learnStyleSide == LEARNSTYLEALTR && layer%2==1))) n += pars->nodesPerLayer;
    
    if(weightType == BACKWEIGHT) {
        if(pars->useBackWeights && layer < pars->numLayers-1 && fromNode < pars->nodesPerLayer)
            return n + fromNode;
        throw std::out_of_range("invalid input number");
    }
    if(pars->useBackWeights && layer < pars->numLayers-1) n += pars->nodesPerLayer;

    if(weightType == BACKMEM) {
        if(pars->useBackMems && layer < pars->numLayers-1 && fromNode < pars->nodesPerLayer)
            return n + fromNode;
        throw std::out_of_range("invalid input number");
    }
    if(pars->useBackMems && layer < pars->numLayers-1) n += pars->nodesPerLayer;

    if(weightType == BLANKWEIGHT) {
        return n;
    }
    throw std::out_of_range("invalid input number");
}

void Cluster::propagateNodeError(int layer, int node, int turnsBack, double** inputError) {
    if(turnsBack >= pars->numTurnsSaved-2) return;
    double error = preNodeError[getTurn(-turnsBack+1)][layer][node];
    double** childInputError = NULL;
    double childOutputError[1] = {error};     //All children must have only one output atm
    Cluster* childCluster = nodeClusters[layer][node];
    if(childCluster != NULL) {
        childInputError = new double*[pars->numTurnsSaved]; //the child cluster MUST have the same number of turns saved!
        for(int t=0; t<pars->numTurnsSaved; t++) {
            childInputError[t] = new double[childCluster->getPars()->numInputs];
            for(int i=0; i<childCluster->getPars()->numInputs; i++) {
                childInputError[t][i] = 0;
            }
        }
        childCluster->propagateError(childOutputError, childInputError, turnsBack);
        for(int i=0; i<childCluster->getPars()->numInputs; i++) {
            nodeError[getTurn(-turnsBack+1)][layer][node][i] = childInputError[turnsBack][i];
        }
    }
    else {
        nodeError[getTurn(-turnsBack+1)][layer][node][0] = error;
    }

    if(childCluster == NULL && error == 0)
        return;
    else if(childCluster != NULL) {
        bool nonZeroError = false;
        for(int i=0; i<childCluster->getPars()->numInputs; i++) {
            if(childInputError[turnsBack][i] != 0)
                nonZeroError = true;
        }
        if(!nonZeroError) {
            for(int i=0; i<pars->numTurnsSaved; i++) {
                delete [] childInputError[i];
            }
            delete [] childInputError;
            return;
        }
    }

    double newerr = 0;
    //propagate to inputs
    if(layer == 0 && inputError != NULL) {
        for(int i=0; i<pars->numInputs; i++) {
            if(childCluster == NULL) {
                newerr = error*transferDerivative(savedNodeStrengths[getTurn(-turnsBack)][layer][node], layer)*inputToNodes[node][i];

            }
            else
                newerr = childInputError[turnsBack][getInputNumber(layer, node, FORWARDWEIGHT, i)]*inputToNodes[node][i];
            inputError[turnsBack][i] += newerr; 
        }
    }
    //propagate up (forward weights)
    else if(layer > 0) {
        for(int i=0; i<pars->nodesPerLayer; i++) {
            if(childCluster == NULL)
                newerr = error*transferDerivative(savedNodeStrengths[getTurn(-turnsBack)][layer][node], layer)*nodesToNodes[layer-1][node][i];
            else
                newerr = childInputError[turnsBack][getInputNumber(layer, node, FORWARDWEIGHT, i)]*nodesToNodes[layer-1][node][i];

            if(pars->useForwardMems) {
                if(childCluster == NULL)
                    newerr += error*transferDerivative(savedNodeStrengths[getTurn(-turnsBack)][layer][node], layer)*memStrengths[layer-1][i]*forwardMems[layer-1][node][i];
                else 
                    newerr += childInputError[turnsBack][getInputNumber(layer, node, FORWARDMEM, i)]*memStrengths[layer-1][i]*forwardMems[layer-1][node][i];
            }
            preNodeError[getTurn(-turnsBack)][layer-1][i] += newerr;
            //if(fabs(newerr) > pars->propThresh)
            //propagateNodeError(newerr, layer-1, i, t+1, inputError);
        }
    }

    //propagate to the sides
    if(pars->learnStyleSide == LEARNSTYLEBP || pars->learnStyleSide == LEARNSTYLEHB || pars->learnStyleSide == LEARNSTYLEALT || pars->learnStyleSide == LEARNSTYLEALTR) {
        for(int i=0; i<pars->nodesPerLayer; i++) {
            newerr = 0;
            if(pars->learnStyleSide == LEARNSTYLEBP || (pars->learnStyleSide == LEARNSTYLEALT && layer%2==1) || (pars->learnStyleSide == LEARNSTYLEALTR && layer%2==0)) {
                if(childCluster == NULL)
                    newerr += error*transferDerivative(savedNodeStrengths[getTurn(-turnsBack)][layer][node], layer)*sideWeights[layer][node][i];
                else
                    newerr += childInputError[turnsBack][getInputNumber(layer, node, SIDEWEIGHT, i)]*sideWeights[layer][node][i];
            }
            if(pars->learnStyleSide == LEARNSTYLEHB || (pars->learnStyleSide == LEARNSTYLEALT && layer%2==0) || (pars->learnStyleSide == LEARNSTYLEALTR && layer%2==1)) {
                if(childCluster == NULL)
                    newerr += error*transferDerivative(savedNodeStrengths[getTurn(-turnsBack)][layer][node], layer)*memStrengths[layer][i]*sideMems[layer][node][i];
                else
                    newerr += childInputError[turnsBack][getInputNumber(layer, node, SIDEMEM, i)]*memStrengths[layer][i]*sideMems[layer][node][i];
            }

            preNodeError[getTurn(-turnsBack)][layer][i] += newerr;
            //if(fabs(newerr) > pars->propThresh)
            //propagateNodeError(newerr, layer, i, t+1, inputError);
        }
    }

    //propagate down (back weights)
    if((pars->useBackMems || pars->useBackWeights) && layer < pars->numLayers-1) {
        for(int i=0; i<pars->nodesPerLayer; i++) {
            newerr = 0;
            if(pars->useBackWeights) {
                if(childCluster == NULL)
                    newerr += error*transferDerivative(savedNodeStrengths[getTurn(-turnsBack)][layer][node], layer)*backWeights[layer][node][i];
                else
                    newerr += childInputError[turnsBack][getInputNumber(layer, node, BACKWEIGHT, i)]*backWeights[layer][node][i];
            }
            if(pars->useBackMems) {
                if(childCluster == NULL)
                    newerr += error*transferDerivative(savedNodeStrengths[getTurn(-turnsBack)][layer][node], layer)*memStrengths[layer+1][i]*backMems[layer][node][i];
                else
                    newerr += childInputError[turnsBack][getInputNumber(layer, node, BACKMEM, i)]*memStrengths[layer+1][i]*backMems[layer][node][i];
            }
            preNodeError[getTurn(-turnsBack)][layer+1][i] += newerr;
            //if(fabs(newerr) > pars->propThresh)
            //propagateNodeError(newerr, layer+1, i, t+1, inputError);
        }
    }

    if(childCluster != NULL) {
        for(int i=0; i<pars->numTurnsSaved; i++) {
            delete [] childInputError[i];
        }
        delete [] childInputError;
    }
}

double Cluster::getRandomErrorWeight() {
    double spread = 0.5;
    double weight = 1-spread+((rand() % (int)(2*10*spread))/10.0);
    return weight;
}

double Cluster::getOutputErrorWeight(double output, double realoutput) {
    return (output - realoutput);
}

//inputError[pars->numTurnsSaved][pars->numInputs]
void Cluster::propagateError(double* outputError, double** inputError, int turnsBack) {
    double newerr;
    for(int i=0; i<pars->numOutputs; i++) {
        nodeError[getTurn(-turnsBack)][pars->numLayers][i][0] += outputError[i];    //the outputError uses a different turn convention I think
    }
    for(int i=0; i<pars->nodesPerLayer; i++) {
        for(int j=0; j<pars->numOutputs; j++) {
            //its just the gradient (derivative of final error function with respect to this node's output)
            newerr = outputError[j]*transferDerivative(savedOutputs[getTurn(-turnsBack)][j],pars->numLayers)*nodesToOutput[j][i];
            preNodeError[getTurn(-turnsBack)][pars->numLayers-1][i] += newerr;
        }
    }
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            propagateNodeError(i, j, turnsBack, inputError);
        }
    }
}

void Cluster::clearNodeError() {
    for(int i=0; i<pars->numTurnsSaved; i++) {
        for(int j=0; j<pars->numLayers; j++) {
            for(int k=0; k<pars->nodesPerLayer; k++) {
                if(nodeClusters[j][k] == NULL)
                    nodeError[i][j][k][0] = 0;
                else {
                    for(int l=0; l<nodeClusters[j][k]->getPars()->numInputs; l++) {
                        nodeError[i][j][k][l] = 0;
                    }
                }
            }
        }
        for(int k=0; k<pars->numOutputs; k++) {
            nodeError[i][pars->numLayers][k][0] = 0;
        }
    }
    for(int i=0; i<pars->numTurnsSaved; i++) {
        for(int j=0; j<pars->numLayers; j++) {
            for(int k=0; k<pars->nodesPerLayer; k++) {
                preNodeError[i][j][k] = 0;
            }
        }
    }
    
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            if(nodeClusters[i][j] != NULL)
                nodeClusters[i][j]->clearNodeError();
        }
    }
}

void Cluster::backPropagate(double *realoutput, double abspp, double** inputError){
    for(int i=0; i<pars->numOutputs; i++) {
        backPropagateOutputError[0][i] = getRandomErrorWeight() * getOutputErrorWeight(transferFunction(savedOutputs[getTurn(0)][i],pars->numLayers), realoutput[i]);
    }

    for(int i=1; i<pars->numTurnsSaved; i++) {
        for(int j=0; j<pars->numOutputs; j++) {
            backPropagateOutputError[i][j] = 0;
        }
    }

    backPropagateError(backPropagateOutputError, abspp, inputError);
}

void Cluster::backPropagateError(double** outputError, double abspp, double** inputError){
    clearNodeError();
    
    for(int t=0; t<pars->numTurnsSaved-2; t++) {
        propagateError(outputError[t], inputError, t);
    }

    updateWeights(abspp);
}

void Cluster::updateWeights(double abspp) {
    //update child weights
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            if(nodeClusters[i][j] != NULL)
                nodeClusters[i][j]->updateWeights(abspp);
        }
    }

    double stepfactor = abspp*pars->stepfactor;

    double normfact;
    for(int t=1; t<pars->numTurnsSaved-1; t++) {
        //first go from output to last layer (note that the numbering convention for nodeError on the output layer differs from other layers by 1)
        //find normalization factor
        normfact = 1;
        for(int i=0; i<pars->nodesPerLayer; i++)
            normfact += pow(transferFunction(savedNodeStrengths[getTurn(-t)][pars->numLayers-1][i],pars->numLayers-1),2);
        normfact = sqrt(normfact);
        for(int k=0; k<pars->numOutputs; k++) {
            for(int i=0; i<pars->nodesPerLayer; i++) {
                nodesToOutput[k][i] -= stepfactor*nodeError[getTurn(-t+1)][pars->numLayers][k][0]*transferDerivative(savedOutputs[getTurn(-t+1)][k],pars->numLayers)*transferFunction(savedNodeStrengths[getTurn(-t)][pars->numLayers-1][i],pars->numLayers-1)/normfact;
            }
            outputThresholds[k] += stepfactor*nodeError[getTurn(-t+1)][pars->numLayers][k][0]*transferDerivative(savedOutputs[getTurn(-t+1)][k],pars->numLayers)/normfact;
        }
        //now do intermediate layers

        for(int i=pars->numLayers-2; i>=0; i--) {
            normfact = 1;
            for(int j=0; j<pars->nodesPerLayer; j++)
                normfact += pow(transferFunction(savedNodeStrengths[getTurn(-t-1)][i][j],i),2);
            if(pars->learnStyleSide == LEARNSTYLEBP || (pars->learnStyleSide == LEARNSTYLEALT && (i+1)%2==1) || (pars->learnStyleSide == LEARNSTYLEALTR && (i+1)%2==0)) {
                for(int j=0; j<pars->nodesPerLayer; j++)
                    normfact += pow(transferFunction(savedNodeStrengths[getTurn(-t-1)][i+1][j],i+1),2);
            }
            if(pars->backPropBackWeights && i < pars->numLayers-2) {
                for(int j=0; j<pars->nodesPerLayer; j++)
                    normfact += pow(transferFunction(savedNodeStrengths[getTurn(-t-1)][i+2][j],i+2),2);
            }

            normfact = sqrt(normfact);

            for(int j=0; j<pars->nodesPerLayer; j++) {
                for(int k=0; k<pars->nodesPerLayer; k++) {
                    if(nodeClusters[i+1][k] == NULL)
                        nodesToNodes[i][k][j] -= stepfactor*nodeError[getTurn(-t+1)][i+1][k][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][i+1][k],i+1)*transferFunction(savedNodeStrengths[getTurn(-t-1)][i][j],i)/normfact;
                    else
                        nodesToNodes[i][k][j] -= stepfactor*nodeError[getTurn(-t+1)][i+1][k][getInputNumber(i+1, k, FORWARDWEIGHT, j)]*transferFunction(savedNodeStrengths[getTurn(-t-1)][i][j],i)/normfact;

                    if(pars->learnStyleSide == LEARNSTYLEBP || (pars->learnStyleSide == LEARNSTYLEALT && (i+1)%2==1) || (pars->learnStyleSide == LEARNSTYLEALTR && (i+1)%2==0)) {
                        if(nodeClusters[i+1][k] == NULL)
                            sideWeights[i+1][k][j] -= stepfactor*nodeError[getTurn(-t+1)][i+1][k][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][i+1][k],i+1)*transferFunction(savedNodeStrengths[getTurn(-t-1)][i+1][j],i+1)/normfact;                
                        else
                            sideWeights[i+1][k][j] -= stepfactor*nodeError[getTurn(-t+1)][i+1][k][getInputNumber(i+1, k, SIDEWEIGHT, j)]*transferFunction(savedNodeStrengths[getTurn(-t-1)][i+1][j],i+1)/normfact;                
                    }

                    if(pars->bpMemStrengths && (pars->learnStyleSide == LEARNSTYLEHB || (pars->learnStyleSide == LEARNSTYLEALT && (i+1)%2==0) || (pars->learnStyleSide == LEARNSTYLEALTR && (i+1)%2==1))) {
                        if(nodeClusters[i+1][k] == NULL)
                            memStrengths[i+1][j] -= stepfactor*nodeError[getTurn(-t+1)][i+1][k][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][i+1][k],i+1)*transferFunction(savedNodeStrengths[getTurn(-t-1)][i+1][j],i+1)*sideMems[i+1][k][j]/(pars->nodesPerLayer);
                        else
                            memStrengths[i+1][j] -= stepfactor*nodeError[getTurn(-t+1)][i+1][k][getInputNumber(i+1, k, SIDEMEM, j)]*transferFunction(savedNodeStrengths[getTurn(-t-1)][i+1][j],i+1)*sideMems[i+1][k][j]/(pars->nodesPerLayer);
                    }

                    if(pars->backPropBackWeights) {
                        if(nodeClusters[i][k] == NULL)
                            backWeights[i][k][j] -= stepfactor*nodeError[getTurn(-t+1)][i][k][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][i][k],i)*transferFunction(savedNodeStrengths[getTurn(-t-1)][i+1][j],i+1)/normfact;
                        else
                            backWeights[i][k][j] -= stepfactor*nodeError[getTurn(-t+1)][i][k][getInputNumber(i, k, BACKWEIGHT, j)]*transferFunction(savedNodeStrengths[getTurn(-t-1)][i+1][j],i+1)/normfact;
                    }

                    if(pars->bpMemStrengths && pars->useBackMems) {
                        if(nodeClusters[i][k] == NULL)
                            memStrengths[i+1][j] -= stepfactor*nodeError[getTurn(-t+1)][i][k][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][i][k],i)*transferFunction(savedNodeStrengths[getTurn(-t-1)][i+1][j],i+1)*backMems[i][k][j]/(pars->nodesPerLayer);
                        else
                            memStrengths[i+1][j] -= stepfactor*nodeError[getTurn(-t+1)][i][k][getInputNumber(i, k, BACKMEM, j)]*transferFunction(savedNodeStrengths[getTurn(-t-1)][i+1][j],i+1)*backMems[i][k][j]/(pars->nodesPerLayer);
                    }

                    if(pars->bpMemStrengths && pars->useForwardMems) {
                        if(nodeClusters[i+1][k] == NULL)
                            memStrengths[i][j] -= stepfactor*nodeError[getTurn(-t+1)][i+1][k][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][i+1][k],i+1)*transferFunction(savedNodeStrengths[getTurn(-t-1)][i][j],i)*forwardMems[i][k][j]/(pars->nodesPerLayer);
                        else
                            memStrengths[i][j] -= stepfactor*nodeError[getTurn(-t+1)][i+1][k][getInputNumber(i+1, k, FORWARDMEM, j)]*transferFunction(savedNodeStrengths[getTurn(-t-1)][i][j],i)*forwardMems[i][k][j]/(pars->nodesPerLayer);

                    }
                }
            }

            for(int k=0; k<pars->nodesPerLayer; k++) {
                if(nodeClusters[i+1][k] == NULL)
                    thresholds[i+1][k] += stepfactor*nodeError[getTurn(-t+1)][i+1][k][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][i+1][k],i+1)/normfact;
                else
                    thresholds[i+1][k] = 0;
            }
        }

        //now do the first layer
        normfact = 1;
        for(int i=0; i<pars->numInputs; i++)
            normfact += pow(savedInputs[getTurn(-t-1)][i],2);
        if(pars->learnStyleSide == LEARNSTYLEBP) {
            for(int j=0; j<pars->nodesPerLayer; j++)
                normfact += pow(transferFunction(savedNodeStrengths[getTurn(-t-1)][0][j],0),2);
        }
        if(pars->backPropBackWeights) {
            for(int j=0; j<pars->nodesPerLayer; j++)
                normfact += pow(transferFunction(savedNodeStrengths[getTurn(-t-1)][1][j],1),2);
        }
        normfact = sqrt(normfact);
        for(int i=0; i<pars->nodesPerLayer; i++) {
            for(int j=0; j<pars->numInputs; j++) {
                if(pars->copyInputsToFirstLevel && i<pars->numInputs) {
                    if(i==j)
                        inputToNodes[i][j] = 5;
                    else
                        inputToNodes[i][j] = 0;
                }
                else {
                    if(nodeClusters[0][i] == NULL)
                        inputToNodes[i][j] -= stepfactor*nodeError[getTurn(-t+1)][0][i][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][0][i],0)*savedInputs[getTurn(-t-1)][j]/normfact;
                    else
                        inputToNodes[i][j] -= stepfactor*nodeError[getTurn(-t+1)][0][i][getInputNumber(0, i, FORWARDWEIGHT, j)]*savedInputs[getTurn(-t-1)][j]/normfact;
                }
            }
            for(int j=0; j<pars->nodesPerLayer; j++) {
                if(pars->learnStyleSide == LEARNSTYLEBP || pars->learnStyleSide == LEARNSTYLEALTR) {
                    if(nodeClusters[0][i] == NULL)
                        sideWeights[0][i][j] -= stepfactor*nodeError[getTurn(-t+1)][0][i][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][0][i],0)*transferFunction(savedNodeStrengths[getTurn(-t-1)][0][j],0)/normfact;                
                    else
                        sideWeights[0][i][j] -= stepfactor*nodeError[getTurn(-t+1)][0][i][getInputNumber(0, i, SIDEWEIGHT, j)]*transferFunction(savedNodeStrengths[getTurn(-t-1)][0][j],0)/normfact;                
                }
                if(pars->bpMemStrengths && (pars->learnStyleSide == LEARNSTYLEHB || pars->learnStyleSide == LEARNSTYLEALT)) {
                    if(nodeClusters[0][i] == NULL)
                        memStrengths[0][j] -= stepfactor*nodeError[getTurn(-t+1)][0][i][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][0][i],0)*transferFunction(savedNodeStrengths[getTurn(-t-1)][0][j],0)*sideMems[0][i][j]/(pars->nodesPerLayer);
                    else
                        memStrengths[0][j] -= stepfactor*nodeError[getTurn(-t+1)][0][i][getInputNumber(0, i, SIDEMEM, j)]*transferFunction(savedNodeStrengths[getTurn(-t-1)][0][j],0)*sideMems[0][i][j]/(pars->nodesPerLayer);
                }
                if(pars->backPropBackWeights) {
                    if(nodeClusters[0][i] == NULL)
                        backWeights[0][i][j] -= stepfactor*nodeError[getTurn(-t+1)][0][i][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][0][i],0)*transferFunction(savedNodeStrengths[getTurn(-t-1)][1][j],1)/normfact;
                    else
                        backWeights[0][i][j] -= stepfactor*nodeError[getTurn(-t+1)][0][i][getInputNumber(0, i, BACKWEIGHT, j)]*transferFunction(savedNodeStrengths[getTurn(-t-1)][1][j],1)/normfact;
                }
                if(pars->bpMemStrengths && pars->useBackMems) {
                    if(nodeClusters[0][i] == NULL)
                        memStrengths[1][j] -= stepfactor*nodeError[getTurn(-t+1)][0][i][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][0][i],0)*transferFunction(savedNodeStrengths[getTurn(-t-1)][1][j],1)*backMems[0][i][j]/(pars->nodesPerLayer);
                    else
                        memStrengths[1][j] -= stepfactor*nodeError[getTurn(-t+1)][0][i][getInputNumber(0, i, BACKMEM, j)]*transferFunction(savedNodeStrengths[getTurn(-t-1)][1][j],1)*backMems[0][i][j]/(pars->nodesPerLayer);
                }
            }
            if(pars->copyInputsToFirstLevel && i<pars->numInputs)
                thresholds[0][i] = 2.5;
            else if(nodeClusters[0][i] == NULL)
                thresholds[0][i] += stepfactor*nodeError[getTurn(-t+1)][0][i][0]*transferDerivative(savedNodeStrengths[getTurn(-t)][0][i],0)/normfact;
            else
                thresholds[0][i] = 0;
        }
    }
}

void Cluster::sleep(int n) {
    double input[pars->numInputs];
    for(int i=0; i<n; i++) {
        for(int j=0; j<pars->numInputs; j++) {
            input[j] = rand()%2;
        }
        setInputs(input);
        calculate();
        //memLearn();
    }
}

double Cluster::memCorrelationChange(double node1, double node2, double pp) {
    return fabs(pp)*pars->memfactor*node1*(node2);
}

//run pseudo-Hebbian learning algorithm
void Cluster::memLearn(double pp) {
    double node1 = 0;
    double node2 = 0;

/*
    //forward
    if(pars->useForwardMems) {
        for(int i=0; i<pars->numLayers-1; i++) {
            for(int k=0; k<pars->nodesPerLayer; k++) {
                norm = 0;
                node1 = transferFunction(savedNodeStrengths[getTurn(i-pars->numLayers-1)][i][k],i);
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    node2 = transferFunction(savedNodeStrengths[getTurn(i-pars->numLayers)][i+1][j],i+1);
                    forwardMems[i][j][k] += memCorrelationChange(node1, node2, pp);
                    norm+=pow(forwardMems[i][j][k],2);
                }
                norm = sqrt(norm);
                if(norm!=0)
                    for(int j=0; j<pars->nodesPerLayer; j++)
                        forwardMems[i][j][k] *= pars->memnorm/norm;
            }
        }
    }
    */

    double memchange;
    //side
    for(int t=0; t<pars->numTurnsSaved-3; t++) {
        for(int i=0; i<pars->numLayers; i++) {
            if(pars->learnStyleSide == LEARNSTYLEHB || (pars->learnStyleSide == LEARNSTYLEALT && i%2==0) || (pars->learnStyleSide == LEARNSTYLEALTR && i%2==1)) {
                for(int k=0; k<pars->nodesPerLayer; k++) {
                    node1 = transferFunction(savedNodeStrengths[getTurn(-t-1)][i][k],i)-transferFunction(savedNodeStrengths[getTurn(-t-2)][i][k],i);
                    for(int j=0; j<pars->nodesPerLayer; j++) {
                        if(j==k) continue;
                        node2 = transferFunction(savedNodeStrengths[getTurn(-t)][i][j],i)-transferFunction(savedNodeStrengths[getTurn(-t-1)][i][j],i);
                        memchange = memCorrelationChange(node1, node2, pp);
                        sideMems[i][j][k] *= pow(pars->memnorm, fabs(node1));
                        sideMems[i][j][k] += memchange;
                    }
                }
            }
        }
    }

/*
    //back mems
    if(pars->useBackMems) {
        for(int i=0; i<pars->numLayers-1; i++) {
            for(int k=0; k<pars->nodesPerLayer; k++) {
                norm = 0;
                node1 = transferFunction(savedNodeStrengths[getTurn(i-pars->numLayers-1)][i+1][k],i);
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    node2 = transferFunction(savedNodeStrengths[getTurn(i-pars->numLayers)][i][j],i+1);
                    backMems[i][j][k] += memCorrelationChange(node1, node2, pp);
                    norm+=pow(backMems[i][j][k],2);
                }
                norm = sqrt(norm);
                if(norm!=0)
                    for(int j=0; j<pars->nodesPerLayer; j++)
                        backMems[i][j][k] *= pars->memnorm/norm;
            }
        }
    }
    */
}

void Cluster::defineArrays() {
    //inputToNodes[pars->nodesPerLayer][pars->numInputs]
    inputToNodes = new double*[pars->nodesPerLayer];
    for(int i=0; i<pars->nodesPerLayer; i++)
        inputToNodes[i] = new double[pars->numInputs];

    //nodesToNodes[pars->numLayers-1][pars->nodesPerLayer][pars->nodesPerLayer]
    nodesToNodes = new double**[pars->numLayers-1];
    for(int i=0; i<pars->numLayers-1; i++) {
        nodesToNodes[i] = new double*[pars->nodesPerLayer];
        for(int j=0; j<pars->nodesPerLayer; j++)
            nodesToNodes[i][j] = new double[pars->nodesPerLayer];
    }

    //thresholds[pars->numLayers][pars->nodesPerLayer]
    thresholds = new double*[pars->numLayers];
    for(int i=0; i<pars->numLayers; i++)
        thresholds[i] = new double[pars->nodesPerLayer];

    //nodesToOutput[pars->numOutputs][pars->nodesPerLayer]
    nodesToOutput = new double*[pars->numOutputs];
    for(int i=0; i<pars->numOutputs; i++)
        nodesToOutput[i] = new double[pars->nodesPerLayer];

    //outputThresholds[pars->numOutputs]
    outputThresholds = new double[pars->numOutputs];

    //savedNodeStrengths[pars->numTurnsSaved][pars->numLayers][pars->nodesPerLayer]
    savedNodeStrengths = new double**[pars->numTurnsSaved];
    for(int i=0; i<pars->numTurnsSaved; i++) {
        savedNodeStrengths[i] = new double*[pars->numLayers];
        for(int j=0; j<pars->numLayers; j++)
            savedNodeStrengths[i][j] = new double[pars->nodesPerLayer];
    }

    //savedInputs[pars->numTurnsSaved][pars->numInputs]
    savedInputs = new double*[pars->numTurnsSaved];
    for(int i=0; i<pars->numTurnsSaved; i++)
        savedInputs[i] = new double[pars->numInputs];

    //savedOutputs[pars->numTurnsSaved][pars->numOutputs]
    savedOutputs = new double*[pars->numTurnsSaved];
    for(int i=0; i<pars->numTurnsSaved; i++)
        savedOutputs[i] = new double[pars->numOutputs];
    
    //sideWeights[pars->numLayers][pars->nodesPerLayer][pars->nodesPerLayer]
    sideWeights = new double**[pars->numLayers];
    for(int i=0; i<pars->numLayers; i++) {
        sideWeights[i] = new double*[pars->nodesPerLayer];
        for(int j=0; j<pars->nodesPerLayer; j++)
            sideWeights[i][j] = new double[pars->nodesPerLayer];
    }

    //backWeights[pars->numLayers-1][pars->nodesPerLayer][pars->nodesPerLayer]
    backWeights = new double**[pars->numLayers-1];
    for(int i=0; i<pars->numLayers-1; i++) {
        backWeights[i] = new double*[pars->nodesPerLayer];
        for(int j=0; j<pars->nodesPerLayer; j++)
            backWeights[i][j] = new double[pars->nodesPerLayer];
    }

    //sideMems[pars->numLayers][pars->nodesPerLayer][pars->nodesPerLayer]
    sideMems = new double**[pars->numLayers];
    for(int i=0; i<pars->numLayers; i++) {
        sideMems[i] = new double*[pars->nodesPerLayer];
        for(int j=0; j<pars->nodesPerLayer; j++)
            sideMems[i][j] = new double[pars->nodesPerLayer];
    }

    //backMems[pars->numLayers-1][pars->nodesPerLayer][pars->nodesPerLayer]
    backMems = new double**[pars->numLayers-1];
    for(int i=0; i<pars->numLayers-1; i++) {
        backMems[i] = new double*[pars->nodesPerLayer];
        for(int j=0; j<pars->nodesPerLayer; j++)
            backMems[i][j] = new double[pars->nodesPerLayer];
    }

    //forwardMems[pars->numLayers-1][pars->nodesPerLayer][pars->nodesPerLayer]
    forwardMems = new double**[pars->numLayers-1];
    for(int i=0; i<pars->numLayers-1; i++) {
        forwardMems[i] = new double*[pars->nodesPerLayer];
        for(int j=0; j<pars->nodesPerLayer; j++)
            forwardMems[i][j] = new double[pars->nodesPerLayer];
    }

    memStrengths = new double*[pars->numLayers];
    for(int i=0; i<pars->numLayers; i++) {
        memStrengths[i] = new double[pars->nodesPerLayer];
    }

    nodeClusters = new Cluster**[pars->numLayers];
    for(int i=0; i<pars->numLayers; i++) {
        nodeClusters[i] = new Cluster*[pars->nodesPerLayer];
        for(int j=0; j<pars->nodesPerLayer; j++) {
            nodeClusters[i][j] = NULL;
        }
    }

    nodeError = new double***[pars->numTurnsSaved];
    for(int i=0; i<pars->numTurnsSaved; i++) {
        nodeError[i] = new double**[pars->numLayers+1];
        for(int j=0; j<pars->numLayers; j++) {
            nodeError[i][j] = new double*[pars->nodesPerLayer];
            for(int k=0; k<pars->nodesPerLayer; k++) {
                nodeError[i][j][k] = new double[1];
                nodeError[i][j][k][0] = 0;
            }
        }
        nodeError[i][pars->numLayers] = new double*[pars->numOutputs];
        for(int k=0; k<pars->numOutputs; k++) {
            nodeError[i][pars->numLayers][k] = new double[1];
            nodeError[i][pars->numLayers][k][0] = 0;
        }
    }

    preNodeError = new double**[pars->numTurnsSaved];
    for(int i=0; i<pars->numTurnsSaved; i++) {
        preNodeError[i] = new double*[pars->numLayers];
        for(int j=0; j<pars->numLayers; j++) {
            preNodeError[i][j] = new double[pars->nodesPerLayer];
        }
    }

    backPropagateOutputError = new double*[pars->numTurnsSaved];
    for(int i=0; i<pars->numTurnsSaved; i++) {
        backPropagateOutputError[i] = new double[pars->numOutputs];
    }
}

void Cluster::addCluster(int layer, int node, Cluster* cluster) {
    if(nodeClusters[layer][node] != NULL)
        delete nodeClusters[layer][node];
    nodeClusters[layer][node] = cluster;
    
    for(int i=0; i<pars->numTurnsSaved; i++) {
        if(nodeError[i][layer][node] != NULL)
            delete [] nodeError[i][layer][node];
        if(cluster != NULL) {
            nodeError[i][layer][node] = new double[cluster->getPars()->numInputs];
            for(int j=0; j<cluster->getPars()->numInputs; j++) {
                nodeError[i][layer][node][j] = 0;
            }
        }
        else {
            nodeError[i][layer][node] = new double[1];
            nodeError[i][layer][node][0] = 0;
        }
    }
}

void Cluster::zeroArrays() {
    for(int i=0; i<pars->nodesPerLayer; i++)
        for(int j=0; j<pars->numInputs; j++)
            inputToNodes[i][j]=0;
    for(int i=0; i<pars->numLayers-1; i++)
        for(int j=0; j<pars->nodesPerLayer; j++)
            for(int k=0; k<pars->nodesPerLayer; k++)
                nodesToNodes[i][j][k]=0;
    for(int i=0; i<pars->numLayers; i++)
        for(int j=0; j<pars->nodesPerLayer; j++)
            thresholds[i][j]=0;
    for(int i=0; i<pars->numOutputs; i++)
        for(int j=0; j<pars->nodesPerLayer; j++)
            nodesToOutput[i][j]=0;
    for(int i=0; i<pars->numOutputs; i++)
        outputThresholds[i] = 0;
    for(int i=0; i<pars->numTurnsSaved; i++)
        for(int j=0; j<pars->numLayers; j++)
            for(int k=0; k<pars->nodesPerLayer; k++)
                savedNodeStrengths[i][j][k] = 0;
    for(int i=0; i<pars->numTurnsSaved; i++)
        for(int j=0; j<pars->numInputs; j++)
            savedInputs[i][j] = 0;
    for(int i=0; i<pars->numTurnsSaved; i++)
        for(int j=0; j<pars->numOutputs; j++)
            savedOutputs[i][j] = 0;
    for(int i=0; i<pars->numLayers; i++)
        for(int j=0; j<pars->nodesPerLayer; j++)
            for(int k=0; k<pars->nodesPerLayer; k++)
                sideWeights[i][j][k] = 0;
    for(int i=0; i<pars->numLayers-1; i++)
        for(int j=0; j<pars->nodesPerLayer; j++)
            for(int k=0; k<pars->nodesPerLayer; k++)
                backWeights[i][j][k] = 0;
    for(int i=0; i<pars->numLayers-1; i++)
        for(int j=0; j<pars->nodesPerLayer; j++)
            for(int k=0; k<pars->nodesPerLayer; k++)
                forwardMems[i][j][k] = 0;
    for(int i=0; i<pars->numLayers; i++)
        for(int j=0; j<pars->nodesPerLayer; j++)
            for(int k=0; k<pars->nodesPerLayer; k++)
                sideMems[i][j][k] = 0;
    for(int i=0; i<pars->numLayers-1; i++)
        for(int j=0; j<pars->nodesPerLayer; j++)
            for(int k=0; k<pars->nodesPerLayer; k++)
                backMems[i][j][k] = 0;
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            nodeClusters[i][j] = NULL;
        }
    }
    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            memStrengths[i][j] = 1.0;
        }
    }

    //
    //nodeError zeroed in defineArrays
    //

    for(int i=0; i<pars->numTurnsSaved; i++) {
        for(int j=0; j<pars->numLayers; j++) {
            for(int k=0; k<pars->nodesPerLayer; k++) {
                preNodeError[i][j][k] = 0;
            }
        }
    }
}

void Cluster::deleteArrays() {
    if(!isConvChild)
        deleteConvolutionSharedArrays();

    for(int i=0; i<pars->numTurnsSaved; i++) {
        delete [] savedInputs[i];
    }
    delete [] savedInputs;

    for(int i=0; i<pars->numTurnsSaved; i++) {
        delete [] savedOutputs[i];
    }
    delete [] savedOutputs;

    for(int i=0; i<pars->numTurnsSaved; i++) {
        for(int j=0; j<pars->numLayers; j++)
            delete [] savedNodeStrengths[i][j];
        delete [] savedNodeStrengths[i];
    }
    delete [] savedNodeStrengths;

    for(int i=0; i<pars->numLayers; i++) {
        delete [] memStrengths[i];
    }
    delete [] memStrengths;

    for(int i=0; i<pars->numLayers; i++) {
        delete [] nodeClusters[i];
    }
    delete [] nodeClusters;

    for(int i=0; i<pars->numTurnsSaved; i++) {
        for(int j=0; j<pars->numLayers; j++) {
            for(int k=0; k<pars->nodesPerLayer; k++) {
                delete [] nodeError[i][j][k];
            }
            delete [] nodeError[i][j];
        }
        for(int k=0; k<pars->numOutputs; k++) {
            delete [] nodeError[i][pars->numLayers][k];
        }
        delete [] nodeError[i][pars->numLayers];
        delete [] nodeError[i];
    }
    delete [] nodeError;

    for(int i=0; i<pars->numTurnsSaved; i++) {
        for(int j=0; j<pars->numLayers; j++) {
            delete [] preNodeError[i][j];
        }
        delete [] preNodeError[i];
    }
    delete [] preNodeError;

    for(int i=0; i<pars->numTurnsSaved; i++) {
        delete [] backPropagateOutputError[i];
    }
    delete [] backPropagateOutputError;
}

//cludge test function
void Cluster::initializeWeights() {
    if(!pars->randomWeights) {
        inputToNodes[0][0] = 0;
        inputToNodes[0][1] = 3;
        thresholds[0][0] = 0;
        nodesToOutput[0][0] = 10;
        outputThresholds[0] = 1;
    }
    else {
        for(int i=0; i<pars->nodesPerLayer; i++)
            for(int j=0; j<pars->numInputs; j++)
                inputToNodes[i][j]=rand() % 5 - 2.5;
        for(int i=0; i<pars->numLayers-1; i++)
            for(int j=0; j<pars->nodesPerLayer; j++)
                for(int k=0; k<pars->nodesPerLayer; k++)
                    nodesToNodes[i][j][k]= rand() % 5 - 2.5;
        for(int i=0; i<pars->numLayers; i++)
            for(int j=0; j<pars->nodesPerLayer; j++)
                thresholds[i][j]= rand() % 5 - 2.5;
        for(int j=0; j<pars->numOutputs; j++) {
            for(int i=0; i<pars->nodesPerLayer; i++)
                nodesToOutput[j][i]= rand() % 5 - 2.5;
            outputThresholds[j] = rand () % 5 - 2.5;
        }
    }
}

//returns false if the destination string can't fit the source string
bool safecat(char* dest, int length, const char* source) {
    int dend = -1;
    for(int i=0; i<length; i++) {
        if(dest[i] == '\0') {
            dend = i;
            break;
        }
    }

    if(dend < 0) return false;

    bool fits = false;
    for(int i=0; i<length-dend; i++) {
        if(source[i] == '\0') {
            fits = true;
            break;
        }
    }

    if(!fits) return false;

    for(int i=0; i<length-dend; i++) {
        dest[dend+i] = source[i];
        if(source[i] == '\0')
            return true;
    }
    dest[length-1] = '\0';
    return false;
}

void intToString(int x, char* str) {
    int predig = 0;
    if(x<0) {
        str[0] = '-';
        predig = 1;
        x = -x;
    }
    int digits;
    if(x!=0)
        digits = (int)log10((double) x)+1;
    else
        digits = 1;
    for(int i=0; i<digits; i++) {
        str[predig + digits-i-1] = (x%10) + '0';
        x /= 10;
    }
    str[predig+digits] = '\0';
}

void doubleToString(double x, char* str, int dec) {
    int predig = 0;
    if(x<0) {
        str[0] = '-';
        predig = 1;
        x = -x;
    }
    int intDigits;
    if(x!=0)
        intDigits = (int)log10(x)+1;
    else
        intDigits = 1;

    if(intDigits < 1) intDigits = 1;
    str[predig + intDigits + dec + 1] = '\0';
    for(int i=0; i<dec; i++) x*=10;
    for(int i=0; i<dec; i++) {
        str[predig + intDigits + dec -i] = (((long long)x)%10) + '0';
        x /= 10;
    }
    str[predig + intDigits] = '.';
    for(int i=0; i<intDigits; i++) {
        str[predig + intDigits-i-1] = (((long long)x)%10) + '0';
        x/=10;
    }
}

void doubleToString(double x, char* str) {
    doubleToString(x, str, 6);
}

bool safecat(char* dest, int length, int source) {
    char strbuff[100];
    intToString(source, strbuff);
    return safecat(dest, length, strbuff);
}

bool safecat(char* dest, int length, double source) {
    char strbuff[100];
    doubleToString(source, strbuff);
    return safecat(dest, length, strbuff);
}

bool Cluster::saveChildCluster(const char* file, int* layer, int* node, int depth) {
    if(depth < 0)
        throw std::out_of_range("asked to load cluster of negative depth");

    if(depth == 0) {
        return saveWeights(file);
    }
    if(layer[0] < 0 || layer[0] >= pars->numLayers || node[0] < 0 || node[0] >= pars->nodesPerLayer)
        throw std::out_of_range("asked for invalid child cluster");
    return nodeClusters[layer[0]][node[0]]->saveChildCluster(file, layer+1, node+1, depth-1);
}

bool Cluster::saveWeights(const char* file) {
    //input to first layer
    std::ofstream outfile(file);
    if(!outfile.is_open()) {
        return false;
    }

    outfile << "in to 0" << std::endl;
    for(int i=0; i<pars->nodesPerLayer; i++) {
        outfile << thresholds[0][i] << "; ";
        for(int j=0; j<pars->numInputs; j++) {
            outfile << " " << inputToNodes[i][j];
        }
        outfile << std::endl;
    }

    //between layers
    for(int k=1; k<pars->numLayers; k++) {
        outfile << k-1 << " to " << k << std::endl;
        for(int i=0; i<pars->nodesPerLayer; i++) {
            outfile << thresholds[k][i] << "; ";
            for(int j=0; j<pars->nodesPerLayer; j++) {
                outfile << " " << nodesToNodes[k-1][i][j];
            }
            outfile << std::endl;
        }
    }

    //mems
    for(int i=0; i<pars->numLayers; i++) {
        if(pars->learnStyleSide == LEARNSTYLEBP || (pars->learnStyleSide == LEARNSTYLEALT && i%2==1) || (pars->learnStyleSide == LEARNSTYLEALTR && i%2==0)) {
            outfile << "side weights " << i << std::endl;
            for(int j=0; j<pars->nodesPerLayer; j++) {
                for(int k=0; k<pars->nodesPerLayer; k++) {
                    outfile << " " << sideWeights[i][j][k];
                }
                outfile << std::endl;
            }
        }
        if(pars->learnStyleSide == LEARNSTYLEHB || (pars->learnStyleSide == LEARNSTYLEALT && i%2==0) || (pars->learnStyleSide == LEARNSTYLEALTR && i%2==1)) {
            outfile << "side mems " << i << std::endl;
            for(int j=0; j<pars->nodesPerLayer; j++) {
                outfile << " " << memStrengths[i][j];
            }
            outfile << std::endl << std::endl;
            for(int j=0; j<pars->nodesPerLayer; j++) {
                for(int k=0; k<pars->nodesPerLayer; k++) {
                    outfile << " " << sideMems[i][j][k];
                }
                outfile << std::endl;
            }
        }
        if(i<pars->numLayers-1) {
            if(pars->useForwardMems) {
                outfile << "forward mems " << i << "\n";
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    outfile << " " << memStrengths[i][j];
                }
                outfile << std::endl << std::endl;
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    for(int k=0; k<pars->nodesPerLayer; k++) {
                        outfile << " " << forwardMems[i][j][k];
                    }
                    outfile << std::endl;
                }
            }
            if(pars->useBackMems) {
                outfile << "back mems " << i << "\n";
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    outfile << " " << memStrengths[i+1][j];
                }
                outfile << std::endl << std::endl;
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    for(int k=0; k<pars->nodesPerLayer; k++) {
                        outfile << " " << backMems[i][j][k];
                    }
                    outfile << std::endl;
                }
            }
        }
    }

    //last layer to output
    outfile << pars->numLayers-1 << " to out\n";
    for(int i=0; i<pars->numOutputs; i++) {
        outfile << outputThresholds[i] << "; ";
        for(int j=0; j<pars->nodesPerLayer; j++) {
            outfile << " " << nodesToOutput[i][j];
        }
        outfile << std::endl;
    }
    outfile << std::endl;
    outfile.close();
    return true;
}

int getWord(char* contents, int length, int pos, char* buf, int* w, int startword) {
    if((*w) < startword) {
        (*w)++;
        return 0;
    }

    for(int i=pos; i<length-1; i++) {
        if(isspace(contents[i]) || contents[i] == ';') pos = i+1;
        else break;
    }

    if(pos >= length-1 || contents[pos] == '\0') {
        buf[0] = '\0';
        return length;
    }

    (*w)++;

    for(int i=pos; i<length-1; i++) {
        if(contents[i] == '\0') {
            buf[i-pos] = '\0';
            return length;
        }
        if(!(isspace(contents[i]) || contents[i] == ';'))
            buf[i-pos] = contents[i];
        else {
            buf[i-pos] = '\0';
            return i+1;
        }
    }
    return 0;
}

int skipWords(char* contents, int length, int pos, char* buf, int num, int* w, int startword) {
    for(int i=0; i<num; i++) {
        pos = getWord(contents, length, pos, buf, w, startword);
    }
    return pos;
}

double stringToDouble(char* str) {
    double d = 0;
    int dp = -1;
    int isNeg = 0;
    if(str[0]=='-')
        isNeg = 1;

    for(int i=isNeg; str[i]!='\0'; i++) {
        if(str[i] == '.') {
            dp = i;
            break;
        }
        d *= 10;
        d += str[i] - '0';
    }
    if(dp>=0) {
        for(int i=dp+1; str[i]!='\0'; i++) {
            d += (str[i]-'0')*pow(10, dp-i);
        }
    }
    if(!isNeg)
        return d;
    else
        return -d;
}

bool Cluster::loadChildCluster(const char* file, int* layer, int* node, int depth) {
    if(depth < 0)
        throw std::out_of_range("asked to load cluster of negative depth");

    if(depth == 0) {
        return loadWeights(file);
    }
    if(layer[0] < 0 || layer[0] >= pars->numLayers || node[0] < 0 || node[0] >= pars->nodesPerLayer)
        throw std::out_of_range("asked for invalid child cluster");
    return nodeClusters[layer[0]][node[0]]->loadChildCluster(file, layer+1, node+1, depth-1);
}

void throwLoadError() {
    throw std::runtime_error("couldn't load cluster file");
}

bool Cluster::loadWeights(const char* file) {
    std::ifstream infile(file);
    std::string dum;
    //input to first layer
    infile >> dum >> dum >> dum;    //"in to 0"
    for(int i=0; i<pars->nodesPerLayer; i++) {
        if(!(infile >> dum)) throwLoadError();
        std::replace(dum.begin(), dum.end(), ';', ' ');
        thresholds[0][i] = std::strtod(dum.c_str(), NULL);
        for(int j=0; j<pars->numInputs; j++) {
            if(!(infile >> inputToNodes[i][j])) throwLoadError();
        }
    }

    //between layers
    for(int k=1; k<pars->numLayers; k++) {
        infile >> dum >> dum >> dum;    //"# to #"
        for(int i=0; i<pars->nodesPerLayer; i++) {
            if(!(infile >> dum)) throwLoadError();
            std::replace(dum.begin(), dum.end(), ';', ' ');
            thresholds[k][i] = std::strtod(dum.c_str(), NULL);
            for(int j=0; j<pars->nodesPerLayer; j++) {
                if(!(infile >> nodesToNodes[k-1][i][j])) throwLoadError();
            }
        }
    }

    //mems
    for(int i=0; i<pars->numLayers; i++) {
        if(pars->learnStyleSide == LEARNSTYLEBP || (pars->learnStyleSide == LEARNSTYLEALT && i%2==1) || (pars->learnStyleSide == LEARNSTYLEALTR && i%2==0)) {
            infile >> dum >> dum >> dum;    //"side weights #"
            for(int j=0; j<pars->nodesPerLayer; j++) {
                for(int k=0; k<pars->nodesPerLayer; k++) {
                    if(!(infile >> sideWeights[i][j][k])) throwLoadError();
                }
            }
        }
        if(pars->learnStyleSide == LEARNSTYLEHB || (pars->learnStyleSide == LEARNSTYLEALT && i%2==0) || (pars->learnStyleSide == LEARNSTYLEALTR && i%2==1)) {
            infile >> dum >> dum >> dum;    //"side mems #"
            for(int j=0; j<pars->nodesPerLayer; j++) {
                if(!(infile >> memStrengths[i][j])) throwLoadError();
            }
            for(int j=0; j<pars->nodesPerLayer; j++) {
                for(int k=0; k<pars->nodesPerLayer; k++) {
                    if(!(infile >> sideMems[i][j][k])) throwLoadError();
                }
            }
        }
        if(i<pars->numLayers-1) {
            if(pars->useForwardMems) {
                infile >> dum >> dum >> dum;    //"forward mems #"
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    if(!(infile >> memStrengths[i][j])) throwLoadError();
                }
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    for(int k=0; k<pars->nodesPerLayer; k++) {
                        if(!(infile >> forwardMems[i][j][k])) throwLoadError();
                    }
                }
            }
            if(pars->useBackMems) {
                infile >> dum >> dum >> dum;    //"back mems #"
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    if(!(infile >> memStrengths[i+1][j])) throwLoadError();
                }
                for(int j=0; j<pars->nodesPerLayer; j++) {
                    for(int k=0; k<pars->nodesPerLayer; k++) {
                        if(!(infile >> backMems[i][j][k])) throwLoadError();
                    }
                }
            }
        }
    }

    //last layer to output
    infile >> dum >> dum >> dum;    //"# to out"
    for(int i=0; i<pars->numOutputs; i++) {
        if(!(infile >> dum)) throwLoadError();
        std::replace(dum.begin(), dum.end(), ';', ' ');
        outputThresholds[i] = std::strtod(dum.c_str(), NULL);
        for(int j=0; j<pars->nodesPerLayer; j++) {
            if(!(infile >> nodesToOutput[i][j])) throwLoadError();
        }
    }
    return true;
}

int Cluster::getMinDepth() {
    int depth = 1+pars->numLayers;
    for(int i=0; i<pars->numLayers; i++) {
        int minlayerdepth = -1;
        for(int j=0; j<pars->nodesPerLayer; j++) {
            if(nodeClusters[i][j] != NULL) {
                int d = nodeClusters[i][j]->getMinDepth()-1;
                if(minlayerdepth < 0 || d < minlayerdepth)
                    minlayerdepth = d;
            }
            else {
                minlayerdepth = 0;
                break;
            }
        }
        depth += minlayerdepth;
    }
    return depth;
}

Cluster* Cluster::getChildCluster(int* layer, int* node, int depth) {
    if(depth < 0)
        throw std::out_of_range("asked to load cluster of negative depth");

    if(depth == 0) {
        return this;
    }
    if(layer[0] < 0 || layer[0] >= pars->numLayers || node[0] < 0 || node[0] >= pars->nodesPerLayer)
        throw std::out_of_range("asked for invalid child cluster");
    return nodeClusters[layer[0]][node[0]]->getChildCluster(layer+1, node+1, depth-1);
}

bool makeChange(double changeChance) {
    return rand() % 10000 < 10000*changeChance;
}

void Cluster::copyWeights(Cluster* source, double changeChance) {
    for(int i=0; i<pars->numLayers && i<source->getPars()->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer && j<source->getPars()->nodesPerLayer; j++) {
            bool changenode = makeChange(changeChance);
            if(i==0) {
                if(changenode) {
                    for(int k=0; k<pars->numInputs && k<source->getPars()->numInputs; k++) {
                        inputToNodes[j][k] = source->getInputToNode(j,k);
                    }
                }
            }
            else {
                if(changenode) {
                    for(int k=0; k<pars->nodesPerLayer && k<source->getPars()->nodesPerLayer; k++) {
                        nodesToNodes[i-1][j][k] = source->getNodeToNode(i-1,j,k);
                        forwardMems[i-1][j][k] = source->getForwardMem(i-1,j,k);
                    }
                }
            }

            if(i<pars->numLayers-1 && i<source->getPars()->numLayers-1) {
                if(changenode) {
                    for(int k=0; k<pars->nodesPerLayer && k<source->getPars()->nodesPerLayer; k++) {
                        backWeights[i][j][k] = source->getBackWeight(i,j,k);
                        backMems[i][j][k] = source->getBackMem(i,j,k);
                    }
                }
            }

            //now stuff for every layer
            if(changenode) {
                if(j<source->getPars()->nodesPerLayer)
                    thresholds[i][j] = source->getThreshold(i,j);

                for(int k=0; k<pars->nodesPerLayer && k<source->getPars()->nodesPerLayer; k++) {
                    sideWeights[i][j][k] = source->getSideWeight(i,j,k);
                    sideWeights[i][j][k] = source->getSideMem(i,j,k);
                }
            }
        }
    }

    for(int i=0; i<pars->numOutputs && i<source->getPars()->numOutputs; i++) {
        if(makeChange(changeChance)) {
            for(int j=0; j<pars->nodesPerLayer && j<source->getPars()->nodesPerLayer; j++) {
                nodesToOutput[i][j] = source->getNodeToOutput(i,j);
            }
            outputThresholds[i] = source->getOutputThreshold(i);
        }
    }
}

double Cluster::getInputToNode(int node, int input) {
    if(node < 0 || node >= pars->nodesPerLayer || input < 0 || input >= pars->numInputs)
        throw std::out_of_range("asked for invalid weight: inputToNodes");
    return inputToNodes[node][input];
}

double Cluster::getNodeToNode(int layer, int node1, int node2) {
    if(layer < 0 || layer >= pars->numLayers-1 || node1 < 0 || node1 >= pars->nodesPerLayer || node2 < 0 || node2 >= pars->nodesPerLayer)
        throw std::out_of_range("asked for invalid weight: nodesToNodes");
    return nodesToNodes[layer][node1][node2];
}

double Cluster::getThreshold(int layer, int node) {
    if(layer < 0 || layer >= pars->numLayers || node < 0 || node > pars->nodesPerLayer)
        throw std::out_of_range("asked for invalid weight: thresholds");
    return thresholds[layer][node];
}

double Cluster::getNodeToOutput(int output, int node) {
    if(output < 0 || output >= pars->numOutputs || node < 0 || node > pars->nodesPerLayer)
        throw std::out_of_range("asked for invalid weight: nodesToOutput");
    return nodesToOutput[output][node];
}

double Cluster::getOutputThreshold(int output) {
    if(output < 0 || output >= pars->numOutputs)
        throw std::out_of_range("asked for invalid weight: outputThresholds");
    return outputThresholds[output];
}

double Cluster::getSideWeight(int layer, int node1, int node2) {
    if(layer < 0 || layer >= pars->numLayers || node1 < 0 || node1 > pars->nodesPerLayer || node2 < 0 || node2 >= pars->nodesPerLayer)
        throw std::out_of_range("asked for invalid weight: sideWeights");
    return sideWeights[layer][node1][node2];
}

double Cluster::getBackWeight(int layer, int node1, int node2) {
    if(layer < 0 || layer >= pars->numLayers-1 || node1 < 0 || node1 > pars->nodesPerLayer || node2 < 0 || node2 >= pars->nodesPerLayer)
        throw std::out_of_range("asked for invalid weight: backWeights");
    return backWeights[layer][node1][node2];
}

double Cluster::getForwardMem(int layer, int node1, int node2) {
    if(layer < 0 || layer >= pars->numLayers-1 || node1 < 0 || node1 > pars->nodesPerLayer || node2 < 0 || node2 >= pars->nodesPerLayer)
        throw std::out_of_range("asked for invalid weight: forwardMems");
    return forwardMems[layer][node1][node2];
}

double Cluster::getSideMem(int layer, int node1, int node2) {
    if(layer < 0 || layer >= pars->numLayers || node1 < 0 || node1 > pars->nodesPerLayer || node2 < 0 || node2 >= pars->nodesPerLayer)
        throw std::out_of_range("asked for invalid weight: sideMems");
    return sideMems[layer][node1][node2];
}

double Cluster::getBackMem(int layer, int node1, int node2) {
    if(layer < 0 || layer >= pars->numLayers-1 || node1 < 0 || node1 > pars->nodesPerLayer || node2 < 0 || node2 >= pars->nodesPerLayer)
        throw std::out_of_range("asked for invalid weight: backMems");
    return backMems[layer][node1][node2];
}


double **Cluster::getInputToNodes() {
    return inputToNodes;
}

double ***Cluster::getNodesToNodes() {
    return nodesToNodes;
}

double **Cluster::getThresholds() {
    return thresholds;
}

double **Cluster::getNodesToOutput() {
    return nodesToOutput;
}

double *Cluster::getOutputThresholds() {
    return outputThresholds;
}

double ***Cluster::getSideWeights() {
    return sideWeights;
}

double ***Cluster::getBackWeights() {
    return backWeights;
}

double ***Cluster::getForwardMems() {
    return forwardMems;
}

double ***Cluster::getSideMems() {
    return sideMems;
}

double ***Cluster::getBackMems() {
    return backMems;
}

//don't mutate the side or back weights at the moment, for questionable reasons
//(because they aren't seeded initially when the bots are created)
void Cluster::mutateWeights(double mutateFactor) {
    for(int i=0; i<pars->nodesPerLayer; i++)
        for(int j=0; j<pars->numInputs; j++)
            if(makeChange(mutateFactor))
                inputToNodes[i][j]=rand() % 5 - 2.5;
    for(int i=0; i<pars->numLayers-1; i++)
        for(int j=0; j<pars->nodesPerLayer; j++)
            for(int k=0; k<pars->nodesPerLayer; k++)
                if(makeChange(mutateFactor))
                    nodesToNodes[i][j][k]= rand() % 5 - 2.5;
    for(int i=0; i<pars->numLayers; i++)
        for(int j=0; j<pars->nodesPerLayer; j++)
            if(makeChange(mutateFactor))
                thresholds[i][j]= rand() % 5 - 2.5;
    for(int j=0; j<pars->numOutputs; j++) {
        for(int i=0; i<pars->nodesPerLayer; i++)
            if(makeChange(mutateFactor))
                nodesToOutput[j][i]= rand() % 5 - 2.5;
        if(makeChange(mutateFactor))
            outputThresholds[j] = rand () % 5 - 2.5;
    }
}

void Cluster::setOutputThreshold(double thresh, int num) {    //for setting the initial ballpark output value when the output is non-binary
    outputThresholds[num] = thresh;
}

void Cluster::setConvolutionBase(Cluster* convBase) {
    deleteConvolutionSharedArrays();

    inputToNodes = convBase->getInputToNodes();
    nodesToNodes = convBase->getNodesToNodes();
    thresholds = convBase->getThresholds();
    nodesToOutput = convBase->getNodesToOutput();
    outputThresholds = convBase->getOutputThresholds();
    sideWeights = convBase->getSideWeights();
    backWeights = convBase->getBackWeights();
    sideMems = convBase->getSideMems();
    backMems = convBase->getBackMems();
    forwardMems = convBase->getForwardMems();
}

void Cluster::deleteConvolutionSharedArrays() {
    for(int i=0; i<pars->nodesPerLayer; i++) {
        delete [] inputToNodes[i];
    }
    delete [] inputToNodes;

    for(int i=0; i<pars->numLayers-1; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            delete [] nodesToNodes[i][j];
        }
        delete [] nodesToNodes[i];
    }
    delete [] nodesToNodes;

    for(int i=0; i<pars->numLayers; i++) {
        delete [] thresholds[i];
    }
    delete [] thresholds;

    for(int i=0; i<pars->numOutputs; i++) {
        delete [] nodesToOutput[i];
    }
    delete [] nodesToOutput;

    delete [] outputThresholds;

    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++)
            delete [] sideWeights[i][j];
        delete [] sideWeights[i];
    }
    delete [] sideWeights;

    for(int i=0; i<pars->numLayers-1; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++)
            delete [] backWeights[i][j];
        delete [] backWeights[i];
    }
    delete [] backWeights;

    for(int i=0; i<pars->numLayers; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++)
            delete [] sideMems[i][j];
        delete [] sideMems[i];
    }
    delete [] sideMems;

    for(int i=0; i<pars->numLayers-1; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++)
            delete [] backMems[i][j];
        delete [] backMems[i];
    }
    delete [] backMems;

    for(int i=0; i<pars->numLayers-1; i++) {
        for(int j=0; j<pars->nodesPerLayer; j++) {
            delete [] forwardMems[i][j];
        }
        delete [] forwardMems[i];
    }
    delete [] forwardMems;
}
