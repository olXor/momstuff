#include "testbots.h"

#define NUMOUTPUTS 1
#define NUMINPUTS 50

static ConvolutionProperties defaultConvProp = {
    1, {1}, 0, NUMINPUTS-1, {NUMINPUTS}, 1, 1, 1, 1
};

const char* datastring = "rawdata/";
double error;
int numerrors;

std::ofstream inputfile;

void runTestSim(Genbot** genbots, int numGenbots, std::string learnsetname) {
    std::stringstream learnsetss;
    learnsetss << datastring << learnsetname;
    std::ifstream learnset(learnsetss.str().c_str());
    std::string line;
    while(getline(learnset, line)) {
        std::stringstream lss(line);
        std::string fname;
        int column;
        int sstart;
        int send;
        double correctoutput[1];

        lss >> fname;
        lss >> column;
        lss >> sstart;
        lss >> send;
        lss >> correctoutput[0];

        std::cout << "Testing " << fname << " (" << sstart << "-" << send << ")" << std::endl;

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
        double inputs[NUMINPUTS] = {0};
        double outputs[NUMOUTPUTS] = {0};
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
                    inputfile << inputs[j] << " ";
                }
                inputfile << std::endl;
                for(int j=0; j<numGenbots; j++) {
                    genbots[j]->setInputs(inputs, NUMINPUTS);
                    genbots[j]->progressTurns(genbots[j]->getMinDepth(),false);
                    genbots[j]->getOutputs(outputs, NUMOUTPUTS);
                    std::cout << outputs[0] << " (correct: " << correctoutput[0] << ")" << std::endl;
                    error += pow(outputs[0]-correctoutput[0],2);
                    numerrors++;
                }

                if(send-i < NUMINPUTS)
                    break;

                curinput = 0;
            }
        }
    }
}

int main() {
    srand(time(NULL));
    std::vector<Genbot*> genbots;

    int id;
    std::string testSet;
    std::cout << "Enter the id of the bot you want to test" << std::endl;
    std::cin >> id;

    std::cout << "Enter the set you want (train=r, test=t, test2=t2)" << std::endl;
    std::cin >> testSet;

    if(testSet=="t")
        testSet = "testset";
    else if(testSet=="r")
        testSet = "trainset";
    else if(testSet=="t2")
        testSet = "test2set";
    else if(testSet=="m")
        testSet = "mocktrainset";

    std::ostringstream savename;

    int numGenbots = 1;
    int botnum = 0;

    if(id!=0) {
        std::ostringstream genomefname;
        genomefname << "savegenbot/" << id << "/genome";

        Genome* genome = new Genome(defaultConvProp);
        genome->loadGenome(genomefname.str().c_str());
        genome->pars[0]->useOutputTransfer = false;
        genbots.push_back(new Genbot(genome, NUMINPUTS, NUMOUTPUTS, id, PRESET_FIXED_BASE_MINIMAL));

        std::ostringstream fname;
        fname << "savegenbot/" << id << "/bot";
        genbots[0]->loadBot(fname.str().c_str());
        genbots[0]->progressTurnsSaved();
        
        savename << id;
        numGenbots = 1;
        error = 0;
        numerrors = 0;
        inputfile.open("testinputs");
        runTestSim(&genbots[0], numGenbots, testSet);
        if(numerrors != 0)
            error = sqrt(error/numerrors);
        else
            error = 0;
        std::cout << "Meansquare error: " << error << std::endl;
    }
}
