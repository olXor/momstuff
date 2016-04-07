#include "genbot.h"
#include "genome.h"
#include "mt4pipegen.h"
#include <iostream>
#include <string>
#include <vector>

#define BARSBACK 100
#define TRADECANDLELIMIT 15
#define TESTPERIOD "M5"

#define NUMOUTPUTS 2
#define LEARNDIVISOR 500
#define NUMINPUTS (4*BARSBACK+1)

#define TIME_PERIOD 3   //2013
#define USE_TRAILING_STOP 1

const char* absdatapath;
const char* datapath;

double checkResults(std::string file) {
    std::ostringstream fname;
    fname << absdatapath << file << ".htm";

    std::ifstream infile(fname.str().c_str());
    std::string line;
    bool profitfound = false;
    double profit = 0;
    std::string profstring;

    while(std::getline(infile, line)) {
        if(line.find("Total net profit") != std::string::npos) {
            profitfound = true;
            profstring = line.substr(56, line.find("<", 56)-56);
            profit = strtod(profstring.c_str(), NULL);
        }
    }

    if(!profitfound) {
        std::ostringstream oss;
        oss << "Couldn't parse result file for " << file;
        throw std::runtime_error(oss.str().c_str());
    }

    return profit;
}

int main() {
    absdatapath = "C:/Users/Thomas/AppData/Roaming/MetaQuotes/Terminal/50CA3DFB510CC5A8F28B48D1BF2A5702/genstockReports/";
    datapath = "/cygdrive/c/Users/Thomas/AppData/Roaming/MetaQuotes/Terminal/50CA3DFB510CC5A8F28B48D1BF2A5702/genstockReports/";
    std::ostringstream rmss;
    rmss << "rm " << datapath << "topbots/*";
    system(rmss.str().c_str());
    std::vector<Genbot*> genbotvector;
    std::ifstream file("topbots/currentbots");
    std::string line;
    int numGenbots = 0;
    while(getline(file, line)) {
        int id;
        if(!(std::istringstream(line) >> id))
            throw std::runtime_error("found something that wasn't an id in the current Genbots file");

        std::ostringstream genomefname;
        genomefname << "topbots/" << id << "/genome";

        Genome* genome = new Genome();
        genome->loadGenome(genomefname.str().c_str());
        genbotvector.push_back(new Genbot(genome, NUMINPUTS, NUMOUTPUTS, id));

        std::ostringstream fname;
        fname << "topbots/" << id << "/bot";
        genbotvector[numGenbots]->loadBot(fname.str().c_str());
        genbotvector[numGenbots]->progressTurnsSaved();
        numGenbots++;
    }

    Genbot** genbots = &genbotvector[0];

    MT4PipeGen* mt4pipe = new MT4PipeGen(NUMINPUTS, NUMOUTPUTS, numGenbots, numGenbots);
    std::string outfilename = "topbots/res2013";
    std::ostringstream ofrm;
    ofrm << "rm " << outfilename;
    system(ofrm.str().c_str());

    for(int i=1; i<=15; i+=2) {
        for(int j=1; j<=i; j++) {
            std::ostringstream rfname;
            rfname << "topbots/" << j << "of" << i;
            mt4pipe->runSim(genbots,
                    -1,                  //botnum
                    BARSBACK,           //barsBack
                    TRADECANDLELIMIT,   //tradeCandleLimit
                    LEARNDIVISOR,       //learnDivisor
                    false,              //train
                    0,                  //testSample
                    TIME_PERIOD,        //timePeriod
                    TESTPERIOD,         //testPeriod
                    USE_TRAILING_STOP,  //useTrailingStop
                    true,               //useTilt
                    rfname.str().c_str(),    //savename
                    AGREE_BOT,          //outputType
                    j,                  //agreeThreshold
                    i);                //botsPolled
            system("wait");
            system("ps | grep wscript.exe | awk '{print $1}' | wait");
            sleep(5);
            std::cout << j << "of" << i << ": " << checkResults(rfname.str()) << std::endl;
            std::ofstream outfile(outfilename.c_str(), std::ios::app);
            outfile << j << "of" << i << ": " << checkResults(rfname.str()) << std::endl;
        }
    }
}
