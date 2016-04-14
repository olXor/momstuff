#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

int main() {
    std::ifstream resultfile("savegenbot/results");
    std::string line;
    std::vector< std::vector<double> > testprofits;
    std::vector< std::vector<double> > trainprofits;
    std::vector< std::vector<double> > test2profits;
    while(std::getline(resultfile, line)) {
        std::stringstream lss(line);
        std::string tok;
        std::vector<double> roundtestprofs;
        std::vector<double> roundtrainprofs;
        std::vector<double> roundtest2profs;

        lss >> tok >> tok >> tok >> tok >> tok >> tok; //"Profits for round # (# cycles):"
        std::size_t prev=0, pos;
        while(lss >> tok) {
            prev = tok.find_first_of("=", 0) + 1;
            pos = tok.find_first_of("(", prev);
            std::string tmp = tok.substr(prev, pos-prev);
            std::stringstream convert(tmp);
            double t;
            convert >> t;
            roundtestprofs.push_back(t);
            prev = pos+1;
            pos = tok.find_first_of(",)", prev);
            tmp = tok.substr(prev, pos-prev);
            convert.clear();
            convert.str(tmp);
            convert >> t;
            roundtrainprofs.push_back(t);
            prev = pos+1;
            if(pos = tok.find_first_of(")",prev) != std::string::npos) {
                tmp = tok.substr(prev, pos-prev);
                convert.clear();
                convert.str(tmp);
                convert >> t;
                roundtest2profs.push_back(t);
            }
        }
        testprofits.push_back(roundtestprofs);
        trainprofits.push_back(roundtrainprofs);
        test2profits.push_back(roundtest2profs);
    }

    std::ofstream testbestfile("resultplot/testbest");
    for(int i=0; i<testprofits.size(); i++) {
        if(testprofits[i].size() > 0)
            testbestfile << i+1 << " " << testprofits[i][0] << std::endl;
        else
            testbestfile << i+1 << " " << 0 << std::endl;
    }

    std::ofstream testfile("resultplot/testaverages");
    for(int i=0; i<testprofits.size(); i++) {
        int size = testprofits[i].size();
        if(size==0)
            continue;
        double avg = 0;
        for(int j=0; j<size; j++) {
            avg += testprofits[i][j];
        }
        avg /= size;
        testfile << i+1 << " " << avg << std::endl;
    }

    std::ofstream testtop15file("resultplot/testtop15averages");
    for(int i=0; i<testprofits.size(); i++) {
        int size = testprofits[i].size();
        if(size==0)
            continue;
        double avg = 0;
        for(int j=0; j<size && j<15; j++) {
            avg += testprofits[i][j];
        }
        if(size <= 15)
            avg /= size;
        else
            avg /= 15;
        testtop15file << i+1 << " " << avg << std::endl;
    }

    std::ofstream testtop5file("resultplot/testtop5averages");
    for(int i=0; i<testprofits.size(); i++) {
        int size = testprofits[i].size();
        if(size==0)
            continue;
        double avg = 0;
        for(int j=0; j<size && j<5; j++) {
            avg += testprofits[i][j];
        }
        if(size <= 5)
            avg /= size;
        else
            avg /= 5;
        testtop5file << i+1 << " " << avg << std::endl;
    }

    std::ofstream trainbestfile("resultplot/trainbest");
    for(int i=0; i<trainprofits.size(); i++) {
        if(trainprofits[i].size() > 0)
            trainbestfile << i+1 << " " << trainprofits[i][0] << std::endl;
        else
            trainbestfile << i+1 << " " << 0 << std::endl;
    }

    std::ofstream trainfile("resultplot/trainaverages");
    for(int i=0; i<trainprofits.size(); i++) {
        int size = trainprofits[i].size();
        if(size==0)
            continue;
        double avg = 0;
        for(int j=0; j<size; j++) {
            avg += trainprofits[i][j];
        }
        avg /= size;
        trainfile << i+1 << " " << avg << std::endl;
    }

    std::ofstream traintop15file("resultplot/traintop15averages");
    for(int i=0; i<trainprofits.size(); i++) {
        int size = trainprofits[i].size();
        if(size==0)
            continue;
        double avg = 0;
        for(int j=0; j<size && j<15; j++) {
            avg += trainprofits[i][j];
        }
        if(size <= 15)
            avg /= size;
        else
            avg /= 15;
        traintop15file << i+1 << " " << avg << std::endl;
    }

    std::ofstream traintop5file("resultplot/traintop5averages");
    for(int i=0; i<trainprofits.size(); i++) {
        int size = trainprofits[i].size();
        if(size==0)
            continue;
        double avg = 0;
        for(int j=0; j<size && j<5; j++) {
            avg += trainprofits[i][j];
        }
        if(size <= 5)
            avg /= size;
        else
            avg /= 5;
        traintop5file << i+1 << " " << avg << std::endl;
    }

    std::ofstream test2bestfile("resultplot/test2best");
    for(int i=0; i<test2profits.size(); i++) {
        if(test2profits[i].size() > 0)
            test2bestfile << i+1 << " " << test2profits[i][0] << std::endl;
        else
            test2bestfile << i+1 << " " << 0 << std::endl;
    }

    std::ofstream test2file("resultplot/test2averages");
    for(int i=0; i<test2profits.size(); i++) {
        int size = test2profits[i].size();
        if(size==0)
            continue;
        double avg = 0;
        for(int j=0; j<size; j++) {
            avg += test2profits[i][j];
        }
        avg /= size;
        test2file << i+1 << " " << avg << std::endl;
    }

    std::ofstream test2top15file("resultplot/test2top15averages");
    for(int i=0; i<test2profits.size(); i++) {
        int size = test2profits[i].size();
        if(size==0)
            continue;
        double avg = 0;
        for(int j=0; j<size && j<15; j++) {
            avg += test2profits[i][j];
        }
        if(size <= 15)
            avg /= size;
        else
            avg /= 15;
        test2top15file << i+1 << " " << avg << std::endl;
    }

    std::ofstream test2top5file("resultplot/test2top5averages");
    for(int i=0; i<test2profits.size(); i++) {
        int size = test2profits[i].size();
        if(size==0)
            continue;
        double avg = 0;
        for(int j=0; j<size && j<5; j++) {
            avg += test2profits[i][j];
        }
        if(size <= 5)
            avg /= size;
        else
            avg /= 5;
        test2top5file << i+1 << " " << avg << std::endl;
    }

    system("gnuplot resultplot/averages.in");
    system("mobapictureviewer resultplot/averages.png");
}
