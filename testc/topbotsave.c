#include <iostream>
#include <string>
#include <sstream>
#include <fstream>

int main() {
    std::ifstream file("savegenbot/currentbots");
    std::string line;
    int id;
    system("rm -r topbots/*");
    while(getline(file,line)) {
        if(!(std::istringstream(line) >> id))
            throw std::runtime_error("found something that wasn't an id in the current Genbots file");

        std::stringstream ss;
        ss << "cp -r savegenbot/" << id << " topbots/";
        system(ss.str().c_str());
    }
    system("cp savegenbot/currentbots topbots/");
}
