#include "../genbot/cluster.h"
#include "../genbot/genbot.h"
#include "../genbot/genome.h"
#include <iostream>
#include <sys/time.h>

ClusterParameters* createClusterParameters(size_t nodes) {
    ClusterParameters* pars = new ClusterParameters();
    pars->numInputs = 2;
    pars->numOutputs = 1;
    pars->numLayers = 1;
    pars->nodesPerLayer = nodes;
}

long secDiff(timeval t1, timeval t2) {
    return t2.tv_sec - t1.tv_sec;
}

long uDiff(timeval t1, timeval t2) {
    return (t2.tv_usec - t1.tv_usec);
}

int main() {
    std::ofstream outfile("resultplot/perftest");
    for(size_t i=10; i<=1000; i+=100) {
        ClusterParameters *pars = createClusterParameters(i);
        Cluster* clust = new Cluster(pars);
        std::cout << i << " nodes" << std::endl;
        long calctime = 0;
        long learntime = 0;
        for(int j=0; j<1000; j++) {
            double inputs[2];
            inputs[0] = (1.0*(rand()%10))/10;
            inputs[1] = (1.0*(rand()%10))/10;
            timeval startcalctime;
            gettimeofday(&startcalctime, NULL);
            clust->setInputs(inputs);
            clust->calculate();
            clust->calculate();
            timeval endcalctime;
            gettimeofday(&endcalctime, NULL);
            calctime += 1000000*secDiff(startcalctime, endcalctime) + uDiff(startcalctime, endcalctime);

            timeval startlearntime;
            gettimeofday(&startlearntime, NULL);
            clust->learn((1.0*(rand()%10))/5-1);
            timeval endlearntime;
            gettimeofday(&endlearntime, NULL);
            learntime += 1000000*secDiff(startlearntime, endlearntime) + uDiff(startlearntime, endlearntime);
        }
        std::cout << "calc: " << calctime << " learn: " << learntime << " us" << std::endl;
        outfile << i << " " << calctime << " " << learntime << std::endl;
        delete clust;
    }

    system("gnuplot perftest.in");
    system("mobapictureviewer resultplot/perftest.png");
}
