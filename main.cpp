
#include <cstdlib>
#include <vector>
#include <utility>
#include <string>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <thread>
#include <armadillo>
#include <utility>
#include "Metrics.hpp"
#include "Knn.hpp"
#include "Kfold.hpp"
#include "Instance.hpp"
#include "Heuristics.hpp"
#include "Metaheuristics.hpp"
#include "utils.h"

using namespace std;
using namespace arma;


int main (int argc, char* argv[]) {
    
    string headerPath, dataPath, resultsPath, probPath, euclidean("euclidean"), ivdm("ivdm");
    if (euclidean.compare(argv[2]) == 0) headerPath = dataPath = resultsPath = "./euclidean/";
    else if (ivdm.compare(argv[2]) == 0) headerPath = dataPath = resultsPath = probPath = "./ivdm/";
    headerPath = headerPath + argv[1] + ".header";
    dataPath = dataPath + argv[1] + ".data";
    resultsPath = resultsPath + argv[1] + ".results";
    if (ivdm.compare(argv[2]) == 0) probPath = probPath + argv[1] + ".prob";

    ifstream headerFile(headerPath), dataFile(dataPath), resultsFile(resultsPath), probFile;
    if (ivdm.compare(argv[2]) == 0) probFile.open(probPath);

    vector <int> dimData(2), dimProb(3);
    headerFile >> dimData[0] >> dimData[1]; //# of files and columns
    if (ivdm.compare(argv[2]) == 0)  headerFile >> dimProb[0] >> dimProb[1] >> dimProb[2]; //Dimensions of prob cube

    mat data(dimData[0],dimData[1]), minmax(3,dimData[1]);
    Col<int> results(dimData[0]);
    cube prob(0,0,0);
    if (ivdm.compare(argv[2]) == 0) prob.set_size(dimProb[0],dimProb[1],dimProb[2]);

    //Read data file
    for (int i=0; i < dimData[0];i++){
        for (int j=0; j < dimData[1];j++){
            dataFile >> data(i,j);
        }
    }

    //Read results
    for (int i=0; i < dimData[0];i++) resultsFile >> results(i);

    int index;
    //Read probabilities file
    if (ivdm.compare(argv[2]) == 0){

        headerFile >> index;

        for (int i=0; i < 3 ; i++){
            for (int j=0; j < dimData[1];j++){
                headerFile >> minmax(i,j);
            }
        }

        for (int i=0; i<dimProb[0];i++){
            for (int j=0;j<dimProb[1];j++){
                for(int k = 0; k < dimProb[2];k++){
                    probFile >> prob(i,j,k);
                }
            }
        }
    }
    string dummy;
    int numHeuristic,numHeuristic2, numMeta,numFolds,knearest,numPop,iterations,stratAux,numPop2,numPop3,numPop4,iterations2,iterations3,iterations4;
    double pNeigh,pCost,pIni,crossP,mutationP,crossP2,mutationP2,crossP3,mutationP3,acept,drop;
    bool strat;

    ifstream configFile(argv[3]);

    configFile >> dummy >> numFolds;
    configFile >> dummy >> knearest;
    configFile >> dummy >> stratAux;
    configFile >> dummy >> pNeigh;
    configFile >> dummy >> pCost;
    configFile >> dummy >> pIni;
    configFile >> dummy >> acept;
    configFile >> dummy >> drop;
    configFile >> dummy >> numPop;
    configFile >> dummy >> iterations;
    configFile >> dummy >> crossP;
    configFile >> dummy >> mutationP;
    configFile >> dummy >> numPop2;
    configFile >> dummy >> iterations2;
    configFile >> dummy >> crossP2;
    configFile >> dummy >> mutationP2;
    configFile >> dummy >> numPop3;
    configFile >> dummy >> iterations3;
    configFile >> dummy >> numPop4;
    configFile >> dummy >> iterations4;
    configFile >> dummy >> crossP3;
    configFile >> dummy >> mutationP3;

    if (stratAux == 0) strat = false;
    else if (stratAux == 1) strat = true;

    string cnn("cnn"),rss("rss"),ib3("ib3"),enn("enn"),genetic("geneticS"),memetic("memetic"),chc("chc"),geneticGen("geneticG");

    if (cnn.compare(argv[4]) == 0) numHeuristic = 0;
    if (rss.compare(argv[4]) == 0) numHeuristic = 1;
    if (ib3.compare(argv[4]) == 0) numHeuristic = 2;
    if (enn.compare(argv[4]) == 0) numHeuristic = 3;

    if (cnn.compare(argv[5]) == 0) numHeuristic2 = 0;
    if (rss.compare(argv[5]) == 0) numHeuristic2 = 1;
    if (ib3.compare(argv[5]) == 0) numHeuristic2 = 2;
    if (enn.compare(argv[5]) == 0) numHeuristic2 = 3;

    if (genetic.compare(argv[6]) == 0) numMeta = 0;
    if (memetic.compare(argv[6]) == 0) numMeta = 1;
    if (chc.compare(argv[6]) == 0) numMeta = 2;
    if (geneticGen.compare(argv[6]) == 0) numMeta = 3;
    
    headerFile.close();
    dataFile.close();
    resultsFile.close();
    if (ivdm.compare(argv[2]) == 0) probFile.close();
    configFile.close();

    IVDM iv(index,minmax,prob);
    Euclidean eu;

    CNN<Euclidean> cnn1(&eu);
    IB3<Euclidean> ib31(&eu,acept,drop);
    RSS<Euclidean> rss1(&eu);
    ENN<Euclidean> enn1(&eu);
    GeneticS<Euclidean> gen1(&eu,iterations,numPop,crossP,mutationP);
    Memetic<Euclidean> mem1(&eu,iterations2,numPop2,crossP2,mutationP2);
    CHC<Euclidean> chc1(&eu,iterations3,numPop3);
    GeneticG<Euclidean> genG1(&eu,iterations4,numPop4,crossP3,mutationP3);

    CNN<IVDM> cnn2(&iv);
    IB3<IVDM> ib32(&iv,acept,drop);
    RSS<IVDM> rss2(&iv);
    ENN<IVDM> enn2(&iv);
    GeneticS<IVDM> gen2(&iv,iterations,numPop,crossP,mutationP);
    Memetic<IVDM> mem2(&iv,iterations2,numPop2,crossP2,mutationP2);
    CHC<IVDM> chc2(&iv,iterations3,numPop3);
    GeneticS<IVDM> genG2(&iv,iterations4,numPop4,crossP3,mutationP3);

    vector<double> resultados;
    ofstream outfile;

    if (euclidean.compare(argv[2]) == 0){
        
        if (numHeuristic == 0){
            if (numHeuristic2 == 0){
                if (numMeta == 0){
                    resultados = kfold(cnn1,cnn1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_cnn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(cnn1,cnn1,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_cnn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(cnn1,cnn1,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_cnn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(cnn1,cnn1,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_cnn_geneticG.txt",ios_base::app); 
                }
            }
            if (numHeuristic2 == 1){
                if (numMeta == 0){
                    resultados = kfold(cnn1,rss1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_rss_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(cnn1,rss1,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_rss_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(cnn1,rss1,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_rss_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(cnn1,rss1,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_rss_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 2){
                if (numMeta == 0){
                    resultados = kfold(cnn1,ib31,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_ib3_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(cnn1,ib31,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_ib3_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(cnn1,ib31,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_ib3_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(cnn1,ib31,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_ib3_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 3){
                if (numMeta == 0){
                    resultados = kfold(cnn1,enn1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_enn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(cnn1,enn1,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_enn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(cnn1,enn1,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_enn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(cnn1,enn1,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_enn_geneticG.txt",ios_base::app);       
                }
            }
        }


        if (numHeuristic == 1){
            if (numHeuristic2 == 0){
                if (numMeta == 0){
                    resultados = kfold(rss1,cnn1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_cnn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(rss1,cnn1,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_cnn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(rss1,cnn1,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_cnn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(rss1,cnn1,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_cnn_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 1){
                if (numMeta == 0){
                    resultados = kfold(rss1,rss1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_rss_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(rss1,rss1,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_rss_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(rss1,rss1,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_rss_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(rss1,rss1,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_rss_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 2){
                if (numMeta == 0){
                    resultados = kfold(rss1,ib31,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_ib3_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(rss1,ib31,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_ib3_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(rss1,ib31,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_ib3_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(rss1,ib31,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_ib3_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 3){
                if (numMeta == 0){
                    resultados = kfold(rss1,enn1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_enn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(rss1,enn1,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_enn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(rss1,enn1,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_enn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(rss1,enn1,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_enn_geneticG.txt",ios_base::app);       
                }
            }
        }
    

        if (numHeuristic == 2){
            if (numHeuristic2 == 0){
                if (numMeta == 0){
                    resultados = kfold(ib31,cnn1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_cnn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(ib31,cnn1,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_cnn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(ib31,cnn1,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_cnn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(ib31,cnn1,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_cnn_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 1){
                if (numMeta == 0){
                    resultados = kfold(ib31,rss1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_rss_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(ib31,rss1,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_rss_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(ib31,rss1,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_rss_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(ib31,rss1,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_rss_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 2){
                if (numMeta == 0){
                    resultados = kfold(ib31,ib31,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_ib3_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(ib31,ib31,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_ib3_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(ib31,ib31,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_ib3_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(ib31,ib31,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_ib3_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 3){
                if (numMeta == 0){
                    resultados = kfold(ib31,enn1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_enn1_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(ib31,enn1,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_enn1_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(ib31,enn1,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_enn1_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(ib31,enn1,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_enn1_geneticG.txt",ios_base::app);       
                }
            }
        }
    

        if (numHeuristic == 3){
            if (numHeuristic2 == 0){
                if (numMeta == 0){
                    resultados = kfold(enn1,cnn1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_cnn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(enn1,cnn1,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_cnn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(enn1,cnn1,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_cnn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(enn1,cnn1,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_cnn_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 1){
                if (numMeta == 0){
                    resultados = kfold(enn1,rss1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_rss_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(enn1,rss1,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_rss_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(enn1,rss1,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_rss_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(enn1,rss1,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_rss_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 2){
                if (numMeta == 0){
                    resultados = kfold(enn1,ib31,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_ib3_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(enn1,ib31,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_ib3_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(enn1,ib31,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_ib3_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(enn1,ib31,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_ib3_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 3){
                if (numMeta == 0){
                    resultados = kfold(enn1,enn1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_enn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(enn1,enn1,mem1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_enn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(enn1,enn1,chc1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_enn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(enn1,enn1,genG1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_enn_geneticG.txt",ios_base::app);       
                }
            }
        }
    }

    if (ivdm.compare(argv[2]) == 0){
        
        if (numHeuristic == 0){
            if (numHeuristic2 == 0){
                if (numMeta == 0){
                    resultados = kfold(cnn2,cnn2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_cnn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(cnn2,cnn2,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_cnn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(cnn2,cnn2,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_cnn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(cnn2,cnn2,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_cnn_geneticG.txt",ios_base::app); 
                }
            }
            if (numHeuristic2 == 1){
                if (numMeta == 0){
                    resultados = kfold(cnn2,rss2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_rss_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(cnn2,rss2,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_rss_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(cnn2,rss2,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_rss_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(cnn2,rss2,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_rss_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 2){
                if (numMeta == 0){
                    resultados = kfold(cnn2,ib32,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_ib3_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(cnn2,ib32,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_ib3_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(cnn2,ib32,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_ib3_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(cnn2,ib32,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/cnn_ib3_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 3){
                if (numMeta == 0){
                    resultados = kfold(cnn2,enn2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_enn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(cnn2,enn2,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_enn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(cnn2,enn2,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_enn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(cnn2,enn2,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/cnn_enn_geneticG.txt",ios_base::app);       
                }
            }
        }

        if (numHeuristic == 1){
            if (numHeuristic2 == 0){
                if (numMeta == 0){
                    resultados = kfold(rss2,cnn2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_cnn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(rss2,cnn2,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_cnn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(rss2,cnn2,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_cnn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(rss2,cnn2,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_cnn_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 1){
                if (numMeta == 0){
                    resultados = kfold(rss2,rss2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_rss_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(rss2,rss2,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_rss_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(rss2,rss2,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_rss_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(rss2,rss2,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_rss_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 2){
                if (numMeta == 0){
                    resultados = kfold(rss2,ib32,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_ib3_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(rss2,ib32,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_ib3_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(rss2,ib32,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_ib3_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(rss2,ib32,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/rss_ib3_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 3){
                if (numMeta == 0){
                    resultados = kfold(rss2,enn2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_enn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(rss2,enn2,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_enn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(rss2,enn2,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_enn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(rss2,enn2,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/rss_enn_geneticG.txt",ios_base::app);       
                }
            }
        }
    

        if (numHeuristic == 2){
            if (numHeuristic2 == 0){
                if (numMeta == 0){
                    resultados = kfold(ib32,cnn2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_cnn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(ib32,cnn2,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_cnn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(ib32,cnn2,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_cnn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(ib32,cnn2,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_cnn_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 1){
                if (numMeta == 0){
                    resultados = kfold(ib32,rss2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_rss_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(ib32,rss2,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_rss_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(ib32,rss2,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_rss_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(ib32,rss2,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_rss_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 2){
                if (numMeta == 0){
                    resultados = kfold(ib32,ib32,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_ib3_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(ib32,ib32,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_ib3_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(ib32,ib32,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_ib3_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(ib32,ib32,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,true);
                    outfile.open("./results/ib3_ib3_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 3){
                if (numMeta == 0){
                    resultados = kfold(ib32,enn2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_enn2_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(ib32,enn2,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_enn2_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(ib32,enn2,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_enn2_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(ib32,enn2,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true,false);
                    outfile.open("./results/ib3_enn2_geneticG.txt",ios_base::app);       
                }
            }
        }
    

        if (numHeuristic == 3){
            if (numHeuristic2 == 0){
                if (numMeta == 0){
                    resultados = kfold(enn2,cnn2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_cnn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(enn2,cnn2,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_cnn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(enn2,cnn2,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_cnn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(enn2,cnn2,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_cnn_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 1){
                if (numMeta == 0){
                    resultados = kfold(enn2,rss2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_rss_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(enn2,rss2,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_rss_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(enn2,rss2,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_rss_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(enn2,rss2,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_rss_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 2){
                if (numMeta == 0){
                    resultados = kfold(enn2,ib32,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_ib3_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(enn2,ib32,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_ib3_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(enn2,ib32,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_ib3_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(enn2,ib32,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,true);
                    outfile.open("./results/enn_ib3_geneticG.txt",ios_base::app);       
                }
            }
            if (numHeuristic2 == 3){
                if (numMeta == 0){
                    resultados = kfold(enn2,enn2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_enn_geneticS.txt",ios_base::app);       
                }
                if (numMeta == 1){
                    resultados = kfold(enn2,enn2,mem2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_enn_memetic.txt",ios_base::app);
                }
                if (numMeta == 2){
                    resultados = kfold(enn2,enn2,chc2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_enn_chc.txt",ios_base::app);
                }
                if (numMeta == 3){
                    resultados = kfold(enn2,enn2,genG2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false,false);
                    outfile.open("./results/enn_enn_geneticG.txt",ios_base::app);       
                }
            }
        }
    }
 
    outfile << argv[1] << "," << resultados[0] << "," << resultados[1] << "," 
            << resultados[2] << "," << resultados[3] << "," << numFolds << ","  << strat<< endl;


    /*
    mat example1 ={{-0.8639, 1.2744, -1.2440, -1.2085},{1.2771, 0.4351, 0.4586, 0.1653},{0.4883, 0.7149, 1.1294, 1.4248}};
    Col<int> resultsEx1 ={0,1,2};
    //mat example1 = {{1.14,-0.114}};
    //Col<int> resultsEx1 = {0};

    

    Col<int> units1(data.n_rows,fill::zeros);
    units1(0) = 1;
    Col<int> units2(data.n_rows,fill::ones);
    Col<int> units3 = initialInstance(0.5,data.n_rows);
    Instance iss(units3,1,0.7,&data,&data,&results,&results,3);

    Knn knn(data,results,3);
    vector<vector<size_t>> respuestas = knn.search2(data,1,eu);

    auto start = chrono::high_resolution_clock::now();

    pair<double,Instance> ajajaja = gen1.find(iss,1);

    auto stop = chrono::high_resolution_clock::now();

    using fpSeconds = chrono::duration<float,chrono::seconds::period>;

    auto duration = fpSeconds(stop - start);

    Knn knn2(iss.training,iss.trainResults,iss.unique);
    double costResult = knn2.score(*(iss.originalTraining),knearest,eu,*(iss.originaltrainResults));

    /*

    cout << "el tamaño original es: " << iss.training.n_rows << endl;
    cout << "el score original es : " <<  costResult << endl;
    cout << "el tamaño reducido es: " << ajajaja.second.training.n_rows << endl;
    cout << "el score reducido es: " << ajajaja.first << endl;
    cout << "el tiempo de ejecucion es: " << duration.count() << endl;

    mat printing(ajajaja.second.training.n_rows,ajajaja.second.training.n_cols+1);

    for (int i=0; i <ajajaja.second.training.n_cols;i++){
        printing.col(i) = ajajaja.second.training.col(i);
    }

    vec v(ajajaja.second.trainResults.n_rows);
    for (int i = 0; i< ajajaja.second.trainResults.n_rows; i++){
        v(i) = (double)ajajaja.second.trainResults(i);
    }

    printing.col(ajajaja.second.training.n_cols) = v;

    ofstream aux1("bananaSet.txt");
    printing.print(aux1);



    /*double aajj = iss.cost2(1,respuestas,knn);

    cout << "el training set es: "<< endl;

    iss.indexesT.print();
    cout << endl;

    cout << "el score es " << aajj << endl;

    /*
    for (int i = 0; i < respuestas.size(); i++){
        for (int j= 0; j < respuestas[i].size();j++){
            cout << respuestas[i][j] << " ";
        }
        cout << endl;
    }
    /*

    Knn knn(iss.training,iss.trainResults,iss.unique);
    double costResult = knn.score(*(iss.originalTraining),knearest,eu,*(iss.originaltrainResults));
    double kappaR1 = knn.kappa(*(iss.originalTraining),knearest,eu,*(iss.originaltrainResults));

    auto start = chrono::high_resolution_clock::now();

    pair<double,Instance> resultados4 = gen1.find(iss,1);

    auto stop = chrono::high_resolution_clock::now();

    using fpSeconds = chrono::duration<float,chrono::seconds::period>;

    auto duration = fpSeconds(stop - start);

    Knn knn2(resultados4.second.training,resultados4.second.trainResults,resultados4.second.unique);
    double kappaR2 = knn2.kappa(*(resultados4.second.originalTraining),knearest,eu,*(resultados4.second.originaltrainResults));

    cout << "el score original es: " << costResult << endl;
    cout << "el kappa original es: " << kappaR1 << endl;
    cout << "el tamaño original es: " << iss.training.n_rows << endl;
    cout << "el score de la instancia es : " << resultados4.first << endl;
    cout << "el kappa de la instancia es: " << kappaR2 << endl;
    cout << "el tamaño de la reduccion es : " << resultados4.second.training.n_rows << endl;
    cout << "el tiempo de ejecucion es: " << duration.count() << endl;
 

    //vector<double> rrees = kfold(ib,data,results,25,1,0.1,0.5,0.3,true,true);
    /*
    ofstream outfile;
    outfile.open("./results/resultados.txt",ios_base::app);
    outfile << argv[1] << "," << rrees[0] << "," << rrees[1] << "," <<  60*rrees[2] << endl;


    /*
    pair<double,Instance> pp = ib.find(iss,1);

    double elapsed_time = Utils::read_time_in_minutes() - start_time;

    cout << "el numer de instancias es : " << pp.second.training.n_rows << endl;
    cout << "el score es: " << pp.first << endl; 
    cout << "el tiempo en minutos es: " << elapsed_time << endl;

    ofstream aux1("bananaSet.txt");
    ofstream aux2("bananaRes.txt");
    pp.second.training.print(aux1);
    pp.second.trainResults.print(aux2);



    //LocalSearch<IVDM> ls(&iv);
    //double abcd = kfold(ls,data,results,4,1,0.1,0.5,0.5);
    //cout << "el score total es: " << abcd << endl;
    
    /* 
    Col<int> units = initialInstance(0.5,data.n_rows);    
    //Col<int> units = {0,0,0,0,0,0,1,0,1};

    Instance iss(units,1,0.5,&data,&example1,&results,&resultsEx1,3);
    cout << "el score es:" << endl;
    cout << iss.cost(iv,1) << endl;

    pair<double,Instance> neighbor = iss.searchNeighborhood(iv,1);
    cout << "el escore es" << endl;
    cout << neighbor.first << endl;


    //cout << kfold(iv,data,results,4,3) << endl;
    cout <<endl;



    //mat example2 = {{0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15, 2},{0.35, 0.265, 0.09, 0.2255, 0.0995, 0.0485, 0.07, 2}};
    //mat a = {{5.5,3.3,2.2,1.0},{4.4,3.2,2.1,1.0},{3.0,2.0,1.0,1.0}};

    //double re = k.score(example1,5,iv,resultsEx1);
    //cout << re << endl;


    /*
    vec example3 = {5.1, 3.5, 1.4, 0.2};
    vec example4 = {4.9, 3.0, 1.4, 0.2};
    vec example5 = {6.2, 3.4, 5.4, 2.3};
    vec example6 = {6.5, 3.0, 5.2, 2.0};
    //vec example1 = {0.455, 0.365, 0.095, 0.514, 0.2245, 0.101, 0.15, 2};
    //vec example2 = {0.35, 0.265, 0.09, 0.2255, 0.0995, 0.0485, 0.07, 2};

    //double re1,re2,re3;
    //re1 = iv.Evaluate(example3,example4);
    //re2 = iv.Evaluate(example3,example5);
    //re3 = iv.Evaluate(example6,example5);
    //cout << "La distancia es: " << endl;
    //cout << re1 << endl;
    //cout << re2 << endl;
    //cout << re3 << endl;
    */
    
    return 0;
}