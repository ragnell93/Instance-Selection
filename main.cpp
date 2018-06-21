
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
#include <boost/program_options.hpp>
#include "Metrics.hpp"
#include "Knn.hpp"
#include "Kfold.hpp"
#include "Instance.hpp"
#include "Heuristics.hpp"
#include "Metaheuristics.hpp"
#include "utils.h"

using namespace std;
using namespace arma;
namespace po = boost::program_options;

int main (int argc, char* argv[]) {

    po::options_description desc("Allowed options");
    desc.add_options()
        ("help","produce help message")
        ("instance,i",po::value<string>(),"instance to run")
        ("distance,d",po::value<string>(), "euclidean or ivdm.Default:euclidean")
        ("crossr",po::value<double>(),"set crossover rate. Default:1")
        ("mutr",po::value<double>(),"set mutation rate.Default:0.001")
        ("tsize",po::value<int>(),"set tournament size.Default:3")
        ("pop",po::value<int>(),"set population size.Default:50")
        ("iter",po::value<int>(),"set number of iterations to run the algorithm.Default:1000")
        ("seed",po::value<int>(),"seed")
        ("strat",po::value<int>(),"1 for using stratification, 0 for no stratification.Default:1")
        ("nfolds",po::value<int>(),"number of folds to use.Default:10")
        ("k",po::value<int>(), "number of nearest neighbors.Deafult:1")
        ("cost",po::value<double>(), "cost function parameter [0,1].Default:0.5")
        ("pini",po::value<double>(), "initial chromosome probability [0,1].Default:0.3")
        ("h1",po::value<string>(),"first heuristic to use: cnn, rss or enn")
        ("h2",po::value<string>(),"second heuristic to use: cnn, rss or enn")
        ("mh",po::value<string>(),"metaheuristic to use: gga, sga, ma or chc")

    ;

    po::variables_map vm;
    po::store(po::parse_command_line(argc,argv,desc),vm);
    po::notify(vm);

   if(vm.count("help")){
      cout << desc << endl;
      return 1;
   }
    
    string headerPath, dataPath, resultsPath, probPath, euclidean("euclidean"), ivdm("ivdm");

    ifstream auxInst(vm["instance"].as<string>());
    string instancia;
    auxInst >> instancia;

    headerPath = dataPath = resultsPath = "./euclidean/";
    if (vm.count("distance")){
        if (euclidean.compare(vm["distance"].as<string>()) == 0) headerPath = dataPath = resultsPath = "./euclidean/";
        else if (ivdm.compare(vm["distance"].as<string>()) == 0) headerPath = dataPath = resultsPath = probPath = "./ivdm/";
    }

    headerPath = headerPath + instancia + ".header";
    dataPath = dataPath + instancia + ".data";
    resultsPath = resultsPath + instancia + ".results";
    if(vm.count("distance")){
        if (ivdm.compare(vm["distance"].as<string>()) == 0) probPath = probPath + vm["instance"].as<string>() + ".prob";
    }

    ifstream headerFile(headerPath), dataFile(dataPath), resultsFile(resultsPath), probFile;
    if (ivdm.compare(argv[2]) == 0) probFile.open(probPath);

    vector <int> dimData(2), dimProb(3);
    headerFile >> dimData[0] >> dimData[1]; //# of files and columns
    if(vm.count("distance")){
        if (ivdm.compare(vm["distance"].as<string>()) == 0)  headerFile >> dimProb[0] >> dimProb[1] >> dimProb[2]; //Dimensions of prob cube
    }

    mat data(dimData[0],dimData[1]), minmax(3,dimData[1]);
    Col<int> results(dimData[0]);
    cube prob(0,0,0);
    if(vm.count("distance")){
        if (ivdm.compare(vm["distance"].as<string>()) == 0) prob.set_size(dimProb[0],dimProb[1],dimProb[2]);
    }

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
    if(vm.count("distance")){
        if (ivdm.compare(vm["distance"].as<string>()) == 0){

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
    }

    string dummy;
    int numHeuristic,numHeuristic2, numMeta,numFolds = 10,knearest = 1,numPop = 50,iterations = 1000,stratAux = 1,tournament = 3;
    double pNeigh = 0.1,pCost = 0.5,pIni=0.3,crossP=1,mutationP=0.001,acept=0.9,drop=0.7;
    bool strat;

    if (vm.count("nfolds")) numFolds = vm["nfolds"].as<int>();
    if (vm.count("k")) knearest = vm["k"].as<int>();
    if (vm.count("strat")) stratAux = vm["strat"].as<int>();
    if (vm.count("cost")) pCost = vm["cost"].as<double>();
    if (vm.count("pini")) pIni = vm["pini"].as<double>();
    if (vm.count("pop")) numPop = vm["pop"].as<int>();
    if (vm.count("iter")) iterations = vm["iter"].as<int>();
    if (vm.count("crossr")) crossP = vm["crossr"].as<double>();
    if (vm.count("mutr")) mutationP = vm["mutr"].as<double>();
    if (vm.count("tsize")) tournament = vm["tsize"].as<int>();

    if (stratAux == 0) strat = false;
    else if (stratAux == 1) strat = true;

    string cnn("cnn"),rss("rss"),ib3("ib3"),enn("enn"),genetic("sga"),memetic("ma"),chc("chc"),geneticGen("gga");

    if (cnn.compare(vm["h1"].as<string>()) == 0) numHeuristic = 0;
    if (rss.compare(vm["h1"].as<string>()) == 0) numHeuristic = 1;
    if (ib3.compare(vm["h1"].as<string>()) == 0) numHeuristic = 2;
    if (enn.compare(vm["h1"].as<string>()) == 0) numHeuristic = 3;

    if (cnn.compare(vm["h2"].as<string>()) == 0) numHeuristic2 = 0;
    if (rss.compare(vm["h2"].as<string>()) == 0) numHeuristic2 = 1;
    if (ib3.compare(vm["h2"].as<string>()) == 0) numHeuristic2 = 2;
    if (enn.compare(vm["h2"].as<string>()) == 0) numHeuristic2 = 3;

    if (genetic.compare(vm["mh"].as<string>()) == 0) numMeta = 0;
    if (memetic.compare(vm["mh"].as<string>()) == 0) numMeta = 1;
    if (chc.compare(vm["mh"].as<string>()) == 0) numMeta = 2;
    if (geneticGen.compare(vm["mh"].as<string>()) == 0) numMeta = 3;
    
    headerFile.close();
    dataFile.close();
    resultsFile.close();
    if(vm.count("distance")){
        if (ivdm.compare(vm["distance"].as<string>()) == 0) probFile.close();
    }

    /*
    if (vm.count("nfolds")) numFolds = vm["nfolds"].as<int>();
    if (vm.count("k")) knearest = vm["k"].as<int>();
    if (vm.count("strat")) stratAux = vm["strat"].as<int>();
    if (vm.count("cost")) pCost = vm["cost"].as<double>();
    if (vm.count("pini")) pIni = vm["pini"].as<double>();
    if (vm.count("pop")) numPop = vm["pop"].as<int>();
    if (vm.count("iter")) iterations = vm["iter"].as<int>();
    if (vm.count("crossr")) crossP = vm["crossr"].as<double>();
    if (vm.count("mutr")) mutationP = vm["mutr"].as<double>();
    if (vm.count("tsize")) tournament = vm["tsize"].as<int>();

    cout << "numfolds " << numFolds << endl;
    cout << "k " << knearest << endl;
    cout << "strat " << stratAux << endl
         << "cost " << pCost << endl
         << "pop " << numPop << endl
         << "iter " << iterations << endl
         << "crossr " << crossP << endl
         << "mutr " << mutationP << endl
         << "tsize " << tournament << endl;

    cout << "heuristiscs " << numHeuristic << " " <<numHeuristic2 << " " << numMeta << endl;

    cout <<"Dimensions " << data.n_rows << " " << data.n_cols << " " << results.n_rows << endl;
    */
    IVDM iv(index,minmax,prob);
    Euclidean eu;

    CNN<Euclidean> cnn1(&eu);
    IB3<Euclidean> ib31(&eu,acept,drop);
    RSS<Euclidean> rss1(&eu);
    ENN<Euclidean> enn1(&eu);
    GeneticS<Euclidean> gen1(&eu,iterations,numPop,crossP,mutationP,tournament);
    Memetic<Euclidean> mem1(&eu,iterations,numPop,crossP,mutationP,tournament);
    CHC<Euclidean> chc1(&eu,iterations,numPop);
    GeneticG<Euclidean> genG1(&eu,iterations,numPop,crossP,mutationP);

    CNN<IVDM> cnn2(&iv);
    IB3<IVDM> ib32(&iv,acept,drop);
    RSS<IVDM> rss2(&iv);
    ENN<IVDM> enn2(&iv);
    GeneticS<IVDM> gen2(&iv,iterations,numPop,crossP,mutationP,tournament);
    Memetic<IVDM> mem2(&iv,iterations,numPop,crossP,mutationP,tournament);
    CHC<IVDM> chc2(&iv,iterations,numPop);
    GeneticG<IVDM> genG2(&iv,iterations,numPop,crossP,mutationP);

    vector<double> resultados;
    ofstream outfile;
    string dis("euclidean");
    if(vm.count("distance")){
        if (ivdm.compare(vm["distance"].as<string>()) == 0) dis = "ivdm";
    }

    if (euclidean.compare(dis) == 0){
        
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

    if (ivdm.compare(dis) == 0){
        
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
 
    outfile << vm["instance"].as<string>() << "," << resultados[0] << "," << resultados[1] << "," 
            << resultados[2] << "," << resultados[3] << "," << resultados[4] << "," 
            << resultados[5] << "," << resultados[6] << endl;

    double objFunc = pCost * (1 - resultados[0]) + (1-pCost) * (1 - resultados[2]);

    cout << objFunc << endl;
  
    return 0;
}