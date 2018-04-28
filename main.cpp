
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
    int numHeuristic,numMeta,numFolds,knearest,numPop,iterations,stratAux;
    double pNeigh,pCost,pIni;
    bool strat;

    ifstream configFile(argv[3]);

    configFile >> dummy >> numHeuristic;
    configFile >> dummy >> numMeta;
    configFile >> dummy >> numFolds;
    configFile >> dummy >> knearest;
    configFile >> dummy >> stratAux;
    configFile >> dummy >> pNeigh;
    configFile >> dummy >> pCost;
    configFile >> dummy >> pIni;
    configFile >> dummy >> numPop;
    configFile >> dummy >> iterations;

    if (stratAux == 0) strat = false;
    else if (stratAux == 1) strat = true;

    headerFile.close();
    dataFile.close();
    resultsFile.close();
    if (ivdm.compare(argv[2]) == 0) probFile.close();


    IVDM iv(index,minmax,prob);
    Euclidean eu;

    CNN<Euclidean> cnn1(&eu);
    IB3<Euclidean> ib31(&eu);
    RSS<Euclidean> rss1(&eu);
    Genetic<Euclidean> gen1(&eu,iterations,numPop);

    CNN<IVDM> cnn2(&iv);
    IB3<IVDM> ib32(&iv);
    RSS<IVDM> rss2(&iv);
    Genetic<IVDM> gen2(&iv,iterations,numPop);

    vector<double> resultados;
    ofstream outfile;

    if (euclidean.compare(argv[2]) == 0){

        if ((numHeuristic == 0) && (numMeta == 0)){ 
            resultados = kfold(cnn1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true);
            outfile.open("./results/cnn_genetic.txt",ios_base::app);
        }

        else if ((numHeuristic == 1) && (numMeta == 0)){
           resultados = kfold(rss1,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false);
           outfile.open("./results/rss_genetic.txt",ios_base::app);
        }

        else if ((numHeuristic == 2) && (numMeta == 0)){
            resultados = kfold(ib31,gen1,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true);
            outfile.open("./results/ib3_genetic.txt",ios_base::app);
        }
    }

    else if (ivdm.compare(argv[2]) == 0){
        
        if ((numHeuristic == 0) && (numMeta == 0)){ 
            resultados = kfold(cnn2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true);
            outfile.open("./results/cnn_genetic.txt",ios_base::app);
        }

        else if ((numHeuristic == 1) && (numMeta == 0)){
           resultados = kfold(rss2,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,false);
           outfile.open("./results/rss_genetic.txt",ios_base::app);
        } 

        else if ((numHeuristic == 2) && (numMeta == 0)){
            resultados = kfold(ib32,gen2,data,results,numFolds,knearest,pNeigh,pCost,pIni,strat,true);
            outfile.open("./results/ib3_genetic.txt",ios_base::app);
        }
    }
    
    outfile << argv[1] << "," << resultados[0] << "," << resultados[1] << "," 
            << resultados[2] << "," << resultados[3] << "," << numFolds << ","  << strat<< endl;


    /*

    mat example1 = {{5.1, 3.5, 1.4, 0.2},{6.0, 2.7, 5.1, 1.6},{6.5, 3.0, 5.2, 2.0}};
    Col<int> resultsEx1 ={0,1,2};
    //mat example1 = {{1.14,-0.114}};
    //Col<int> resultsEx1 = {0};

    

    Col<int> units(data.n_rows,fill::zeros);
    units(0) = 1;
    Col<int> units2(data.n_rows,fill::ones);
    Col<int> units3 = initialInstance(0.3,data.n_rows);
    Instance iss(units3,1,0.1,&data,&example1,&results,&resultsEx1,3);

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