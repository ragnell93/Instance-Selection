
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

    headerFile.close();
    dataFile.close();
    resultsFile.close();
    if (ivdm.compare(argv[2]) == 0) probFile.close();

 

 

    IVDM iv(index,minmax,prob);
    Euclidean eu;
    
    //mat example1 = {{5.1, 3.5, 1.4, 0.2},{6.0, 2.7, 5.1, 1.6},{6.5, 3.0, 5.2, 2.0}};
    //Col<int> resultsEx1 ={0,1,2};
    mat example1 = {{1.14,-0.114}};
    Col<int> resultsEx1 = {0};

    double start_time = Utils::read_time_in_minutes();

    Col<int> units(data.n_rows,fill::zeros);
    units(0) = 1;
    Col<int> units2(data.n_rows,fill::ones);
    Instance iss(units2,1,0.1,&data,&example1,&results,&resultsEx1,3);
    CNN<Euclidean> cnn(&eu);
    ENN<Euclidean> enn(&eu);
    IB3<Euclidean> ib(&eu);
    RSS<Euclidean> rss(&eu);
    pair<double,Instance> pp = rss.find(iss,1);

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