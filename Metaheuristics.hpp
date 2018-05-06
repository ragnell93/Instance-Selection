#ifndef Metaheuristics_H
#define Metaheuristics_H
#include <cmath>
#include <armadillo>
#include <utility>
#include <random>
#include <algorithm>
#include <chrono>
#include "Knn.hpp"
#include "Metrics.hpp"
#include "Instance.hpp"
#include "utils.h"

using namespace std;
using namespace arma;

template <typename MetricType>
struct Genetic{

    MetricType* metric;
    int iterations;
    int numPopulation;

    Genetic(MetricType* met,int i,int np):metric(met),iterations(i),numPopulation(np){}

    pair<Instance,Instance> cross(Instance &x,Instance &y,int point){

        Instance resx = x, resy = y;
        for (int i = 0; i < point; i++){
            resx.units(i) = x.units(i);
            resy.units(i) = y.units(i);
        }
        for (int i = point; i < x.units.n_rows; i++){
            resx.units(i) = y.units(i);
            resy.units(i) = x.units(i);
        }
        resx.changeTrainingSet();
        resy.changeTrainingSet();

        return make_pair(resx,resy);
    }

    Instance mutate(Instance &x){

        Col<int> mask = initialInstance(0.05,x.units.n_rows);
        Instance res = x;
        for (int i = 0; i < x.units.n_rows; i++){
            if (mask(i) == 1){
                if (res.units(i) == 0) res.units(i) = 1;
                else if (res.units(i) == 1) res.units(i) = 0;
            }
        }
        res.changeTrainingSet();
        return res;        
    }

    pair<double,Instance> find(Instance &initial,int knearest){

        random_device rd; // obtain a random number from hardware
        mt19937 eng(rd()); // seed
        uniform_real_distribution<> disReal(0,1);
        uniform_int_distribution<> disInt(0,numPopulation-1);
        uniform_int_distribution<> disInt2(0,initial.units.n_rows-1);

        vector<Instance> population(numPopulation);
        vector<Instance> newPopulation(numPopulation);
        Instance bestInstance;
        double bestCost;
        int numRandom = floor(numPopulation*0.1); //number of chromosomes that will be random
        population[0] = initial; //mantain the initial instance
        vector<double> vecCost(numPopulation);

        Knn knn2(*(initial.originalTraining),*(initial.originaltrainResults),initial.unique);
        vector<vector<size_t>> ordIndex = knn2.search2(*(initial.originalTraining),knearest,*metric);

        for (int i = 1; i < (numPopulation - numRandom);i++){

            Col<int> c = initialInstance(0.5,initial.units.n_rows);
            Col<int> aux(c.n_rows,fill::zeros);
            for (int j = 0; j < c.n_rows;j++){
                if ((initial.units(j) == 1) && (c(j) == 1))
                    aux(j) = 1;
            }
            population[i] = initial;
            population[i].units = aux;
            population[i].changeTrainingSet();
        }

        //Filling with random samples 
        for (int i = 0; i < numRandom; i++){
            Col<int> c = initialInstance(0.1,initial.units.n_rows);
            population[numPopulation-i-1] = initial;
            population[numPopulation-i-1].units = c;
            population[numPopulation-i-1].changeTrainingSet();
        }

        for (int f = 0; f < numPopulation; f++){
            if (population[f].training.n_rows == 0){
                int onBit = disInt2(eng);
                population[f].units(onBit) = 1;
                population[f].changeTrainingSet();
            }
        }

        for (int i = 0; i < numPopulation; i++) vecCost[i] = population[i].cost2(knearest,ordIndex,knn2);
        bestInstance = population[0];
        bestCost = vecCost[0];

        for (int i = 1; i < numPopulation; i++){
            if (vecCost[i] < bestCost){
                bestInstance = population[i];
                bestCost = vecCost[i];
            }
        }
        
        for (int i = 0; i < iterations;i++){

            int j = 0;
            while (j < numPopulation){
                if (disReal(eng) < 0.9){

                    int f1 = disInt(eng);
                    for (int u = 0; u < 2; u++){
                        int faux = disInt(eng);
                        if (vecCost[faux] < vecCost[f1]) f1 = faux;
                    }
                    int f2 = disInt(eng);
                    for (int u = 0; u < 2; u++){
                        int faux = disInt(eng);
                        if (vecCost[faux] < vecCost[f2]) f2 = faux;
                    }

                    int point = disInt2(eng);
                    pair<Instance,Instance> auxPair = cross(population[f1],population[f2],point);

                    if (auxPair.first.training.n_rows == 0){
                        int onBit = disInt2(eng);
                        auxPair.first.units(onBit) = 1;
                        auxPair.first.changeTrainingSet();
                    }

                    if (auxPair.second.training.n_rows == 0){
                        int onBit = disInt2(eng);
                        auxPair.second.units(onBit) = 1;
                        auxPair.second.changeTrainingSet();
                    }

                    double costAux1 = auxPair.first.cost2(knearest,ordIndex,knn2);
                    double costAux2 = auxPair.second.cost2(knearest,ordIndex,knn2);


                    if (costAux1 < vecCost[f1]){
                        population[f1] = auxPair.first;
                        vecCost[f1] = costAux1;
                    }
                    else if (costAux1 < vecCost[f2]){
                        population[f2] = auxPair.first;
                        vecCost[f2] = costAux1;
                    }
                    if (costAux2 < vecCost[f1]){
                        population[f1] = auxPair.second;
                        vecCost[f1] = costAux2;
                    }
                    else if (costAux2 < vecCost[f2]){
                        population[f2] = auxPair.second;
                        vecCost[f2] = costAux2;
                    }

                    if (costAux1 < bestCost){
                        bestInstance = auxPair.first;
                        bestCost = costAux1;
                    }

                    if (costAux2 < bestCost){
                        bestInstance = auxPair.second;
                        bestCost = costAux2;
                    }

                    j += 2;
                }
            }

            for (int z=0; z < numPopulation; z++){
                if (disReal(eng) < 0.01){
                    population[z] = mutate(population[z]);
                    if (population[z].training.n_rows == 0){
                        int onBit = disInt2(eng);
                        population[z].units(onBit) = 1;
                        population[z].changeTrainingSet();
                    }
                    vecCost[z] = population[z].cost2(knearest,ordIndex,knn2);
                    if (vecCost[z] < bestCost){
                        bestInstance = population[z];
                        bestCost = vecCost[z];
                    }
                }
            }
        }

        Knn knn(bestInstance.training,bestInstance.trainResults,bestInstance.unique);
        double costResult = knn.score(*(bestInstance.originalTraining),knearest,*metric,*(bestInstance.originaltrainResults));

        return make_pair(costResult,bestInstance);

    }
};

template <typename MetricType>
struct Memetic{

    MetricType* metric;
    int iterations;
    int numPopulation;

    Memetic(MetricType* met,int i,int np):metric(met),iterations(i),numPopulation(np){}

    Instance cross(Instance &x,Instance &y,int point){

        Instance resx = x;
        for (int i = 0; i < point; i++) resx.units(i) = x.units(i);
        for (int i = point; i < x.units.n_rows; i++) resx.units(i) = y.units(i);   
        resx.changeTrainingSet();
        return resx;
    }

    Instance mutate(Instance &x){

        Col<int> mask = initialInstance(0.05,x.units.n_rows);
        Instance res = x;
        for (int i = 0; i < x.units.n_rows; i++){
            if (mask(i) == 1){
                if (res.units(i) == 0) res.units(i) = 1;
                else if (res.units(i) == 1) res.units(i) = 0;
            }
        }
        res.changeTrainingSet();
        return res;        
    }

    int findNeighbor(vector<vector<size_t>> &ind,Instance &ins,int index){

        int neighbor = -1;
        bool flag = false;
        for (int i = 0;i < ind[index].size() && !flag;i++){
            if (binary_search(ins.indexesT.begin(),ins.indexesT.end(),ind[index][i])){
                neighbor = ind[index][i];
                flag = true;
            }
        }
        return neighbor;
    }

    Instance localSearch(Instance &initial, vector<vector<size_t>> &ind,double threshold){

        Instance S = initial;
        vector<int> U(initial.units.n_rows);
        vector<int> R;
        vector<int> Uaux(initial.units.n_rows);
        for (int i=0; i<U.size(); i++) U[i] = findNeighbor(ind,initial,i);

        int i = 0,iteration = 0;
        double gain = 0;
        while (i < S.training.n_rows){

            int in = S.indexesT(i);
            if (std::find(R.begin(),R.end(),in) == R.end()){
                
                gain = 0;
                S.units(in) = 0;
                S.changeTrainingSet();

                if (S.training.n_rows == 0){
                    S.units(in) = 1;
                    S.changeTrainingSet();
                    break;
                }

                Uaux = U;

                for (int j=0; j<U.size();j++){
                    if (U[j] == in){
                        U[j] = findNeighbor(ind,S,j);

                        if (((*(S.originaltrainResults))(U[j]) != (*(S.originaltrainResults))(j)) 
                            && ((*(S.originaltrainResults))(Uaux[j]) == (*(S.originaltrainResults))(j))){
                            gain--;
                        }
                        else if (((*(S.originaltrainResults))(U[j]) == (*(S.originaltrainResults))(j)) 
                            && ((*(S.originaltrainResults))(Uaux[j]) != (*(S.originaltrainResults))(j))){
                            gain++;
                        }
                    }
                }

                if (gain >= threshold){
                    while (!R.empty()) R.pop_back();
                    i = 0;
                }
                else{
                    U = Uaux;
                    S.units(in) = 1;
                    S.changeTrainingSet();
                    R.push_back(in);
                }
            }
            i++;   

        }

        return S;
    }

    pair<double,Instance> find(Instance &initial,int knearest){

        random_device rd; // obtain a random number from hardware
        mt19937 eng(rd()); // seed
        uniform_real_distribution<> disReal(0,1);
        uniform_int_distribution<> disInt(0,numPopulation-1);
        uniform_int_distribution<> disInt2(0,initial.units.n_rows-1);

        vector<Instance> population(numPopulation);
        vector<Instance> newPopulation(numPopulation);
        Instance bestInstance, worstInstance;
        double bestCost, worstCost;
        int worstIndex;
        double costAux1;
        int numRandom = floor(numPopulation*0.1); //number of chromosomes that will be random
        population[0] = initial; //mantain the initial instance
        vector<double> vecCost(numPopulation);

        double threshold, actualAcc, pastAcc, actualRed, pastRed;
        int AccInd = 0, RedInd = 0;

        Knn knn2(*(initial.originalTraining),*(initial.originaltrainResults),initial.unique);
        vector<vector<size_t>> ordIndex = knn2.search2(*(initial.originalTraining),knearest,*metric);

        for (int i = 1; i < (numPopulation - numRandom);i++){

            Col<int> c = initialInstance(0.5,initial.units.n_rows);
            Col<int> aux(c.n_rows,fill::zeros);
            for (int j = 0; j < c.n_rows;j++){
                if ((initial.units(j) == 1) && (c(j) == 1))
                    aux(j) = 1;
            }
            population[i] = initial;
            population[i].units = aux;
            population[i].changeTrainingSet();
        }

        //Filling with random samples 
        for (int i = 0; i < numRandom; i++){
            Col<int> c = initialInstance(0.1,initial.units.n_rows);
            population[numPopulation-i-1] = initial;
            population[numPopulation-i-1].units = c;
            population[numPopulation-i-1].changeTrainingSet();
        }

        for (int f = 0; f < numPopulation; f++){
            if (population[f].training.n_rows == 0){
                int onBit = disInt2(eng);
                population[f].units(onBit) = 1;
                population[f].changeTrainingSet();
            }
        }

        for (int i = 0; i < numPopulation; i++) vecCost[i] = population[i].cost2(knearest,ordIndex,knn2);
        bestInstance = population[0];
        bestCost = vecCost[0];
        worstInstance = population[0];
        worstCost = vecCost[0];

        for (int i = 1; i < numPopulation; i++){
            if (vecCost[i] < bestCost){
                bestInstance = population[i];
                bestCost = vecCost[i];
            }
            if (vecCost[i] > worstCost){
                worstInstance = population[i];
                worstCost = vecCost[i];
                worstIndex = i;
            }
        }

        actualAcc = bestCost;
        actualRed = bestInstance.training.n_rows;

        for (int i = 0; i < iterations;i++){

            pastRed = actualRed;
            pastAcc = actualAcc;

            int f1 = disInt(eng);
            for (int u = 0; u < 2; u++){
                int faux = disInt(eng);
                if (vecCost[faux] > vecCost[f1]) f1 = faux;
            }
            int f2 = disInt(eng);
            for (int u = 0; u < 2; u++){
                int faux = disInt(eng);
                if (vecCost[faux] > vecCost[f2]) f2 = faux;
            }

            int point = disInt2(eng);
            Instance auxPair = cross(population[f1],population[f2],point);

            if (disReal(eng) < 0.01) auxPair = mutate(auxPair);

            if (auxPair.training.n_rows == 0){
                int onBit = disInt2(eng);
                auxPair.units(onBit) = 1;
                auxPair.changeTrainingSet();
            }

            costAux1 = auxPair.cost2(knearest,ordIndex,knn2);
            double pls = 0.0625;
            if (costAux1 < worstCost) pls = 1;

            Instance improvement = auxPair;

            if (disReal(eng) < pls) improvement = localSearch(auxPair,ordIndex,threshold);

            if (improvement.training.n_rows == 0){
                int onBit = disInt2(eng);
                improvement.units(onBit) = 1;
                improvement.changeTrainingSet();
            }

            costAux1 = improvement.cost2(knearest,ordIndex,knn2);

            if (costAux1 < worstCost){
                population[worstIndex] = improvement;
                vecCost[worstIndex] = costAux1;
            }

            if (costAux1 < bestCost){
                bestInstance = improvement;
                bestCost = costAux1;
            }

            worstInstance = population[0];
            worstCost = vecCost[0];
            worstIndex = 0;
            for (int i = 0; i < numPopulation; i++){
                if (vecCost[i] > worstCost){
                    worstInstance = population[i];
                    worstCost = vecCost[i];
                    worstIndex = i;
                }
            }

            actualAcc = bestCost;
            actualRed = bestInstance.training.n_rows;

            if (abs(actualAcc - pastAcc) < 0.00001) AccInd++;
            else AccInd = 0;
            if (abs(actualRed - pastRed) < 0.00001) RedInd++;
            else RedInd = 0;

            if (AccInd >= 10) threshold++;
            if (RedInd >= 10) threshold--;

        }

        Knn knn(bestInstance.training,bestInstance.trainResults,bestInstance.unique);
        double costResult = knn.score(*(bestInstance.originalTraining),knearest,*metric,*(bestInstance.originaltrainResults));

        return make_pair(costResult,bestInstance);

    }
};

template <typename MetricType>
struct CHC{

    MetricType* metric;
    int iterations;
    int numPopulation;

    CHC(MetricType* met,int i,int np):metric(met),iterations(i),numPopulation(np){}

    double hamming(Col<int> &x, Col<int> &y){
        double distance = 0;
        for (int i=0; i < x.n_rows; i++){
            if (x(i) != y(i)) distance++;
        }
        return distance;
    }

    pair<Instance,Instance> cross(Instance &x,Instance &y){

        random_device rd; // obtain a random number from hardware
        mt19937 eng(rd()); // seed
        uniform_real_distribution<> disReal(0,1);

        int dis = hamming(x.units,y.units)/2;
        Instance resx = x, resy = y;
        for (int i=0; i<x.units.n_rows;i++){
            if (dis > 0 && x.units(i)!=y.units(i) && disReal(eng)<0.5){
                resx.units(i) = y.units(i);
                resy.units(i) = x.units(i);
                dis--;

            }
        }
        resx.changeTrainingSet();
        resy.changeTrainingSet();

        return make_pair(resx,resy);
    }

    void initPopulation(Instance initial,vector<vector<size_t>> ordIndex,vector<Instance> &population,vector<double> &vecCost,Instance &bestInstance,double &bestCost,Instance &worstInstance,double &worstCost,int &worstIndex,Knn knn2,int knearest,double p,bool re){

        random_device rd; // obtain a random number from hardware
        mt19937 eng(rd()); // seed
        uniform_real_distribution<> disReal(0,1);
        uniform_int_distribution<> disInt(0,numPopulation-1);
        uniform_int_distribution<> disInt2(0,initial.units.n_rows-1);

        population[0] = initial; //mantain the initial instance

        for (int i = 1; i < numPopulation ;i++){

            Col<int> c = initialInstance(p,initial.units.n_rows);
            Col<int> aux(c.n_rows,fill::zeros);
            for (int j = 0; j < c.n_rows;j++){
                if ((initial.units(j) == 1) && (c(j) == 1)) aux(j) = 1;
            }
            population[i] = initial;
            population[i].units = aux;
            population[i].changeTrainingSet();
        }

        for (int f = 0; f < numPopulation; f++){
            if (population[f].training.n_rows == 0){
                int onBit = disInt2(eng);
                population[f].units(onBit) = 1;
                population[f].changeTrainingSet();
            }
        }

        for (int i = 0; i < numPopulation; i++) vecCost[i] = population[i].cost2(knearest,ordIndex,knn2);
        bestInstance = population[0];
        bestCost = vecCost[0];
        worstInstance = population[0];
        worstCost = vecCost[0];

        for (int i = 1; i < numPopulation; i++){
            if (vecCost[i] < bestCost){
                bestInstance = population[i];
                bestCost = vecCost[i];
            }
            if (vecCost[i] > worstCost){
                worstInstance = population[i];
                worstCost = vecCost[i];
                worstIndex = i;
            }
        }
    }

    pair<double,Instance> find(Instance &initial,int knearest){

        random_device rd; // obtain a random number from hardware
        mt19937 eng(rd()); // seed
        uniform_real_distribution<> disReal(0,1);
        uniform_int_distribution<> disInt(0,numPopulation-1);
        uniform_int_distribution<> disInt2(0,initial.units.n_rows-1);

        vector<Instance> population(numPopulation);
        Instance bestInstance, worstInstance;
        double bestCost, worstCost;
        int worstIndex;
        double costAux1,costAux2;
        int numRandom = floor(numPopulation*0.1); //number of chromosomes that will be random
        vector<double> vecCost(numPopulation);
        bool flag;

        double threshold = initial.units.n_rows/8;

        Knn knn2(*(initial.originalTraining),*(initial.originaltrainResults),initial.unique);
        vector<vector<size_t>> ordIndex = knn2.search2(*(initial.originalTraining),knearest,*metric);

        initPopulation(initial,ordIndex,population,vecCost,bestInstance,bestCost,worstInstance,worstCost,worstIndex,knn2,knearest,0.5,false);

        for (int i=0;i<iterations;i++){
            
            int f1 = disInt(eng);
            int f2 = disInt(eng);
            double h = hamming(population[f1].units,population[f2].units);
            flag = false; //if a child enters in the population

            if (h > threshold){

                pair<Instance,Instance> auxPair = cross(population[f1],population[f2]);

                costAux1 = auxPair.first.cost2(knearest,ordIndex,knn2);
                costAux2 = auxPair.second.cost2(knearest,ordIndex,knn2);

                if (costAux1 < worstCost){    
                    population[worstIndex] = auxPair.first;
                    vecCost[worstIndex] = costAux1;
                    flag = true;
                    worstInstance = population[0];
                    worstCost = vecCost[0];
                    worstIndex = 0;
                    for (int i = 0; i < numPopulation; i++){
                        if (vecCost[i] > worstCost){
                            worstInstance = population[i];
                            worstCost = vecCost[i];
                            worstIndex = i;
                        }
                    }
                }

                if (costAux1 < bestCost){
                    bestInstance = auxPair.first;
                    bestCost = costAux1;
                }

                if (costAux2 < worstCost){    
                    population[worstIndex] = auxPair.second;
                    vecCost[worstIndex] = costAux2;
                    flag = true;
                    worstInstance = population[0];
                    worstCost = vecCost[0];
                    worstIndex = 0;
                    for (int i = 0; i < numPopulation; i++){
                        if (vecCost[i] > worstCost){
                            worstInstance = population[i];
                            worstCost = vecCost[i];
                            worstIndex = i;
                        }
                    }
                }

                if (costAux2 < bestCost){
                    bestInstance = auxPair.second;
                    bestCost = costAux2;
                }
            }

            if (!flag){
                if (threshold <= 0){
                    initPopulation(initial,ordIndex,population,vecCost,bestInstance,bestCost,worstInstance,worstCost,worstIndex,knn2,knearest,0.5,true);
                    threshold = initial.units.n_rows/8;
                } 
                else threshold--;
            }
        }

        Knn knn(bestInstance.training,bestInstance.trainResults,bestInstance.unique);
        double costResult = knn.score(*(bestInstance.originalTraining),knearest,*metric,*(bestInstance.originaltrainResults));

        return make_pair(costResult,bestInstance);
    }
        

};

#endif