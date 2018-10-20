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
struct GeneticS{

    /*Stationary genetic algorithm.
    
    metric: metric used to calculate distances
    iterations: number of iterations the algorithm will run
    numPopulation: number of elements in the population
    crossP: probability that 2 selected chromosomes will cross
    mutationP: probability that a chromosome will undergo mutation

    cross(x,y,point): cross the given instances x & y using point as the pivot for the one point cross.
                      two childs are returned.

    mutate(x): will flip bits of x randomly with 5% probability per bit

    find(initial,knearest): search the solution space using the stationary genetic algorithm
    */


    MetricType* metric;
    int iterations;
    int numPopulation;
    double crossP;
    double mutationP;
    int tournament;

    GeneticS(MetricType* met,int i,int np,double cp,double mp,int t):metric(met),iterations(i),
        numPopulation(np),crossP(cp),mutationP(mp),tournament(t){}

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
        int numRandom = floor(numPopulation*0); //number of chromosomes that will be random
        population[0] = initial; //mantain the initial instance
        vector<double> vecCost(numPopulation);

        Knn knn2(*(initial.originalTraining),*(initial.originaltrainResults),initial.unique);
        vector<vector<size_t>> ordIndex = knn2.search2(*(initial.originalTraining),knearest,*metric);

        //Fill population using the initial as template
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

        //if an instance have an empty subset
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

        float numIt21 = (iterations*crossP*numPopulation)/100;
        
        for (int i = 0; i < numIt21;i++){

            //Select the 2 parents from a 3-way tournament

            int f1 = disInt(eng);
            for (int u = 0; u < tournament; u++){
                int faux = disInt(eng);
                if (vecCost[faux] < vecCost[f1]) f1 = faux;
            }
            int f2 = disInt(eng);
            for (int u = 0; u < tournament; u++){
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

            //Update the costs
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


            //Mutation
            for (int z=0; z < numPopulation; z++){
                if (disReal(eng) < mutationP){
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

    /*Memetic algorithm.
    
    metric: metric used to calculate distances
    iterations: number of iterations the algorithm will run
    numPopulation: number of elements in the population
    crossP: probability that 2 selected chromosomes will cross
    mutationP: probability that a chromosome will undergo mutation

    cross(x,y,point): cross the given instances x & y using point as the pivot for the one point cross.
                      1 child is returned.

    mutate(x): will flip bits of x randomly with 5% probability per bit

    findNeighbor(ind,ins,index): will find the nearest neighbor of the "index" instance from the set formet by ins
                                using ind as a reference of distance matrix

    localSearch(initial,ind,threshold): perform local search on the "initial" instance using "ind" as the distance matrix
                                        and threshold as the point in which if the given gain is above, then the changes are
                                        accepted

    find(initial,knearest): search the solution space using a memetic algorithm 
    */

    MetricType* metric;
    int iterations;
    int numPopulation;
    double crossP;
    double mutationP;
    int tournament;

    Memetic(MetricType* met,int i,int np,double cp,double mp,int t):metric(met),iterations(i),
        numPopulation(np),crossP(cp),mutationP(mp),tournament(t){}

    vector<Instance> cross(Instance &x,Instance &y,int point){

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

        vector<Instance> resp(2);
        resp[0] = resx;
        resp[1] = resy;
        return resp;
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

        Instance S = initial; //current set
        vector<int> U(initial.units.n_rows); //current vector of neighbors 
        vector<int> R; //prohibited units
        vector<int> Uaux(initial.units.n_rows); //backup of U
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
        vector<double> costAux(2);
        int numRandom = floor(numPopulation*0); //number of chromosomes that will be random
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

            if (disReal(eng) < crossP){

                pastRed = actualRed;
                pastAcc = actualAcc;

                int f1 = disInt(eng);
                for (int u = 0; u < tournament; u++){
                    int faux = disInt(eng);
                    if (vecCost[faux] > vecCost[f1]) f1 = faux;
                }
                int f2 = disInt(eng);
                for (int u = 0; u < tournament; u++){
                    int faux = disInt(eng);
                    if (vecCost[faux] > vecCost[f2]) f2 = faux;
                }

                int point = disInt2(eng);
                vector<Instance> auxPair(2);
                auxPair = cross(population[f1],population[f2],point);
                if (disReal(eng) < mutationP) auxPair[0] = mutate(auxPair[0]);
                if (disReal(eng) < mutationP) auxPair[1] = mutate(auxPair[1]);

                for (int y = 0; y < 2; y++){
                    if (auxPair[y].training.n_rows == 0){
                        int onBit = disInt2(eng);
                        auxPair[y].units(onBit) = 1;
                        auxPair[y].changeTrainingSet();
                    }
                }

                costAux[0] = auxPair[0].cost2(knearest,ordIndex,knn2);
                costAux[1] = auxPair[1].cost2(knearest,ordIndex,knn2);

                for (int y = 0; y < 2; y++){

                    double pls = 0.0625;
                    if (costAux[y] < worstCost) pls = 1;

                    Instance improvement = auxPair[y];

                    if (disReal(eng) < pls) improvement = localSearch(auxPair[y],ordIndex,threshold);

                    if (improvement.training.n_rows == 0){
                        int onBit = disInt2(eng);
                        improvement.units(onBit) = 1;
                        improvement.changeTrainingSet();
                    }

                    costAux[y] = improvement.cost2(knearest,ordIndex,knn2);

                    if (costAux[y] < worstCost){
                        population[worstIndex] = improvement;
                        vecCost[worstIndex] = costAux[y];
                    }

                    if (costAux[y] < bestCost){
                        bestInstance = improvement;
                        bestCost = costAux[y];
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
            }
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
        uniform_int_distribution<> disInt2(0,x.units.n_rows-1);

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

        if (resx.training.n_rows == 0){
            int onBit = disInt2(eng);
            resx.units(onBit) = 1;
            resx.changeTrainingSet();
        }

        if (resy.training.n_rows == 0){
            int onBit = disInt2(eng);
            resy.units(onBit) = 1;
            resy.changeTrainingSet();
        }

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

        double threshold = initial.units.n_rows/4;

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
                    initPopulation(bestInstance,ordIndex,population,vecCost,bestInstance,bestCost,worstInstance,worstCost,worstIndex,knn2,knearest,0.5,true);
                    threshold = initial.units.n_rows/4;
                } 
                else threshold--;
            }
        }

        Knn knn(bestInstance.training,bestInstance.trainResults,bestInstance.unique);
        double costResult = knn.score(*(bestInstance.originalTraining),knearest,*metric,*(bestInstance.originaltrainResults));

        return make_pair(costResult,bestInstance);
    }
        

};

template <typename MetricType>
struct GeneticG{

    /*Generational genetic algorithm.
    
    metric: metric used to calculate distances
    iterations: number of iterations the algorithm will run
    numPopulation: number of elements in the population
    crossP: probability that 2 selected chromosomes will cross
    mutationP: probability that a chromosome will undergo mutation

    cross(x,y,point): cross the given instances x & y using point as the pivot for the one point cross.
                      two childs are returned.

    mutate(x): will flip bits of x randomly with 5% probability per bit

    find(initial,knearest): search the solution space using the stationary genetic algorithm
    */


    MetricType* metric;
    int iterations;
    int numPopulation;
    double crossP;
    double mutationP;

    GeneticG(MetricType* met,int i,int np,double cp,double mp):metric(met),iterations(i),
        numPopulation(np),crossP(cp),mutationP(mp){}

    pair<Instance,Instance> cross(Instance &x,Instance &y,int point){

        random_device rd; // obtain a random number from hardware
        mt19937 eng(rd()); // seed
        uniform_real_distribution<> disReal(0,1);
        uniform_int_distribution<> disInt2(0,x.units.n_rows-1);

        Instance resx = x, resy = y;
        for (int i = 0; i < point; i++){
            resx.units(i) = x.units(i);
            resy.units(i) = y.units(i);
        }
        for (int i = point; i < x.units.n_rows; i++){
            resx.units(i) = y.units(i);
            resy.units(i) = x.units(i);
        }
        if (resx.training.n_rows == 0){
            int onBit = disInt2(eng);
            resx.units(onBit) = 1;
            resx.changeTrainingSet();
        }

        if (resy.training.n_rows == 0){
            int onBit = disInt2(eng);
            resy.units(onBit) = 1;
            resy.changeTrainingSet();
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
        Instance bestInstance = initial;
        double bestCost  = initial.cost(*metric,knearest);
        int bestIndex = 0;
        int numRandom = floor(numPopulation*0); //number of chromosomes that will be random
        population[0] = initial; //mantain the initial instance

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
    
        for (int i = 1; i < numPopulation; i++){
            double costAux = population[i].cost2(knearest,ordIndex,knn2);
            if (costAux < bestCost){
                bestInstance = population[i];
                bestCost = costAux;
                bestIndex = i;
            }
        }
        
        float numIt21 = (iterations*crossP*numPopulation)/100;
        
        for (int i = 0; i < numIt21;i++){

            int f1 = disInt(eng);
            int f2 = disInt(eng);
            int point = disInt2(eng);
            pair<Instance,Instance> auxPair = cross(population[f1],population[f2],point);
            
            if (bestIndex != f1){
                population[f1] = auxPair.first;
            }

            if (bestIndex == f1){
                double costAux2 = population[f1].cost2(knearest,ordIndex,knn2);
                if (costAux2 < bestCost){
                    bestInstance = population[f1];
                    bestCost = costAux2;
                    bestIndex = f1;
                }
            }

            if (bestIndex != f2){
                population[f2] = auxPair.second;
            }
            
            if (bestIndex == f2){
                double costAux3 = population[f2].cost2(knearest,ordIndex,knn2);
                if (costAux3 < bestCost){
                    bestInstance = population[f2];
                    bestCost = costAux3;
                    bestIndex = f2;
                }
            }

            if (i % numPopulation == 0){
                for (int z=0; z < numPopulation; z++){
                    if (disReal(eng) < mutationP) population[z] = mutate(population[z]);
                }
            }
        }

        Knn knn(bestInstance.training,bestInstance.trainResults,bestInstance.unique);
        double costResult = knn.score(*(bestInstance.originalTraining),knearest,*metric,*(bestInstance.originaltrainResults));

        return make_pair(costResult,bestInstance);

    }
};

#endif