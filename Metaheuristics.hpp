#ifndef Metaheuristics_H
#define Metaheuristics_H
#include <cmath>
#include <armadillo>
#include <utility>
#include <random>
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
        Instance bestInstance = initial;
        double bestCost = 0;
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
            if (vecCost[i] > bestCost){
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
                        if (vecCost[faux] > vecCost[f1]) f1 = faux;
                    }
                    int f2 = disInt(eng);
                    for (int u = 0; u < 2; u++){
                        int faux = disInt(eng);
                        if (vecCost[faux] > vecCost[f2]) f2 = faux;
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


                    if (costAux1 > vecCost[f1]){
                        population[f1] = auxPair.first;
                        vecCost[f1] = costAux1;
                    }
                    else if (costAux1 > vecCost[f2]){
                        population[f2] = auxPair.first;
                        vecCost[f2] = costAux1;
                    }
                    if (costAux2 > vecCost[f1]){
                        population[f1] = auxPair.second;
                        vecCost[f1] = costAux2;
                    }
                    else if (costAux2 > vecCost[f2]){
                        population[f2] = auxPair.second;
                        vecCost[f2] = costAux2;
                    }

                    if (costAux1 > bestCost){
                        bestInstance = auxPair.first;
                        bestCost = costAux1;
                    }

                    if (costAux2 > bestCost){
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
                    if (vecCost[z] > bestCost){
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

#endif