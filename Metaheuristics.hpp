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

    pair<Instance,Instance> cross(Instance &x,Instance &y){
        
        Col<int> mask = initialInstance(0.5,x.units.n_rows);
        Instance resx = x, resy = y;
        for (int i = 0; i < x.units.n_rows; i++){
            if (mask(i) == 0){
                resx.units(i) = x.units(i);
                resy.units(i) = y.units(i);
            }
            else if (mask(i) == 1){
                resx.units(i) = y.units(i);
                resy.units(i) = x.units(i);
            }
        }
        resx.changeTrainingSet();
        resy.changeTrainingSet();

        return make_pair(resx,resy);

    }

    Instance mutate(Instance &x){

        Col<int> mask = initialInstance(0.2,x.units.n_rows);
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

        vector<Instance> population(numPopulation);
        vector<Instance> newPopulation(numPopulation);
        Instance bestInstance = initial;
        double bestCost  = initial.cost(*metric,knearest);
        int bestIndex = 0;
        int numRandom = round(numPopulation*0.1); //number of chromosomes that will be random
        population[0] = initial; //mantain the initial instance

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
            Col<int> c = initialInstance(0.3,initial.units.n_rows);
            population[numPopulation-i-1] = initial;
            population[numPopulation-i-1].units = c;
            population[numPopulation-i-1].changeTrainingSet();
        }

        for (int i = 1; i < numPopulation; i++){
            double costAux = population[i].cost(*metric,knearest);
            if (costAux > bestCost){
                bestInstance = population[i];
                bestCost = costAux;
                bestIndex = i;
            }
        }
        
        for (int i = 0; i < iterations;i++){

            int j = 0;
            while (j < numPopulation){
                if (disReal(eng) < 0.9){
                    int f1 = disInt(eng);
                    int f2 = disInt(eng);
                    pair<Instance,Instance> auxPair = cross(population[f1],population[f2]);
                    newPopulation[j] = auxPair.first;
                    if ((numPopulation % 2 == 0) || (j != numPopulation-1)) 
                        newPopulation[j+1] = auxPair.second;
                    j += 2;
                }
            }

            for (int z=0; z < numPopulation; z++){
                if (disReal(eng) < 0.1) newPopulation[z] = mutate(newPopulation[z]);
            }

            for (int z=0; z < numPopulation; z++){
                if (z != bestIndex) population[z] = newPopulation[z];
                else{
                    double costoAux = newPopulation[z].cost(*metric,knearest);
                    if (costoAux > bestCost){
                        bestInstance = newPopulation[z];
                        bestCost = costoAux;
                        bestIndex = z;
                        population[z] = newPopulation[z];
                    }
                }
            }

            for (int z = 0; z < numPopulation; z++){
                double costAux = population[z].cost(*metric,knearest);
                if (costAux > bestCost){
                    bestInstance = population[z];
                    bestCost = costAux;
                    bestIndex = z;
                }
            }

            //cout << "iteration " << i << endl;
        }

        Knn knn(bestInstance.training,bestInstance.trainResults,bestInstance.unique);
        double costResult = knn.score(*(bestInstance.originalTraining),knearest,*metric,*(bestInstance.originaltrainResults));

        return make_pair(costResult,bestInstance);

    }
};

#endif