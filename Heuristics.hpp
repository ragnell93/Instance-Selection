#ifndef Heuristics_H
#define Heuristics_H
#include <cmath>
#include <armadillo>
#include <utility>
#include <random>
#include <chrono>
#include "Knn.hpp"
#include "Metrics.hpp"
#include "Instance.hpp"

using namespace std;
using namespace arma;

template <typename MetricType>
struct LocalSearch{

    MetricType* metric;
    LocalSearch(MetricType* met):metric(met){}

    pair<double,Instance> find(Instance &initial,int knearest){

        bool flag=true;
        pair<double,Instance> oldOne,neighbor;
        oldOne.second=initial;
        oldOne.first = initial.cost(*metric,knearest);
        chrono::steady_clock::time_point tend = chrono::steady_clock::now() + chrono::minutes(1);
        
        while (flag && (chrono::steady_clock::now() < tend)){
            neighbor = oldOne.second.searchNeighborhood(*metric,knearest,tend);
            if (neighbor.first < 0) flag = false;
            else oldOne = neighbor;
        }

        return oldOne;
    }

};

#endif