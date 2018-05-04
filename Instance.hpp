#ifndef Instance_H
#define Instance_H
#include <cmath>
#include <armadillo>
#include <utility>
#include <random>
#include <chrono>
#include <algorithm>
#include <vector>
#include "Knn.hpp"
#include "Metrics.hpp"

using namespace std;
using namespace arma;

/* Function that given a probability and the data size gives an initial chromosome*/

Col<int> initialInstance(double prob,int size){
    random_device rd; // obtain a random number from hardware
    mt19937 eng(rd()); // seed
    uniform_real_distribution<> disReal(0,1);
    Col<int> units(size,fill::zeros);
    for (int i = 0; i < size; i++){ if (disReal(eng) < prob) units(i)=1; }
    return units;
}

struct Instance{

    /*Chromosome for the Instance Selection problem. Member attributes and functions:

    units = Bit string, 1 if the instance is in the subset 0 otherwise
    percenVecinity = percentage of the vicinity that will be explored
    percenCost = weight given to the accuracy and instance reduction 
    originalTraining =  pointer to the original training dataset so one can create new Instances
    originaltrainResults = pointer to the classes of the original training set
    test = Test set that will be used when evaluating the instance
    testResults = classes of the test set
    training = Reduced dataset represented by units
    trainResults = Classes of the reduced dataset
    unique = number of classes
    totalInstances = number of instances in the original dataset

    changeTrainingSet() = computes again the reduced set for this cromosome

    cost(MetricType &met,int knearest) = the cost of the instance in base to the accuracy and reduction

    searchNeighborhood(MetricType &met, int knearest, chrono::steady_clock::time_point &tend) =
        search for the best contiguous Instance using the cost function in the perecentage defined

    */

    Col <int> units;
    double percenVecinity, percenCost;
    mat* originalTraining;
    mat* test;
    Col<int>* originaltrainResults;
    Col<int>* testResults;
    mat training;
    Col<int> trainResults;
    int unique,totalInstances;
    Col<int> indexesT;

    Instance(){}

    Instance(Col<int> &u,double p1,double p2,mat* tr,mat* te,Col<int>* trr,Col<int>* ter,int un):
        units(u),percenVecinity(p1),percenCost(p2),test(te),testResults(ter),unique(un),
        originalTraining(tr),originaltrainResults(trr){

            totalInstances = tr->n_rows;
            int count = 0,index=0;
            for (int i=0;i<u.n_rows;i++){ if (u(i)==1) count++; }
            training.set_size(count,tr->n_cols);
            trainResults.set_size(count);
            indexesT.set_size(count);

            for (int i=0;i<u.n_rows;i++){
                if (u(i)==1){
                    training.row(index) = tr->row(i);
                    trainResults(index) = (*trr)(i);
                    indexesT(index) = i;
                    index++;
                }
            }
        }

    void changeTrainingSet(){
            totalInstances = originalTraining->n_rows;
            int count = 0,index=0;
            for (int i=0;i<units.n_rows;i++){ if (units(i)==1) count++; }
            training.set_size(count,originalTraining->n_cols);
            trainResults.set_size(count);
            indexesT.set_size(count);

            for (int i=0;i<units.n_rows;i++){
                if (units(i)==1){
                    training.row(index) = originalTraining->row(i);
                    trainResults(index) = (*originaltrainResults)(i);
                    indexesT(index) = i;
                    index++;
                }
            }
    }

    template<typename MetricType>
    double cost(MetricType &met,int knearest){

        Knn knn(training,trainResults,unique);
        double accuracy = 100 * knn.score(*test,knearest,met,*testResults);
        double reduction = 100 * (totalInstances - training.n_rows) / totalInstances;

        return percenCost * accuracy + (1-percenCost)*reduction; 
    }

    double cost2(int knearest, vector<vector<size_t>> &ind, Knn &knn){

        Mat<int> comparison(test->n_rows,unique,fill::zeros);

        bool flag;
        int l;
        for (int i = 0; i < ind.size(); i++){
            l = 0;
            for (int j = 0; j < knearest; j++){
                flag = false;
                while (l < ind[i].size() && !flag){
                    if (binary_search(indexesT.begin(),indexesT.end(),ind[i][l])){
                        int indexAux = (*(originaltrainResults))(ind[i][l]);
                        comparison(i,indexAux)++;
                        flag = true;
                    }
                    l++;
                }
            }
        }

        Col<int> resultClass(test->n_rows);
        for (int i = 0; i < test->n_rows; i++) resultClass(i) = comparison.row(i).index_max();

        Mat<int> conf(knn.confMatrix(resultClass,*(testResults)));
        double sc = 0;
        for (int i=0; i < unique; i++) sc+=conf(i,i);
        sc = sc/test->n_rows;
        
        double reduction = 100 * (totalInstances - training.n_rows) / totalInstances;

        return percenCost * sc + (1-percenCost)*reduction;

    }
    
    template<typename MetricType>
    pair <double,Instance> searchNeighborhood(MetricType &met, int knearest, chrono::steady_clock::time_point &tend){

        bool flag = false;
        Instance bestNeighbor = *this;
        double bestCost = bestNeighbor.cost(met,knearest);
        double neighborCost;
        int aux = floor(percenVecinity * units.n_rows);

        vector <int> indexes(units.n_rows);
        for (int i=0;i<units.n_rows;i++) indexes[i] = i;
        random_shuffle(indexes.begin(),indexes.end());

        for (int i = 0; (i < aux) && (chrono::steady_clock::now() < tend); i++){
            
            Col<int> nunits(units);
            if (nunits(indexes[i])==0) nunits(indexes[i]) = 1;
            else if (nunits(indexes[i])==1) nunits(indexes[i]) = 0;

            Instance neighbor(nunits,percenVecinity,percenCost,originalTraining,test,originaltrainResults,testResults,unique);
            neighborCost = neighbor.cost(met,knearest);

            if (neighborCost > bestCost){
                bestNeighbor = neighbor;
                bestCost = neighborCost;
                flag = true;

            }
        }

        if (!flag) bestCost = -1;

        return make_pair(bestCost,bestNeighbor);
    }
};

#endif