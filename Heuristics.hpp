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
#include "utils.h"

using namespace std;
using namespace arma;

template <typename MetricType>
struct LocalSearch{

    /*Class for the local search algorithm:

    metric = metric used in the search

    find(initial,knearest) = given an initial Instance perform a local serach using knearest neighbor
    */

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


template<typename MetricType>
struct CNN{

    /*Perform CNN, constructs a reduced data set beginning from one instance and adding each instance not 
     classified correctly */ 
 
    MetricType* metric;
    CNN(MetricType* met):metric(met){}

    pair<double,Instance> find(Instance &initial, int knearest){

        bool flag = false, flag2 = false;
        Instance current = initial;
        int j = 0;

        Knn knn2(*(initial.originalTraining),*(initial.originaltrainResults),initial.unique);
        vector<vector<size_t>> ordIndex = knn2.search2(*(initial.originalTraining),knearest,*metric);

        vector <int> indexes(initial.units.n_rows);
        for (int i=0;i<initial.units.n_rows;i++) indexes[i] = i;
        random_shuffle(indexes.begin(),indexes.end());
 
        for (int i = 0; (i < initial.units.n_rows) && !flag; i++){

            flag2 = false;
            while ((j < current.originalTraining->n_rows) && !flag2){

                rowvec query = current.originalTraining->row(indexes[j]);
                int prediction = knn2.predict(query,knearest,ordIndex,indexes[j],(*(current.originaltrainResults)),current.indexesT);

                if (prediction != (*(current.originaltrainResults))(indexes[j])){
                    Col<int> nunits(current.units);
                    nunits(indexes[j]) = 1;
                    current.units = nunits;
                    current.changeTrainingSet();
                    flag2 = true;
                }
                j++;

            }
            if (!flag2) flag = true; //flag becomes true when every point is classified correctly
        }

        Knn knn(current.training,current.trainResults,current.unique);
        double costResult = knn.score(*(current.originalTraining),knearest,*metric,*(current.originaltrainResults));

        return make_pair(costResult,current);
    }
};


template<typename MetricType>
struct RSS{

    /*Orders instances by nearest enemy and adds to the subset the instance that doesn't have a instance nearer than its enemy*/

    MetricType* metric;
    RSS(MetricType* met):metric(met){}

    pair<double,Instance> find(Instance &initial, int knearest){

        mat data = initial.training;
        mat query = initial.training;
        mat distances(data.n_rows,query.n_rows);
        rowvec aux1, aux2;

        for (int i = 0; i < query.n_rows; i++){
            aux2 = query.row(i);
            for (int j = 0; j < data.n_rows; j++){ 
                aux1 = data.row(j);
                distances(j,i) = metric->Evaluate(aux1,aux2);
            }
        }

        mat ordered = sort(distances);
        vector<double> disNE(data.n_rows);
        uvec index;
        bool flag;

        for (int i = 0; i < query.n_rows; i++){
            flag = false;
            for (int j = 0; j < data.n_rows && !flag; j++){
                index = arma::find(abs(distances.col(i)-ordered(j,i))<0.0000000001);
                if (initial.trainResults(index(0)) != initial.trainResults(i)){
                    disNE[i] = ordered(j,i);
                    flag = true;
                }
            }
        }

        vector<size_t> orderedIndex = Utils::sort_indexes(disNE);

        Col<int> nunits(data.n_rows,fill::zeros);
        nunits(0) = 1;
        Instance current = initial;
        current.units = nunits;
        current.changeTrainingSet();

        for (int i = 0;i < data.n_rows;i++){
            
            flag = false;
            for (int j = 0; j<current.training.n_rows && !flag; j++){

                aux2 = current.training.row(j);
                aux1 = data.row(orderedIndex[i]);
                int k = current.indexesT(j);

                if (metric->Evaluate(aux1,aux2) < disNE[k]) flag = true;
            }
            
            if (!flag){
                Col<int> newunits(current.units);
                newunits(i) = 1;
                current.units = newunits;
                current.changeTrainingSet();
            }
        }

        Knn knn(current.training,current.trainResults,current.unique);
        double costResult = knn.score(*(current.originalTraining),knearest,*metric,*(current.originaltrainResults));

        return make_pair(costResult,current);
    }
};

template<typename MetricType>
struct ENN{

    /*Removes instances that aren't the same class as their nearest neighbor*/

    MetricType* metric;
    ENN(MetricType* met):metric(met){}

    pair<double,Instance> find(Instance &initial, int knearest){

        Instance oldOne;
        Instance current = initial;
        Knn knn2(*(initial.originalTraining),*(initial.originaltrainResults),initial.unique);
        vector<vector<size_t>> ordIndex = knn2.search2(*(initial.originalTraining),knearest,*metric);

        for (int i = 0; i < initial.units.n_rows; i++){

            oldOne = current;
            Col<int> nunits(current.units);
            nunits(i) = 0;
            current.units = nunits;
            current.changeTrainingSet();

            rowvec query = current.originalTraining->row(i);
            int prediction = knn2.predict(query,knearest,ordIndex,i,(*(current.originaltrainResults)),current.indexesT);
            Col<int> newTestRes = (current.originaltrainResults)->row(i);
            if (prediction == newTestRes(0)) current = oldOne;

        }

        Knn knn(current.training,current.trainResults,current.unique);
        double costResult = knn.score(*(current.originalTraining),knearest,*metric,*(current.originaltrainResults));

        return make_pair(costResult,current);

    }

};

template<typename MetricType>
struct IB3{

    MetricType* metric;
    double acept;
    double drop;
    IB3(MetricType* met,double a, double d):metric(met),acept(a),drop(d){}

    bool aceptable(Row<int> acp, int frecuency,int numProcessed,bool mode){

        //Mode True if one is verifying that the instance is to be accepted with 0.9 confidence and
        //Mode false if checking yif the instance will be deprecated with 0.7 confidence
        
        //Accuracy intervals
        double z;
        if (mode) z = acept;
        else z = drop;
        int n = acp(0)+acp(1);
        if (n==0) n++;
        double p = acp(0)/n;

        double lowerAcc = (p + pow(z,2)/(2*n) - z*sqrt(p*(1-p)/n + pow(z,2)/(4*pow(n,2)))) / (1 + pow(z,2)/n);
        double upperAcc = (p + pow(z,2)/(2*n) + z*sqrt(p*(1-p)/n + pow(z,2)/(4*pow(n,2)))) / (1 + pow(z,2)/n);

        //frecuency intervals
        n = numProcessed;
        if (n == 0) n++;
        p = frecuency/n;

        double lowerFrec = (p + pow(z,2)/(2*n) - z*sqrt(p*(1-p)/n + pow(z,2)/(4*pow(n,2)))) / (1 + pow(z,2)/n);
        double upperFrec = (p + pow(z,2)/(2*n) + z*sqrt(p*(1-p)/n + pow(z,2)/(4*pow(n,2)))) / (1 + pow(z,2)/n);

        bool resp;
        if (mode){
            resp = false;
            if (lowerAcc >= upperFrec) resp = true;
        }
        else if (!mode){
            resp = true;
            if (upperAcc < lowerFrec) resp = false;
        }

        return resp;

    }
    
    pair<double,Instance> find(Instance &initial, int knearest){

        vec sim(initial.originalTraining->n_rows);
    
        Mat<int> classRecord(initial.originalTraining->n_rows,2,fill::zeros); //first col is accerts, second mistakes
        Col<int> frecuencyRecord(initial.unique); //frecuency of the each class
        int numProcessed = 0;

        Instance current = initial;
        rowvec aux1, aux2;
        random_device rd; // obtain a random number from hardware
        mt19937 eng(rd()); // seed
        int ymax,orderedIndex;

        //For each x in Training Set
        for (int i = 0; i < initial.units.n_rows; i++){

            Col<int> auxUnits(current.units);

            mat data = current.training; //obtaining current dataset from the chromosome
            mat query = current.originalTraining->row(i); //getting current instance
            frecuencyRecord((*(current.originaltrainResults))(i))++; //update the frecuency of the class
            numProcessed++; //adding total number of instance processed
            mat distances(data.n_rows,query.n_rows);
            aux2 = query.row(0);

            //Fill the distance matrix for the current instance 
            for (int j = 0; j < data.n_rows; j++){
                aux1 = data.row(j);
                distances(j,0) = metric->Evaluate(aux1,aux2);
            }

            mat ordered = sort(distances);
            uvec index;
            bool flag = false;
            ymax = -1;

            for (int j = 0; (j < distances.n_rows) && !flag; j++){

                //find real index in original dataset 
                index = arma::find(abs(distances.col(0)-ordered(j,0))<0.0000000001);
                int k = 0,count = 0;
                bool flag2 = false;
                while (count < index(0)+1){
                    if (current.units(k)==1) count++;
                    k++;
                }

                //If the instance is acceptable with 0.9 confidence level
                Row<int> acp = classRecord.row(k-1);
                int aux5 = (*(current.originaltrainResults))(k-1);
                if (aceptable(acp,frecuencyRecord(aux5),numProcessed,true)){
                    ymax = k-1;
                    flag = true;
                    orderedIndex = j;
                }
            }
            //If it isn't acceptable find a random one
            if (!flag){
                uniform_int_distribution<> disInt(0,data.n_rows-1);
                orderedIndex = disInt(eng);
                index = arma::find(abs(distances.col(0)-ordered(orderedIndex,0))<0.0000000001);
                int k = 0,count = 0;
                bool flag2 = false;
                while (count < index(0)+1){
                    if (current.units(k)==1) count++;
                    k++;
                }
                ymax = k-1;
            }

            //If the accepted neighboor is of the same class as the query instance then update the
            //classification record as positive. Else count it as negative and add the missclassified 
            //instace to the current dataset.
            if ((*(current.originaltrainResults))(i) == (*(current.originaltrainResults))(ymax))
                classRecord(ymax,0)++;
            else{
                classRecord(ymax,1)++;
                Col<int> nunits(auxUnits);
                nunits[i] = 1;
                auxUnits = nunits;
            }


            //For each neighboor that is at a smaller or equal distance that the accepted one
            for (int o = 0; o <orderedIndex; o++){

                //Find it's real index
                index = arma::find(abs(distances.col(0)-ordered(o,0))<0.0000000001);
                int k = 0,count = 0;
                bool flag2 = false;
                while (count < index(0)+1){
                    if (current.units(k)==1) count++;
                    k++;
                }
                k--;
                
                int aux5 = (*(current.originaltrainResults))(k);
                Row<int> acp = classRecord.row(k);

                //Update the classification frecuency 
                if ((*(current.originaltrainResults))(i) == aux5)
                    classRecord(k,0)++;
                else classRecord(k,1)++;

                //If said instance isn't acceptable with 0.7 confidence level, deprecate it.
                if (!aceptable(acp,frecuencyRecord(aux5),numProcessed,false)){
                    Col<int> nunits(auxUnits);
                    nunits[k] = 0;
                    auxUnits = nunits;
                }
            }

            current.units = auxUnits;
            current.changeTrainingSet();
        }

        //Return the current chromosome and it's score
        Knn knn(current.training,current.trainResults,current.unique);
        double costResult = knn.score(*(current.originalTraining),knearest,*metric,*(current.originaltrainResults));

        return make_pair(costResult,current);
    }
};

#endif