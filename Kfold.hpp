#ifndef KFOLD_H
#define KFOLD_H
#include <vector>
#include <algorithm>
#include <cmath>
#include <armadillo>
#include "Metrics.hpp"
#include "Knn.hpp"
#include "Heuristics.hpp"
#include "Metaheuristics.hpp"
#include "Instance.hpp"
#include "utils.h"

using namespace std;
using namespace arma;

/*kfold(met,data,results,k,kn): function that perform cross validation with k folds.
  First construct the k folds from the data and results. Then for each partition k
  a training set is constructed with the k-1 remaining folds for the knn estimator,
  using the kth fold as the query to obtain a score for the estimator.After getting all 
  k scores using the kth partition as query set it calculates the mean accuracy
 */

template <typename Heuristic, typename Meta>
vector<double> kfold(Heuristic &heu,Meta &metaheu,mat &data,Col<int> &results,int k,int knearest,double pNeigh,double pCost,double pIni,bool st,bool units){

    Col<int> un = unique(results); //number of classes
    vector <int> indexes(data.n_rows); //indexes of the instances
    for (int i=0;i<data.n_rows;i++) indexes[i] = i;
    random_shuffle(indexes.begin(),indexes.end());

    random_device rd; // obtain a random number from hardware
    mt19937 eng(rd()); // seed
    uniform_int_distribution<> disInt(0,data.n_rows-1);
    
    int number = floor(data.n_rows/k); //number of elements for each fold
    cube folds(number,data.n_cols,k-1);
    Mat<int> res(number,k-1);
    mat lastFold(data.n_rows-number*(k-1),data.n_cols);
    Col<int> lastRes(data.n_rows-number*(k-1));

    vector<vector<rowvec>> strata(un.n_rows,vector<rowvec>(0,rowvec(data.n_cols)));

    for (int i = 0; i < data.n_rows; i++) strata[results(indexes[i])].push_back(data.row(indexes[i]));

    //Filling each fold with the given permutation of the indexes
    int numberSample = 0;
    double strataProportion;
    vector<int> stratIndex(un.n_rows,0);

    for (int q=0;q<k-1;q++){
        numberSample = 0;
        for (int r=0; r < strata.size();r++){
            strataProportion = floor(number*strata[r].size()/data.n_rows);
            for (int t = 0; t < strataProportion; t++){
                folds.slice(q).row(numberSample) = strata[r][stratIndex[r]];
                res(numberSample,q) = r;
                stratIndex[r]++;
                numberSample++;
            }
        }

        //Fill the remaining spots with randoms instances
        while (numberSample < number){
            int auxI = disInt(eng);
            folds.slice(q).row(numberSample) = data.row(auxI);
            res(numberSample,q) = results(auxI);
            numberSample++;
        }
    }

    //filling the last fold with the remaining data
    int aux = 0, aux2;
    numberSample = 0;
    for (int r=0; r < strata.size();r++){
        strataProportion = floor(lastFold.n_rows*strata[r].size()/data.n_rows);
        for (int t = 0; (t < strataProportion) && (stratIndex[r] < strata[r].size()); t++){
            lastFold.row(numberSample) = strata[r][stratIndex[r]];
            lastRes(numberSample) = r;
            stratIndex[r]++;
            numberSample++;
        }
    }

    //Fill the remaining spots with randoms instances
    while (numberSample < lastFold.n_rows){
        int auxI = disInt(eng);
        lastFold.row(numberSample) = data.row(auxI);
        lastRes(numberSample) = results(auxI);
        numberSample++;
    }

    vec scores(k);
    vec kappas(k);
    vec reductionScore(k);
    vec tiempo(k);

    if (st){

        vector<Instance> reduction(k);
        Col<int> auxU;
    
        for (int j=0; j<k-1;j++){
            
            if (units){
                Col<int> auxU2(folds.slice(j).n_rows,fill::zeros);
                auxU2(0) = 1;
                auxU = auxU2;
            }
            else if (!units){
                Col<int> auxU2(folds.slice(j).n_rows,fill::ones);
                auxU = auxU2;
            }

            mat auxMat = folds.slice(j);
            Col<int> auxCol = res.col(j);

            Instance auxI(auxU,pNeigh,pCost,&auxMat,&auxMat,&auxCol,&auxCol,un.n_rows);
            auto start = chrono::high_resolution_clock::now();

            pair<double,Instance> pairAux2 = heu.find(auxI,knearest);
            pair<double,Instance> pairAux = metaheu.find(pairAux2.second,knearest);

            auto stop = chrono::high_resolution_clock::now();
            using fpSeconds = chrono::duration<float,chrono::seconds::period>;
            tiempo(j) = (double)(fpSeconds(stop - start).count());

            reduction[j] = pairAux.second;

            cout << "el indice j es: " << j << endl;
        }

        if (units){
            Col<int> auxU2(lastFold.n_rows,fill::zeros);
            auxU2(0) = 1;
            auxU = auxU2;
        }
        else if (!units){
            Col<int> auxU2(lastFold.n_rows,fill::ones);
            auxU = auxU2;
        }

        Instance auxI(auxU,pNeigh,pCost,&lastFold,&lastFold,&lastRes,&lastRes,un.n_rows);
        auto start = chrono::high_resolution_clock::now();

        pair<double,Instance> pairAux2 = heu.find(auxI,knearest);
        pair<double,Instance> pairAux = metaheu.find(pairAux2.second,knearest);
        
        auto stop = chrono::high_resolution_clock::now();
        using fpSeconds = chrono::duration<float,chrono::seconds::period>;
        tiempo(k-1) = (double)(fpSeconds(stop - start).count());

        reduction[k-1] = pairAux.second;

        for (int l=0; l<k; l++){

            mat trainSet;
            Col<int> trainRes;
            for (int z=0; z<k;z++){
                if (z != l){
                    trainSet.insert_rows(trainSet.n_rows,reduction[z].training);
                    trainRes.insert_rows(trainRes.n_rows,reduction[z].trainResults);
               }
            }

            mat auxMat;
            Col<int> auxCol;
            if (l != k-1){  
                auxMat = folds.slice(l);
                auxCol = res.col(l);
            }
            else if (l == k-1){
                auxMat = lastFold;
                auxCol = lastRes;
            }
            
            Knn knn(trainSet,trainRes,un.n_rows);
            scores(l) = knn.score(auxMat,knearest,*(heu.metric),auxCol);
            kappas(l) = knn.kappa(auxMat,knearest,*(heu.metric),auxCol);
            reductionScore(l) = ((double)data.n_rows - (double)trainSet.n_rows) / (double)data.n_rows;

            cout << "index l is : " << l << endl;

        }
    }

    else if (!st){

        int notin=0;
        mat trainSet(number*(k-2)+lastFold.n_rows,data.n_cols);
        Col<int> trainRes(number*(k-2)+lastFold.n_rows);

        //for each fold constructs a training set without the kth fold 
        for (int j=0;j<k-1;j++){

            aux = 0;
            aux2=number;
            for (int i=0;i<k-1;i++){
                if (i != notin){
                    trainSet.rows(aux,aux2-1) = folds.slice(i);
                    trainRes.subvec(aux,aux2-1) = res.col(i);
                    aux = aux2;
                    aux2 += number;
                }
            }

            trainSet.rows(aux,trainSet.n_rows-1) = lastFold;
            trainRes.subvec(aux,trainSet.n_rows-1) = lastRes;
            mat testSet(folds.slice(notin));
            Col<int> testRes(res.col(notin));

            Col<int> auxU;
            if (units){
                Col<int> auxU2(trainSet.n_rows,fill::zeros);
                auxU2(0) = 1;
                auxU = auxU2;
            }
            else if (!units){
                Col<int> auxU2(trainSet.n_rows,fill::ones);
                auxU = auxU2;
            }

            Instance initial(auxU,pNeigh,pCost,&trainSet,&testSet,&trainRes,&testRes,un.n_rows);
            auto start = chrono::high_resolution_clock::now();

            pair <double,Instance> pairAux2 = heu.find(initial,knearest);
            pair<double,Instance> obtained = metaheu.find(pairAux2.second,knearest);

            auto stop = chrono::high_resolution_clock::now();
            using fpSeconds = chrono::duration<float,chrono::seconds::period>;
            tiempo(j) = (double)(fpSeconds(stop - start).count());

            Knn knn(obtained.second.training,obtained.second.trainResults,un.n_rows);
            scores(j) = knn.score(testSet,knearest,*(heu.metric),testRes);
            kappas(j) = knn.kappa(testSet,knearest,*(heu.metric),testRes);
            reductionScore(j) = ((double)data.n_rows - (double)obtained.second.training.n_rows) / (double)data.n_rows;
            
            notin++;
        } 

        aux = 0;
        aux2=number;
        trainSet.set_size(number*(k-1),data.n_cols);
        trainRes.set_size(number*(k-1));
        for (int i=0;i<k-1;i++){
                trainSet.rows(aux,aux2-1) = folds.slice(i);
                trainRes.subvec(aux,aux2-1) = res.col(i);
                aux = aux2;
                aux2 += number;
        }

        Col<int> auxU;
        if (units){
            Col<int> auxU2(trainSet.n_rows,fill::zeros);
            auxU2(0) = 1;
            auxU = auxU2;
        }
        else if (!units){
            Col<int> auxU2(trainSet.n_rows,fill::ones);
            auxU = auxU2;
        }


        Instance initial(auxU,pNeigh,pCost,&trainSet,&lastFold,&trainRes,&lastRes,un.n_rows);
        auto start = chrono::high_resolution_clock::now();

        pair <double,Instance> pairAux2 = heu.find(initial,knearest);
        pair<double,Instance> obtained = metaheu.find(pairAux2.second,knearest);

        auto stop = chrono::high_resolution_clock::now();
        using fpSeconds = chrono::duration<float,chrono::seconds::period>;
        tiempo(k-1) = (double)(fpSeconds(stop - start).count());

        Knn knn(obtained.second.training,obtained.second.trainResults,un.n_rows);
        scores(k-1) = knn.score(lastFold,knearest,*(heu.metric),lastRes);
        kappas(k-1) = knn.kappa(lastFold,knearest,*(heu.metric),lastRes);
        reductionScore(k-1) = ((double)data.n_rows - (double)obtained.second.training.n_rows) / (double)data.n_rows;
    }

    vector<double> mean_results(4);
    mean_results[0] = sum(scores)/(double)scores.n_rows;
    mean_results[1] = sum(kappas)/(double)kappas.n_rows;
    mean_results[2] = sum(reductionScore)/(double)reductionScore.n_rows;
    mean_results[3] = sum(tiempo)/(double)tiempo.n_rows;

    return mean_results;  
}

#endif