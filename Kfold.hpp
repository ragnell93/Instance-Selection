#ifndef KFOLD_H
#define KFOLD_H
#include <vector>
#include <algorithm>
#include <cmath>
#include <thread>
#include <armadillo>
#include "Metrics.hpp"
#include "Knn.hpp"
#include "Heuristics.hpp"
#include "Metaheuristics.hpp"
#include "Instance.hpp"
#include "utils.h"

using namespace std;
using namespace arma;

template <typename Heuristic, typename Meta>
void searchInstanceStrat(Heuristic heu,Meta metaheu, Instance auxI, int knearest, vec &tiempo, vector<Instance> &reduction,int index){

    auto start = chrono::high_resolution_clock::now();
    pair<double,Instance> pairAux2 = heu.find(auxI,knearest);
    pair<double,Instance> pairAux = metaheu.find(pairAux2.second,knearest);

    auto stop = chrono::high_resolution_clock::now();
    using fpSeconds = chrono::duration<float,chrono::seconds::period>;
    tiempo(index) = (double)(fpSeconds(stop - start).count());
    reduction[index] = pairAux.second;
}

template <typename Heuristic>
void calcMetrics(Heuristic heu,mat trainSet,Col<int> trainRes,mat auxMat,Col<int>auxCol, int knearest, double total,int un, vec &scores,vec &kappas, vec &reductionScore,int index){

    Knn knn(trainSet,trainRes,un);
    scores(index) = knn.score(auxMat,knearest,*(heu.metric),auxCol);
    kappas(index) = knn.kappa(auxMat,knearest,*(heu.metric),auxCol);
    reductionScore(index) = ((double)total - (double)trainSet.n_rows) / (double)total;
}

template <typename Heuristic, typename Meta>
void calcNoStrat(Heuristic heu,Meta metaheu, Instance initial, mat testSet, Col<int>testRes, int knearest, double total, int un,vec &tiempo, vec &scores,vec &kappas,vec &reductionScore, int index){

    auto start = chrono::high_resolution_clock::now();

    pair <double,Instance> pairAux2 = heu.find(initial,knearest);
    pair<double,Instance> obtained = metaheu.find(pairAux2.second,knearest);

    auto stop = chrono::high_resolution_clock::now();
    using fpSeconds = chrono::duration<float,chrono::seconds::period>;
    tiempo = (double)(fpSeconds(stop - start).count());

    Knn knn(obtained.second.training,obtained.second.trainResults,un);
    scores(index) = knn.score(testSet,knearest,*(heu.metric),testRes);
    kappas(index) = knn.kappa(testSet,knearest,*(heu.metric),testRes);
    reductionScore(index) = ((double)total - (double)obtained.second.training.n_rows) / (double)total;

}

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
    vector<thread> threads1(k);
    vector<thread> threads2(k);
    vector<thread> threads3(k);

    if (st){

        vector<Instance> reduction(k);
        Col<int> auxU;
        vector<mat> auxMat(k);
        vector<Col<int>> auxCol(k);

    
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

            auxMat[j] = folds.slice(j);
            auxCol[j] = res.col(j);

            Instance auxI(auxU,pNeigh,pCost,&(auxMat[j]),&(auxMat[j]),&(auxCol[j]),&(auxCol[j]),un.n_rows);

            threads1[j] = thread(searchInstanceStrat<Heuristic,Meta>,heu,metaheu,auxI,knearest,ref(tiempo),ref(reduction),j);

            //cout << "el indice j es: " << j << endl;
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

        threads1[k-1] = thread(searchInstanceStrat<Heuristic,Meta>,heu,metaheu,auxI,knearest,ref(tiempo),ref(reduction),k-1);

        for (int b = 0; b<k; b++) threads1[b].join();

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

            threads2[l] = thread(calcMetrics<Heuristic>,heu,trainSet,trainRes,auxMat,auxCol,knearest,data.n_rows,un.n_rows, ref(scores),ref(kappas),ref(reductionScore),l);

            //cout << "index l is : " << l << endl;
        }

        for (int t = 0; t < k; t++) threads2[t].join();
    }

    else if (!st){

        int notin=0;
        mat trainSet(number*(k-2)+lastFold.n_rows,data.n_cols);
        Col<int> trainRes(number*(k-2)+lastFold.n_rows);
        vector<mat> trainSet2(k-1);
        vector<Col<int>> trainRes2(k-1);
        vector<mat> testSet2(k-1);
        vector<Col<int>> testRes2(k-1);
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
            testSet2[j] = folds.slice(notin);
            testRes2[j] = res.col(notin);

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

            trainSet2[j] = trainSet;
            trainRes2[j] = trainRes;

            Instance initial(auxU,pNeigh,pCost,&(trainSet2[j]),&(trainSet2[j]),&(trainRes2[j]),&(trainRes2[j]),un.n_rows);

            threads3[j] = thread(calcNoStrat<Heuristic,Meta>,heu,metaheu,initial,testSet2[j],testRes2[j],knearest,data.n_rows,un.n_rows,ref(tiempo),ref(scores),ref(kappas),ref(reductionScore),j);

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

        threads3[k-1] = thread(calcNoStrat<Heuristic,Meta>,heu,metaheu,initial,lastFold,lastRes,knearest,data.n_rows,un.n_rows,ref(tiempo),ref(scores),ref(kappas),ref(reductionScore),k-1);

        for (int b = 0; b < k ; b++) threads3[b].join();
    }

    vector<double> mean_results(4);
    mean_results[0] = sum(scores)/(double)scores.n_rows;
    mean_results[1] = sum(kappas)/(double)kappas.n_rows;
    mean_results[2] = sum(reductionScore)/(double)reductionScore.n_rows;
    mean_results[3] = sum(tiempo)/(double)tiempo.n_rows;

    return mean_results;  
}

#endif