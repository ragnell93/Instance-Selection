#ifndef KFOLD_H
#define KFOLD_H
#include <vector>
#include <algorithm>
#include <cmath>
#include <armadillo>
#include "Metrics.hpp"
#include "Knn.hpp"
#include "Heuristics.hpp"
#include "Instance.hpp"

using namespace std;
using namespace arma;

/*kfold(met,data,results,k,kn): function that perform cross validation with k folds.
  First construct the k folds from the data and results. Then for each partition k
  a training set is constructed with the k-1 remaining folds for the knn estimator,
  using the kth fold as the query to obtain a score for the estimator.After getting all 
  k scores using the kth partition as query set it calculates the mean accuracy
 */

template <typename Heuristic>
double kfold(Heuristic &heu,mat &data,Col<int> &results,int k,int knearest,double pNeigh,double pCost,double pIni){

    Col<int> un = unique(results); //number of classes
    vector <int> indexes(data.n_rows); //indexes of the instances
    for (int i=0;i<data.n_rows;i++) indexes[i] = i;
    random_shuffle(indexes.begin(),indexes.end());
    
    int number = floor(data.n_rows/k); //number of elements for each fold
    cube folds(number,data.n_cols,k-1);
    Mat<int> res(number,k-1);
    mat lastFold(data.n_rows-number*(k-1),data.n_cols);
    Col<int> lastRes(data.n_rows-number*(k-1));

    //Filling each fold with the given permutation of the indexes
    int numRow = 0;
    for (int q=0;q<k-1;q++){
        for (int w=0;w<number;w++){
            folds.slice(q).row(w) = data.row(indexes[numRow]);
            res(w,q) = results(indexes[numRow]);
            numRow++;
        }
    }

    //filling the last fold with the remaining data
    int aux = 0, aux2;
    while (numRow < data.n_rows){
        lastFold.row(aux) = data.row(indexes[numRow]);
        lastRes(aux) = results(indexes[numRow]);
        aux++;
        numRow++;
    }

    vec scores(k);
    mat trainSet(number*(k-2)+lastFold.n_rows,data.n_cols);
    Col<int> trainRes(number*(k-2)+lastFold.n_rows);
    int notin=0;

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

        Col<int> iConfig = initialInstance(pIni,trainSet.n_rows);
        Instance initial(iConfig,pNeigh,pCost,&trainSet,&testSet,&trainRes,&testRes,un.n_rows);
        pair <double,Instance> obtained = heu.find(initial,knearest);
        
        scores(j) = obtained.first;
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

    Col<int> iConfig = initialInstance(pIni,trainSet.n_rows);
    Instance initial(iConfig,pNeigh,pCost,&trainSet,&lastFold,&trainRes,&lastRes,un.n_rows);
    pair <double,Instance> obtained = heu.find(initial,knearest);
    scores(k-1) = obtained.first;
    
    cout << "scores " << endl;
    scores.print();

    return sum(scores)/scores.n_rows;
    
}

#endif