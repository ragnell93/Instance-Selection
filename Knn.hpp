#ifndef KNN_H
#define KNN_H
#include <cmath>
#include <armadillo>
#include "Metrics.hpp"
#include "utils.h"

using namespace std;
using namespace arma;

struct Knn{

    /* K-nearest neighbor estimator. Calculates the k nearest points and classify the query according
        to the majority class. Member functions and attributes:

        data: trainning dataset that will be used to classify a query

        results: results of the training dataset

        uniqueClasses: number of unique classes in the results

        search(query,k,met): find the k nearest instances for each and everyone of the queries using the 
                             given metric met.

        confMatrix(predicted,actual): given the predicted classes of a query, compares them to the actual classes and
                                     returns the confusion matrix

        score(query,k,met,queryResults): given a query set, calls search with k neighbors and metric met to obtain the
                                        predicted classes. Then calls confMatrix with the result of search and queryResults
                                        to obtain the confusion matrix and finally returns the accurarcy

    */

    mat data;
    Col<int> results;
    int uniqueClasses;
    Knn (mat d, Col<int> r, int u):data(d), results(r), uniqueClasses(u){}

    template<typename MetricType>
    vector<vector<size_t>> search2(mat &query,int k, MetricType &met){
        
        vector<vector<double>> distances(query.n_rows,vector<double>(data.n_rows));
        rowvec aux1,aux2;
        for (int i = 0; i < query.n_rows; i++){
            aux2 = query.row(i);
            for (int j = 0; j < data.n_rows; j++){ 
                aux1 = data.row(j);
                distances[i][j] = met.Evaluate(aux1,aux2);
            }
        }

        vector<vector<size_t>> ordered(query.n_rows,vector<size_t>(data.n_rows));
        for (int i = 0; i < query.n_rows; i++) ordered[i] = Utils::sort_indexes(distances[i]);

        return ordered;
    }

    template<typename MetricType>
    Col<int> search(mat &query,int k, MetricType &met){

        mat distances(data.n_rows,query.n_rows);
        rowvec aux1, aux2;
        for (int i = 0; i < query.n_rows; i++){
            aux2 = query.row(i);
            for (int j = 0; j < data.n_rows; j++){ 
                aux1 = data.row(j);
                distances(j,i) = met.Evaluate(aux1,aux2);
            }
        }

        mat ordered = sort(distances);
        Mat<int> comparison(query.n_rows,uniqueClasses,fill::zeros);
        uvec index;

        for (int i = 0; i < query.n_rows; i++){
            for (int j = 0; j < k; j++){
                index = find(abs(distances.col(i)-ordered(j,i))<0.0000000001);
                comparison(i,results(index(0)))++;
            }
        }

        Col<int> resultClass(query.n_rows);
        for (int i = 0; i < query.n_rows; i++) resultClass(i) = comparison.row(i).index_max();

        return resultClass;     

    }

    Mat<int> confMatrix(Col<int> &predicted,Col<int> &actual){

        Mat<int> matrix(uniqueClasses,uniqueClasses,fill::zeros);
        for (int i=0; i < predicted.n_rows; i++) matrix(predicted(i),actual(i))++;
        return matrix;
    }

    template<typename MetricType>
    double score(mat &query,int k, MetricType &met, Col<int> &queryResults){

        Col<int> predicted(search(query,k,met));
        Mat<int> conf(confMatrix(predicted,queryResults));
        double sc = 0;
        for (int i=0; i < uniqueClasses; i++) sc+=conf(i,i);
        sc = sc/query.n_rows;
        return sc;
    }

    template<typename MetricType>
    double kappa(mat &query,int k, MetricType &met, Col<int> &queryResults){

        Col<int> predicted(search(query,k,met));
        Mat<int> conf(confMatrix(predicted,queryResults));

        double diagonal;
        for (int i=0; i < uniqueClasses; i++) diagonal+=conf(i,i);

        double term2;
        for (int i=0; i < uniqueClasses; i++) term2 += sum(conf.row(i)) * sum(conf.col(i));

        double result = (query.n_rows*diagonal - term2) / (pow(query.n_rows,2) - term2);

        return result;
    }

};

#endif