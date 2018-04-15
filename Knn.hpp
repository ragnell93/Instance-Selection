#ifndef KNN_H
#define KNN_H
#include <cmath>
#include <armadillo>
#include "Metrics.hpp"

using namespace std;
using namespace arma;

struct Knn{

    mat data;
    Col<int> results;
    int uniqueClasses;
    Knn (mat d, Col<int> r, int u):data(d), results(r), uniqueClasses(u){}

    template<typename MetricType>
    Col<int> search(mat &query,int k, MetricType &met){

        mat distances(data.n_rows,query.n_rows);

        rowvec aux1, aux2;
        for (int i = 0; i < query.n_rows; i++){
            aux2 = query.row(i);
            for (int j = 0; j < data.n_rows; j++){
                //cout << "Hola mundo " << i <<  endl; 
                aux1 = data.row(j);
                distances(j,i) = met.Evaluate(aux1,aux2);
            }
        }

        //distances.print();

        mat ordered = sort(distances);
        Mat<int> comparison(query.n_rows,uniqueClasses,fill::zeros);
        uvec index;

        //cout << endl;
        //ordered.print();

        for (int i = 0; i < query.n_rows; i++){
            for (int j = 0; j < k; j++){
                index = find(abs(distances.col(i)-ordered(j,i))<0.0000000001);
                comparison(i,results(index(0)))++;
            }
        }
        //cout << endl;
        //comparison.print();

        Col<int> resultClass(query.n_rows);
        for (int i = 0; i < query.n_rows; i++) resultClass(i) = comparison.row(i).index_max();

        return resultClass;     

    }

};

#endif