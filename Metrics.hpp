#ifndef METRICS_H
#define METRICS_H
#include <cmath>
#include <armadillo>

using namespace std;
using namespace arma;

struct Euclidean{

    /* Euclidean distance */

    Euclidean(){}

    template<typename VecTypeA, typename VecTypeB>
    double Evaluate(VecTypeA  &a,VecTypeB &b){
        return sqrt(sum(pow((a - b),2)));
    }
};

struct IVDM{

    /*Interpolated Value Difference Metric. It needs the conditional probabilities for each atribute, value and class
     *then do an interpolation calculating the probability of a given continuous value and adding every value for every
     *atribute. Member attributes and functions:

     minmax: the matrix where the first row is the max values of each attribute, the second row the min values and 
             the 3 row the width of each attribute.

     prob: matrix with the conditional probabilitie p(a,x,c)

     index: index where the discrete attributes begin

     discretize(x,j): takes x as the number to discretize and j the number of the attribute

     Evaluate(a,b): takes 2 vectors and calculates their distance

     */

    mat minmax;
    cube prob;
    int index;
    IVDM(int i, mat mm, cube p): index(i), minmax(mm), prob(p) {}

    int discretize(double x, int j){
        if (x >= minmax(0,j)) return prob.n_rows;
        else return (floor((x-minmax(1,j))/minmax(2,j)) + 1);
    }

    template<typename VecTypeA, typename VecTypeB>
    double Evaluate(VecTypeA& a,VecTypeB& b){

        int u1,u2;
        double mid1a,mid2a,mid1b,mid2b,lowerProb1,upperProb1,lowerProb2,upperProb2;
        rowvec inter1(prob.n_slices), inter2(prob.n_slices), iv(minmax.n_cols);
        typename VecTypeA::iterator aux1 = a.begin();
        typename VecTypeB::iterator aux2 = b.begin();
        
        int i = 0;
        //Continuous values
        while (i < index){ 

            u1 = discretize(*aux1,i);
            if (u1 < 1) u1 = 1;
            mid1a = minmax(1,i) + minmax(2,i) * (u1+0.5);
            if (*aux1 < mid1a) u1--;
            mid1a = minmax(1,i) + minmax(2,i) * (u1+0.5);
            mid1b = minmax(1,i) + minmax(2,i) * (u1+1.5);

            u2 = discretize(*aux2,i);
            if (u2 < 1) u2 = 1;
            mid2a = minmax(1,i) + minmax(2,i) * (u2+0.5);
            if (*aux2 < mid2a) u2--;
            mid2a = minmax(1,i) + minmax(2,i) * (u2+0.5);
            mid2b= minmax(1,i) + minmax(2,i) * (u2+1.5);
  
            for (int k = 0; k < prob.n_slices; k++){

                if (u1 == 0) {lowerProb1 = 0; upperProb1 = prob(u1,i,k);}
                else if (u1 == prob.n_rows) {lowerProb1 = prob(u1-1,i,k); upperProb1 = 0;}
                else {lowerProb1 = prob(u1-1,i,k); upperProb1 = prob(u1,i,k);}

                if (u2 == 0) {lowerProb2 = 0; upperProb2 = prob(u2,i,k);}
                else if (u2 == prob.n_rows) {lowerProb2 = prob(u2-1,i,k); upperProb2 = 0;}
                else {lowerProb2 = prob(u2-1,i,k); upperProb2 = prob(u2,i,k);}
                    
                inter1(k) = lowerProb1 + ((*aux1 - mid1a)/(mid1b-mid1a)) * (upperProb1 - lowerProb1);
                inter2(k) = lowerProb2 + ((*aux2 - mid2a)/(mid2b-mid2a)) * (upperProb2 - lowerProb2);
            }

            iv(i) = sum(pow((inter1 - inter2),2));
            aux1++;
            aux2++;
            i++;
        }    
 
        //Discrete values
        while (i < minmax.n_cols){

            for (int k = 0; k < prob.n_slices; k++){
                inter1(k) = prob((int)*aux1,i,k);
                inter2(k) = prob((int)*aux2,i,k);
            }

            iv(i) = sum(pow((inter1 - inter2),2));
            aux1++;
            aux2++;
            i++;

        }
        
        double distance=0;
        for (int g=0; g < minmax.n_cols; g++) distance += iv(g)*iv(g);
        return distance;
    }
};

#endif 