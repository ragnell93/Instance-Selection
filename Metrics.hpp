#ifndef METRICS_H
#define METRICS_H
#include <cmath>
#include <armadillo>

using namespace std;
using namespace arma;

struct Euclidean{

    Euclidean(){}

    template<typename VecTypeA, typename VecTypeB>
    double Evaluate(VecTypeA  &a,VecTypeB &b){
        return sqrt(sum(pow((a - b),2)));
    }
};

struct IVDM{

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
            mid1a = minmax(1,i) + minmax(2,i) * (u1+0.5);
            if (*aux1 < mid1a) u1--;
            mid1a = minmax(1,i) + minmax(2,i) * (u1+0.5);
            mid1b = minmax(1,i) + minmax(2,i) * (u1+1.5);

            u2 = discretize(*aux2,i);
            mid2a = minmax(1,i) + minmax(2,i) * (u2+0.5);
            if (*aux2 < mid2a) u2--;
            mid2a = minmax(1,i) + minmax(2,i) * (u2+0.5);
            mid2b= minmax(1,i) + minmax(2,i) * (u2+1.5);

            /*
            cout << "Para el punto 1 se tiene: " << endl;
            cout << "el valor discreto es: " << u1 << endl;
            cout << "el punto medio es: " << mid1a << endl;
            cout << "el punto medio alto es: " << mid1b << endl << endl;

            cout << "Para el punto 2 se tiene: " << endl;
            cout << "el valor discreto es: " << u2 << endl;
            cout << "el punto medio es: " << mid2a << endl;
            cout << "el punto medio alto es: " << mid2b << endl << endl;
            */
  
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
            /*
            cout << "La interpolacion 1 es:"<< endl;
            inter1.print();
            cout << endl;
            cout << "La interpolacion 2 es:" << endl;
            inter2.print();
            */
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

        //cout << "EL vector con los ivs" << endl;
        //iv.print();
        
        double distance=0;
        for (int g=0; g < minmax.n_cols; g++) distance += iv(g)*iv(g);
        //cout << endl << "la distancia es: " << distance << endl;

        return distance;
    }
};

#endif 