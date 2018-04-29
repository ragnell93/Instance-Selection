main: main.cpp Metrics.hpp Knn.hpp Metaheuristics.hpp Heuristics.hpp Instance.hpp Kfold.hpp
	g++ main.cpp -o main -O3 -std=c++11 -larmadillo -pthread

gd: main.cpp Metrics.hpp Knn.hpp Metaheuristics.hpp Heuristics.hpp Instance.hpp Kfold.hpp
	g++ -O0 -Wall -std=c++11 -ggdb -larmadillo -pthread -o main main.cpp

prueba: prueba.cpp
	g++ prueba.cpp -o prueba -O3 -std=c++11 -larmadillo -pthread
