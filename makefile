main: main.cpp Metrics.hpp Knn.hpp
	g++ main.cpp -o main -O3 -std=c++11 -larmadillo

gd: main.cpp Metrics.hpp Knn.hpp
	g++ -O0 -Wall -std=c++11 -ggdb -larmadillo -o main main.cpp

prueba: prueba.cpp
	g++ prueba.cpp -o prueba -O3 -std=c++11 -larmadillo
