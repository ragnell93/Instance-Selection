main: main.cpp Metrics.hpp Knn.hpp Metaheuristics.hpp Heuristics.hpp Instance.hpp Kfold.hpp
	g++ main.cpp -o main -O3 -std=c++11 -larmadillo -pthread -lboost_program_options

main2: main2.cpp Metrics.hpp Knn.hpp Metaheuristics2.hpp Heuristics.hpp Instance.hpp Kfold2.hpp
	g++ main2.cpp -o main2 -O3 -std=c++11 -larmadillo -pthread -lboost_program_options

gd: main.cpp Metrics.hpp Knn.hpp Metaheuristics.hpp Heuristics.hpp Instance.hpp Kfold.hpp
	g++ -O0 -Wall -std=c++11 -ggdb -larmadillo -pthread -lboost_program_options -o main main.cpp

prueba: prueba.cpp
	g++ prueba.cpp -o prueba -O3 -std=c++11 -larmadillo -pthread
