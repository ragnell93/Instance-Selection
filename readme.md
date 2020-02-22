# Instance Selection

Code for my undergrad research project. It consisted on mixing different heuristics and metaheuristics to see their performance on the [Instance Selection Problem](https://en.wikipedia.org/wiki/Instance_selection)

The heuristics can be found in **heuristics.hpp**, they are: 

- [CNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm#CNN_for_data_reduction)
- [ENN](https://ieeexplore.ieee.org/abstract/document/4309137)
- RSSS
- IB3

The meaheuristics used can be found in **Metaheuristics.hpp**, they are: 

- [CHC](https://www.sciencedirect.com/science/article/pii/B9780080506845500203)
- [Genetic Algorithm](https://en.wikipedia.org/wiki/Genetic_algorithm)
- [Memetic Algorithm](https://en.wikipedia.org/wiki/Memetic_algorithm)

The distance functions used can be found in **Metrics.hpp**, the are:

- [Euclidean](https://en.wikipedia.org/wiki/Euclidean_distance)
- [IVDM](https://jair.org/index.php/jair/article/view/10182)

The optimized configuration was done using [IRACE](http://iridia.ulb.ac.be/irace/)

To run it simply use **runAll.sh**
