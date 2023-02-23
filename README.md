# Shuffled Frog Leaping Algorithm for Travelling Salesman Problem

## About
This repository contains Python code files for implementing the Shuffled Frog Leaping Algorithm (SFLA) to solve the Travelling Salesman Problem (TSP).

The **SFLA** is a metaheuristic optimization algorithm inspired by the behavior of a group of frogs. The algorithm starts with an initial population of solutions, which are then iteratively improved by combining and shuffling the solutions to generate new ones.

The **TSP** is a classic problem in computer science and operations research, where the goal is to find the shortest possible route that visits a set of cities and returns to the starting city. It is a well-known NP-hard problem, which means that it is not feasible to find an optimal solution in polynomial time for large instances.

## Usage

The main script of this repository is main.py, which contains the implementation of the SFLA algorithm for TSP. The code uses a distance matrix to represent the cost of traveling between any two cities in the problem instance. The user can customize the algorithm parameters, such as the population size, the number of iterations, and the mutation rate, to fine-tune the performance of the algorithm for different TSP instances.

The repository also includes a set of benchmark instances berlin10.tsp and berlin52.tsp, which can be used to test the algorithm and compare its performance with other TSP solvers. The benchmark instances are taken from the TSPLIB library, which is a collection of TSP instances widely used in the research community.

## How to run the code
To run the code, the user needs to have Python 3 and the NumPy library installed on their machine. The code can be executed from the command line with the following syntax:

```sh
python main.py berlinx.tsp
```
where berlinx.tsp is the path to the TSP instance file. The code will output the best found solution and its cost, as well as the execution time of the algorithm.

## Contributing
If you find any bugs or have suggestions for improving the code, feel free to open an issue or submit a pull request. Contributions are always welcome!

## License
This code is released under the MIT License. See the LICENSE file for more details.
