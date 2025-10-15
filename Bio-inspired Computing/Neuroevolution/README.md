# NeuroEvolution of Augmenting Topologies - XOR Example

This project provides a demonstration of the NeuroEvolution of Augmenting Topologies (NEAT) algorithm by training a neural network to solve the XOR boolean function. It is designed as a didactic introduction to the topic, showing how a network's structure and weights can be evolved simultaneously.

This code is based on the XOR example from the neat-python library (v0.92). The primary modification is the inclusion of a custom reporter that saves the best-performing neural network from each generation. This allows for a clear, step-by-step visualization of how the network topology evolves from a simple structure to a more complex one capable of solving the problem.

## Requirements

To run this simulation, you will need Python 3 and the following libraries: ```neat-python```, ```matplotlib```, and ```graphviz```. You will also need to install Graphviz at the system level, as the Python library is just a wrapper.

## How to Run the Code
Execute the main script from your terminal.
```bash
python evolve-feedforward.py
```

The core parameters of the NEAT algorithm can be adjusted in the **`config-feedforward`** file. For a complete list and detailed explanation of all available parameters, please refer to the official [library documentation](https://neat-python.readthedocs.io/en/latest/).

The script will run for up to 300 generations, or until a solution with a fitness of 3.98 is found. As it runs, it will generate the following files:

- **`topology`** A directory where you will find SVG images (gen_0.svg, gen_1.svg, etc.) showing the best network of each generation.
- **`winner.svg`**: shows the topology of the final, successful network that solved the XOR problem.
- **`avg_fitness.svg`**: A plot of the population's average and best fitness across all generations.
- **`speciation.svg`**: A visualization of how species sizes changed throughout the evolution process.

## Reference
For a deeper understanding of the NEAT algorithm, we highly recommend reading the original scientific paper that introduced it: *Stanley, K. O., & Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary computation, 10(2), 99-127*.