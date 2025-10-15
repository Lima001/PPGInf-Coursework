# Objective vs. Divergent Search

The core of this project is a simulation where agents learn to navigate from a start point to a goal. A **deceptive trap** is placed in the world, offering a local reward to lure agents away from the optimal path. This setup provides a classic scenario to compare how different evolutionary strategies handle exploration versus exploitation.

The simulation implements and contrasts three distinct search drivers:
* **Objective Search (Fitness):** Guides evolution by rewarding agents for getting closer to the goal.
* **Novelty Search (Spatial Exploration):** Rewards agents for exploring new areas of the map, based on the final position (endpoint) of their trajectory.
* **Surprise Search (Temporal Deviation):** Rewards agents for moving in ways that are different from the behavior of recent generations.

## Repository Contents

1.  **`main.py`**:  A self-contained Python script for running the simulation. It handles the GA, simulates agent movement, calculates all metrics, and generates visualizations for each generation.
2.  **`reference.pdf`**: A self-contained reference document that provides the explanations and mathematical formulas behind the concepts implemented in the code.

## How to Run the Simulation

### Dependencies
You'll need the following Python libraries. You can install them via pip:
```bash
pip install numpy matplotlib imageio
```

### Execution
You can run the simulation from your terminal. The behavior of the GA is controlled by weighting the three search metrics using command-line arguments.

Pure Objective Search:
```bash
python main.py --fw 1.0 --nw 0.0 --sw 0.0
```

Pure Novelty Search:
```bash
python main.py --fw 0.0 --nw 1.0 --sw 0.0
```

Combined Fitness and Surprise:
```bash
python main.py --fw 0.5 --sw 0.5
```

Key Arguments:

- ```--fw```: Sets the weight for fitness (objective).
- ```--nw```: Sets the weight for novelty.
- ```--sw```: Sets the weight for surprise.
- ```--gens```: Number of generations to run (e.g., --gens 100).
- ```--pop```: Population size (e.g., --pop 50).
- ```--gif```: If included, the script will generate an animated GIF of the entire run.

All outputs, including generation plots and the final GIF, will be saved in a timestamped subdirectory inside the ```run_out/``` folder