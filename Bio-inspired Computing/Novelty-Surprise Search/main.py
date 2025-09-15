#!/usr/bin/env python3
"""
Didactic Genetic Algorithm demo:
 - Scenario: A single, central deceptive trap.
 - This setup is designed to show how a pure objective search can get stuck, while novelty/surprise search can help escape deceptiveness scenario.
 - A K-NN surprise metric is introduced to measure deviation from a local historical neighborhood.
 - Fitness is based on a robot's proximity to a goal and a deceptive trap.
 - Novelty is K-NN over endpoints (population + FIFO archive).
 - Early stop if any individual reaches the goal.
 - This version is designed for graphical output to clearly illustrate the search process.
"""

import os
import argparse
import random
import sys
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from datetime import datetime, timezone
from glob import glob

try:
    import imageio
    HAS_IMAGEIO = True
except Exception:
    HAS_IMAGEIO = False

# ---------------- CLI ----------------
parser = argparse.ArgumentParser(description="GA demo: objective vs. novelty search with a single deceptive trap.")
parser.add_argument("--path", action="store_true", help="Plot full trajectories (light gray) in per-generation images")
parser.add_argument("--seed", type=int, default=1, help="Random seed")
parser.add_argument("--gif", action="store_true", help="Attempt to create a GIF of generations (requires imageio)")
parser.add_argument("--out", type=str, default="run_out", help="Output directory")
parser.add_argument("--pop", type=int, default=50, help="Population size")
parser.add_argument("--gens", type=int, default=400, help="Number of generations")
parser.add_argument("--seq", type=int, default=40, help="Timesteps per individual (sequence length)")
parser.add_argument("--step", type=float, default=0.2, help="Step scale (movement multiplier)")
parser.add_argument("--fw", type=float, default=1.0, help="Fitness weight (default 1.0)")
parser.add_argument("--nw", type=float, default=0.0, help="Novelty weight (default 0.0)")
parser.add_argument("--sw", type=float, default=0.0, help="Surprise weight (default 0.0)")
parser.add_argument("--run-until-end", action="store_true", help="Do not stop when goal is found; run for all generations")
args = parser.parse_args()

# ---------------- Configurable parameters ----------------
POP = args.pop
GENS = args.gens
SEQ_LEN = args.seq
STEP_SIZE = args.step
WORLD = 5.0

# Signal parameters for a realistic scenario
DECEPTIVE_BONUS = 20
GOAL_BONUS = 50

# Standard, fair parameters
MUT_PROB = 0.6
PER_GENE_MUT_RATE = 0.1
MUT_SCALE = 0.25
ELITISM = 1
TOURNAMENT_K = 5

K_NEAR = 5
SURPRISE_K = 10
SURPRISE_HISTORY_M = 50
ARCHIVE_MAX = 5000
ARCHIVE_ADD_PER_GEN = 10

FW = args.fw
NW = args.nw
SW = args.sw
RUN_UNTIL_END = args.run_until_end

PLOT_PATHS = args.path
MAKE_GIF = args.gif and HAS_IMAGEIO

OUTDIR = args.out
os.makedirs(OUTDIR, exist_ok=True)

np.random.seed(args.seed)
random.seed(args.seed)

EPS = 1e-8

# ---------------- World geometry ----------------

# Fixed start and goal points
START = np.array([-WORLD * 0.8, -WORLD * 0.8])
GOAL = np.array([WORLD * 0.8, WORLD * 0.8])

# Single, central deceptive trap
TRAP_CENTER = np.array([0.0, 0.0])
TRAP_RADIUS = WORLD * 0.3
GOAL_REACH_RADIUS = WORLD * 0.05

# ---------------- Utility functions ----------------
def random_genome():
    """Returns a random genome: a flat vector of length SEQ_LEN*2."""
    return np.random.uniform(-1.0, 1.0, size=(SEQ_LEN * 2)).astype(np.float32)

def pairwise_distances(A, B):
    """Euclidean pairwise distances between rows of A and rows of B (safe with empty arrays)."""
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=np.float64)
    aa = (A**2).sum(axis=1)[:, None]
    bb = (B**2).sum(axis=1)[None, :]
    ab = A.dot(B.T)
    d2 = aa + bb - 2 * ab
    d2 = np.maximum(d2, 0.0)
    return np.sqrt(d2)

# ---------------- Simulation & Descriptors ----------------
def simulate_genome(genome):
    """Simulates a genome's trajectory: position moves according to genes."""
    seq = genome.reshape(SEQ_LEN, 2)
    pos = START.copy()
    traj = [pos.copy()]
    for a in seq:
        pos = pos + (a * STEP_SIZE)
        pos = np.clip(pos, -WORLD, WORLD)
        traj.append(pos.copy())
        if np.linalg.norm(pos - GOAL) < GOAL_REACH_RADIUS:
            break
    while len(traj) < SEQ_LEN + 1:
        traj.append(traj[-1].copy())
    return np.array(traj)

def fitness_for_traj(traj):
    """Calculates fitness with a single deceptive trap bonus."""
    final = traj[-1]
    dist_to_goal = np.linalg.norm(final - GOAL)
    fit = -dist_to_goal
    
    # Check for the single deceptive trap
    min_dt = np.min(np.linalg.norm(traj - TRAP_CENTER, axis=1))
    if min_dt <= TRAP_RADIUS:
        fit += DECEPTIVE_BONUS * (1.0 - (min_dt / TRAP_RADIUS))
    
    # Check if the goal was reached
    if dist_to_goal < GOAL_REACH_RADIUS:
        fit += GOAL_BONUS
    
    return float(fit)

def endpoint_descriptor(traj):
    """Returns the final (x,y) coordinates of a trajectory, used for novelty."""
    return traj[-1].astype(np.float32)

def surprise_for_traj(trajectories, prev_gen_trajectories_hist, k=SURPRISE_K, m=SURPRISE_HISTORY_M):
    """
    Calculates surprise as average Euclidean distance to k-nearest neighbors from a history of past generations.
    - If there are not enough generations in the history, it uses all available data.
    - If there is no history, surprise is 0.
    """
    if not prev_gen_trajectories_hist or prev_gen_trajectories_hist[0].size == 0:
        return np.zeros(trajectories.shape[0])
    
    # Flatten trajectories for distance calculation
    current_trajectories_flat = trajectories.reshape(trajectories.shape[0], -1)
    
    # Combine the history of trajectories
    history_trajectories_flat = np.vstack([t.reshape(t.shape[0], -1) for t in prev_gen_trajectories_hist[-m:]])

    dmat = pairwise_distances(current_trajectories_flat, history_trajectories_flat)
    
    total_candidates = history_trajectories_flat.shape[0]
    k_eff = min(k, total_candidates) if total_candidates > 0 else 0
    if k_eff <= 0:
        return np.zeros(trajectories.shape[0], dtype=np.float64)
    
    knn = np.sort(dmat, axis=1)[:, :k_eff]
    return knn.mean(axis=1)

# ---------------- Novelty (endpoints K-NN + FIFO archive) ----------------
def novelty_from_population(endpoints, archive_array, k=K_NEAR):
    """Average distance to k nearest among combined (current endpoints + archive)."""
    if endpoints.size == 0:
        return np.array([])
    if archive_array.size == 0:
        combined = endpoints
    else:
        combined = np.vstack([endpoints, archive_array])
    dmat = pairwise_distances(endpoints, combined)
    # prevent self-match
    for i in range(endpoints.shape[0]):
        dmat[i, i] = np.inf
    total_candidates = combined.shape[0] - 1
    k_eff = min(k, total_candidates) if total_candidates > 0 else 0
    if k_eff <= 0:
        return np.zeros(endpoints.shape[0], dtype=np.float64)
    knn = np.sort(dmat, axis=1)[:, :k_eff]
    return knn.mean(axis=1)

# ---------------- GA operators ----------------
def tournament_select(scores, k=TOURNAMENT_K):
    """Selects the best individual from a random sample of k."""
    k_eff = min(k, len(scores))
    participants = np.random.choice(len(scores), size=k_eff, replace=False)
    best = participants[np.argmax(scores[participants])]
    return int(best)

def one_point_crossover(a, b):
    """Creates two new genomes by swapping a segment from the parents."""
    L = a.size
    if L <= 1:
        return a.copy(), b.copy()
    pt = np.random.randint(1, L)
    child1 = np.concatenate([a[:pt], b[pt:]])
    child2 = np.concatenate([b[:pt], a[pt:]])
    return child1, child2

def mutate(genome):
    """Randomly modifies genes in a genome."""
    g = genome.copy()
    mask = np.random.rand(*g.shape) < PER_GENE_MUT_RATE
    if np.any(mask):
        g[mask] += np.random.randn(np.sum(mask)) * MUT_SCALE
        g = np.clip(g, -3.0, 3.0)
    return g

def evolve(population, composite_scores):
    """Evolves the population for one generation."""
    pop_size = len(population)
    new_pop = []
    if ELITISM > 0:
        elite_idx = np.argsort(composite_scores)[-ELITISM:]
        for i in elite_idx:
            new_pop.append(population[int(i)].copy())
    while len(new_pop) < pop_size:
        p1 = tournament_select(composite_scores)
        p2 = tournament_select(composite_scores)
        c1, c2 = one_point_crossover(population[p1], population[p2])
        if np.random.rand() < MUT_PROB:
            c1 = mutate(c1)
        if np.random.rand() < MUT_PROB:
            c2 = mutate(c2)
        new_pop.append(c1)
        if len(new_pop) < pop_size:
            new_pop.append(c2)
    return new_pop

# ---------------- Plot helpers ----------------
def plot_two_series(best_series, avg_series, title, ylabel, outpath):
    """Plots two series (e.g., best and average fitness) on the same graph."""
    gens = np.arange(1, len(best_series) + 1)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(gens, best_series, '-o', label=f'Best {ylabel}')
    ax.plot(gens, avg_series, '-.', label=f'Avg {ylabel}')
    ax.set_xlabel('Generation'); ax.set_ylabel(f'{ylabel}')
    ax.set_title(title); ax.grid(True); ax.legend(loc='upper left', fontsize='small')
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close(fig)

def plot_single_series(series, title, ylabel, outpath):
    """Plots a single series over generations."""
    gens = np.arange(1, len(series) + 1)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(gens, series, '-o', label=f'{ylabel}')
    ax.set_xlabel('Generation'); ax.set_ylabel(f'{ylabel}')
    ax.set_title(title); ax.grid(True); ax.legend(loc='upper left', fontsize='small')
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close(fig)

def plot_generation(trajs, endpoints, best_idx, gen_num, out_dir, run_id, show_paths=False):
    """Generates a plot of the current generation's trajectories and endpoints."""
    fig, ax = plt.subplots(figsize=(6,6))
    if show_paths:
        for t in trajs:
            ax.plot(t[:,0], t[:,1], linewidth=0.9, alpha=0.35, color=(0.8,0.8,0.8))
    ax.scatter(endpoints[:,0], endpoints[:,1], s=30, alpha=0.8, label='population endpoints')
    ax.scatter([START[0]], [START[1]], c='green', s=80, edgecolors='k', label='start')
    ax.scatter([GOAL[0]], [GOAL[1]], c='blue', s=80, edgecolors='k', label='goal')
    
    # Plot the single deceptive trap
    trap_circ = Circle(tuple(TRAP_CENTER), TRAP_RADIUS, color='red', alpha=0.25)
    ax.add_patch(trap_circ)
    ax.plot([], [], 'o', color='red', alpha=0.25, label='deceptive trap')
        
    if 0 <= best_idx < len(endpoints):
        b = endpoints[best_idx]; ax.scatter([b[0]], [b[1]], marker='*', s=200, c='red', edgecolors='k', label='best')
    
    ax.set_xlim(-WORLD*1.1, WORLD*1.1); ax.set_ylim(-WORLD*1.1, WORLD*1.1); ax.set_aspect('equal', 'box')
    ax.set_title(f"Generation {gen_num}"); ax.legend(loc='upper left', fontsize='small')
    plt.tight_layout()
    outp = os.path.join(out_dir, f"{run_id}_gen_{gen_num:03d}.png")
    plt.savefig(outp, dpi=150); plt.close(fig)
    return outp

def plot_all_endpoints_scatter(all_endpoints_list, out_dir, run_id):
    """Plots all endpoints from all generations on a single scatter plot."""
    fig, ax = plt.subplots(figsize=(7,6))
    cmap = plt.cm.viridis
    total = len(all_endpoints_list)
    for gi, arr in enumerate(all_endpoints_list):
        if arr is None or arr.size == 0: continue
        color = cmap(gi / max(1, total - 1))
        ax.scatter(arr[:,0], arr[:,1], s=12, alpha=0.5, color=color)
    ax.scatter([START[0]], [START[1]], c='green', s=80, edgecolors='k', label='start')
    ax.scatter([GOAL[0]], [GOAL[1]], c='blue', s=80, edgecolors='k', label='goal')
    
    trap_circ = Circle(tuple(TRAP_CENTER), TRAP_RADIUS, color='red', alpha=0.25); ax.add_patch(trap_circ)
    ax.plot([], [], 'o', color='red', alpha=0.25, label='deceptive trap')

    ax.set_xlim(-WORLD*1.1, WORLD*1.1); ax.set_ylim(-WORLD*1.1, WORLD*1.1); ax.set_aspect('equal', 'box')
    ax.set_title('All endpoints across generations (colored by generation)'); plt.tight_layout()
    outp = os.path.join(out_dir, f"{run_id}_all_endpoints_scatter.png"); plt.savefig(outp, dpi=150); plt.close(fig)

# ---------------- Main Loop ----------------
def run():
    run_id = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    out_subdir = os.path.join(OUTDIR, run_id)
    os.makedirs(out_subdir, exist_ok=True)

    population = [random_genome() for _ in range(POP)]
    archive = deque(maxlen=ARCHIVE_MAX)
    prev_gen_trajectories_hist = deque(maxlen=SURPRISE_HISTORY_M)
    
    hist_best_fit = []; hist_avg_fit = []
    hist_best_nov = []; hist_avg_nov = []
    hist_best_sur = []; hist_avg_sur = []
    hist_div = []
    
    all_endpoints_across = []
    
    goal_found_gen = -1
    
    for gen in range(1, GENS + 1):
        trajs = []; endpoints = []; fitnesses = []

        for ind in population:
            traj = simulate_genome(ind)
            trajs.append(traj)
            endpoints.append(endpoint_descriptor(traj))
            fitnesses.append(fitness_for_traj(traj))

        trajs = np.array(trajs)
        endpoints = np.vstack(endpoints)
        fitnesses = np.array(fitnesses)
        
        # Calculate Surprise based on deviation from the previous generations' history
        surprise = surprise_for_traj(trajs, list(prev_gen_trajectories_hist), k=SURPRISE_K, m=SURPRISE_HISTORY_M)
        
        # Novelty: endpoints + FIFO archive
        archive_array = np.vstack(archive) if len(archive) > 0 else np.empty((0,2))
        novelty = novelty_from_population(endpoints, archive_array, k=K_NEAR)
        
        # Normalize scores to have equal footing for composite score calculation
        norm_fitness = (fitnesses - np.min(fitnesses)) / (np.max(fitnesses) - np.min(fitnesses) + EPS)
        norm_novelty = (novelty - np.min(novelty)) / (np.max(novelty) - np.min(novelty) + EPS)
        norm_surprise = (surprise - np.min(surprise)) / (np.max(surprise) - np.min(surprise) + EPS)
        
        composite = FW * norm_fitness + NW * norm_novelty + SW * norm_surprise

        best_idx = int(np.argmax(composite))
        
        best_fit_val = float(np.max(fitnesses)); avg_fit_val = float(np.mean(fitnesses))
        best_nov_val = float(np.max(novelty)) if novelty.size > 0 else 0.0
        avg_nov_val = float(np.mean(novelty)) if novelty.size > 0 else 0.0
        best_sur_val = float(np.max(surprise)) if surprise.size > 0 else 0.0
        avg_sur_val = float(np.mean(surprise)) if surprise.size > 0 else 0.0

        hist_best_fit.append(best_fit_val); hist_avg_fit.append(avg_fit_val)
        hist_best_nov.append(best_nov_val); hist_avg_nov.append(avg_nov_val)
        hist_best_sur.append(best_sur_val); hist_avg_sur.append(avg_sur_val)
        
        if endpoints.shape[0] > 1:
            dmat = pairwise_distances(endpoints, endpoints)
            tri = np.triu_indices(endpoints.shape[0], k=1)
            diversity = float(np.mean(dmat[tri]))
        else:
            diversity = 0.0
        hist_div.append(diversity)
        
        all_endpoints_across.append(endpoints.copy())
        
        if ARCHIVE_ADD_PER_GEN > 0 and novelty.size > 0:
            top_idx = np.argsort(-novelty)[:ARCHIVE_ADD_PER_GEN]
            for idx in top_idx:
                archive.append(endpoints[idx])
        
        goal_reachers = int(np.sum(np.linalg.norm(endpoints - GOAL[None, :], axis=1) < GOAL_REACH_RADIUS))

        print(f"Gen {gen:03d}/{GENS} | fit {best_fit_val:.3f} (avg {avg_fit_val:.3f}) | nov {best_nov_val:.3f} (avg {avg_nov_val:.3f}) | sur {best_sur_val:.3f} (avg {avg_sur_val:.3f}) | goals {goal_reachers}")
        
        _ = plot_generation(trajs, endpoints, best_idx, gen, out_subdir, run_id, show_paths=PLOT_PATHS)
        
        if goal_reachers > 0 and not RUN_UNTIL_END:
            goal_found_gen = gen
            print(f"\n GOAL REACHED at generation {gen}! Stopping early.\n")
            break

        # Store all trajectories for the next generation's surprise calculation
        prev_gen_trajectories_hist.append(trajs)
        
        population = evolve(population, composite)

    outp1 = os.path.join(out_subdir, f"{run_id}_metric_fitness.png")
    plot_two_series(hist_best_fit, hist_avg_fit, "Fitness (best vs avg)", "Fitness", outp1)
    
    outp2 = os.path.join(out_subdir, f"{run_id}_metric_novelty.png")
    plot_two_series(hist_best_nov, hist_avg_nov, "Novelty (best vs avg)", "Novelty", outp2)

    outp3 = os.path.join(out_subdir, f"{run_id}_metric_surprise.png")
    plot_two_series(hist_best_sur, hist_avg_sur, "Surprise (best vs avg)", "Surprise", outp3)
    
    outp4 = os.path.join(out_subdir, f"{run_id}_metric_diversity.png")
    plot_single_series(hist_div, "Population Diversity", "Diversity", outp4)
    
    plot_all_endpoints_scatter(all_endpoints_across, out_subdir, run_id)

    if MAKE_GIF:
        try:
            pngs = sorted(glob(os.path.join(out_subdir, f"{run_id}_gen_*.png")))
            images = [imageio.v2.imread(p) for p in pngs]
            gif_path = os.path.join(out_subdir, f"{run_id}_evolution.gif")
            imageio.v2.mimsave(gif_path, images, fps=6)
            print("GIF saved to:", gif_path)
        except Exception as e:
            print("Could not create GIF:", e)
    
    if goal_found_gen != -1:
        print("Run complete — goal reached at generation", goal_found_gen)
    else:
        print("Run complete — goal not reached within max generations.")
    print("Outputs saved in:", out_subdir)

if __name__ == "__main__":
    cfg = {
        'POP': POP, 'GENS': GENS, 'SEQ_LEN': SEQ_LEN,
        'FW': FW, 'NW': NW, 'SW': SW
    }
    run()