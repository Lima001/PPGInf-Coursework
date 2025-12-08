"""
Render the topology graph given a NN-checkpoint file.
"""

import argparse
import os
import pickle
import gzip
import graphviz
import neat
from settings import INCLUDE_VELOCITY_IN_INPUT

INPUT_KEYS = [-1, -2, -3, -4, -5]   # Inputs: 5 sensors (-1 to -5).
if INCLUDE_VELOCITY_IN_INPUT:
    INPUT_KEYS.append(-6)           # If velocity information enabled, include -6.

OUTPUT_KEYS = [0, 1, 2, 3]  # Outputs: ACC, BRK, LFT, RGT 

# Label mapping
NODE_LABELS = {
    -1: 'S1', -2: 'S2', -3: 'S3', -4: 'S4', -5: 'S5', 
    0: 'ACC', 1: 'BRK', 2: 'LFT', 3: 'RGT'
}

if INCLUDE_VELOCITY_IN_INPUT:
    NODE_LABELS[-6] = 'VEL'

def load_genome(checkpoint_file):
    """Loads the genome assuming"""
    
    try:
        with open(checkpoint_file, 'rb') as f:
            genome = pickle.load(f)
        return genome
    
    except Exception as e:
        print(f"Error loading pickle: {e}")
        print("Ensure this file was created by evolve.py.")
        exit(1)

def draw_net(genome, view=False, filename=None, fmt='png'):
    """Draws the neural network topology using Graphviz."""
    
    dot = graphviz.Digraph(format=fmt)
    dot.attr(rankdir='TB') 
    dot.attr(splines='true')
    dot.attr(nodesep='0.6', ranksep='0.8')

    # Filter enabled connections
    enabled_connections = [cg for cg in genome.connections.values() if cg.enabled]
    
    # Identify active nodes (Used Inputs/Outputs + Hidden)
    active_nodes = set()
    
    # Add connected nodes
    for cg in enabled_connections:
        active_nodes.add(cg.key[0]) 
        active_nodes.add(cg.key[1]) 
    
    # Always include all outputs to see the full architecture, even if disconnected
    active_nodes.update(OUTPUT_KEYS)
    
    # Only include Inputs if they are actually connected
    used_inputs = [i for i in INPUT_KEYS if i in active_nodes]

    # --- Inputs (Top) ---
    with dot.subgraph(name='cluster_inputs') as c:
        c.attr(rank='source') 
        c.attr(style='invis')
        
        for node_id in used_inputs:
            c.node(str(node_id), label=NODE_LABELS.get(node_id, str(node_id)), style='filled', shape='circle', fillcolor='lightgray', fontsize='10')

    # --- Outputs (Bottom) ---
    with dot.subgraph(name='cluster_outputs') as c:
        c.attr(rank='sink') 
        c.attr(style='invis')
        
        for node_id in OUTPUT_KEYS:
            c.node(str(node_id), label=NODE_LABELS.get(node_id, str(node_id)), style='filled', shape='circle', fillcolor='lightblue', fontsize='10')

    # --- Hidden Nodes (Middle) ---
    for node_id in active_nodes:
        if node_id not in INPUT_KEYS and node_id not in OUTPUT_KEYS:
            dot.node(str(node_id), label=str(node_id), style='filled', shape='circle', fillcolor='white', fontsize='10')

    # --- Edges ---
    for cg in enabled_connections:
        width = str(0.5 + abs(cg.weight)) 
        color = 'green' if cg.weight > 0 else 'red'
        
        dot.edge(str(cg.key[0]), str(cg.key[1]), label=f"{cg.weight:.2f}", color=color, penwidth=width, fontsize='8')

    output_path = dot.render(filename, view=view, cleanup=True)
    
    return output_path

def main():
    
    parser = argparse.ArgumentParser(description="Render topology from raw genome pickle.")
    parser.add_argument("checkpoint", help="Path to the .pkl file (e.g., training_output/checkpoints/gen_5.pkl)")
    parser.add_argument("--output", "-o", help="Output filename (without extension)")
    parser.add_argument("--format", default="png", choices=["png", "pdf", "svg"], help="Output format")
    parser.add_argument("--no-view", action="store_true", help="Do not open image immediately")

    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: File '{args.checkpoint}' not found.")
        return

    print(f"Loading: {args.checkpoint}...")
    best_genome = load_genome(args.checkpoint)
    
    if not hasattr(best_genome, 'connections'):
        print("Error: Loaded object does not look like a NEAT Genome.")
        return

    print(f"Genome ID: {best_genome.key} | Fitness: {best_genome.fitness}")

    if args.output:
        output_name = args.output
    else:
        output_name = os.path.splitext(args.checkpoint)[0] + "_topology"

    try:
        path = draw_net(best_genome, view=not args.no_view, filename=output_name, fmt=args.format)
        print(f"Graph generated: {path}")
    
    except Exception as e:
        print(f"Error rendering graph: {e}")
        print("Note: Ensure Graphviz is installed on your system (not just the python library).")

if __name__ == "__main__":
    main()