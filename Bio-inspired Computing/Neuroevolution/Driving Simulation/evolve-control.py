"""
NEAT-based evolutionary training.
Combines logic for both velocity-aware and non-velocity-aware training based on settings.py configuration.
"""
import pygame
import sys
import os
import neat
import pickle
import statistics
import cv2 

from libs.settings import *
from libs.car import Car
from libs.ui import draw_interface

# --- Setup Directories ---
os.makedirs(NN_CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

if HEADLESS_MODE:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# Global Pygame contexts
screen = None
clock = None
font = None
track_surface = None

# Generation counter
generation = 0

def run_simulation(genomes, config):
    """Simulation loop for a generation."""
    
    global screen

    if not HEADLESS_MODE:
        screen.fill(COLOR_BACKGROUND)
        
        msg = font.render(f"Simulating Gen {generation}", True, COLOR_TEXT)
        rect = msg.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2))
        
        screen.blit(msg, rect)
        pygame.display.flip()        
        pygame.event.pump()

    nets = []
    cars = []
    ge = []
    
    # Initialize genomes and cars
    for _, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        cars.append(Car(600, 575, 270.0))
        ge.append(genome)
        
    # Frame Loop
    for frame in range(MAX_GENERATION_TIME_SEC * FPS):
        if len(cars) == 0: break
        
        surviving_cars = []
        surviving_nets = []
        surviving_ge = []
        
        for i, car in enumerate(cars):
            # Get inputs
            inputs = car.get_sensor_data(track_surface)
            
            # Get NN output
            output = nets[i].activate(inputs)
            
            # Convert NN output to activations (threshold 0.5)
            controls = [
                output[0] > 0.5, # Acc
                output[1] > 0.5, # Brake
                output[2] > 0.5, # Left
                output[3] > 0.5  # Right
            ]
            
            # Update Physics
            car.update(controls)
            
            # Check for collisions
            car.check_wall_collision(track_surface)
            
            # Check for checkpoints
            car.check_checkpoints()
            
            # Assign fitness to NEAT
            ge[i].fitness = car.fitness
            
            if car.is_alive:
                surviving_cars.append(car)
                surviving_nets.append(nets[i])
                surviving_ge.append(ge[i])
        
        cars = surviving_cars
        nets = surviving_nets
        ge = surviving_ge

def replay_simulation(best_genome, config):
    """Replays the best genome of the generation and saves to video."""
    
    vid_path = os.path.join(VIDEOS_DIR, f"gen_{generation}_best.mp4")
    out_video = cv2.VideoWriter(vid_path, cv2.VideoWriter_fourcc(*'mp4v'), FPS, (SCREEN_WIDTH, SCREEN_HEIGHT))
    
    net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    car = Car(600, 575, 270.0)
    
    frame_count = 0
    max_frames = MAX_GENERATION_TIME_SEC * FPS
    
    while car.is_alive and frame_count < max_frames:
        if not HEADLESS_MODE:
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()

        inputs = car.get_sensor_data(track_surface)
        output = net.activate(inputs)
        controls = [o > 0.5 for o in output]
        
        car.update(controls)
        car.check_wall_collision(track_surface)
        car.check_checkpoints()
        
        # Draw for video
        screen.fill(COLOR_BACKGROUND)
        screen.blit(track_surface, (0, 0))
        car.draw(screen)
        draw_interface(screen, font, frame_count/FPS, car, generation)
        
        # Capture Frame
        view = pygame.surfarray.array3d(screen).transpose([1, 0, 2])
        out_video.write(cv2.cvtColor(view, cv2.COLOR_RGB2BGR))
        
        if not HEADLESS_MODE:
            pygame.display.flip()
            clock.tick(FPS)
            
        frame_count += 1
    
    out_video.release()

def eval_genomes(genomes, config):
    global generation
    
    generation += 1
    print(f"Gen {generation}...")
    
    # Run generation
    run_simulation(genomes, config)
    
    # Save best
    best_genome = max(genomes, key=lambda g: g[1].fitness)[1]
    
    cp_path = os.path.join(NN_CHECKPOINTS_DIR, f"gen_{generation}.pkl")
    
    with open(cp_path, "wb") as f:
        pickle.dump(best_genome, f)
        
    # Replay best individual for that generation
    replay_simulation(best_genome, config)

def run(config_file):
    
    global track_surface, screen, clock, font
    
    pygame.init()
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.Font(None, 24)
    
    pygame.display.set_caption("NEAT Evolution")
    
    # Try to load the track   
    try:
        track_surface = pygame.image.load(TRACK_IMAGE_FILE).convert() 
    
    except FileNotFoundError:
        print(f"Error: '{TRACK_IMAGE_FILE}' not found.")
        sys.exit()
    
    if track_surface.get_size() != (SCREEN_WIDTH, SCREEN_HEIGHT):
        track_surface = pygame.transform.scale(track_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))

    # NEAT Config
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_file)
    
    print(f"Loading Config: {config_path}")
    
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    
    p = neat.Population(config)
    
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
        
    p.run(eval_genomes, MAX_GENERATIONS)

if __name__ == "__main__":
    
    # Select config file based on settings
    config_file = '' 
    if INCLUDE_VELOCITY_IN_INPUT:
        config_file = "neat-configs/feedforward-with-velocity.txt"
    else:
        config_file = 'neat-configs/feedforward.txt'
    
    print(f"Starting Training")    
    run(config_file)