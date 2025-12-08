"""
Handles the drawing of the User Interface.
"""

import pygame
from settings import *

def draw_interface(surface, font, simulation_time, car, gen=None):
    """
    Draws the complete user interface including stats, checkpoints, and actuator (input) bars.
    
    Args:
        surface: Pygame surface to draw on.
        font: Pygame font object.
        simulation_time: Current time in seconds.
        car: The active Car object.
        gen (optional): Current generation number (for NEAT).
    """
    
    # Text stats
    y_offset = 10
    x_offset = 10
    
    if gen is not None:
        gen_text = font.render(f"Gen: {gen}", True, COLOR_TEXT)
        surface.blit(gen_text, (x_offset, y_offset))
        y_offset += 20

    time_text = font.render(f"Time: {simulation_time:.1f}s", True, COLOR_TEXT)
    surface.blit(time_text, (x_offset, y_offset))
    y_offset += 20
    
    laps_text = font.render(f"Laps: {car.laps}", True, COLOR_TEXT)
    surface.blit(laps_text, (x_offset, y_offset))
    y_offset += 20
    
    checkpoints_text = font.render(f"Checkpoint: {car.checkpoint_index}", True, COLOR_TEXT)
    surface.blit(checkpoints_text, (x_offset, y_offset))
    y_offset += 20
    
    fitness_text = font.render(f"Score: {car.fitness}", True, (0, 150, 0))
    surface.blit(fitness_text, (x_offset, y_offset))

    # Draw Checkpoints
    _draw_checkpoints(surface, car)

    # Draw Input Bars
    _draw_input_bars(surface, font, car)

def _draw_checkpoints(surface, car):
    """Helper to draw track checkpoints."""
    
    for i, checkpoint in enumerate(CHECKPOINTS):
        color = COLOR_CHECKPOINT
        width = 2
        
        # Highlight target checkpoint 
        if i == car.checkpoint_index:
            color = COLOR_TARGET_CHECKPOINT
            width = 4
        
        pygame.draw.line(surface, color, checkpoint[0], checkpoint[1], width)

def _draw_input_bars(surface, font, car):
    """Helper to draw the acceleration/steering visualization bars."""
    
    bar_width = 40  
    max_bar_height = 50
    bar_x_start = 10
    bar_y_start = SCREEN_HEIGHT - max_bar_height - 30 
    bar_spacing = 15 

    labels = ["SPD", "ACC", "BRK", "LFT", "RGT"]
    colors = [COLOR_BAR_SPEED, COLOR_BAR_ACC, COLOR_BAR_BRK, COLOR_BAR_STEER, COLOR_BAR_STEER]

    # Calculate speed percentage based on physics constant
    max_v = PHYSICS["MAX_VELOCITY"]
    speed_percent = 0.0 if max_v == 0 else car.velocity / max_v
    
    values = [speed_percent] + car.activations 

    for i, (label, color) in enumerate(zip(labels, colors)):
        value = values[i]
        bar_height = value * max_bar_height
        bar_x = bar_x_start + i * (bar_width + bar_spacing)
        bar_y = bar_y_start + (max_bar_height - bar_height)
        
        # Draw Background
        pygame.draw.rect(surface, COLOR_BAR_BG, (bar_x, bar_y_start, bar_width, max_bar_height))
        
        # Draw Active Level
        pygame.draw.rect(surface, color, (bar_x, bar_y, bar_width, bar_height))
        
        # Draw Label
        label_text = font.render(label, True, COLOR_TEXT)
        text_rect = label_text.get_rect(centerx=bar_x + bar_width/2, top=bar_y_start + max_bar_height + 5)
        surface.blit(label_text, text_rect)