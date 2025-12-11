"""
Manual driving simulation.
"""

import pygame
import sys

from libs.settings import *
from libs.car import Car
from libs.ui import draw_interface

def main():
    
    pygame.init()
    
    clock = pygame.time.Clock()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    font = pygame.font.Font(None, 24)
    
    pygame.display.set_caption("Driving Simulation - Manual Mode")
  
    # Try to load the track   
    try:
        track_surface = pygame.image.load(TRACK_IMAGE_FILE).convert() 
    
    except FileNotFoundError:
        print(f"Error: '{TRACK_IMAGE_FILE}' not found.")
        sys.exit()
    
    if track_surface.get_size() != (SCREEN_WIDTH, SCREEN_HEIGHT):
        track_surface = pygame.transform.scale(track_surface, (SCREEN_WIDTH, SCREEN_HEIGHT))
    
    car = Car(600, 575, 270.0)
    
    start_time = pygame.time.get_ticks()
    simulation_time = 0

    running = True
    while running:
        
        clock.tick(FPS)
        
        if car.is_alive:
            simulation_time = (pygame.time.get_ticks() - start_time) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): 
                running = False

        keys = pygame.key.get_pressed()
        
        # Reset simulation when R is pressed
        if keys[pygame.K_r]:
            car = Car(600, 575, 270.0)
            start_time = pygame.time.get_ticks() 
            simulation_time = 0
        
        inputs = [
            keys[pygame.K_UP],
            keys[pygame.K_DOWN],
            keys[pygame.K_LEFT],
            keys[pygame.K_RIGHT]
        ]

        # Update Physics
        car.update(inputs, enforce_timeout=False)
        
        # Update Sensors
        car.get_sensor_data(track_surface)
        
        # Check for collisions
        car.check_wall_collision(track_surface)
        
        # Check for checkpoints
        car.check_checkpoints()

        # Drawings
        screen.blit(track_surface, (0, 0))
        car.draw(screen, draw_sensors=True)
        draw_interface(screen, font, simulation_time, car)
        
        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()