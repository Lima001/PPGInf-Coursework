"""
The Car class containing physics, sensor logic, and collision detection.
"""

import math
import pygame
from settings import *
from utils import line_intersect

class Car:
    
    def __init__(self, x, y, start_angle=0.0):
        # Position
        self.x = x
        self.y = y
        self.angle = start_angle
        self.velocity = 0.0
        
        # State
        self.is_alive = True
        
        # Sensors
        self.sensor_angles = SENSOR_ANGLES
        self.sensor_max_length = SENSOR_MAX_LENGTH
        self.sensor_distances = []
        self.sensor_endpoints = []

        # Activations [ACC, BRK, LFT, RGT]
        self.activations = [0.0, 0.0, 0.0, 0.0]
        
        # Racing mechanics 
        self.checkpoint_index = 0
        self.laps = 0
        self.prev_pos = (x, y)
        self.fitness = 0
        self.time_since_last_checkpoint = 0
        
        # Graphic representation
        self.image = pygame.Surface((CAR_WIDTH, CAR_HEIGHT), pygame.SRCALPHA)
        self.image.fill(COLOR_CAR)

    
    def get_sensor_data(self, track_surface):
        """
        Casts rays and returns normalized data.
        Returns: [d1, d2, d3, d4, d5] OR [d1, d2, d3, d4, d5, velocity] based on settings.INCLUDE_VELOCITY_IN_INPUT.
        """
        
        if not self.is_alive:
             self.sensor_distances = [0] * len(self.sensor_angles)
             self.sensor_endpoints = []
             result = [0] * len(self.sensor_angles)
             if INCLUDE_VELOCITY_IN_INPUT:
                 result.append(0)
             return result

        self.sensor_distances = []
        self.sensor_endpoints = [] 
        
        car_rad = math.radians(self.angle)
        
        for angle in self.sensor_angles:
            sensor_angle_rad = car_rad + math.radians(angle)
            
            dist = 0
            for i in range(1, int(self.sensor_max_length) + 1):
                dist = i
                
                sensor_x = self.x - math.sin(sensor_angle_rad) * i
                sensor_y = self.y - math.cos(sensor_angle_rad) * i
                
                # Check bounds
                if not (0 <= sensor_x < SCREEN_WIDTH and 0 <= sensor_y < SCREEN_HEIGHT):
                    dist -= 1; 
                    break
                
                else:
                    # Check off-track (background color)
                    pixel_color = track_surface.get_at((int(sensor_x), int(sensor_y)))    
                    
                    if pixel_color[:3] == COLOR_BACKGROUND[:3]:
                        dist -= 1; 
                        break
            
            self.sensor_distances.append(dist)
            end_x = self.x - math.sin(sensor_angle_rad) * dist
            end_y = self.y - math.cos(sensor_angle_rad) * dist
            self.sensor_endpoints.append((end_x, end_y))
            
        # Normalize
        normalized_data = [d / self.sensor_max_length for d in self.sensor_distances]

        if INCLUDE_VELOCITY_IN_INPUT:
            normalized_data.append(self.velocity / PHYSICS["MAX_VELOCITY"])

        return normalized_data


    def check_wall_collision(self, track_surface):
        """Checks if the car center is on a wall color."""
        
        if not self.is_alive: 
            return

        pixel_color = track_surface.get_at((int(self.x), int(self.y)))
        if pixel_color[:3] == COLOR_BACKGROUND[:3]:
            self.is_alive = False

    
    def check_checkpoints(self):
        """Checks if the car crossed the next target checkpoint."""
        
        if not self.is_alive: return

        curr_pos = (self.x, self.y)
        checkpoint_P1, checkpoint_P2 = CHECKPOINTS[self.checkpoint_index]
        
        # Check intersection between car movement vector and checkpoint line
        if line_intersect(self.prev_pos, curr_pos, checkpoint_P1, checkpoint_P2):
            self.fitness += REWARDS["CHECKPOINT_PASS"]
            self.time_since_last_checkpoint = 0
            self.checkpoint_index += 1
            
            if self.checkpoint_index >= len(CHECKPOINTS):
                self.checkpoint_index = 0
                self.laps += 1
                self.fitness += REWARDS["LAP_COMPLETE"]

    
    def update(self, control_inputs, enforce_timeout=True):
        """
        Updates physics based on inputs using global PHYSICS constants.
        """
        
        if not self.is_alive:
            #self.velocity = 0
            return
        
        # Store pos for next frame checkpoint check
        self.prev_pos = (self.x, self.y)

        accelerate, brake, steer_left, steer_right = control_inputs
        
        # Visualization data
        self.activations = [float(accelerate), float(brake), float(steer_left), float(steer_right)]

        # Physics computation
        is_accelerating = False
        
        if accelerate:
            self.velocity += PHYSICS["ACCELERATION"]
            is_accelerating = True
        
        if brake:
            if self.velocity > 0: 
                self.velocity -= PHYSICS["BRAKE"]
            
            if self.velocity < 0: 
                self.velocity = 0

        self.velocity = max(0, min(self.velocity, PHYSICS["MAX_VELOCITY"])) 
            
        if self.velocity > 0.5: 
            if steer_left and not steer_right: 
                self.angle += PHYSICS["STEER_SPEED"]
            
            elif steer_right and not steer_left: 
                self.angle -= PHYSICS["STEER_SPEED"]
            
            self.angle %= 360 

        if not is_accelerating:
            
            if self.velocity > 0:
                self.velocity -= (PHYSICS["DRAG"] * self.velocity) + PHYSICS["FRICTION"]
                
                if self.velocity < 0: 
                    self.velocity = 0

        # Move
        rad = math.radians(self.angle)
        self.x -= self.velocity * math.sin(rad)
        self.y -= self.velocity * math.cos(rad)
        
        # Timeout Logic
        self.time_since_last_checkpoint += 1
        if enforce_timeout and self.time_since_last_checkpoint > (TIMEOUT_SECONDS * FPS):
            self.is_alive = False

    
    def draw(self, surface, draw_sensors=True):
        """Draws the car sprite and sensor rays."""
        
        rotated_image = pygame.transform.rotate(self.image, self.angle)
        new_rect = rotated_image.get_rect(center=(self.x, self.y))
        
        surface.blit(rotated_image, new_rect.topleft)
        
        if draw_sensors and self.is_alive:
            for endpoint in self.sensor_endpoints:
                pygame.draw.line(surface, COLOR_SENSOR, (self.x, self.y), endpoint, 1)