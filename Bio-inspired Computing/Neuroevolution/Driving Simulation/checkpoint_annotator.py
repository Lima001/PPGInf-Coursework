"""
Helper program for defining/drawing the track checkpoints.
"""

import pygame
import sys

# Screen configuration
TRACK_IMAGE = "track.png" 
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800

# World coordinate of track start
CAR_START_POS = (600, 575)

# Colors
COLOR_TEXT = (255, 255, 255)
COLOR_CHECKPOINT = (0, 255, 255)
COLOR_ACTIVE = (255, 0, 0)
COLOR_START_DOT = (0, 255, 0)

class CheckpointEditor:
    
    def __init__(self):
        pygame.init()
        
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16, bold=True)

        pygame.display.set_caption("Checkpoint Editor | Scroll: Zoom | Middle Click: Pan")

        # Load Track
        try:
            self.original_image = pygame.image.load(TRACK_IMAGE).convert()
        except FileNotFoundError:
            print(f"Error: Could not load {TRACK_IMAGE}")
            sys.exit()

        # State Variables
        self.checkpoints = []
        self.current_start_point = None # In World Coordinates
        
        # Viewport / Camera State
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.is_panning = False
        self.last_mouse_pos = (0, 0)

        # Fit image to screen initially
        img_rect = self.original_image.get_rect()
        scale_w = SCREEN_WIDTH / img_rect.width
        scale_h = SCREEN_HEIGHT / img_rect.height
        self.zoom = min(scale_w, scale_h)
        
        # Center it
        self.offset_x = (SCREEN_WIDTH - (img_rect.width * self.zoom)) / 2
        self.offset_y = (SCREEN_HEIGHT - (img_rect.height * self.zoom)) / 2

    # --- Coordinate conversion ---
    def screen_to_world(self, pos):
        sx, sy = pos
        wx = (sx - self.offset_x) / self.zoom
        wy = (sy - self.offset_y) / self.zoom
        return (wx, wy)

    def world_to_screen(self, pos):
        wx, wy = pos
        sx = (wx * self.zoom) + self.offset_x
        sy = (wy * self.zoom) + self.offset_y
        return (sx, sy)

    def run(self):
        running = True
        
        while running:
            
            # Handle Input
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                # --- ZOOMING (Mouse Wheel) ---
                elif event.type == pygame.MOUSEWHEEL:
                    old_zoom = self.zoom
                    zoom_factor = 1.1 if event.y > 0 else 0.9
                    self.zoom *= zoom_factor
                    self.zoom = max(0.1, min(self.zoom, 5.0)) # Clamp zoom
                    
                    # Zoom towards mouse pointer
                    mx, my = pygame.mouse.get_pos()
                    
                    # Get world pos under mouse before zoom
                    wx = (mx - self.offset_x) / old_zoom
                    wy = (my - self.offset_y) / old_zoom
                    
                    # Calculate new offset to keep that world pos under mouse
                    self.offset_x = mx - (wx * self.zoom)
                    self.offset_y = my - (wy * self.zoom)

                # --- PANNING (Middle Click) ---
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    
                    # Middle Mouse
                    if event.button == 2:
                        self.is_panning = True
                        self.last_mouse_pos = event.pos
                    
                    # Left Click (Draw)
                    elif event.button == 1:
                        world_pos = self.screen_to_world(event.pos)
                        if self.current_start_point is None:
                            self.current_start_point = world_pos
                        else:
                            self.checkpoints.append((self.current_start_point, world_pos))
                            self.current_start_point = None
                       
                    # Right Click (Undo)     
                    elif event.button == 3: 
                        if self.current_start_point:
                            self.current_start_point = None
                        elif self.checkpoints:
                            self.checkpoints.pop()

                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 2:
                        self.is_panning = False

                elif event.type == pygame.MOUSEMOTION:
                    if self.is_panning:
                        mx, my = event.pos
                        dx = mx - self.last_mouse_pos[0]
                        dy = my - self.last_mouse_pos[1]
                        self.offset_x += dx
                        self.offset_y += dy
                        self.last_mouse_pos = (mx, my)

                # SAVE
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        self.save_checkpoints()
                        running = False

            # Draw track
            self.screen.fill((30, 30, 30))
            scaled_w = int(self.original_image.get_width() * self.zoom)
            scaled_h = int(self.original_image.get_height() * self.zoom)
            
            if scaled_w > 0 and scaled_h > 0:
                scaled_surf = pygame.transform.scale(self.original_image, (scaled_w, scaled_h))
                self.screen.blit(scaled_surf, (self.offset_x, self.offset_y))

            # Draw car start
            start_screen = self.world_to_screen(CAR_START_POS)
            pygame.draw.circle(self.screen, (255, 165, 0), start_screen, 8 * self.zoom)
            
            # Draw existing checkpoints
            for i, checkpoint in enumerate(self.checkpoints):
                p1_world, p2_world = checkpoint
                p1_screen = self.world_to_screen(p1_world)
                p2_screen = self.world_to_screen(p2_world)
                
                pygame.draw.line(self.screen, COLOR_CHECKPOINT, p1_screen, p2_screen, max(2, int(3 * self.zoom)))
                
                # Text
                mid_x = (p1_screen[0] + p2_screen[0]) / 2
                mid_y = (p1_screen[1] + p2_screen[1]) / 2
                lbl = self.font.render(str(i), True, COLOR_TEXT)
                self.screen.blit(lbl, (mid_x, mid_y))

            # Draw rubber band
            if self.current_start_point:
                p1_screen = self.world_to_screen(self.current_start_point)
                mouse_screen = pygame.mouse.get_pos()
                
                pygame.draw.circle(self.screen, COLOR_START_DOT, p1_screen, 6 * self.zoom)
                pygame.draw.line(self.screen, COLOR_ACTIVE, p1_screen, mouse_screen, 2)

            # UI overlay
            info = f"Zoom: {self.zoom:.2f}x | Checkpoints: {len(self.checkpoints)}"
            ui = self.font.render(info, True, (255, 255, 0))
            self.screen.blit(ui, (10, 10))

            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()

    def save_checkpoints(self):
        
        # Round coordinates to integers for cleaner code
        clean_checkpoints = []
        
        for g in self.checkpoints:
            p1 = (int(g[0][0]), int(g[0][1]))
            p2 = (int(g[1][0]), int(g[1][1]))
            clean_checkpoints.append((p1, p2))
            
        print("\n" + "="*40)
        print("COPY THIS INTO settings.py:")
        print("="*40)
        print(f"CHECKPOINTS = {clean_checkpoints}")
        print("="*40 + "\n")

if __name__ == "__main__":
    CheckpointEditor().run()
