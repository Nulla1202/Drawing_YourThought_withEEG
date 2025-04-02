import socket
import numpy as np
import pygame
import sys
import threading
import time
from datetime import datetime

# Simple neural network class for EEG data
class SimpleEEGNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Random initialization of weights
        np.random.seed(42)  # For reproducible results
        self.w1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.w2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def forward(self, x):
        # Reshape input to ensure it's a row vector
        x = x.reshape(1, -1)
        
        # First layer
        z1 = np.dot(x, self.w1) + self.b1
        a1 = self.sigmoid(z1)
        
        # Output layer
        z2 = np.dot(a1, self.w2) + self.b2
        a2 = self.sigmoid(z2)
        
        return a2.flatten()  # Return as 1D array

# Parse EEG data from the message
def parse_eeg_data(message):
    try:
        # Extract only the numerical part (remove "/EEG,ss" prefix)
        if ",ss" in message:
            data_part = message.split(",ss")[1]
        else:
            data_part = message  # In case the format is different
        
        # Find where the timestamp starts (assuming it starts with "2025-")
        timestamp_idx = data_part.find("2025-")
        if timestamp_idx != -1:
            data_part = data_part[:timestamp_idx]
        
        # Split by semicolons and convert to float
        # Remove any null bytes before converting to float
        values = [float(val.replace('\x00', '')) for val in data_part.split(";") if val.strip()]
        
        return np.array(values)
    except Exception as e:
        print(f"Error parsing EEG data: {e}")
        return None

# Interpret neural network outputs
def interpret_outputs(outputs):
    if outputs is None or len(outputs) < 5:
        return {
            "status": "error",
            "message": "Invalid outputs"
        }
    
    # Output 1: Binary on/off (0.5 threshold)
    on_off = "On" if outputs[0] >= 0.5 else "Off"
    
    # Output 2-5: Directional values (up, down, left, right)
    directions = ["up", "down", "left", "right"]
    direction_values = outputs[1:5]
    
    # Get the two highest direction values
    sorted_indices = np.argsort(direction_values)[::-1]
    primary_direction = directions[sorted_indices[0]]
    secondary_direction = directions[sorted_indices[1]]
    primary_value = direction_values[sorted_indices[0]]
    secondary_value = direction_values[sorted_indices[1]]
    
    return {
        "status": "success",
        "on_off": on_off,
        "primary_direction": primary_direction,
        "primary_value": primary_value,
        "secondary_direction": secondary_direction,
        "secondary_value": secondary_value,
        "raw_outputs": outputs,
        "timestamp": datetime.now()
    }

# Global variables to store EEG data
latest_outputs = None
output_timestamp = None
connection_status = "disconnected"
drawing_history = []
current_history_index = -1

# Function to receive EEG data via UDP
def udp_receiver(ip_address, port_number):
    global latest_outputs, output_timestamp, connection_status
    
    # Create UDP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Initialize neural network settings
    hidden_size = 10
    output_size = 5  # 1 for on/off, 4 for directions (up, down, left, right)
    nn = None
    
    try:
        s.bind((ip_address, port_number))
        print(f"Listening for EEG data on UDP {ip_address}:{port_number}")
        connection_status = "waiting"
        
        while True:
            # Receive data (up to 1024 bytes)
            data, addr = s.recvfrom(1024)
            
            # Decode and parse the message
            message = data.decode()
            print(f"\nReceived message from {addr[0]}:{addr[1]}")
            print(f"Raw message: {message}")
            print(f"Message size: {len(data)} bytes")
            
            # Parse EEG data
            eeg_data = parse_eeg_data(message)
            
            if eeg_data is not None and len(eeg_data) > 0:
                # Initialize network if this is the first data or if input size changed
                if nn is None or nn.w1.shape[0] != len(eeg_data):
                    input_size = len(eeg_data)
                    print(f"Initializing neural network with input size: {input_size}")
                    nn = SimpleEEGNetwork(input_size, hidden_size, output_size)
                
                # Process the data
                output = nn.forward(eeg_data)
                latest_outputs = output
                output_timestamp = datetime.now()
                connection_status = "connected"
                
                # Print the output and interpretation
                interpretation = interpret_outputs(output)
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                print(f"[{timestamp}] Neural Network Output:")
                print(f"  On/Off: {interpretation['on_off']}")
                print(f"  Primary Direction: {interpretation['primary_direction']} ({interpretation['primary_value']:.4f})")
                print(f"  Secondary Direction: {interpretation['secondary_direction']} ({interpretation['secondary_value']:.4f})")
            
    except KeyboardInterrupt:
        print("\nStopping UDP server...")
    except Exception as e:
        print(f"Error: {e}")
        connection_status = "error"
    finally:
        # Close socket
        s.close()
        print("Socket closed")

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
CANVAS_WIDTH = 900
CANVAS_HEIGHT = 600
SIDEBAR_WIDTH = 300
FONT_SIZE = 20
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 40
BUTTON_MARGIN = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)

# Main function
def main():
    global latest_outputs, output_timestamp, connection_status, drawing_history, current_history_index
    
    # Create screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("EEG Drawing App")
    
    # Create font
    font = pygame.font.SysFont(None, FONT_SIZE)
    
    # Create canvas surface
    canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
    canvas.fill(WHITE)
    
    # Add initial canvas state to history
    drawing_history.append(pygame.Surface.copy(canvas))
    current_history_index = 0
    
    # Start UDP receiver thread
    ip_address = '192.168.0.247'  # Default IP
    port_number = 8001  # Default port
    
    udp_thread = threading.Thread(target=udp_receiver, args=(ip_address, port_number))
    udp_thread.daemon = True
    udp_thread.start()
    
    # Drawing state variables
    drawing_mode = "manual"  # or "auto"
    tool = "pencil"  # or "eraser"
    cursor_pos = [CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2]
    is_drawing = False
    
    # Buttons
    buttons = {
        "pencil": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT),
        "eraser": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN + BUTTON_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT),
        "undo": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*2 + BUTTON_HEIGHT, BUTTON_WIDTH, BUTTON_HEIGHT),
        "clear": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN + BUTTON_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*2 + BUTTON_HEIGHT, BUTTON_WIDTH, BUTTON_HEIGHT),
        "mode": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*3 + BUTTON_HEIGHT*2, BUTTON_WIDTH*2 + BUTTON_MARGIN, BUTTON_HEIGHT),
        "save": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*4 + BUTTON_HEIGHT*3, BUTTON_WIDTH, BUTTON_HEIGHT),
    }
    
    # Main loop
    running = True
    clock = pygame.time.Clock()
    last_auto_save = time.time()
    
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check if mouse is on canvas
                    if event.pos[0] < CANVAS_WIDTH and drawing_mode == "manual":
                        is_drawing = True
                        cursor_pos = list(event.pos)
                    
                    # Check if mouse is on buttons
                    if buttons["pencil"].collidepoint(event.pos):
                        tool = "pencil"
                    elif buttons["eraser"].collidepoint(event.pos):
                        tool = "eraser"
                    elif buttons["undo"].collidepoint(event.pos):
                        if current_history_index > 0:
                            current_history_index -= 1
                            canvas = pygame.Surface.copy(drawing_history[current_history_index])
                    elif buttons["clear"].collidepoint(event.pos):
                        canvas.fill(WHITE)
                        drawing_history = [pygame.Surface.copy(canvas)]
                        current_history_index = 0
                    elif buttons["mode"].collidepoint(event.pos):
                        drawing_mode = "auto" if drawing_mode == "manual" else "manual"
                    elif buttons["save"].collidepoint(event.pos):
                        save_path = f"eeg_drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        pygame.image.save(canvas, save_path)
                        print(f"Canvas saved to {save_path}")
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and is_drawing:  # Left click release
                    is_drawing = False
                    # Save to history
                    if current_history_index < len(drawing_history) - 1:
                        drawing_history = drawing_history[:current_history_index+1]
                    drawing_history.append(pygame.Surface.copy(canvas))
                    current_history_index = len(drawing_history) - 1
            
            elif event.type == pygame.MOUSEMOTION:
                if is_drawing and drawing_mode == "manual":
                    # Get current position
                    x, y = event.pos
                    if x >= CANVAS_WIDTH:
                        x = CANVAS_WIDTH - 1
                    
                    # Draw line from previous position to current position
                    color = BLACK if tool == "pencil" else WHITE
                    line_width = 2 if tool == "pencil" else 20
                    pygame.draw.line(canvas, color, cursor_pos, (x, y), line_width)
                    cursor_pos = [x, y]
        
        # Handle auto drawing mode
        if drawing_mode == "auto" and latest_outputs is not None:
            interpretation = interpret_outputs(latest_outputs)
            
            if interpretation["status"] == "success":
                # Move cursor based on EEG output
                move_step = 1  # pixels per frame
                
                # Primary direction
                if interpretation["primary_direction"] == "up":
                    cursor_pos[1] -= move_step * interpretation["primary_value"]
                elif interpretation["primary_direction"] == "down":
                    cursor_pos[1] += move_step * interpretation["primary_value"]
                elif interpretation["primary_direction"] == "left":
                    cursor_pos[0] -= move_step * interpretation["primary_value"]
                elif interpretation["primary_direction"] == "right":
                    cursor_pos[0] += move_step * interpretation["primary_value"]
                
                # Secondary direction
                if interpretation["secondary_direction"] == "up":
                    cursor_pos[1] -= move_step * interpretation["secondary_value"]
                elif interpretation["secondary_direction"] == "down":
                    cursor_pos[1] += move_step * interpretation["secondary_value"]
                elif interpretation["secondary_direction"] == "left":
                    cursor_pos[0] -= move_step * interpretation["secondary_value"]
                elif interpretation["secondary_direction"] == "right":
                    cursor_pos[0] += move_step * interpretation["secondary_value"]
                
                # Ensure cursor stays within canvas bounds
                cursor_pos[0] = max(0, min(CANVAS_WIDTH-1, cursor_pos[0]))
                cursor_pos[1] = max(0, min(CANVAS_HEIGHT-1, cursor_pos[1]))
                
                # Draw if "On"
                if interpretation["on_off"] == "On":
                    color = BLACK if tool == "pencil" else WHITE
                    line_width = 2 if tool == "pencil" else 20
                    
                    # Draw a dot at cursor position
                    pygame.draw.circle(canvas, color, (int(cursor_pos[0]), int(cursor_pos[1])), line_width // 2)
                
                # Periodically save in auto mode
                current_time = time.time()
                if current_time - last_auto_save > 5:  # Save every 5 seconds in auto mode
                    if current_history_index < len(drawing_history) - 1:
                        drawing_history = drawing_history[:current_history_index+1]
                    drawing_history.append(pygame.Surface.copy(canvas))
                    current_history_index = len(drawing_history) - 1
                    last_auto_save = current_time
        
        # Draw the screen
        screen.fill(LIGHT_GRAY)
        
        # Draw canvas background border
        pygame.draw.rect(screen, BLACK, (0, 0, CANVAS_WIDTH, CANVAS_HEIGHT), 1)
        
        # Draw canvas
        screen.blit(canvas, (0, 0))
        
        # Draw cursor in auto mode
        if drawing_mode == "auto":
            cursor_color = RED if latest_outputs is not None and latest_outputs[0] >= 0.5 else YELLOW
            pygame.draw.circle(screen, cursor_color, (int(cursor_pos[0]), int(cursor_pos[1])), 5, 1)
        
        # Draw sidebar
        pygame.draw.rect(screen, GRAY, (CANVAS_WIDTH, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT))
        
        # Draw buttons
        for btn_name, btn_rect in buttons.items():
            if btn_name == "mode":
                color = GREEN if drawing_mode == "auto" else BLUE
            elif btn_name == tool:
                color = BLUE
            else:
                color = GRAY
            
            pygame.draw.rect(screen, color, btn_rect)
            
            # Button text
            if btn_name == "mode":
                label = f"Mode: {drawing_mode.upper()}"
            else:
                label = btn_name.capitalize()
            
            text = font.render(label, True, WHITE)
            text_rect = text.get_rect(center=btn_rect.center)
            screen.blit(text, text_rect)
        
        # Draw connection status
        status_text = f"Status: {connection_status}"
        status_color = GREEN if connection_status == "connected" else YELLOW if connection_status == "waiting" else RED
        status_surf = font.render(status_text, True, status_color)
        screen.blit(status_surf, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 150))
        
        # Draw NN output visualization
        if latest_outputs is not None:
            interpretation = interpret_outputs(latest_outputs)
            
            if interpretation["status"] == "success":
                # Draw "On/Off" status
                on_off_text = f"State: {interpretation['on_off']}"
                on_off_color = GREEN if interpretation['on_off'] == 'On' else GRAY
                on_off_surf = font.render(on_off_text, True, on_off_color)
                screen.blit(on_off_surf, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 120))
                
                # Draw primary direction
                primary_text = f"Primary: {interpretation['primary_direction']} ({interpretation['primary_value']:.2f})"
                primary_surf = font.render(primary_text, True, BLACK)
                screen.blit(primary_surf, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 90))
                
                # Draw primary direction bar
                bar_width = int(interpretation['primary_value'] * 200)
                pygame.draw.rect(screen, BLUE, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 70, bar_width, 10))
                pygame.draw.rect(screen, BLACK, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 70, 200, 10), 1)
                
                # Draw secondary direction
                secondary_text = f"Secondary: {interpretation['secondary_direction']} ({interpretation['secondary_value']:.2f})"
                secondary_surf = font.render(secondary_text, True, BLACK)
                screen.blit(secondary_surf, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 50))
                
                # Draw secondary direction bar
                bar_width = int(interpretation['secondary_value'] * 200)
                pygame.draw.rect(screen, GREEN, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 30, bar_width, 10))
                pygame.draw.rect(screen, BLACK, (CANVAS_WIDTH + 10, CANVAS_HEIGHT - 30, 200, 10), 1)
                
                # Draw raw values
                for i, value in enumerate(interpretation['raw_outputs']):
                    bar_height = int(value * 60)
                    bar_rect = pygame.Rect(
                        CANVAS_WIDTH + 10 + i * 40, 
                        CANVAS_HEIGHT - 200 - bar_height, 
                        30, 
                        bar_height
                    )
                    pygame.draw.rect(screen, (100, 100, 255), bar_rect)
                    pygame.draw.rect(screen, BLACK, bar_rect, 1)
                    
                    # Label
                    output_label = font.render(str(i), True, BLACK)
                    screen.blit(output_label, (CANVAS_WIDTH + 20 + i * 40, CANVAS_HEIGHT - 190))
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(60)
    
    # Quit pygame
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
