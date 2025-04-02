import socket
import numpy as np
import pygame
import sys
import threading
import time
import json
import os
from datetime import datetime

# Simple neural network class for EEG data
class SimpleEEGNetwork:
    def __init__(self, input_size, hidden_size, output_size, weights=None):
        # Initialize with provided weights or random weights
        if weights is not None:
            self.w1 = weights['w1']
            self.b1 = weights['b1']
            self.w2 = weights['w2']
            self.b2 = weights['b2']
        else:
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
    
    def get_weights(self):
        return {
            'w1': self.w1.tolist(),
            'b1': self.b1.tolist(),
            'w2': self.w2.tolist(),
            'b2': self.b2.tolist()
        }
    
    def train(self, training_data):
        """
        Simple training method using the collected data samples.
        This is a very basic approach - in a real application, you would use
        a more sophisticated training algorithm.
        
        training_data: dict with keys 'on', 'off', 'up', 'down', 'left', 'right'
                      each containing lists of EEG samples
        """
        # Calculate average EEG pattern for each state
        state_averages = {}
        for state, samples in training_data.items():
            if samples:  # Only process if we have samples
                # Convert list of samples to numpy array
                samples_array = np.array(samples)
                # Calculate average
                state_averages[state] = np.mean(samples_array, axis=0)
        
        # If we have enough state averages, use them to adjust weights
        required_states = ['on', 'off', 'up', 'down', 'left', 'right']
        if all(state in state_averages for state in required_states):
            # This is a very simplified approach to weight adjustment
            # In a real application, you'd use proper backpropagation
            
            # For output 1 (on/off), adjust to recognize the difference between on and off states
            on_off_diff = state_averages['on'] - state_averages['off']
            
            # For outputs 2-5 (up, down, left, right), adjust to recognize respective patterns
            direction_diffs = [
                state_averages['up'] - np.mean([state_averages[d] for d in ['down', 'left', 'right']], axis=0),
                state_averages['down'] - np.mean([state_averages[d] for d in ['up', 'left', 'right']], axis=0),
                state_averages['left'] - np.mean([state_averages[d] for d in ['up', 'down', 'right']], axis=0),
                state_averages['right'] - np.mean([state_averages[d] for d in ['up', 'down', 'left']], axis=0)
            ]
            
            # Very simple weight adjustment - in reality, you'd need a proper training algorithm
            # This is just a proof of concept
            input_size = self.w1.shape[0]
            hidden_size = self.w1.shape[1]
            
            # Initialize new weights - we'll completely replace the random weights
            self.w1 = np.random.randn(input_size, hidden_size) * 0.01
            
            # Incorporate the state differences into the weights
            for i in range(input_size):
                # For on/off detection
                if i < len(on_off_diff):
                    self.w1[i, 0] += on_off_diff[i] * 0.1
                
                # For direction detection
                for j, diff in enumerate(direction_diffs):
                    if i < len(diff):
                        self.w1[i, j+1] += diff[i] * 0.1
            
            # Output layer weights - simplified approach
            self.w2 = np.zeros_like(self.w2)
            for i in range(5):  # 5 outputs
                self.w2[i, i] = 1.0  # Direct mapping
            
            return True  # Training successful
        
        return False  # Not enough data

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
latest_eeg_data = None
training_data = {
    'on': [],
    'off': [],
    'up': [],
    'down': [],
    'left': [],
    'right': []
}
current_training_state = None
nn = None  # Neural network instance

# Function to save training data to file
def save_training_data(filename="brain_canvas_training.json"):
    global training_data
    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for state, samples in training_data.items():
            serializable_data[state] = [sample.tolist() for sample in samples]
        
        with open(filename, 'w') as f:
            json.dump(serializable_data, f)
        
        print(f"Training data saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving training data: {e}")
        return False

# Function to load training data from file
def load_training_data(filename="brain_canvas_training.json"):
    global training_data
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                loaded_data = json.load(f)
            
            # Convert lists back to numpy arrays
            for state, samples in loaded_data.items():
                training_data[state] = [np.array(sample) for sample in samples]
            
            print(f"Training data loaded from {filename}")
            return True
        else:
            print(f"Training file {filename} not found")
            return False
    except Exception as e:
        print(f"Error loading training data: {e}")
        return False

# Function to save neural network weights
def save_network_weights(nn, filename="brain_canvas_weights.json"):
    try:
        weights = nn.get_weights()
        with open(filename, 'w') as f:
            json.dump(weights, f)
        
        print(f"Neural network weights saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving neural network weights: {e}")
        return False

# Function to load neural network weights
def load_network_weights(input_size, hidden_size, output_size, filename="brain_canvas_weights.json"):
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                weights = json.load(f)
            
            # Convert lists back to numpy arrays
            for key in weights:
                weights[key] = np.array(weights[key])
            
            nn = SimpleEEGNetwork(input_size, hidden_size, output_size, weights)
            print(f"Neural network weights loaded from {filename}")
            return nn
        else:
            print(f"Weights file {filename} not found, initializing with random weights")
            return SimpleEEGNetwork(input_size, hidden_size, output_size)
    except Exception as e:
        print(f"Error loading neural network weights: {e}")
        return SimpleEEGNetwork(input_size, hidden_size, output_size)

# Function to receive EEG data via UDP
def udp_receiver(ip_address, port_number):
    global latest_outputs, output_timestamp, connection_status, latest_eeg_data, nn
    
    # Create UDP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    # Initialize neural network settings
    hidden_size = 10
    output_size = 5  # 1 for on/off, 4 for directions (up, down, left, right)
    
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
                # Store the latest EEG data (for training)
                latest_eeg_data = eeg_data
                
                # Initialize network if this is the first data or if input size changed
                if nn is None:
                    input_size = len(eeg_data)
                    print(f"Initializing neural network with input size: {input_size}")
                    
                    # Try to load saved weights first
                    nn = load_network_weights(input_size, hidden_size, output_size)
                elif nn.w1.shape[0] != len(eeg_data):
                    # If input size has changed, reinitialize the network
                    input_size = len(eeg_data)
                    print(f"Reinitializing neural network with new input size: {input_size}")
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
PURPLE = (128, 0, 128)

# Main function
def main():
    global latest_outputs, output_timestamp, connection_status, drawing_history, current_history_index
    global training_data, current_training_state, latest_eeg_data, nn
    
    # Create screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BrainCanvas - EEG Drawing App")
    
    # Create fonts
    font = pygame.font.SysFont(None, FONT_SIZE)
    large_font = pygame.font.SysFont(None, FONT_SIZE * 2)
    
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
    drawing_mode = "manual"  # "manual", "auto", or "training"
    tool = "pencil"  # or "eraser"
    cursor_pos = [CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2]
    is_drawing = False
    
    # Training variables
    training_sequence = ['on', 'off', 'up', 'down', 'left', 'right']
    training_index = 0
    training_samples_per_state = 5
    training_current_samples = 0
    training_timer = 0
    training_state = "ready"  # "ready", "collect", "complete"
    
    # Buttons
    buttons = {
        "pencil": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT),
        "eraser": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN + BUTTON_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT),
        "undo": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*2 + BUTTON_HEIGHT, BUTTON_WIDTH, BUTTON_HEIGHT),
        "clear": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN + BUTTON_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*2 + BUTTON_HEIGHT, BUTTON_WIDTH, BUTTON_HEIGHT),
        "mode": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*3 + BUTTON_HEIGHT*2, BUTTON_WIDTH*2 + BUTTON_MARGIN, BUTTON_HEIGHT),
        "save": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*4 + BUTTON_HEIGHT*3, BUTTON_WIDTH, BUTTON_HEIGHT),
        "train_start": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*5 + BUTTON_HEIGHT*4, BUTTON_WIDTH*2 + BUTTON_MARGIN, BUTTON_HEIGHT),
        "save_training": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*6 + BUTTON_HEIGHT*5, BUTTON_WIDTH, BUTTON_HEIGHT),
        "load_training": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN + BUTTON_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN*6 + BUTTON_HEIGHT*5, BUTTON_WIDTH, BUTTON_HEIGHT),
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
                        # Cycle through modes: manual -> auto -> training -> manual
                        if drawing_mode == "manual":
                            drawing_mode = "auto"
                        elif drawing_mode == "auto":
                            drawing_mode = "training"
                            training_state = "ready"
                            training_index = 0
                            training_current_samples = 0
                        else:
                            drawing_mode = "manual"
                    elif buttons["save"].collidepoint(event.pos):
                        save_path = f"eeg_drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                        pygame.image.save(canvas, save_path)
                        print(f"Canvas saved to {save_path}")
                    elif buttons["train_start"].collidepoint(event.pos) and drawing_mode == "training":
                        if training_state == "ready":
                            training_state = "collect"
                            training_timer = time.time()
                        elif training_state == "complete":
                            # Train the neural network
                            if nn is not None and nn.train(training_data):
                                # Save the trained weights
                                save_network_weights(nn)
                                training_state = "ready"
                                training_index = 0
                                training_current_samples = 0
                    elif buttons["save_training"].collidepoint(event.pos):
                        save_training_data()
                    elif buttons["load_training"].collidepoint(event.pos):
                        if load_training_data():
                            # Re-train the network with loaded data
                            if nn is not None:
                                nn.train(training_data)
                                save_network_weights(nn)
            
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
        
        # Handle training mode
        if drawing_mode == "training" and training_state == "collect" and latest_eeg_data is not None:
            current_state = training_sequence[training_index]
            current_training_state = current_state
            
            # Collect data samples
            current_time = time.time()
            # Allow 3 seconds between samples to stabilize
            if current_time - training_timer > 3:
                # Add the current EEG data to the training set
                training_data[current_state].append(latest_eeg_data)
                training_current_samples += 1
                training_timer = current_time
                
                # Check if we have enough samples for the current state
                if training_current_samples >= training_samples_per_state:
                    training_index += 1
                    training_current_samples = 0
                    
                    # Check if training is complete
                    if training_index >= len(training_sequence):
                        training_state = "complete"
                        current_training_state = None
        
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
        
        # Draw training instructions in training mode
        if drawing_mode == "training" and training_state == "collect":
            # Draw an overlay with the current instruction
            overlay = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))  # Semi-transparent black
            screen.blit(overlay, (0, 0))
            
            current_state = training_sequence[training_index]
            
            # Display the instruction
            if current_state == "on":
                instruction = "Think about DRAWING (On state)"
            elif current_state == "off":
                instruction = "Think about NOT DRAWING (Off state)"
            else:
                instruction = f"Think about moving {current_state.upper()} direction"
            
            text = large_font.render(instruction, True, WHITE)
            text_rect = text.get_rect(center=(CANVAS_WIDTH//2, CANVAS_HEIGHT//2))
            screen.blit(text, text_rect)
            
            # Display the sample counter
            counter_text = font.render(f"Sample {training_current_samples + 1}/{training_samples_per_state} for {current_state}", True, WHITE)
            counter_rect = counter_text.get_rect(center=(CANVAS_WIDTH//2, CANVAS_HEIGHT//2 + 40))
            screen.blit(counter_text, counter_rect)
            
            # Display the countdown
            countdown = max(0, int(3 - (time.time() - training_timer)))
            countdown_text = large_font.render(str(countdown), True, WHITE)
            countdown_rect = countdown_text.get_rect(center=(CANVAS_WIDTH//2, CANVAS_HEIGHT//2 + 80))
            screen.blit(countdown_text, countdown_rect)
        
        # Draw sidebar
        pygame.draw.rect(screen, GRAY, (CANVAS_WIDTH, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT))
        
        # Draw buttons
        for btn_name, btn_rect in buttons.items():
            # Skip training-specific buttons if not in training mode
            if btn_name.startswith("train_") and drawing_mode != "training":
                continue
            
            if btn_name == "mode":
                if drawing_mode == "auto":
                    color = BLUE
                elif drawing_mode == "training":
                    color = PURPLE
                else:
                    color = GREEN
            elif btn_name == "train_start":
                if training_state == "ready":
                    color = GREEN
                elif training_state == "collect":
                    color = YELLOW
                elif training_state == "complete":
                    color = PURPLE
                else:
                    color = GRAY
            elif btn_name == tool:
                color = BLUE
            else:
                color = GRAY
            
            pygame.draw.rect(screen, color, btn_rect)
            
            # Button text
            if btn_name == "mode":
                label = f"Mode: {drawing_mode.upper()}"
            elif btn_name == "train_start":
                if training_state == "ready":
                    label = "Start Training"
                elif training_state == "collect":
                    label = "Training..."
                elif training_state == "complete":
                    label = "Save Model"
                else:
                    label = "Train"
            else:
                label = btn_name.replace("_", " ").capitalize()
            
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
        
        # Draw training data statistics in training mode
        if drawing_mode == "training":
            y_pos = 250
            training_stats_title = font.render("Training Data Stats:", True, BLACK)
            screen.blit(training_stats_title, (CANVAS_WIDTH + 10, y_pos))
            y_pos += 25
            
            for state, samples in training_data.items():
                samples_text = font.render(f"{state}: {len(samples)} samples", True, BLACK)
                screen.blit(samples_text, (CANVAS_WIDTH + 10, y_pos))
                y_pos += 20
        
        # Update the display
        pygame.display.flip()
        
        # Cap the frame rate
        clock.tick(60)
    
    # Save training data before exit if training was done
    if any(len(samples) > 0 for samples in training_data.values()):
        save_training_data()
    
    # Quit pygame
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()