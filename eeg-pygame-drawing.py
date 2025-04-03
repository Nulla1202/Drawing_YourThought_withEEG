import socket
import numpy as np
import pygame
import sys
import threading
import time
import json
import os
import random
from datetime import datetime

# EEG Network class with Reinforcement Learning capabilities
class RLEEGNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, weights=None):
        # Initialize with existing weights or generate new random weights
        if weights is not None:
            self.w1 = weights['w1']
            self.b1 = weights['b1']
            self.w2 = weights['w2']
            self.b2 = weights['b2']
        else:
            # Initialize weights randomly
            np.random.seed(42)  # For reproducible results
            self.w1 = np.random.randn(input_size, hidden_size) * 0.1
            self.b1 = np.zeros((1, hidden_size))
            self.w2 = np.random.randn(hidden_size, output_size) * 0.1
            self.b2 = np.zeros((1, output_size))

        # Reinforcement learning parameters
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        # Clip input to prevent overflow in exp
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, x):
        # Reshape input to a row vector
        x = x.reshape(1, -1)

        # Layer 1
        self.layer1_input = x
        self.layer1_z = np.dot(x, self.w1) + self.b1
        self.layer1_output = self.sigmoid(self.layer1_z)

        # Output Layer
        self.layer2_z = np.dot(self.layer1_output, self.w2) + self.b2
        self.layer2_output = self.sigmoid(self.layer2_z)

        return self.layer2_output.flatten()  # Return as a 1D array

    def update_with_reward(self, target, reward_scale=0.1):
        """
        Reinforcement learning function to update weights in real-time
        based on the difference between current output and target.

        target: Expected output (e.g., [1, 0, 0, 1, 0] = On, Up, Right)
        reward_scale: Scaling factor for the reward
        """
        # Calculate output error
        target = np.array(target).reshape(1, -1)
        output_error = target - self.layer2_output

        # Scale error based on reward (positive or negative)
        # Using Mean Squared Error as a basis for reward (lower error is better, but here we use error directly for update)
        # A simple reward could be based on how close the output is to the target
        # For simplicity, we use the error directly, scaled. A more sophisticated reward function could be used.
        reward = -np.sum(output_error ** 2) # Negative MSE, higher is better (closer to 0)
        error_scaled = output_error * reward_scale # Using error directly for gradient calculation

        # Output layer error gradient
        delta_output = error_scaled * self.sigmoid_derivative(self.layer2_output)

        # Hidden layer error gradient
        delta_hidden = np.dot(delta_output, self.w2.T) * self.sigmoid_derivative(self.layer1_output)

        # Update weights and biases
        self.w2 += np.dot(self.layer1_output.T, delta_output) * self.learning_rate
        self.b2 += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
        self.w1 += np.dot(self.layer1_input.T, delta_hidden) * self.learning_rate
        self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate

        return reward # Return the calculated reward

    def get_weights(self):
        return {
            'w1': self.w1.tolist(),
            'b1': self.b1.tolist(),
            'w2': self.w2.tolist(),
            'b2': self.b2.tolist()
        }

    def train_batch(self, training_data):
        """
        Simple batch training method using collected data samples.
        This is a very simplified approach and might need refinement.
        """
        # Calculate average EEG patterns for each state
        state_averages = {}
        min_samples = 1 # Minimum samples required to calculate average
        states_with_data = 0
        for state, samples in training_data.items():
            if len(samples) >= min_samples:  # Only process if there are enough samples
                samples_array = np.array(samples)
                state_averages[state] = np.mean(samples_array, axis=0)
                states_with_data += 1

        # If we have averages for enough required states, adjust weights
        required_states = ['on', 'off', 'up', 'down', 'left', 'right']
        if all(state in state_averages for state in required_states):
            print("Training network with averaged state data...")
            # This is a very simplified weight adjustment - real training would use backpropagation over all samples
            inputs = []
            targets = []

            # Map states to target vectors
            state_to_target = {
                'on':    [1, 0, 0, 0, 0],
                'off':   [0, 0, 0, 0, 0],
                'up':    [1, 1, 0, 0, 0], # Assuming 'on' and direction
                'down':  [1, 0, 1, 0, 0],
                'left':  [1, 0, 0, 1, 0],
                'right': [1, 0, 0, 0, 1]
            }

            for state in required_states:
                inputs.append(state_averages[state])
                # Adjust target based on whether it's a direction (implies 'on') or just on/off
                if state in ['on', 'off']:
                    targets.append(state_to_target[state])
                else: # Directions imply 'on'
                     # Ensure target uses floats
                    target_vector = state_to_target[state]
                    targets.append([float(x) for x in target_vector])


            inputs = np.array(inputs)
            targets = np.array(targets)

            # Simple iterative training (like multiple epochs)
            epochs = 100 # Number of training iterations
            initial_learning_rate = self.learning_rate
            for epoch in range(epochs):
                total_error = 0
                # Adjust learning rate slightly over epochs if needed
                # self.learning_rate = initial_learning_rate / (1 + epoch * 0.01)

                for i in range(len(inputs)):
                    x = inputs[i]
                    target = targets[i]

                    # Forward pass
                    output = self.forward(x)

                    # Calculate error (simple difference for update direction)
                    error = target - output

                    # Update using a simplified backpropagation logic (similar to update_with_reward but without external reward scaling)
                    output_error = error.reshape(1, -1)
                    delta_output = output_error * self.sigmoid_derivative(self.layer2_output)
                    delta_hidden = np.dot(delta_output, self.w2.T) * self.sigmoid_derivative(self.layer1_output)

                    # Update weights and biases
                    self.w2 += np.dot(self.layer1_output.T, delta_output) * self.learning_rate
                    self.b2 += np.sum(delta_output, axis=0, keepdims=True) * self.learning_rate
                    self.w1 += np.dot(self.layer1_input.T, delta_hidden) * self.learning_rate
                    self.b1 += np.sum(delta_hidden, axis=0, keepdims=True) * self.learning_rate

                    total_error += np.sum(error**2)

                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Error: {total_error / len(inputs):.6f}")

            self.learning_rate = initial_learning_rate # Reset learning rate
            print("Batch training finished.")
            return True # Training successful

        else:
            print(f"Not enough data for training. Have data for {states_with_data}/{len(required_states)} required states.")
            return False # Insufficient data

# Parse EEG data from message
def parse_eeg_data(message):
    try:
        # Extract only the numerical part (remove prefix like "/EEG,ss")
        if ",ss" in message:
            data_part = message.split(",ss")[1]
        else:
            data_part = message # Handle different formats

        # Find where the timestamp starts (assume it starts with "2025-")
        timestamp_idx = data_part.find("2025-") # Or adjust based on actual timestamp format
        if timestamp_idx != -1:
            data_part = data_part[:timestamp_idx]

        # Split by semicolon and convert to float
        # Remove null bytes before conversion
        values = [float(val.replace('\x00', '')) for val in data_part.split(";") if val.strip()]

        return np.array(values)
    except Exception as e:
        print(f"Error parsing EEG data: {e}")
        print(f"Raw message causing error: {message}")
        return None

# Interpret neural network outputs
def interpret_outputs(outputs):
    if outputs is None or len(outputs) < 5:
        return {
            "status": "error",
            "message": "Invalid outputs"
        }

    # Output 1: Binary On/Off (0.5 threshold)
    on_off = "On" if outputs[0] >= 0.5 else "Off"

    # Outputs 2-5: Direction values (Up, Down, Left, Right)
    directions = ["up", "down", "left", "right"]
    direction_values = outputs[1:5]

    # Get the top two direction values
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
        "raw_outputs": outputs.tolist(), # Convert numpy array to list for consistency
        "timestamp": datetime.now()
    }

# Function to generate random task for reinforcement learning
def generate_random_task():
    """
    Generates a random training task.
    Task is a combination of [on/off, up, down, left, right].
    Returns: (task_description, target_output_array)
    """
    # Randomly decide on/off state
    on_off = random.choice(["On", "Off"])

    # Randomly decide primary direction
    primary_dir = random.choice(["up", "down", "left", "right"])

    # Create target output array [on/off, up, down, left, right]
    target = [1.0 if on_off == "On" else 0.0, 0.0, 0.0, 0.0, 0.0] # Use floats

    # Set the direction index
    dir_indices = {"up": 1, "down": 2, "left": 3, "right": 4}
    if on_off == "On": # Only set direction if 'On'
        target[dir_indices[primary_dir]] = 1.0

    # Create task description
    if on_off == "On":
        description = f"Focus on {primary_dir.upper()}"
    else:
        description = f"Focus on OFF" # No direction needed if Off

    return description, target

# Global variables
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
current_training_state = None # Tracks the state being trained ('up', 'down', etc.)
nn = None  # Neural network instance

# Reinforcement Learning variables
rl_task_description = ""
rl_target_output = None
rl_task_start_time = 0
rl_reward_history = []
rl_is_training = False
rl_task_duration = 10 # Duration for each RL task in seconds

# Function to save training data to a file
def save_training_data(filename="brain_canvas_training.json"):
    global training_data
    try:
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for state, samples in training_data.items():
            serializable_data[state] = [sample.tolist() for sample in samples]

        with open(filename, 'w') as f:
            json.dump(serializable_data, f, indent=4) # Add indent for readability

        print(f"Training data saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving training data: {e}")
        return False

# Function to load training data from a file
def load_training_data(filename="brain_canvas_training.json"):
    global training_data
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                loaded_data = json.load(f)

            # Convert lists back to numpy arrays
            loaded_count = 0
            training_data = {state: [] for state in training_data} # Clear existing data first
            for state, samples in loaded_data.items():
                if state in training_data: # Only load recognized states
                    training_data[state] = [np.array(sample) for sample in samples]
                    loaded_count += len(samples)

            print(f"Loaded {loaded_count} samples from {filename}")
            return True
        else:
            print(f"Training file {filename} not found.")
            return False
    except Exception as e:
        print(f"Error loading training data: {e}")
        # Reset training_data if loading fails to avoid partial loads
        training_data = {state: [] for state in ['on', 'off', 'up', 'down', 'left', 'right']}
        return False

# Function to save neural network weights
def save_network_weights(nn_instance, filename="brain_canvas_weights.json"):
    if nn_instance is None:
        print("Network not initialized, cannot save weights.")
        return False
    try:
        weights = nn_instance.get_weights()
        with open(filename, 'w') as f:
            json.dump(weights, f, indent=4) # Add indent

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

            # Basic validation of weight dimensions
            if weights['w1'].shape[0] != input_size or \
               weights['w1'].shape[1] != hidden_size or \
               weights['w2'].shape[0] != hidden_size or \
               weights['w2'].shape[1] != output_size:
                print(f"Weight dimensions in {filename} do not match network parameters.")
                print(f"Expected input: {input_size}, hidden: {hidden_size}, output: {output_size}")
                print(f"Loaded w1: {weights['w1'].shape}, w2: {weights['w2'].shape}")
                print("Initializing with random weights instead.")
                return RLEEGNetwork(input_size, hidden_size, output_size)


            nn_instance = RLEEGNetwork(input_size, hidden_size, output_size, weights=weights)
            print(f"Loaded neural network weights from {filename}")
            return nn_instance
        else:
            print(f"Weights file {filename} not found, initializing with random weights.")
            return RLEEGNetwork(input_size, hidden_size, output_size)
    except Exception as e:
        print(f"Error loading neural network weights: {e}")
        print("Initializing with random weights.")
        return RLEEGNetwork(input_size, hidden_size, output_size)


# Function to receive EEG data via UDP
def udp_receiver(ip_address, port_number):
    global latest_outputs, output_timestamp, connection_status, latest_eeg_data, nn
    global rl_is_training, rl_task_description, rl_target_output, rl_task_start_time, rl_reward_history
    global rl_task_duration

    # Create UDP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(1.0) # Set a timeout for recvfrom

    # Initialize neural network settings
    hidden_size = 10 # Can be adjusted
    output_size = 5  # 1 for on/off, 4 for directions (up, down, left, right)
    input_size = None # Determined by the first valid data received

    try:
        s.bind((ip_address, port_number))
        print(f"Waiting for EEG data on UDP {ip_address}:{port_number}")
        connection_status = "waiting"

        while True:
            try:
                # Receive data (max 1024 bytes)
                data, addr = s.recvfrom(1024)
                connection_status = "connected" # Update status once data is received

                # Decode and parse the message
                message = data.decode(errors='ignore') # Ignore decoding errors
                # print(f"\nReceived message from {addr[0]}:{addr[1]}")
                # print(f"Raw message: {message[:100]}...") # Print only the start
                # print(f"Message size: {len(data)} bytes")

                # Parse EEG data
                eeg_data = parse_eeg_data(message)

                if eeg_data is not None and len(eeg_data) > 0:
                    # Store latest EEG data (for training collection)
                    latest_eeg_data = eeg_data

                    # Initialize or re-initialize the network if needed
                    current_input_size = len(eeg_data)
                    if nn is None:
                        input_size = current_input_size
                        print(f"Initializing neural network with input size: {input_size}")
                        # Try loading saved weights first
                        nn = load_network_weights(input_size, hidden_size, output_size)
                    elif nn.w1.shape[0] != current_input_size:
                        # Re-initialize if input size changes unexpectedly
                        input_size = current_input_size
                        print(f"Re-initializing neural network due to changed input size: {input_size}")
                        nn = RLEEGNetwork(input_size, hidden_size, output_size)

                    # Process data through the network
                    output = nn.forward(eeg_data)
                    latest_outputs = output
                    output_timestamp = datetime.now()

                    # Display output and interpretation in console (optional)
                    # interpretation = interpret_outputs(output)
                    # timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    # print(f"[{timestamp_str}] NN Output:")
                    # print(f"  On/Off: {interpretation['on_off']}")
                    # print(f"  Primary Dir: {interpretation['primary_direction']} ({interpretation['primary_value']:.4f})")
                    # print(f"  Secondary Dir: {interpretation['secondary_direction']} ({interpretation['secondary_value']:.4f})")

                    # If in reinforcement learning training mode, update the model in real-time
                    if rl_is_training and rl_target_output is not None:
                        current_time = time.time()
                        # Generate a new task if the time for the current task has elapsed
                        if current_time - rl_task_start_time >= rl_task_duration:
                            rl_task_description, rl_target_output = generate_random_task()
                            rl_task_start_time = current_time
                            print(f"New RL Task: {rl_task_description}, Target: {rl_target_output}")

                        # Update the model and record the reward
                        reward = nn.update_with_reward(rl_target_output)
                        rl_reward_history.append(reward)
                        # print(f"Reward: {reward:.6f}") # Can be too verbose

            except socket.timeout:
                # No data received within the timeout period
                if connection_status == "connected":
                    print("Connection timed out, waiting for data...")
                    connection_status = "waiting"
                latest_outputs = None # Reset output if connection lost
                latest_eeg_data = None
                continue # Continue listening
            except KeyboardInterrupt:
                print("\nStopping UDP receiver...")
                break
            except Exception as e:
                print(f"Error in UDP loop: {e}")
                connection_status = "error"
                # Optionally add a delay before retrying
                time.sleep(1)

    except OSError as e:
        print(f"Failed to bind UDP socket: {e}")
        print("Please check if the IP address is correct and the port is available.")
        connection_status = "error"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        connection_status = "error"
    finally:
        # Close the socket
        s.close()
        print("Socket closed.")
        connection_status = "disconnected"

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
CANVAS_WIDTH = 900
CANVAS_HEIGHT = 600
SIDEBAR_WIDTH = 300
FONT_SIZE = 18 # Slightly smaller font size
BUTTON_WIDTH = 130 # Adjusted button width
BUTTON_HEIGHT = 35 # Adjusted button height
BUTTON_MARGIN = 10

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (230, 230, 230)
RED = (255, 0, 0)
GREEN = (0, 200, 0) # Slightly darker green
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)
ORANGE = (255, 165, 0)
DARK_GRAY = (100, 100, 100)

# Main function
def main():
    global latest_outputs, output_timestamp, connection_status, drawing_history, current_history_index
    global training_data, current_training_state, latest_eeg_data, nn
    global rl_is_training, rl_task_description, rl_target_output, rl_task_start_time, rl_reward_history
    global rl_task_duration
    global cursor_pos, last_draw_pos # Make cursor_pos and last_draw_pos global for reset access

    # Create the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("BrainCanvas - EEG Drawing App (RL Enabled)")

    # Create fonts (Using system default font)
    try:
        font = pygame.font.Font(None, FONT_SIZE)
        large_font = pygame.font.Font(None, FONT_SIZE * 2)
        # font = pygame.font.SysFont("Arial", FONT_SIZE) # Alternative: Specify common font
        # large_font = pygame.font.SysFont("Arial", FONT_SIZE * 2)
    except Exception as e:
        print(f"Font loading error: {e}. Using default font.")
        font = pygame.font.Font(None, FONT_SIZE)
        large_font = pygame.font.Font(None, FONT_SIZE * 2)


    # Create canvas surface
    canvas = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT))
    canvas.fill(WHITE)

    # Add initial canvas state to history
    drawing_history.append(pygame.Surface.copy(canvas))
    current_history_index = 0

    # Start UDP receiver thread
    # ip_address = '192.168.0.247'  # Default IP - CHANGE IF NEEDED
    ip_address = '0.0.0.0' # Listen on all available interfaces
    port_number = 8001  # Default port

    udp_thread = threading.Thread(target=udp_receiver, args=(ip_address, port_number), daemon=True)
    # udp_thread.daemon = True # Set as daemon thread
    udp_thread.start()

    # Drawing state variables
    drawing_mode = "manual"  # "manual", "auto", "training", "rl_training"
    tool = "pencil"  # or "eraser"
    cursor_pos = [CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2] # Cursor for auto/rl modes
    is_drawing = False # For manual drawing
    last_draw_pos = None # For continuous lines in auto/rl mode

    # Training variables
    training_sequence = ['on', 'off', 'up', 'down', 'left', 'right']
    training_index = 0
    training_samples_per_state = 5 # Number of samples per state
    training_current_samples = 0
    training_timer = 0
    training_state = "ready"  # "ready", "collect", "complete"
    training_pause_duration = 3 # Seconds pause between samples

    # Buttons layout (Adjusted for better fit)
    buttons = {
        # Row 1
        "pencil": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT),
        "eraser": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN * 2 + BUTTON_WIDTH, BUTTON_MARGIN, BUTTON_WIDTH, BUTTON_HEIGHT),
        # Row 2
        "undo": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN * 2 + BUTTON_HEIGHT, BUTTON_WIDTH, BUTTON_HEIGHT),
        "clear": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN * 2 + BUTTON_WIDTH, BUTTON_MARGIN * 2 + BUTTON_HEIGHT, BUTTON_WIDTH, BUTTON_HEIGHT),
        # Row 3
        "mode": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN * 3 + BUTTON_HEIGHT * 2, BUTTON_WIDTH * 2 + BUTTON_MARGIN, BUTTON_HEIGHT),
        # Row 4
        "save_img": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN * 4 + BUTTON_HEIGHT * 3, BUTTON_WIDTH, BUTTON_HEIGHT),
        "reset_cursor": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN * 2 + BUTTON_WIDTH, BUTTON_MARGIN * 4 + BUTTON_HEIGHT * 3, BUTTON_WIDTH, BUTTON_HEIGHT), ########## ADDED ##########
        # Row 5: Training Controls
        "start_train": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN * 5 + BUTTON_HEIGHT * 4, BUTTON_WIDTH * 2 + BUTTON_MARGIN, BUTTON_HEIGHT),
        "save_data": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN * 6 + BUTTON_HEIGHT * 5, BUTTON_WIDTH, BUTTON_HEIGHT),
        "load_data": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN * 2 + BUTTON_WIDTH, BUTTON_MARGIN * 6 + BUTTON_HEIGHT * 5, BUTTON_WIDTH, BUTTON_HEIGHT),
        # Row 6: RL Controls
        "start_rl": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN * 7 + BUTTON_HEIGHT * 6, BUTTON_WIDTH * 2 + BUTTON_MARGIN, BUTTON_HEIGHT),
        "save_weights": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN, BUTTON_MARGIN * 8 + BUTTON_HEIGHT * 7, BUTTON_WIDTH, BUTTON_HEIGHT),
        "load_weights": pygame.Rect(CANVAS_WIDTH + BUTTON_MARGIN * 2 + BUTTON_WIDTH, BUTTON_MARGIN * 8 + BUTTON_HEIGHT * 7, BUTTON_WIDTH, BUTTON_HEIGHT),
    }

    # Main loop
    running = True
    clock = pygame.time.Clock()
    last_history_save = time.time()
    history_save_interval = 3 # Save history every 3 seconds in auto/rl mode

    while running:
        # Event processing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check if mouse is on canvas for manual drawing
                    if event.pos[0] < CANVAS_WIDTH and drawing_mode == "manual":
                        is_drawing = True
                        # Start drawing from the click position
                        last_draw_pos = list(event.pos)
                        # Draw a single point if just clicking
                        color = BLACK if tool == "pencil" else WHITE
                        line_width = 2 if tool == "pencil" else 20
                        pygame.draw.circle(canvas, color, last_draw_pos, line_width // 2)

                    # Check if mouse is on buttons
                    for btn_name, btn_rect in buttons.items():
                         if btn_rect.collidepoint(event.pos):
                            # Handle button clicks based on name
                            if btn_name == "pencil":
                                tool = "pencil"
                            elif btn_name == "eraser":
                                tool = "eraser"
                            elif btn_name == "undo":
                                if current_history_index > 0:
                                    current_history_index -= 1
                                    # Load the previous state directly onto the canvas
                                    canvas.blit(drawing_history[current_history_index], (0, 0))
                            elif btn_name == "clear":
                                canvas.fill(WHITE)
                                # Add the clear state to history
                                drawing_history = drawing_history[:current_history_index+1]
                                drawing_history.append(pygame.Surface.copy(canvas))
                                current_history_index = len(drawing_history) - 1
                            elif btn_name == "mode":
                                # Cycle through modes: manual -> auto -> training -> rl_training -> manual
                                if drawing_mode == "manual":
                                    drawing_mode = "auto"
                                    last_draw_pos = None # Reset last draw position for auto mode
                                elif drawing_mode == "auto":
                                    drawing_mode = "training"
                                    training_state = "ready"
                                    training_index = 0
                                    training_current_samples = 0
                                elif drawing_mode == "training":
                                    drawing_mode = "rl_training"
                                    rl_is_training = False # Start RL in stopped state
                                else: # rl_training
                                    drawing_mode = "manual"
                            elif btn_name == "save_img":
                                try:
                                    save_path = f"eeg_drawing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                                    pygame.image.save(canvas, save_path)
                                    print(f"Canvas saved to {save_path}")
                                except Exception as e:
                                    print(f"Error saving image: {e}")
                            ########## ADDED ##########
                            elif btn_name == "reset_cursor":
                                cursor_pos = [CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2]
                                last_draw_pos = None # Stop any drawing line
                                print("Cursor reset to center.")
                            ###########################
                            elif btn_name == "start_train" and drawing_mode == "training":
                                if training_state == "ready":
                                    training_state = "collect"
                                    training_index = 0 # Start from the beginning
                                    training_current_samples = 0
                                    current_training_state = training_sequence[training_index]
                                    training_timer = time.time() # Start timer for the first sample pause
                                    print(f"Starting training data collection for: {current_training_state}")
                                elif training_state == "collect":
                                     # Allow stopping collection manually
                                     training_state = "ready"
                                     current_training_state = None
                                     print("Training data collection stopped.")
                                elif training_state == "complete":
                                    # Train the neural network
                                    if nn is not None:
                                        print("Attempting to train the network with collected data...")
                                        if nn.train_batch(training_data):
                                            # Save the trained weights automatically
                                            print("Training successful. Saving weights...")
                                            save_network_weights(nn)
                                        else:
                                            print("Training failed or not enough data.")
                                    else:
                                        print("Network not initialized. Cannot train.")
                                    # Reset training state after attempt
                                    training_state = "ready"
                                    training_index = 0
                                    training_current_samples = 0
                            elif btn_name == "save_data":
                                save_training_data()
                            elif btn_name == "load_data":
                                if load_training_data():
                                    # Optionally, re-train the network immediately after loading data
                                    if nn is not None:
                                        print("Retraining network with loaded data...")
                                        if nn.train_batch(training_data):
                                             save_network_weights(nn) # Save weights after retraining
                                        else:
                                            print("Retraining failed or not enough data.")
                                    else:
                                        print("Network not initialized. Cannot train.")
                            elif btn_name == "start_rl" and drawing_mode == "rl_training":
                                # Toggle Reinforcement Learning
                                rl_is_training = not rl_is_training
                                if rl_is_training:
                                    if nn is None:
                                        print("Error: Neural network not initialized. Cannot start RL.")
                                        rl_is_training = False
                                    else:
                                        # Start a new random task
                                        rl_task_description, rl_target_output = generate_random_task()
                                        rl_task_start_time = time.time()
                                        rl_reward_history = [] # Reset reward history
                                        last_draw_pos = None # Reset draw position for RL mode
                                        print(f"Starting RL Training. Task: {rl_task_description}, Target: {rl_target_output}")
                                else:
                                    # Save weights when stopping RL training
                                    print("Stopping RL Training.")
                                    if nn is not None:
                                        save_network_weights(nn)
                            elif btn_name == "save_weights":
                                save_network_weights(nn)
                            elif btn_name == "load_weights":
                                if nn is not None:
                                     # Need input size to load weights correctly
                                     loaded_nn = load_network_weights(nn.w1.shape[0], nn.w1.shape[1], nn.w2.shape[1])
                                     if loaded_nn:
                                         nn = loaded_nn # Replace current nn with loaded one
                                else:
                                     print("Network not initialized. Cannot load weights yet.")
                                     print("Please connect EEG first.")


            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1 and is_drawing:  # Left click release
                    is_drawing = False
                    last_draw_pos = None # Reset last position
                    # Save to history after manual drawing stroke
                    if current_history_index < len(drawing_history) - 1:
                        drawing_history = drawing_history[:current_history_index+1]
                    # Make sure not to exceed a reasonable history size
                    if len(drawing_history) > 50: # Limit history size
                         drawing_history.pop(0)
                         current_history_index -=1
                    drawing_history.append(pygame.Surface.copy(canvas))
                    current_history_index = len(drawing_history) - 1


            elif event.type == pygame.MOUSEMOTION:
                if is_drawing and drawing_mode == "manual":
                    # Get current mouse position
                    current_pos = list(event.pos)
                    # Ensure position is within canvas bounds
                    current_pos[0] = max(0, min(CANVAS_WIDTH - 1, current_pos[0]))
                    current_pos[1] = max(0, min(CANVAS_HEIGHT - 1, current_pos[1]))

                    # Draw a line from the last position to the current position
                    if last_draw_pos is not None:
                        color = BLACK if tool == "pencil" else WHITE
                        line_width = 2 if tool == "pencil" else 20
                        pygame.draw.line(canvas, color, last_draw_pos, current_pos, line_width)

                    # Update the last position
                    last_draw_pos = current_pos

        # --- Update logic based on mode ---
        current_time = time.time()

        # Process Auto or RL drawing based on NN output
        if (drawing_mode == "auto" or (drawing_mode == "rl_training" and rl_is_training)) and latest_outputs is not None:
            interpretation = interpret_outputs(latest_outputs)

            if interpretation["status"] == "success":
                # Move cursor based on EEG output
                move_step = 2 # Speed of cursor movement (pixels per frame)

                delta_x = 0
                delta_y = 0

                # Apply movement based on primary direction
                if interpretation["primary_direction"] == "up":
                    delta_y -= move_step * interpretation["primary_value"]
                elif interpretation["primary_direction"] == "down":
                    delta_y += move_step * interpretation["primary_value"]
                elif interpretation["primary_direction"] == "left":
                    delta_x -= move_step * interpretation["primary_value"]
                elif interpretation["primary_direction"] == "right":
                    delta_x += move_step * interpretation["primary_value"]

                # Optional: Add secondary direction influence (can make control jittery)
                # if interpretation["secondary_direction"] == "up":
                #     delta_y -= move_step * interpretation["secondary_value"] * 0.5 # Reduced influence
                # ... and so on for other secondary directions

                # Update cursor position
                cursor_pos[0] += delta_x
                cursor_pos[1] += delta_y

                # Keep cursor within canvas boundaries
                cursor_pos[0] = max(0, min(CANVAS_WIDTH - 1, cursor_pos[0]))
                cursor_pos[1] = max(0, min(CANVAS_HEIGHT - 1, cursor_pos[1]))

                current_draw_pos = (int(cursor_pos[0]), int(cursor_pos[1]))

                # Draw if "On" state is active
                if interpretation["on_off"] == "On":
                    color = BLACK if tool == "pencil" else WHITE
                    line_width = 2 if tool == "pencil" else 20

                    if last_draw_pos is not None:
                        # Draw line from last position to current
                         pygame.draw.line(canvas, color, last_draw_pos, current_draw_pos, line_width)
                    else:
                        # Draw a circle if it's the first point or line wasn't possible
                        pygame.draw.circle(canvas, color, current_draw_pos, line_width // 2)

                    last_draw_pos = current_draw_pos # Update last position for next frame's line
                else:
                    last_draw_pos = None # Stop drawing line if state is "Off"

                # Save history periodically in auto/rl mode
                if current_time - last_history_save > history_save_interval:
                    if current_history_index < len(drawing_history) - 1:
                        drawing_history = drawing_history[:current_history_index+1]
                    # Limit history size
                    if len(drawing_history) > 50:
                         drawing_history.pop(0)
                         current_history_index -=1
                    drawing_history.append(pygame.Surface.copy(canvas))
                    current_history_index = len(drawing_history) - 1
                    last_history_save = current_time

        # Process Training mode
        if drawing_mode == "training" and training_state == "collect":
             # Check if it's time to collect the next sample
             if current_time - training_timer > training_pause_duration:
                 if latest_eeg_data is not None:
                     # Add the current EEG data to the training set
                    training_data[current_training_state].append(latest_eeg_data)
                    training_current_samples += 1
                    print(f"Collected sample {training_current_samples}/{training_samples_per_state} for state: {current_training_state}")

                    # Check if enough samples are collected for the current state
                    if training_current_samples >= training_samples_per_state:
                         training_index += 1
                         training_current_samples = 0

                         # Check if training sequence is complete
                         if training_index >= len(training_sequence):
                             training_state = "complete"
                             current_training_state = None
                             print("Training data collection complete. Ready to train model.")
                         else:
                             # Move to the next state
                             current_training_state = training_sequence[training_index]
                             print(f"Next state: {current_training_state}")
                             # Reset timer for the pause before the next sample collection starts
                             training_timer = current_time
                    else:
                        print("Waiting for EEG data to collect sample...")
                        # Keep timer paused until data arrives
                        training_timer = current_time # Reset timer to wait again


        # --- Drawing section ---
        screen.fill(LIGHT_GRAY)

        # Draw canvas background border
        pygame.draw.rect(screen, BLACK, (0, 0, CANVAS_WIDTH, CANVAS_HEIGHT), 1)

        # Draw the canvas onto the screen
        screen.blit(canvas, (0, 0))

        # Draw cursor in auto or RL mode (only if NN output is available)
        ########## MODIFIED ##########
        if (drawing_mode == "auto" or drawing_mode == "rl_training") and latest_outputs is not None:
             interpretation = interpret_outputs(latest_outputs)
             if interpretation["status"] == "success":
                 # Cursor color indicates On/Off state
                 cursor_color = GREEN if interpretation['on_off'] == 'On' else RED
                 pygame.draw.circle(screen, cursor_color, (int(cursor_pos[0]), int(cursor_pos[1])), 5)
                 pygame.draw.circle(screen, BLACK, (int(cursor_pos[0]), int(cursor_pos[1])), 5, 1) # Border
        elif (drawing_mode == "auto" or drawing_mode == "rl_training"): # Draw placeholder if no output yet
             pygame.draw.circle(screen, YELLOW, (int(cursor_pos[0]), int(cursor_pos[1])), 5)
             pygame.draw.circle(screen, BLACK, (int(cursor_pos[0]), int(cursor_pos[1])), 5, 1) # Border
        ###########################


        # Draw training instructions overlay
        if drawing_mode == "training" and training_state == "collect":
            # Draw semi-transparent overlay
            overlay = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))  # Semi-transparent black
            screen.blit(overlay, (0, 0))

            state_to_instruct = {
                'on': "Think about DRAWING (ON)",
                'off': "Think about NOT drawing (OFF)",
                'up': "Focus on moving UP",
                'down': "Focus on moving DOWN",
                'left': "Focus on moving LEFT",
                'right': "Focus on moving RIGHT"
            }
            instruction = state_to_instruct.get(current_training_state, "Prepare...")

            # Display instruction
            text = large_font.render(instruction, True, WHITE)
            text_rect = text.get_rect(center=(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2 - 20))
            screen.blit(text, text_rect)

            # Display sample counter
            counter_text_str = f"Sample {training_current_samples + 1}/{training_samples_per_state} for {current_training_state.upper()}"
            counter_text = font.render(counter_text_str, True, WHITE)
            counter_rect = counter_text.get_rect(center=(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2 + 30))
            screen.blit(counter_text, counter_rect)

            # Display countdown/waiting message
            time_since_last_action = current_time - training_timer
            if time_since_last_action < training_pause_duration:
                countdown = max(0, int(training_pause_duration - time_since_last_action))
                countdown_text_str = f"Prepare... {countdown}"
            else:
                countdown_text_str = "Capturing..."

            countdown_text = large_font.render(countdown_text_str, True, YELLOW)
            countdown_rect = countdown_text.get_rect(center=(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2 + 70))
            screen.blit(countdown_text, countdown_rect)

        # Draw RL training instructions overlay
        if drawing_mode == "rl_training" and rl_is_training:
            # Draw semi-transparent overlay
            overlay = pygame.Surface((CANVAS_WIDTH, CANVAS_HEIGHT), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))  # Semi-transparent black
            screen.blit(overlay, (0, 0))

            # Display task instruction
            task_text_str = f"Task: {rl_task_description}"
            text = large_font.render(task_text_str, True, WHITE)
            text_rect = text.get_rect(center=(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2 - 20))
            screen.blit(text, text_rect)

            # Display time remaining
            time_elapsed = current_time - rl_task_start_time
            remaining = max(0, int(rl_task_duration - time_elapsed))
            countdown_text_str = f"Time left: {remaining}s"
            countdown_text = large_font.render(countdown_text_str, True, WHITE)
            countdown_rect = countdown_text.get_rect(center=(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2 + 30))
            screen.blit(countdown_text, countdown_rect)

            # Display current reward (if available)
            if rl_reward_history:
                reward_text_str = f"Last Reward: {rl_reward_history[-1]:.4f}" # Show last reward
                reward_text = font.render(reward_text_str, True, YELLOW)
                reward_rect = reward_text.get_rect(center=(CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2 + 70))
                screen.blit(reward_text, reward_rect)

        # --- Draw Sidebar ---
        pygame.draw.rect(screen, DARK_GRAY, (CANVAS_WIDTH, 0, SIDEBAR_WIDTH, SCREEN_HEIGHT))

        # Draw buttons
        # Helper to draw button text
        def draw_button_text(text_str, rect):
             text_surf = font.render(text_str, True, WHITE)
             text_rect = text_surf.get_rect(center=rect.center)
             screen.blit(text_surf, text_rect)

        ########## MODIFIED ##########
        # Draw buttons based on defined layout
        for btn_name, btn_rect in buttons.items():
             # Skip mode-specific buttons if not in that mode
             is_training_button = btn_name in ["start_train", "save_data", "load_data"]
             is_rl_button = btn_name in ["start_rl", "save_weights", "load_weights"]
             # Reset cursor button should be visible in auto and rl modes
             is_cursor_button = btn_name == "reset_cursor"

             if is_training_button and drawing_mode != "training":
                 continue
             if is_rl_button and drawing_mode != "rl_training":
                  continue
             if is_cursor_button and drawing_mode not in ["auto", "rl_training"]:
                  continue

             # Determine button color
             color = GRAY # Default color
             label = btn_name.replace("_", " ").capitalize() # Default label

             if btn_name == "pencil" and tool == "pencil": color = BLUE
             elif btn_name == "eraser" and tool == "eraser": color = BLUE
             elif btn_name == "mode":
                 if drawing_mode == "manual": color = GREEN; label = "Mode: Manual"
                 elif drawing_mode == "auto": color = BLUE; label = "Mode: Auto"
                 elif drawing_mode == "training": color = PURPLE; label = "Mode: Training"
                 elif drawing_mode == "rl_training": color = ORANGE; label = "Mode: RL Training"
             elif btn_name == "start_train":
                 if training_state == "ready": color = GREEN; label = "Start Training"
                 elif training_state == "collect": color = YELLOW; label = "Stop Collection"
                 elif training_state == "complete": color = PURPLE; label = "Train Model"
             elif btn_name == "start_rl":
                 color = RED if rl_is_training else GREEN
                 label = "Stop RL" if rl_is_training else "Start RL"
             elif btn_name == "reset_cursor":
                 color = ORANGE # Give reset button a distinct color
                 label = "Reset Cursor"
             elif btn_name == tool: # Highlight selected tool (only if not already colored)
                 if color == GRAY: color = BLUE

             pygame.draw.rect(screen, color, btn_rect)
             pygame.draw.rect(screen, BLACK, btn_rect, 1) # Add border
             draw_button_text(label, btn_rect)
        ###########################


        # --- Draw Status Information in Sidebar ---
        ########## MODIFIED ##########
        # Find the lowest button Y position to start drawing info below it
        max_button_y = 0
        for btn_rect in buttons.values():
             if btn_rect.bottom > max_button_y:
                 max_button_y = btn_rect.bottom

        info_y_start = max_button_y + BUTTON_MARGIN * 2 # Start below the lowest button
        info_x = CANVAS_WIDTH + 10
        line_height = FONT_SIZE + 5
        ###########################

        # Connection Status
        status_text = f"Status: {connection_status}"
        status_color = GREEN if connection_status == "connected" else YELLOW if connection_status == "waiting" else RED
        status_surf = font.render(status_text, True, status_color)
        screen.blit(status_surf, (info_x, info_y_start))
        info_y_start += line_height

        # NN Output Visualization
        if latest_outputs is not None:
            interpretation = interpret_outputs(latest_outputs)
            if interpretation["status"] == "success":
                # On/Off State
                on_off_text = f"State: {interpretation['on_off']}"
                on_off_color = GREEN if interpretation['on_off'] == 'On' else GRAY
                on_off_surf = font.render(on_off_text, True, on_off_color)
                screen.blit(on_off_surf, (info_x, info_y_start))
                info_y_start += line_height

                # Primary Direction
                primary_text = f"Primary: {interpretation['primary_direction']} ({interpretation['primary_value']:.2f})"
                primary_surf = font.render(primary_text, True, WHITE)
                screen.blit(primary_surf, (info_x, info_y_start))
                info_y_start += line_height // 2
                # Primary Direction Bar
                bar_max_width = SIDEBAR_WIDTH - 20 # Max width for bar
                bar_width = int(interpretation['primary_value'] * bar_max_width)
                pygame.draw.rect(screen, BLUE, (info_x, info_y_start, bar_width, 10))
                pygame.draw.rect(screen, BLACK, (info_x, info_y_start, bar_max_width, 10), 1)
                info_y_start += line_height

                # Secondary Direction
                secondary_text = f"Secondary: {interpretation['secondary_direction']} ({interpretation['secondary_value']:.2f})"
                secondary_surf = font.render(secondary_text, True, WHITE)
                screen.blit(secondary_surf, (info_x, info_y_start))
                info_y_start += line_height // 2
                # Secondary Direction Bar
                bar_width = int(interpretation['secondary_value'] * bar_max_width)
                pygame.draw.rect(screen, GREEN, (info_x, info_y_start, bar_width, 10))
                pygame.draw.rect(screen, BLACK, (info_x, info_y_start, bar_max_width, 10), 1)
                info_y_start += line_height * 1.5 # Add more space

                # Raw Output Bars
                raw_title_surf = font.render("Raw Outputs:", True, WHITE)
                screen.blit(raw_title_surf, (info_x, info_y_start))
                info_y_start += line_height
                bar_x_start = info_x
                # Ensure raw_outputs is a list before len()
                raw_outputs_list = interpretation.get('raw_outputs', [])
                if raw_outputs_list: # Check if list is not empty
                    bar_width_raw = (SIDEBAR_WIDTH - 20) // len(raw_outputs_list) - 5
                    bar_max_height = 40
                    output_labels = ["O/F", "U", "D", "L", "R"] # Labels for outputs
                    for i, value in enumerate(raw_outputs_list):
                        bar_height = int(value * bar_max_height)
                        # Prevent negative height errors
                        bar_height = max(0, bar_height)
                        bar_rect = pygame.Rect(
                            bar_x_start + i * (bar_width_raw + 5),
                            info_y_start + bar_max_height - bar_height, # Draw upwards from bottom line
                            bar_width_raw,
                            bar_height
                        )
                        pygame.draw.rect(screen, (100, 100, 255), bar_rect)
                        # Draw border around the max possible height
                        pygame.draw.rect(screen, BLACK, (bar_x_start + i * (bar_width_raw + 5), info_y_start, bar_width_raw, bar_max_height), 1)
                        # Label
                        if i < len(output_labels): # Avoid index error if output size changes
                           label_surf = font.render(output_labels[i], True, WHITE)
                           label_rect = label_surf.get_rect(center=(bar_rect.centerx, info_y_start + bar_max_height + 10))
                           screen.blit(label_surf, label_rect)
                info_y_start += bar_max_height + 20 # Update y position


        # Training Data Stats (only in Training mode)
        if drawing_mode == "training":
             stats_title = font.render("Training Data Samples:", True, WHITE)
             screen.blit(stats_title, (info_x, info_y_start))
             info_y_start += line_height
             for state, samples in training_data.items():
                 samples_text = font.render(f"  {state}: {len(samples)}", True, WHITE)
                 screen.blit(samples_text, (info_x, info_y_start))
                 info_y_start += line_height

        # RL Stats (only in RL Training mode)
        if drawing_mode == "rl_training":
             rl_title = font.render("RL Status:", True, WHITE)
             screen.blit(rl_title, (info_x, info_y_start))
             info_y_start += line_height

             rl_status_str = f"  Mode: {'Running' if rl_is_training else 'Stopped'}"
             rl_status_surf = font.render(rl_status_str, True, GREEN if rl_is_training else YELLOW)
             screen.blit(rl_status_surf, (info_x, info_y_start))
             info_y_start += line_height

             if rl_is_training and rl_task_description:
                 rl_task_surf = font.render(f"  Task: {rl_task_description}", True, WHITE)
                 screen.blit(rl_task_surf, (info_x, info_y_start))
                 info_y_start += line_height

             if rl_reward_history:
                 last_reward_str = f"  Last Reward: {rl_reward_history[-1]:.4f}"
                 last_reward_surf = font.render(last_reward_str, True, YELLOW)
                 screen.blit(last_reward_surf, (info_x, info_y_start))
                 info_y_start += line_height

                 # Calculate moving average of reward
                 window_size = 20
                 avg_reward = sum(rl_reward_history[-window_size:]) / min(window_size, len(rl_reward_history))
                 avg_reward_str = f"  Avg Reward ({min(window_size, len(rl_reward_history))}): {avg_reward:.4f}"
                 avg_reward_surf = font.render(avg_reward_str, True, YELLOW)
                 screen.blit(avg_reward_surf, (info_x, info_y_start))
                 info_y_start += line_height


        # Update the display
        pygame.display.flip()

        # Limit frame rate
        clock.tick(30) # Lower frame rate might be sufficient and less CPU intensive

    # --- End of main loop ---

    # Save training data and weights before exiting if modified
    print("Exiting application...")
    if any(len(samples) > 0 for samples in training_data.values()):
         print("Saving collected training data...")
         save_training_data()
    if nn is not None:
         print("Saving final network weights...")
         save_network_weights(nn)

    # Quit pygame
    pygame.quit()
    # Signal the UDP thread to stop (though it's a daemon, explicit stop is cleaner if needed)
    # Ideally, use an event or flag to signal the thread to exit its loop.
    print("Exiting.")
    sys.exit()


if __name__ == "__main__":
    main()