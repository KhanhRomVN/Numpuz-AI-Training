import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
import time
from datetime import datetime
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from collections import deque

class PuzzleDataset(Dataset):
    """Dataset for sliding puzzle training data"""
    
    def __init__(self, data_file, board_size=3):
        self.board_size = board_size
        self.total_tiles = board_size * board_size
        
        # Load training data
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        # Filter for correct board size
        self.data = [item for item in self.data if item[4] == board_size]
        
        print(f"Loaded {len(self.data)} training samples for {board_size}x{board_size}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        state, action_logits, value, difficulty, size = self.data[idx]
        
        # Convert to tensors
        state_tensor = torch.FloatTensor(state)
        action_tensor = torch.FloatTensor(action_logits)
        value_tensor = torch.FloatTensor([value])
        difficulty_tensor = torch.LongTensor([difficulty])
        
        return state_tensor, action_tensor, value_tensor, difficulty_tensor

class NumpuzNetwork(nn.Module):
    """Neural network for sliding puzzle"""
    
    def __init__(self, board_size=3, hidden_layers=[256, 128, 64]):
        super(NumpuzNetwork, self).__init__()
        self.board_size = board_size
        self.total_tiles = board_size * board_size
        
        # Input layer
        self.input_size = self.total_tiles
        
        # Hidden layers
        layers = []
        prev_size = self.input_size
        
        for hidden_size in hidden_layers:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output heads
        self.policy_head = nn.Linear(prev_size, self.total_tiles)
        self.value_head = nn.Linear(prev_size, 1)
        self.difficulty_head = nn.Linear(prev_size, 4)  # 4 difficulty classes
        
    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Hidden layers
        x = self.hidden_layers(x)
        
        # Outputs
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        difficulty = self.difficulty_head(x)
        
        return policy, value, difficulty

class Phase1Trainer:
    """Phase 1: Foundation Training for 3x3 puzzles"""
    
    def __init__(self, config):
        self.config = config
        self.board_size = config["board_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = NumpuzNetwork(
            board_size=config["board_size"],
            hidden_layers=config["hidden_layers"]
        ).to(self.device)
        
        # Optimizer and loss functions
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config["learning_rate"]
        )
        
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        self.difficulty_criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'policy_loss': [],
            'value_loss': [], 
            'difficulty_loss': [],
            'train_accuracy': [],
            'epoch_times': []
        }
    
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        difficulty_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (states, action_targets, value_targets, difficulty_targets) in enumerate(pbar):
            # Move to device
            states = states.to(self.device)
            action_targets = action_targets.to(self.device)
            value_targets = value_targets.to(self.device)
            difficulty_targets = difficulty_targets.squeeze().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            policy_pred, value_pred, difficulty_pred = self.model(states)
            
            # Calculate losses
            # Convert action_targets from one-hot to class indices
            _, action_indices = torch.max(action_targets, 1)
            policy_loss_batch = self.policy_criterion(policy_pred, action_indices)
            value_loss_batch = self.value_criterion(value_pred.squeeze(), value_targets.squeeze())
            difficulty_loss_batch = self.difficulty_criterion(difficulty_pred, difficulty_targets)
            
            # Combined loss (weighted)
            loss = (policy_loss_batch + 0.5 * value_loss_batch + 0.2 * difficulty_loss_batch)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            policy_loss += policy_loss_batch.item()
            value_loss += value_loss_batch.item()
            difficulty_loss += difficulty_loss_batch.item()
            
            # Accuracy
            _, predicted = torch.max(policy_pred, 1)
            _, targets = torch.max(action_targets, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += states.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct_predictions/total_samples*100:.2f}%'
            })
        
        avg_loss = total_loss / len(dataloader)
        avg_policy_loss = policy_loss / len(dataloader)
        avg_value_loss = value_loss / len(dataloader)
        avg_difficulty_loss = difficulty_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, avg_policy_loss, avg_value_loss, avg_difficulty_loss, accuracy
    
    def train(self, train_loader, val_loader=None):
        """Main training loop"""
        print(f"\nStarting Phase 1 Training for {self.board_size}x{self.board_size}")
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Learning rate: {self.config['learning_rate']}")
        print("-" * 60)
        
        best_accuracy = 0.0
        start_time = time.time()
        
        for epoch in range(self.config["epochs"]):
            epoch_start = time.time()
            
            # Train
            avg_loss, policy_loss, value_loss, difficulty_loss, accuracy = self.train_epoch(train_loader, epoch)
            epoch_time = time.time() - epoch_start
            
            # Update history
            self.history['train_loss'].append(avg_loss)
            self.history['policy_loss'].append(policy_loss)
            self.history['value_loss'].append(value_loss)
            self.history['difficulty_loss'].append(difficulty_loss)
            self.history['train_accuracy'].append(accuracy)
            self.history['epoch_times'].append(epoch_time)
            
            # Print epoch summary
            print(f"Epoch {epoch+1:3d}/{self.config['epochs']}: "
                  f"Loss: {avg_loss:.4f} | "
                  f"Policy: {policy_loss:.4f} | "
                  f"Value: {value_loss:.4f} | "
                  f"Acc: {accuracy*100:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                self.save_model(f"numpuz_{self.board_size}x{self.board_size}_best.pth")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best accuracy: {best_accuracy*100:.2f}%")
        
        # Save final model
        self.save_model(f"numpuz_{self.board_size}x{self.board_size}_foundation.pth")
        self.save_training_history()
        
        return self.history
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'history': self.history
        }
        
        os.makedirs('models', exist_ok=True)
        torch.save(checkpoint, f"models/{filename}")
        print(f"Model saved: models/{filename}")
    
    def save_training_history(self):
        """Save training history"""
        history_file = f"models/training_history_{self.board_size}x{self.board_size}.json"
        
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Plot training history"""
        plt.figure(figsize=(15, 5))
        
        # Loss plot
        plt.subplot(1, 3, 1)
        plt.plot(self.history['train_loss'], label='Total Loss')
        plt.plot(self.history['policy_loss'], label='Policy Loss')
        plt.plot(self.history['value_loss'], label='Value Loss')
        plt.plot(self.history['difficulty_loss'], label='Difficulty Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(1, 3, 2)
        plt.plot(self.history['train_accuracy'])
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Training Accuracy')
        plt.grid(True)
        
        # Time plot
        plt.subplot(1, 3, 3)
        plt.plot(self.history['epoch_times'])
        plt.xlabel('Epoch')
        plt.ylabel('Time (s)')
        plt.title('Epoch Training Time')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'models/training_curves_{self.board_size}x{self.board_size}.png', dpi=300, bbox_inches='tight')
        plt.close()

def run_phase1():
    """Run Phase 1 training with automatic data generation"""
    
    # Configuration for Phase 1 (3x3)
    config_3x3 = {
        "board_size": 3,
        "training_samples": 50_000,
        "epochs": 200,
        "batch_size": 128,
        "learning_rate": 0.001,
        "model_size": "small",
        "hidden_layers": [256, 128, 64]
    }
    
    print("=" * 80)
    print("PHASE 1: FOUNDATION TRAINING (3x3)")
    print("=" * 80)
    
    # Check if training data exists, if not generate it
    data_file = 'puzzle_data/3x3_training_data.pkl'
    if not os.path.exists(data_file):
        print("Training data not found. Generating dataset...")
        generator = PuzzleGenerator(board_size=3)
        generator.generate_dataset(
            num_samples=config_3x3["training_samples"],
            output_file=data_file
        )
        print("Dataset generation completed!")
    else:
        print(f"Found existing dataset: {data_file}")
    
    # Load training data
    try:
        dataset = PuzzleDataset(data_file, board_size=3)
        print(f"Loaded {len(dataset)} training samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create data loader
    train_loader = DataLoader(
        dataset, 
        batch_size=config_3x3["batch_size"], 
        shuffle=True,
        num_workers=4
    )
    
    # Initialize trainer
    trainer = Phase1Trainer(config_3x3)
    
    # Start training
    history = trainer.train(train_loader)
    
    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    
    return history

class PuzzleGenerator:
    """Generate sliding puzzle training data using BFS solver"""
    
    def __init__(self, board_size=3):
        self.board_size = board_size
        self.total_tiles = board_size * board_size
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    def is_solvable(self, puzzle):
        """Check if puzzle is solvable using inversion count"""
        flat_puzzle = [tile for row in puzzle for tile in row if tile != 0]
        inversions = 0
        for i in range(len(flat_puzzle)):
            for j in range(i + 1, len(flat_puzzle)):
                if flat_puzzle[i] > flat_puzzle[j]:
                    inversions += 1
        
        if self.board_size % 2 == 1:  # odd board size
            return inversions % 2 == 0
        else:  # even board size
            blank_row = self.board_size - (puzzle[-1].index(0) if 0 in puzzle[-1] else 0)
            return (inversions + blank_row) % 2 == 1
    
    def get_blank_position(self, puzzle):
        """Find the position of blank tile (0)"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if puzzle[i][j] == 0:
                    return i, j
        return -1, -1
    
    def move_tile(self, puzzle, direction):
        """Move blank tile in given direction"""
        i, j = self.get_blank_position(puzzle)
        di, dj = direction
        
        new_i, new_j = i + di, j + dj
        if 0 <= new_i < self.board_size and 0 <= new_j < self.board_size:
            # Create new puzzle state
            new_puzzle = [row[:] for row in puzzle]
            new_puzzle[i][j], new_puzzle[new_i][new_j] = new_puzzle[new_i][new_j], new_puzzle[i][j]
            return new_puzzle
        return None
    
    def solve_bfs(self, start_state, max_depth=30):
        """Solve puzzle using BFS with limited depth"""
        target_state = self.create_target_state()
        
        queue = deque([(start_state, [])])
        visited = {self.state_to_tuple(start_state): True}
        
        while queue:
            current_state, path = queue.popleft()
            
            if len(path) > max_depth:
                continue
                
            if current_state == target_state:
                return path
            
            # Try all possible moves
            for move_idx, move in enumerate(self.moves):
                new_state = self.move_tile(current_state, move)
                if new_state and self.state_to_tuple(new_state) not in visited:
                    queue.append((new_state, path + [move_idx]))
                    visited[self.state_to_tuple(new_state)] = True
        
        return None  # No solution found within max_depth
    
    def create_target_state(self):
        """Create solved puzzle state"""
        state = [[0] * self.board_size for _ in range(self.board_size)]
        num = 1
        for i in range(self.board_size):
            for j in range(self.board_size):
                if i == self.board_size - 1 and j == self.board_size - 1:
                    state[i][j] = 0
                else:
                    state[i][j] = num
                    num += 1
        return state
    
    def state_to_tuple(self, state):
        """Convert state to tuple for hashing"""
        return tuple(tuple(row) for row in state)
    
    def generate_training_sample(self, difficulty=10):
        """Generate one training sample with given difficulty"""
        # Start from solved state and make random moves
        state = self.create_target_state()
        
        for _ in range(difficulty):
            valid_moves = []
            for move_idx, move in enumerate(self.moves):
                if self.move_tile(state, move) is not None:
                    valid_moves.append((move_idx, move))
            
            if not valid_moves:
                break
                
            move_idx, move = random.choice(valid_moves)
            state = self.move_tile(state, move)
        
        # Solve from this state to get training data
        solution = self.solve_bfs(state)
        if solution and len(solution) > 0:
            # Convert state to neural network input format
            state_vector = self.state_to_vector(state)
            
            # Create action probabilities (one-hot for first move)
            action_probs = [0.0] * 4  # 4 possible moves
            first_move = solution[0]
            action_probs[first_move] = 1.0
            
            # Value: closer to 1 if easier (shorter solution)
            value = max(0.1, 1.0 - len(solution) / 30.0)
            
            # Difficulty class: 0=easy, 1=medium, 2=hard, 3=expert
            difficulty_class = min(3, len(solution) // 8)
            
            return state_vector, action_probs, value, difficulty_class
        
        return None
    
    def state_to_vector(self, state):
        """Convert puzzle state to neural network input vector"""
        # One-hot encoding for each position
        vector = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                tile = state[i][j]
                # Simple encoding: normalize tile values
                vector.append(tile / (self.total_tiles - 1))
        return vector
    
    def generate_dataset(self, num_samples=50000, output_file='puzzle_data/3x3_training_data.pkl'):
        """Generate complete training dataset"""
        print(f"Generating {num_samples} training samples for {self.board_size}x{self.board_size}...")
        
        os.makedirs('puzzle_data', exist_ok=True)
        
        dataset = []
        difficulties = [5, 10, 15, 20]  # Different difficulty levels
        
        with tqdm(total=num_samples) as pbar:
            while len(dataset) < num_samples:
                difficulty = random.choice(difficulties)
                sample = self.generate_training_sample(difficulty)
                
                if sample is not None:
                    state, action_probs, value, difficulty_class = sample
                    dataset.append((state, action_probs, value, difficulty_class, self.board_size))
                    pbar.update(1)
        
        # Save dataset
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved to {output_file}")
        print(f"Generated {len(dataset)} samples")
        
        return dataset

if __name__ == "__main__":
    run_phase1()