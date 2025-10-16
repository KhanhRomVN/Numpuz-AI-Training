import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import json
import time
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import torch.nn.functional as F
import random
from collections import deque
import heapq

class PuzzleGenerator:
    """Generate 4x4 sliding puzzle training data using IDA* solver"""
    
    def __init__(self, board_size=4):
        self.board_size = board_size
        self.total_tiles = board_size * board_size
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        self.move_names = ['up', 'down', 'left', 'right']
        
        # Precompute target positions for Manhattan distance
        self.target_positions = {}
        for num in range(1, self.total_tiles):
            i = (num - 1) // self.board_size
            j = (num - 1) % self.board_size
            self.target_positions[num] = (i, j)
        self.target_positions[0] = (self.board_size - 1, self.board_size - 1)
    
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
    
    def manhattan_distance(self, state):
        """Calculate Manhattan distance heuristic"""
        distance = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                tile = state[i][j]
                if tile != 0:
                    target_i, target_j = self.target_positions[tile]
                    distance += abs(i - target_i) + abs(j - target_j)
        return distance
    
    def linear_conflict(self, state):
        """Calculate linear conflict heuristic (enhances Manhattan)"""
        conflict = 0
        
        # Check rows
        for i in range(self.board_size):
            row = state[i]
            for j in range(self.board_size):
                tile1 = row[j]
                if tile1 == 0:
                    continue
                target_i1, target_j1 = self.target_positions[tile1]
                if target_i1 == i:  # Tile belongs in this row
                    for k in range(j + 1, self.board_size):
                        tile2 = row[k]
                        if tile2 == 0:
                            continue
                        target_i2, target_j2 = self.target_positions[tile2]
                        if target_i2 == i and target_j1 > target_j2:
                            conflict += 2
        
        # Check columns
        for j in range(self.board_size):
            for i in range(self.board_size):
                tile1 = state[i][j]
                if tile1 == 0:
                    continue
                target_i1, target_j1 = self.target_positions[tile1]
                if target_j1 == j:  # Tile belongs in this column
                    for k in range(i + 1, self.board_size):
                        tile2 = state[k][j]
                        if tile2 == 0:
                            continue
                        target_i2, target_j2 = self.target_positions[tile2]
                        if target_j2 == j and target_i1 > target_i2:
                            conflict += 2
        
        return conflict
    
    def heuristic(self, state):
        """Combined heuristic: Manhattan distance + linear conflict"""
        return self.manhattan_distance(state) + self.linear_conflict(state)
    
    def ida_star_solve(self, start_state, max_depth=80):
        """Solve using IDA* algorithm with Manhattan + Linear Conflict heuristic"""
        target_state = self.create_target_state()
        
        if start_state == target_state:
            return []
        
        # Initial bound
        bound = self.heuristic(start_state)
        path = []
        
        while True:
            result = self._ida_star_search(start_state, 0, bound, path, set(), target_state, max_depth)
            if result == "FOUND":
                return path
            if result == float('inf'):
                return None  # No solution
            bound = result
    
    def _ida_star_search(self, state, g, bound, path, visited, target_state, max_depth):
        """IDA* recursive search"""
        f = g + self.heuristic(state)
        
        if f > bound:
            return f
        if state == target_state:
            return "FOUND"
        if g >= max_depth:
            return float('inf')
        
        state_tuple = self.state_to_tuple(state)
        if state_tuple in visited:
            return float('inf')
        
        visited.add(state_tuple)
        min_cost = float('inf')
        
        # Try all possible moves
        for move_idx, move in enumerate(self.moves):
            new_state = self.move_tile(state, move)
            if new_state and self.state_to_tuple(new_state) not in visited:
                path.append(move_idx)
                result = self._ida_star_search(new_state, g + 1, bound, path, visited, target_state, max_depth)
                
                if result == "FOUND":
                    return "FOUND"
                if result < min_cost:
                    min_cost = result
                
                path.pop()
        
        visited.remove(state_tuple)
        return min_cost
    
    def solve_bfs_optimized(self, start_state, max_depth=50):
        """Optimized BFS with heuristic pruning for smaller puzzles"""
        target_state = self.create_target_state()
        
        if start_state == target_state:
            return []
        
        # Use priority queue with heuristic
        queue = []
        heapq.heappush(queue, (self.heuristic(start_state), 0, start_state, []))
        visited = {self.state_to_tuple(start_state): True}
        
        while queue:
            _, cost, current_state, path = heapq.heappop(queue)
            
            if len(path) > max_depth:
                continue
                
            if current_state == target_state:
                return path
            
            # Try all possible moves
            for move_idx, move in enumerate(self.moves):
                new_state = self.move_tile(current_state, move)
                if new_state:
                    new_state_tuple = self.state_to_tuple(new_state)
                    if new_state_tuple not in visited:
                        new_cost = cost + 1
                        priority = new_cost + self.heuristic(new_state)
                        heapq.heappush(queue, (priority, new_cost, new_state, path + [move_idx]))
                        visited[new_state_tuple] = True
        
        return None
    
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
    
    def generate_training_sample(self, difficulty=25):
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
        
        # Ensure puzzle is solvable
        if not self.is_solvable(state):
            # If not solvable, make one additional random move
            valid_moves = []
            for move_idx, move in enumerate(self.moves):
                if self.move_tile(state, move) is not None:
                    valid_moves.append((move_idx, move))
            if valid_moves:
                move_idx, move = random.choice(valid_moves)
                state = self.move_tile(state, move)
        
        # Solve from this state to get training data
        solution = self.ida_star_solve(state)
        if not solution:
            # Fallback to optimized BFS
            solution = self.solve_bfs_optimized(state)
        
        if solution and len(solution) > 0:
            # Convert state to neural network input format
            state_vector = self.state_to_vector(state)
            
            # Create action probabilities (one-hot for first move)
            action_probs = [0.0] * 4  # 4 possible moves
            first_move = solution[0]
            action_probs[first_move] = 1.0
            
            # Value: closer to 1 if easier (shorter solution)
            value = max(0.1, 1.0 - len(solution) / 50.0)
            
            # Difficulty class: 0=easy, 1=medium, 2=hard, 3=expert
            solution_length = len(solution)
            if solution_length <= 15:
                difficulty_class = 0
            elif solution_length <= 30:
                difficulty_class = 1
            elif solution_length <= 45:
                difficulty_class = 2
            else:
                difficulty_class = 3
            
            return state_vector, action_probs, value, difficulty_class
        
        return None
    
    def state_to_vector(self, state):
        """Convert puzzle state to neural network input vector"""
        # Enhanced encoding with position information
        vector = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                tile = state[i][j]
                # Normalized tile value
                vector.append(tile / (self.total_tiles - 1))
                # Position encoding
                vector.append(i / (self.board_size - 1))
                vector.append(j / (self.board_size - 1))
        return vector
    
    def generate_dataset(self, num_samples=100000, output_file='puzzle_data/4x4_training_data.pkl'):
        """Generate complete training dataset for 4x4 with progress tracking"""
        print(f"Generating {num_samples} training samples for {self.board_size}x{self.board_size}...")
        
        os.makedirs('puzzle_data', exist_ok=True)
        
        dataset = []
        difficulties = [8, 12, 16, 20, 24]  # Adjusted difficulties for 4x4
        
        success_count = 0
        fail_count = 0
        
        with tqdm(total=num_samples, desc="Generating samples") as pbar:
            while len(dataset) < num_samples:
                difficulty = random.choice(difficulties)
                sample = self.generate_training_sample(difficulty)
                
                if sample is not None:
                    state, action_probs, value, difficulty_class = sample
                    dataset.append((state, action_probs, value, difficulty_class, self.board_size))
                    success_count += 1
                    pbar.update(1)
                else:
                    fail_count += 1
                
                # Update progress bar with success rate
                if (success_count + fail_count) % 100 == 0:
                    success_rate = success_count / (success_count + fail_count) * 100
                    pbar.set_postfix({
                        'success_rate': f'{success_rate:.1f}%',
                        'failed': fail_count
                    })
        
        # Save dataset
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        print(f"Dataset saved to {output_file}")
        print(f"Generated {len(dataset)} samples (Success rate: {success_count/(success_count+fail_count)*100:.1f}%)")
        
        return dataset

class OptimizedPuzzleDataset(Dataset):
    """Optimized Dataset for sliding puzzle training data with caching"""
    
    def __init__(self, data_file, board_size=4):
        self.board_size = board_size
        self.total_tiles = board_size * board_size
        
        # Load training data with progress bar
        print(f"Loading training data from {data_file}...")
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        # Filter for correct board size and cache tensors
        self.cached_data = []
        for item in tqdm(self.data, desc="Processing data"):
            if item[4] == board_size:
                state, action_logits, value, difficulty, size = item
                # Pre-convert to tensors
                self.cached_data.append((
                    torch.FloatTensor(state),
                    torch.FloatTensor(action_logits),
                    torch.FloatTensor([value]),
                    torch.LongTensor([difficulty])
                ))
        
        print(f"Loaded {len(self.cached_data)} training samples for {self.board_size}x{self.board_size}")
    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        return self.cached_data[idx]

class EnhancedNumpuzNetwork(nn.Module):
    def __init__(self, board_size=4, hidden_layers=[1024, 512, 256, 128]):  # Tăng layers
        super(EnhancedNumpuzNetwork, self).__init__()
        self.board_size = board_size
        self.total_tiles = board_size * board_size
        self.input_size = self.total_tiles * 3
        
        # Thêm residual connections
        self.input_bn = nn.BatchNorm1d(self.input_size)
        
        # Tăng số layer và units
        self.layer1 = nn.Sequential(
            nn.Linear(self.input_size, hidden_layers[0]),
            nn.BatchNorm1d(hidden_layers[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.BatchNorm1d(hidden_layers[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25)
        )
        
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_layers[1], hidden_layers[2]),
            nn.BatchNorm1d(hidden_layers[2]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2)
        )
        
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_layers[2], hidden_layers[3]),
            nn.BatchNorm1d(hidden_layers[3]),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15)
        )
        
        # Enhanced output heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_layers[3], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_layers[3], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        self.difficulty_head = nn.Sequential(
            nn.Linear(hidden_layers[3], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Flatten input if needed
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Input batch normalization
        x = self.input_bn(x)
        
        # Hidden layers
        x = self.hidden_layers(x)
        
        # Outputs
        policy = self.policy_head(x)
        value = self.value_head(x)
        difficulty = self.difficulty_head(x)
        
        return policy, value, difficulty

class OptimizedPhase2Trainer:
    """Optimized Phase 2 trainer that handles both dataset generation and training"""
    
    def __init__(self, config):
        self.config = config
        self.board_size = config["board_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize model with transfer learning
        self.model = self._initialize_model_with_transfer()
        
        # Setup optimizer and scheduler
        self.optimizer, self.scheduler = self._setup_optimizer_scheduler()
        
        # Enhanced loss functions with label smoothing
        self.policy_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.value_criterion = nn.MSELoss()
        self.difficulty_criterion = nn.CrossEntropyLoss()
        
        # Gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # Training history with more metrics
        self.history = defaultdict(list)
        self.best_accuracy = 0.0
        
        # Create models directory
        os.makedirs('models', exist_ok=True)
        os.makedirs('puzzle_data', exist_ok=True)
    
    def _initialize_model_with_transfer(self):
        """Initialize 4x4 model with transfer learning from 3x3 model"""
        
        model = EnhancedNumpuzNetwork(
            board_size=self.config["board_size"],
            hidden_layers=self.config["hidden_layers"],
            transfer_config=self.config["transfer_learning"]
        ).to(self.device)
        
        # Load pre-trained 3x3 model for transfer learning
        pretrained_path = self.config["transfer_learning"]["pretrained_path"]
        if os.path.exists(pretrained_path):
            try:
                print(f"Loading pre-trained weights from {pretrained_path}...")
                checkpoint = torch.load(pretrained_path, map_location=self.device)
                pretrained_dict = checkpoint['model_state_dict']
                model_dict = model.state_dict()
                
                # Filter and transfer compatible weights
                transferred_params = 0
                for name, param in pretrained_dict.items():
                    if name in model_dict:
                        # Handle different input sizes for first layer
                        if name == 'input_bn.weight' or name == 'input_bn.bias':
                            # Batch norm for input - partial transfer
                            min_size = min(param.shape[0], model_dict[name].shape[0])
                            model_dict[name][:min_size] = param[:min_size]
                            transferred_params += 1
                            print(f"  ↻ Partially transferred: {name}")
                        elif name == 'hidden_layers.0.weight':
                            # First hidden layer - partial transfer
                            min_rows = min(param.shape[0], model_dict[name].shape[0])
                            min_cols = min(param.shape[1], model_dict[name].shape[1])
                            model_dict[name][:min_rows, :min_cols] = param[:min_rows, :min_cols]
                            transferred_params += 1
                            print(f"  ↻ Partially transferred: {name}")
                        elif name == 'hidden_layers.0.bias':
                            # First hidden layer bias
                            min_size = min(param.shape[0], model_dict[name].shape[0])
                            model_dict[name][:min_size] = param[:min_size]
                            transferred_params += 1
                            print(f"  ↻ Partially transferred: {name}")
                        elif model_dict[name].shape == param.shape:
                            # Exact match
                            model_dict[name] = param
                            transferred_params += 1
                            print(f"  ✓ Transferred: {name}")
                        else:
                            print(f"  ✗ Shape mismatch: {name} ({param.shape} -> {model_dict[name].shape})")
                    else:
                        print(f"  ✗ Not found: {name}")
                
                model.load_state_dict(model_dict)
                print(f"Transfer learning: {transferred_params} parameters transferred")
                
            except Exception as e:
                print(f"Warning: Error loading pre-trained model: {e}")
                print("Continuing with random initialization...")
        else:
            print(f"Warning: Pre-trained model not found at {pretrained_path}")
            print("Training from scratch...")
        
        return model
    
    def _setup_optimizer_scheduler(self):
        """Setup optimizer with layer-wise learning rates and scheduler"""
        
        # Parameter groups for different learning rates
        param_groups = []
        
        # New parameters (higher learning rate)
        new_params = []
        # Fine-tune parameters (medium learning rate)  
        fine_tune_params = []
        # Frozen parameters (lower learning rate or frozen)
        frozen_params = []
        
        transfer_config = self.config["transfer_learning"]
        
        for name, param in self.model.named_parameters():
            if any(frozen_layer in name for frozen_layer in transfer_config["frozen_layers"]):
                frozen_params.append(param)
            elif any(fine_tune_layer in name for fine_tune_layer in transfer_config["fine_tune_layers"]):
                fine_tune_params.append(param)
            else:
                new_params.append(param)
        
        # Add parameter groups with different learning rates
        if new_params:
            param_groups.append({
                'params': new_params, 
                'lr': self.config["learning_rate"],
                'weight_decay': self.config.get("weight_decay", 1e-4)
            })
        
        if fine_tune_params:
            param_groups.append({
                'params': fine_tune_params,
                'lr': self.config["learning_rate"] * 0.5,
                'weight_decay': self.config.get("weight_decay", 1e-4)
            })
        
        if frozen_params:
            param_groups.append({
                'params': frozen_params, 
                'lr': self.config["learning_rate"] * 0.1,
                'weight_decay': self.config.get("weight_decay", 1e-4)
            })
        
        optimizer = optim.AdamW(param_groups)
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[pg['lr'] for pg in param_groups],
            epochs=self.config["epochs"],
            steps_per_epoch=self.config.get("steps_per_epoch", 100),
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        return optimizer, scheduler
    
    def generate_training_data(self):
        """Generate 4x4 training data if not exists"""
        data_file = self.config["data_file"]
        
        if os.path.exists(data_file):
            print(f"Training data already exists: {data_file}")
            return True
        
        print("Generating 4x4 training data...")
        generator = PuzzleGenerator(board_size=4)
        
        try:
            dataset = generator.generate_dataset(
                num_samples=self.config["training_samples"],
                output_file=data_file
            )
            print("✅ Dataset generation completed!")
            return True
        except Exception as e:
            print(f"❌ Error generating dataset: {e}")
            return False
    
    def train_epoch(self, dataloader, epoch):
        """Enhanced training with mixed precision and gradient accumulation"""
        self.model.train()
        
        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        difficulty_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Gradient accumulation
        accum_steps = self.config.get("gradient_accumulation_steps", 1)
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (states, action_targets, value_targets, difficulty_targets) in enumerate(pbar):
            # Move to device
            states = states.to(self.device, non_blocking=True)
            action_targets = action_targets.to(self.device, non_blocking=True)
            value_targets = value_targets.to(self.device, non_blocking=True)
            difficulty_targets = difficulty_targets.squeeze().to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            with torch.cuda.amp.autocast(enabled=self.device.type == 'cuda'):
                policy_pred, value_pred, difficulty_pred = self.model(states)
                
                # Calculate losses
                _, action_indices = torch.max(action_targets, 1)
                policy_loss_batch = self.policy_criterion(policy_pred, action_indices)
                value_loss_batch = self.value_criterion(value_pred.squeeze(), value_targets.squeeze())
                difficulty_loss_batch = self.difficulty_criterion(difficulty_pred, difficulty_targets)
                
                # Combined loss with adaptive weighting
                loss_weights = self.config.get("loss_weights", {"policy": 1.0, "value": 0.5, "difficulty": 0.2})
                loss = (loss_weights["policy"] * policy_loss_batch + 
                       loss_weights["value"] * value_loss_batch + 
                       loss_weights["difficulty"] * difficulty_loss_batch)
                
                # Scale loss for gradient accumulation
                loss = loss / accum_steps
            
            # Backward pass with gradient scaling
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % accum_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                                 max_norm=self.config.get("max_grad_norm", 1.0))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                 max_norm=self.config.get("max_grad_norm", 1.0))
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
            
            # Statistics
            total_loss += loss.item() * accum_steps
            policy_loss += policy_loss_batch.item()
            value_loss += value_loss_batch.item()
            difficulty_loss += difficulty_loss_batch.item()
            
            # Accuracy
            _, predicted = torch.max(policy_pred, 1)
            correct_predictions += (predicted == action_indices).sum().item()
            total_samples += states.size(0)
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'Loss': f'{loss.item() * accum_steps:.4f}',
                'Acc': f'{correct_predictions/total_samples*100:.2f}%',
                'LR': f'{current_lr:.2e}'
            })
        
        # Handle remaining gradients
        if total_samples % accum_steps != 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                             max_norm=self.config.get("max_grad_norm", 1.0))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                             max_norm=self.config.get("max_grad_norm", 1.0))
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Calculate averages
        avg_loss = total_loss / len(dataloader)
        avg_policy_loss = policy_loss / len(dataloader)
        avg_value_loss = value_loss / len(dataloader)
        avg_difficulty_loss = difficulty_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        return avg_loss, avg_policy_loss, avg_value_loss, avg_difficulty_loss, accuracy
    
    def train(self):
        """Complete training pipeline: generate data and train model"""
        print(f"\nStarting Complete Phase 2 Pipeline for {self.board_size}x{self.board_size}")
        print("=" * 60)
        
        # Step 1: Generate training data
        print("📊 Step 1: Generating 4x4 training data...")
        if not self.generate_training_data():
            print("❌ Failed to generate training data. Exiting.")
            return None
        
        # Step 2: Load training data
        print("📥 Step 2: Loading training data...")
        try:
            dataset = OptimizedPuzzleDataset(self.config["data_file"], board_size=4)
            
            # Create data loader with optimized settings
            train_loader = DataLoader(
                dataset, 
                batch_size=self.config["batch_size"], 
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
            
            # Update steps_per_epoch for scheduler
            self.config["steps_per_epoch"] = len(train_loader)
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
        
        # Step 3: Training loop
        print("🎯 Step 3: Starting training...")
        print(f"Training samples: {len(dataset)}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print("-" * 60)
        
        start_time = time.time()
        patience = self.config.get("early_stopping_patience", 20)
        patience_counter = 0
        
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
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Print epoch summary
            print(f"Epoch {epoch+1:3d}/{self.config['epochs']}: "
                  f"Loss: {avg_loss:.4f} | "
                  f"Policy: {policy_loss:.4f} | "
                  f"Value: {value_loss:.4f} | "
                  f"Acc: {accuracy*100:.2f}% | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.2f}s")
            
            # Save best model
            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                self.save_model(f"numpuz_{self.board_size}x{self.board_size}_best.pth")
                patience_counter = 0
                print(f"🎯 New best accuracy: {accuracy*100:.2f}%")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= patience:
                print(f"🛑 Early stopping after {epoch+1} epochs")
                break
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.get("checkpoint_interval", 10) == 0:
                self.save_model(f"numpuz_{self.board_size}x{self.board_size}_epoch_{epoch+1}.pth")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        print(f"Best accuracy: {self.best_accuracy*100:.2f}%")
        
        # Save final model
        self.save_model(f"numpuz_{self.board_size}x{self.board_size}_final.pth")
        self.save_training_history()
        
        return self.history
    
    def save_model(self, filename):
        """Save model checkpoint with metadata"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': dict(self.history),
            'best_accuracy': self.best_accuracy,
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, f"models/{filename}")
        print(f"💾 Model saved: models/{filename}")
    
    def save_training_history(self):
        """Save training history and plots"""
        history_file = f"models/training_history_{self.board_size}x{self.board_size}.json"
        
        with open(history_file, 'w') as f:
            json.dump(dict(self.history), f, indent=2)
        
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Enhanced training visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Total Loss', linewidth=2)
        ax1.plot(self.history['policy_loss'], label='Policy Loss', alpha=0.7)
        ax1.plot(self.history['value_loss'], label='Value Loss', alpha=0.7)
        ax1.plot(self.history['difficulty_loss'], label='Difficulty Loss', alpha=0.7)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.history['train_accuracy'], linewidth=2, color='green')
        ax2.axhline(y=self.best_accuracy, color='red', linestyle='--', 
                   label=f'Best: {self.best_accuracy*100:.1f}%')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax3.plot(self.history['learning_rates'], linewidth=2, color='purple')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Time plot
        ax4.plot(self.history['epoch_times'], linewidth=2, color='orange')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (s)')
        ax4.set_title('Epoch Training Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'models/training_curves_{self.board_size}x{self.board_size}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()

def run_complete_phase2():
    """Run complete Phase 2 pipeline: data generation + training"""
    
    # Enhanced configuration for Phase 2 (4x4)
    config_4x4 = {
        "board_size": 4,
        "data_file": "puzzle_data/4x4_training_data.pkl",
        "training_samples": 100000,  # Reduced for faster testing
        "epochs": 300,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "early_stopping_patience": 30,
        "checkpoint_interval": 10,
        "model_size": "medium",
        "hidden_layers": [512, 256, 128],
        "loss_weights": {
            "policy": 1.0,
            "value": 0.3,
            "difficulty": 0.1
        },
        "transfer_learning": {
            "pretrained_path": "models/numpuz_3x3_foundation.pth",
            "frozen_layers": ["hidden_layers.0", "hidden_layers.3"],  # First two hidden layers
            "fine_tune_layers": ["hidden_layers.6", "policy_head", "value_head"],
        }
    }
    
    print("=" * 80)
    print("🚀 OPTIMIZED PHASE 2: 4x4 SCALING TRAINING PIPELINE")
    print("=" * 80)
    print("This pipeline will:")
    print("1. ✅ Generate 4x4 training data using IDA* solver (FAST)")
    print("2. ✅ Load pre-trained 3x3 model for transfer learning") 
    print("3. ✅ Train 4x4 model with optimized settings")
    print("4. ✅ Save final 4x4 model and training history")
    print("=" * 80)
    
    # Check if phase1 model exists
    if not os.path.exists(config_4x4["transfer_learning"]["pretrained_path"]):
        print(f"❌ Pre-trained 3x3 model not found: {config_4x4['transfer_learning']['pretrained_path']}")
        print("Please run Phase 1 training first to generate the 3x3 foundation model.")
        return None
    
    try:
        # Initialize and run complete pipeline
        trainer = OptimizedPhase2Trainer(config_4x4)
        history = trainer.train()
        
        if history is not None:
            print("\n" + "=" * 80)
            print("🎉 PHASE 2 COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"📊 Final model: models/numpuz_4x4_final.pth")
            print(f"🏆 Best accuracy: {trainer.best_accuracy*100:.2f}%")
            print("=" * 80)
        
        return history
        
    except Exception as e:
        print(f"❌ Error during Phase 2 pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_complete_phase2()