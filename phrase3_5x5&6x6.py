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
import glob
import sys

class EnhancedPuzzleGenerator:
    """Enhanced puzzle generator with better heuristics and parallel processing support"""
    
    def __init__(self, board_size=5):
        self.board_size = int(board_size)  # Đảm bảo là integer
        self.total_tiles = self.board_size * self.board_size
        self.moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.move_names = ['up', 'down', 'left', 'right']
        
        # Precompute target positions
        self.target_positions = {}
        for num in range(1, self.total_tiles):
            i = (num - 1) // self.board_size
            j = (num - 1) % self.board_size
            self.target_positions[num] = (i, j)
        self.target_positions[0] = (self.board_size - 1, self.board_size - 1)
        
        # Pattern database cache for performance
        self.pattern_db = {}
        self._init_pattern_databases()
    
    def _init_pattern_databases(self):
        """Initialize pattern databases for common subproblems"""
        if self.board_size >= 5:
            # Cache common patterns for corners
            self.corner_patterns = self._generate_corner_patterns()
    
    def _generate_corner_patterns(self):
        """Generate corner pattern databases for better heuristics"""
        patterns = {}
        # Implement corner pattern databases for 5x5 and larger
        return patterns
    
    def is_solvable(self, puzzle):
        """Enhanced solvability check with better inversion counting"""
        flat_puzzle = []
        blank_row = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                tile = puzzle[i][j]
                if tile != 0:
                    flat_puzzle.append(tile)
                else:
                    blank_row = self.board_size - i
        
        inversions = 0
        n = len(flat_puzzle)
        for i in range(n):
            for j in range(i + 1, n):
                if flat_puzzle[i] > flat_puzzle[j]:
                    inversions += 1
        
        if self.board_size % 2 == 1:
            return inversions % 2 == 0
        else:
            return (inversions + blank_row) % 2 == 1
    
    def get_blank_position(self, puzzle):
        """Optimized blank position finder"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if puzzle[i][j] == 0:
                    return i, j
        return -1, -1
    
    def move_tile(self, puzzle, direction):
        """Optimized tile movement"""
        i, j = self.get_blank_position(puzzle)
        di, dj = direction
        
        new_i, new_j = i + di, j + dj
        if 0 <= new_i < self.board_size and 0 <= new_j < self.board_size:
            # Use list operations instead of numpy for compatibility
            new_puzzle = [list(row) for row in puzzle]
            new_puzzle[i][j], new_puzzle[new_i][new_j] = new_puzzle[new_i][new_j], new_puzzle[i][j]
            return new_puzzle
        return None
    
    def manhattan_distance(self, state):
        """Optimized Manhattan distance calculation"""
        distance = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                tile = state[i][j]
                if tile != 0:
                    target_i, target_j = self.target_positions[tile]
                    distance += abs(i - target_i) + abs(j - target_j)
        return distance
    
    def linear_conflict(self, state):
        """Enhanced linear conflict detection"""
        conflict = 0
        
        # Row conflicts
        for i in range(self.board_size):
            row_tiles = []
            for j in range(self.board_size):
                tile = state[i][j]
                if tile != 0 and (tile - 1) // self.board_size == i:
                    row_tiles.append((j, tile))
            
            # Sort by current position and check conflicts
            row_tiles.sort()
            for idx1 in range(len(row_tiles)):
                for idx2 in range(idx1 + 1, len(row_tiles)):
                    pos1, tile1 = row_tiles[idx1]
                    pos2, tile2 = row_tiles[idx2]
                    target1 = (tile1 - 1) % self.board_size
                    target2 = (tile2 - 1) % self.board_size
                    if target1 > target2:
                        conflict += 2
        
        # Column conflicts
        for j in range(self.board_size):
            col_tiles = []
            for i in range(self.board_size):
                tile = state[i][j]
                if tile != 0 and (tile - 1) % self.board_size == j:
                    col_tiles.append((i, tile))
            
            col_tiles.sort()
            for idx1 in range(len(col_tiles)):
                for idx2 in range(idx1 + 1, len(col_tiles)):
                    pos1, tile1 = col_tiles[idx1]
                    pos2, tile2 = col_tiles[idx2]
                    target1 = (tile1 - 1) // self.board_size
                    target2 = (tile2 - 1) // self.board_size
                    if target1 > target2:
                        conflict += 2
        
        return conflict
    
    def corner_tiles_heuristic(self, state):
        """Additional heuristic for corner tiles"""
        corner_score = 0
        corners = [(0, 0), (0, self.board_size-1), 
                  (self.board_size-1, 0), (self.board_size-1, self.board_size-1)]
        
        for i, j in corners:
            tile = state[i][j]
            if tile != 0:
                target_i, target_j = self.target_positions[tile]
                if (i, j) != (target_i, target_j):
                    corner_score += 2
        
        return corner_score
    
    def heuristic(self, state):
        """Combined heuristic with weights optimized for board size"""
        manhattan = self.manhattan_distance(state)
        conflict = self.linear_conflict(state)
        corners = self.corner_tiles_heuristic(state)
        
        # Weight heuristics based on board size
        if self.board_size <= 4:
            return manhattan + conflict
        else:
            return int(manhattan + conflict * 1.5 + corners)  # Ép kiểu integer
    
    def ida_star_solve(self, start_state, max_depth=150):
        """Enhanced IDA* with better memory management"""
        target_state = self.create_target_state()
        
        if start_state == target_state:
            return []
        
        bound = int(self.heuristic(start_state))  # Ép kiểu integer
        path = []
        
        for depth_limit in range(bound, max_depth + 1, 5):  # Step by 5 for efficiency
            result = self._ida_star_search(start_state, 0, depth_limit, path, set(), target_state)
            if result == "FOUND":
                return path
            if result == float('inf'):
                break
        
        return None
    
    def _ida_star_search(self, state, g, bound, path, visited, target_state):
        """Optimized IDA* search with depth limiting"""
        h = self.heuristic(state)
        f = g + h
        
        if f > bound:
            return f
        if state == target_state:
            return "FOUND"
        if g > bound * 2:  # Prevent excessive depth
            return float('inf')
        
        state_tuple = self.state_to_tuple(state)
        if state_tuple in visited:
            return float('inf')
        
        visited.add(state_tuple)
        min_cost = float('inf')
        
        # Try moves in order of likely success (based on blank position)
        blank_i, blank_j = self.get_blank_position(state)
        move_order = self._prioritize_moves(blank_i, blank_j)
        
        for move_idx in move_order:
            move = self.moves[move_idx]
            new_state = self.move_tile(state, move)
            if new_state and self.state_to_tuple(new_state) not in visited:
                path.append(move_idx)
                result = self._ida_star_search(new_state, g + 1, bound, path, visited, target_state)
                
                if result == "FOUND":
                    return "FOUND"
                if result < min_cost:
                    min_cost = result
                
                path.pop()
        
        visited.remove(state_tuple)
        return min_cost
    
    def _prioritize_moves(self, blank_i, blank_j):
        """Prioritize moves that move toward solution"""
        # Moves that bring blank closer to bottom-right
        priority = []
        if blank_i < self.board_size - 1:
            priority.append(1)  # down
        if blank_j < self.board_size - 1:
            priority.append(3)  # right
        if blank_i > 0:
            priority.append(0)  # up
        if blank_j > 0:
            priority.append(2)  # left
        
        # Add remaining moves
        all_moves = [0, 1, 2, 3]
        for move in all_moves:
            if move not in priority:
                priority.append(move)
        
        return priority
    
    def solve_with_fallback(self, start_state, max_depth=120):
        """Multi-stage solver with fallbacks"""
        # Try IDA* first
        solution = self.ida_star_solve(start_state, max_depth)
        if solution:
            return solution
        
        # Fallback to limited A*
        return self.limited_a_star_solve(start_state, max_depth // 2)
    
    def limited_a_star_solve(self, start_state, max_nodes=50000):
        """A* with node limit for larger puzzles"""
        target_state = self.create_target_state()
        
        if start_state == target_state:
            return []
        
        open_set = []
        heapq.heappush(open_set, (self.heuristic(start_state), 0, start_state, []))
        visited = {self.state_to_tuple(start_state): True}
        nodes_expanded = 0
        
        while open_set and nodes_expanded < max_nodes:
            _, cost, current_state, path = heapq.heappop(open_set)
            nodes_expanded += 1
            
            if current_state == target_state:
                return path
            
            if len(path) > 100:  # Limit path length
                continue
                
            # Try all possible moves
            for move_idx, move in enumerate(self.moves):
                new_state = self.move_tile(current_state, move)
                if new_state:
                    new_state_tuple = self.state_to_tuple(new_state)
                    if new_state_tuple not in visited:
                        new_cost = cost + 1
                        priority = new_cost + self.heuristic(new_state)
                        heapq.heappush(open_set, (priority, new_cost, new_state, path + [move_idx]))
                        visited[new_state_tuple] = True
        
        return None
    
    def create_target_state(self):
        """Create solved state"""
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
        """Fast state to tuple conversion"""
        return tuple(tuple(row) for row in state)
    
    def generate_training_sample(self, difficulty=30):
        """Enhanced training sample generation with better difficulty scaling"""
        try:
            # Start from solved state
            state = self.create_target_state()
            
            # Make random moves with some intelligence
            last_move = -1
            for step in range(difficulty):
                valid_moves = []
                for move_idx, move in enumerate(self.moves):
                    if self.move_tile(state, move) is not None and move_idx != last_move:
                        valid_moves.append((move_idx, move))
                
                if not valid_moves:
                    break
                    
                # Prefer moves that don't immediately undo previous move
                move_idx, move = random.choice(valid_moves)
                state = self.move_tile(state, move)
                last_move = self._get_opposite_move(move_idx)
            
            # Ensure solvability
            if not self.is_solvable(state):
                # Make corrective moves
                state = self._make_solvable(state)
            
            # Solve and generate training data
            solution = self.solve_with_fallback(state)
            
            if solution and len(solution) > 0:
                state_vector = self.state_to_enhanced_vector(state)
                
                # Action probabilities with some exploration
                action_probs = [0.1] * 4  # Small baseline probability
                first_move = solution[0]
                action_probs[first_move] = 0.7  # Higher probability for correct move
                
                # Normalize probabilities
                total = sum(action_probs)
                action_probs = [p / total for p in action_probs]
                
                # Value based on solution length and complexity
                max_expected_moves = 100 if self.board_size == 5 else 150
                base_value = max(0.1, 1.0 - len(solution) / max_expected_moves)
                
                # Adjust value based on board size and complexity
                complexity = self.heuristic(state) / (self.board_size * 10)
                value = base_value * (1.0 - complexity * 0.3)
                
                # Enhanced difficulty classification
                solution_length = len(solution)
                if solution_length <= 15:
                    difficulty_class = 0  # Easy
                elif solution_length <= 30:
                    difficulty_class = 1  # Medium
                elif solution_length <= 50:
                    difficulty_class = 2  # Hard
                else:
                    difficulty_class = 3  # Expert
                
                return state_vector, action_probs, value, difficulty_class
            
            return None
        except Exception as e:
            print(f"Error in generate_training_sample: {e}")
            return None
    
    def _get_opposite_move(self, move_idx):
        """Get opposite move index"""
        opposites = {0: 1, 1: 0, 2: 3, 3: 2}
        return opposites.get(move_idx, -1)
    
    def _make_solvable(self, state):
        """Make puzzle solvable by making additional moves"""
        for _ in range(3):  # Try up to 3 corrective moves
            valid_moves = []
            for move_idx, move in enumerate(self.moves):
                if self.move_tile(state, move) is not None:
                    valid_moves.append((move_idx, move))
            
            if valid_moves:
                move_idx, move = random.choice(valid_moves)
                state = self.move_tile(state, move)
                
                if self.is_solvable(state):
                    break
        
        return state
    
    def state_to_enhanced_vector(self, state):
        """Enhanced state representation with more features"""
        vector = []
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                tile = state[i][j]
                
                # Normalized tile value
                vector.append(tile / (self.total_tiles - 1))
                
                # Position encoding
                vector.append(i / (self.board_size - 1))
                vector.append(j / (self.board_size - 1))
                
                # Distance to target
                if tile != 0:
                    target_i, target_j = self.target_positions[tile]
                    dist = abs(i - target_i) + abs(j - target_j)
                    vector.append(dist / (2 * self.board_size))
                else:
                    vector.append(0.0)
                
                # Is tile in correct row/column
                if tile != 0:
                    correct_row = 1.0 if i == (tile - 1) // self.board_size else 0.0
                    correct_col = 1.0 if j == (tile - 1) % self.board_size else 0.0
                else:
                    correct_row = 1.0 if i == self.board_size - 1 else 0.0
                    correct_col = 1.0 if j == self.board_size - 1 else 0.0
                
                vector.append(correct_row)
                vector.append(correct_col)
        
        return vector
    
    def generate_dataset(self, num_samples=100000, output_file='puzzle_data/5x5_training_data.pkl'):
        """Optimized dataset generation with progress tracking"""
        print(f"Generating {num_samples} training samples for {self.board_size}x{self.board_size}...")
        
        os.makedirs('puzzle_data', exist_ok=True)
        
        dataset = []
        if self.board_size == 5:
            difficulties = [12, 18, 25, 32, 40]
        else:  # 6x6
            difficulties = [18, 25, 35, 45, 55]
        
        success_count = 0
        fail_count = 0
        batch_size = 1000
        
        # Calculate actual number of samples per difficulty
        samples_per_difficulty = num_samples // len(difficulties)
        
        with tqdm(total=num_samples, desc=f"Generating {self.board_size}x{self.board_size}") as pbar:
            for difficulty in difficulties:
                for _ in range(samples_per_difficulty):
                    sample = self.generate_training_sample(difficulty)
                    
                    if sample is not None:
                        state, action_probs, value, difficulty_class = sample
                        dataset.append((state, action_probs, value, difficulty_class, self.board_size))
                        success_count += 1
                        pbar.update(1)
                    else:
                        fail_count += 1
                    
                    # Update progress
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
        print(f"Generated {len(dataset)} samples (Success rate: {success_rate:.1f}%)")
        
        return dataset

class OptimizedCurriculumDataset(Dataset):
    """Optimized dataset for curriculum learning with caching"""
    
    def __init__(self, data_files, board_sizes):
        self.board_sizes = board_sizes
        self.max_board_size = max(board_sizes)
        
        # Calculate input size based on enhanced features
        self.input_size_per_tile = 6  # tile_value, row, col, dist, correct_row, correct_col
        self.max_input_size = self.max_board_size * self.max_board_size * self.input_size_per_tile
        
        self.data = []
        self.size_indices = {size: [] for size in board_sizes}
        
        for data_file, board_size in zip(data_files, board_sizes):
            if not os.path.exists(data_file):
                print(f"Warning: Data file {data_file} not found")
                continue
                
            print(f"Loading {board_size}x{board_size} data from {data_file}...")
            with open(data_file, 'rb') as f:
                file_data = pickle.load(f)
            
            start_idx = len(self.data)
            for item in tqdm(file_data, desc=f"Processing {board_size}x{board_size}"):
                if item[4] == board_size:
                    state, action_logits, value, difficulty, size = item
                    
                    # Pad state vector to max input size
                    current_size = len(state)
                    if current_size < self.max_input_size:
                        padded_state = state + [0.0] * (self.max_input_size - current_size)
                    else:
                        padded_state = state[:self.max_input_size]
                    
                    # Convert to tensors
                    self.data.append((
                        torch.FloatTensor(padded_state),
                        torch.FloatTensor(action_logits),
                        torch.FloatTensor([value]),
                        torch.LongTensor([difficulty]),
                        torch.LongTensor([board_size])
                    ))
            
            # Record indices for this board size
            end_idx = len(self.data)
            self.size_indices[board_size] = list(range(start_idx, end_idx))
        
        print(f"Loaded {len(self.data)} total training samples for sizes {board_sizes}")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    def get_indices_for_size(self, board_size):
        """Get indices for specific board size"""
        return self.size_indices.get(board_size, [])

class AdvancedNumpuzNetwork(nn.Module):
    """Advanced neural network with adaptive architecture for multiple board sizes"""
    
    def __init__(self, max_board_size=6, hidden_layers=[1024, 512, 256]):
        super(AdvancedNumpuzNetwork, self).__init__()
        self.max_board_size = max_board_size
        self.max_tiles = max_board_size * max_board_size
        self.input_size_per_tile = 6
        self.input_size = self.max_tiles * self.input_size_per_tile
        self.hidden_layers_config = hidden_layers
        
        # Enhanced input processing
        self.input_bn = nn.BatchNorm1d(self.input_size)
        
        # Dynamic hidden layers with residual connections
        layers = []
        prev_size = self.input_size
        dropout_rates = [0.3, 0.25, 0.2]
        
        for idx, hidden_size in enumerate(hidden_layers):
            dropout = dropout_rates[min(idx, len(dropout_rates)-1)]
            
            # Residual block for deeper networks
            block = nn.Sequential(
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout * 0.5)
            )
            layers.append(block)
            prev_size = hidden_size
        
        self.hidden_blocks = nn.ModuleList(layers)
        
        # Adaptive projection for residual connections
        self.residual_projections = nn.ModuleList()
        for hidden_size in hidden_layers:
            if hidden_size != self.input_size:
                self.residual_projections.append(nn.Linear(self.input_size, hidden_size))
            else:
                self.residual_projections.append(nn.Identity())
        
        # Enhanced output heads
        last_hidden_size = hidden_layers[-1]
        
        # Policy head with attention mechanism
        self.policy_head = nn.Sequential(
            nn.Linear(last_hidden_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 4),
            nn.Softmax(dim=-1)  # Output probabilities
        )
        
        # Value head with confidence estimation
        self.value_head = nn.Sequential(
            nn.Linear(last_hidden_size, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # Enhanced difficulty head
        self.difficulty_head = nn.Sequential(
            nn.Linear(last_hidden_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4)
        )
        
        # Board size adaptation
        self.size_embedding = nn.Embedding(20, 32)  # Support up to 20x20
        self.size_projection = nn.Linear(32, last_hidden_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.1)
    
    def forward(self, x, board_size=None):
        # Flatten input
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        
        # Input normalization
        x = self.input_bn(x)
        residual = x
        
        # Pass through hidden blocks with residual connections
        for i, block in enumerate(self.hidden_blocks):
            # Project residual if needed
            if residual.size(1) != block[0].weight.size(1):
                residual_proj = self.residual_projections[i](residual)
            else:
                residual_proj = residual
            
            # Forward through block
            x = block(x)
            
            # Add residual connection
            if x.size(1) == residual_proj.size(1):
                x = x + residual_proj
            
            residual = x
        
        # Add board size information
        if board_size is not None:
            size_embed = self.size_embedding(board_size)
            if size_embed.dim() == 3:
                size_embed = size_embed.squeeze(1)
            size_projected = self.size_projection(size_embed)
            x = x + size_projected * 0.1  # Small scaling factor
        
        # Outputs
        policy = self.policy_head(x)
        value = self.value_head(x)
        difficulty = self.difficulty_head(x)
        
        return policy, value, difficulty

class OptimizedPhase3Trainer:
    """Optimized Phase 3 trainer with enhanced training strategies"""
    
    def __init__(self, config):
        self.config = config
        self.board_sizes = config["board_sizes"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize advanced model
        self.model = AdvancedNumpuzNetwork(
            max_board_size=max(self.board_sizes),
            hidden_layers=config["hidden_layers"]
        ).to(self.device)
        
        # Load pre-trained model from Phase 2
        if config.get("pretrained_path"):
            self._load_phase2_model(config["pretrained_path"])
        
        # Training state
        self.curriculum_schedule = config["curriculum_schedule"]
        self.current_phase = 0
        
        # Will be set after data loading
        self.optimizer = None
        self.scheduler = None
        
        # Enhanced loss functions
        self.policy_criterion = nn.KLDivLoss(reduction='batchmean')  # For probability distributions
        self.value_criterion = nn.MSELoss()
        self.difficulty_criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        
        # Training metrics
        self.history = defaultdict(list)
        self.best_accuracy = 0.0
        self.size_metrics = {size: defaultdict(list) for size in self.board_sizes}
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('puzzle_data', exist_ok=True)
    
    def _load_phase2_model(self, pretrained_path):
        """Load Phase 2 model with enhanced compatibility handling"""
        if not os.path.exists(pretrained_path):
            print(f"Warning: Pre-trained model not found at {pretrained_path}")
            return
        
        try:
            print(f"Loading Phase 2 model from {pretrained_path}...")
            checkpoint = torch.load(pretrained_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    pretrained_dict = checkpoint['model_state_dict']
                else:
                    pretrained_dict = checkpoint
            else:
                pretrained_dict = checkpoint
            
            model_dict = self.model.state_dict()
            
            # Enhanced weight transfer with shape adaptation
            transferred_params = 0
            skipped_params = 0
            
            for name, param in pretrained_dict.items():
                if name in model_dict:
                    target_param = model_dict[name]
                    
                    if target_param.shape == param.shape:
                        # Exact match
                        model_dict[name] = param
                        transferred_params += 1
                        print(f"  ✓ Transferred: {name}")
                    
                    elif len(target_param.shape) == len(param.shape) and target_param.shape[0] == param.shape[0]:
                        # Partial transfer for weight matrices
                        if len(param.shape) == 2:
                            min_cols = min(target_param.shape[1], param.shape[1])
                            target_param.data[:, :min_cols] = param[:, :min_cols]
                            transferred_params += 1
                            print(f"  ↻ Partially transferred: {name} ({param.shape} -> {target_param.shape})")
                        elif len(param.shape) == 1:
                            min_size = min(target_param.shape[0], param.shape[0])
                            target_param.data[:min_size] = param[:min_size]
                            transferred_params += 1
                            print(f"  ↻ Partially transferred: {name} ({param.shape} -> {target_param.shape})")
                        else:
                            skipped_params += 1
                            print(f"  ⚠ Shape mismatch: {name} ({param.shape} -> {target_param.shape})")
                    else:
                        skipped_params += 1
                        print(f"  ⚠ Shape mismatch: {name} ({param.shape} -> {target_param.shape})")
                else:
                    skipped_params += 1
                    print(f"  ✗ Not found: {name}")
            
            self.model.load_state_dict(model_dict)
            print(f"Transfer learning completed: {transferred_params} transferred, {skipped_params} skipped")
            
        except Exception as e:
            print(f"Error loading Phase 2 model: {e}")
            print("Continuing with random initialization...")
    
    def _setup_optimizer_scheduler(self, steps_per_epoch):
        """Setup optimizer with curriculum-aware learning rates"""
        
        # Parameter groups for different learning rates
        param_groups = []
        
        # New parameters (higher learning rate)
        new_params = []
        # Fine-tune parameters (medium learning rate)  
        fine_tune_params = []
        # Frozen parameters (lower learning rate)
        frozen_params = []
        
        # Classify parameters
        for name, param in self.model.named_parameters():
            if 'size_' in name or 'policy_head' in name:
                new_params.append(param)
            elif 'value_head' in name or 'difficulty_head' in name:
                fine_tune_params.append(param)
            else:
                frozen_params.append(param)
        
        base_lr = self.config["learning_rate"]
        
        if new_params:
            param_groups.append({
                'params': new_params, 
                'lr': base_lr,
                'weight_decay': self.config.get("weight_decay", 1e-4)
            })
        
        if fine_tune_params:
            param_groups.append({
                'params': fine_tune_params,
                'lr': base_lr * 0.5,
                'weight_decay': self.config.get("weight_decay", 1e-4)
            })
        
        if frozen_params:
            param_groups.append({
                'params': frozen_params,
                'lr': base_lr * 0.1,
                'weight_decay': self.config.get("weight_decay", 1e-4)
            })
        
        optimizer = optim.AdamW(param_groups)
        
        # OneCycle scheduler for fast convergence
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=[pg['lr'] for pg in param_groups],
            epochs=sum(phase["iterations"] for phase in self.curriculum_schedule),
            steps_per_epoch=steps_per_epoch,
            pct_start=0.1,
            div_factor=10.0,
            final_div_factor=100.0
        )
        
        return optimizer, scheduler
    
    def generate_training_data(self):
        """Generate training data for all board sizes"""
        data_files = []
        
        for board_size in self.board_sizes:
            data_file = f'puzzle_data/{board_size}x{board_size}_training_data.pkl'
            data_files.append(data_file)
            
            if os.path.exists(data_file):
                print(f"Training data exists: {data_file}")
                continue
            
            print(f"Generating {board_size}x{board_size} training data...")
            generator = EnhancedPuzzleGenerator(board_size=board_size)
            
            try:
                samples_per_size = self.config["training_samples"] // len(self.board_sizes)
                dataset = generator.generate_dataset(
                    num_samples=samples_per_size,
                    output_file=data_file
                )
                print(f"✅ {board_size}x{board_size} dataset generation completed!")
            except Exception as e:
                print(f"❌ Error generating {board_size}x{board_size} dataset: {e}")
                return False
        
        return True
    
    def train_epoch(self, dataloader, epoch, current_size=None):
        """Enhanced training for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        difficulty_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        size_correct = {size: 0 for size in self.board_sizes}
        size_total = {size: 0 for size in self.board_sizes}
        
        accum_steps = self.config.get("gradient_accumulation_steps", 1)
        
        desc = f"Epoch {epoch+1}"
        if current_size:
            desc += f" | Size {current_size}"
        pbar = tqdm(dataloader, desc=desc)
        
        for batch_idx, (states, action_targets, value_targets, difficulty_targets, sizes) in enumerate(pbar):
            # Move to device
            states = states.to(self.device, non_blocking=True)
            action_targets = action_targets.to(self.device, non_blocking=True)
            value_targets = value_targets.to(self.device, non_blocking=True)
            difficulty_targets = difficulty_targets.squeeze().to(self.device, non_blocking=True)
            sizes = sizes.to(self.device, non_blocking=True)
            
            # Mixed precision forward pass
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    policy_pred, value_pred, difficulty_pred = self.model(states, sizes)
                    
                    # Calculate losses
                    policy_loss_batch = self.policy_criterion(
                        torch.log(policy_pred + 1e-8),  # Add epsilon for numerical stability
                        action_targets
                    )
                    value_loss_batch = self.value_criterion(value_pred.squeeze(), value_targets.squeeze())
                    difficulty_loss_batch = self.difficulty_criterion(difficulty_pred, difficulty_targets)
                    
                    # Weighted loss
                    loss_weights = self.config.get("loss_weights", {"policy": 1.0, "value": 0.3, "difficulty": 0.1})
                    loss = (loss_weights["policy"] * policy_loss_batch + 
                           loss_weights["value"] * value_loss_batch + 
                           loss_weights["difficulty"] * difficulty_loss_batch)
                    
                    loss = loss / accum_steps
            else:
                # CPU training
                policy_pred, value_pred, difficulty_pred = self.model(states, sizes)
                
                policy_loss_batch = self.policy_criterion(
                    torch.log(policy_pred + 1e-8),
                    action_targets
                )
                value_loss_batch = self.value_criterion(value_pred.squeeze(), value_targets.squeeze())
                difficulty_loss_batch = self.difficulty_criterion(difficulty_pred, difficulty_targets)
                
                loss_weights = self.config.get("loss_weights", {"policy": 1.0, "value": 0.3, "difficulty": 0.1})
                loss = (loss_weights["policy"] * policy_loss_batch + 
                       loss_weights["value"] * value_loss_batch + 
                       loss_weights["difficulty"] * difficulty_loss_batch)
                
                loss = loss / accum_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % accum_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config.get("max_grad_norm", 1.0)
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        max_norm=self.config.get("max_grad_norm", 1.0)
                    )
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                if self.scheduler:
                    self.scheduler.step()
            
            # Statistics
            total_loss += loss.item() * accum_steps
            policy_loss += policy_loss_batch.item()
            value_loss += value_loss_batch.item()
            difficulty_loss += difficulty_loss_batch.item()
            
            # Accuracy (using max probability)
            _, predicted = torch.max(policy_pred, 1)
            _, targets = torch.max(action_targets, 1)
            batch_correct = (predicted == targets)
            correct_predictions += batch_correct.sum().item()
            total_samples += states.size(0)
            
            # Per-size accuracy
            for size in self.board_sizes:
                size_mask = (sizes.squeeze() == size)
                if size_mask.any():
                    size_correct[size] += batch_correct[size_mask].sum().item()
                    size_total[size] += size_mask.sum().item()
            
            # Update progress
            current_lr = self.optimizer.param_groups[0]['lr']
            overall_acc = correct_predictions / total_samples * 100
            
            size_acc_text = ""
            for size in self.board_sizes:
                if size_total[size] > 0:
                    acc = size_correct[size] / size_total[size] * 100
                    size_acc_text += f" {size}x{size}:{acc:.1f}%"
            
            pbar.set_postfix({
                'Loss': f'{loss.item() * accum_steps:.4f}',
                'Acc': f'{overall_acc:.2f}%',
                'LR': f'{current_lr:.2e}',
                'Sizes': size_acc_text.strip()
            })
        
        # Final gradient step if needed
        if total_samples % accum_steps != 0:
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get("max_grad_norm", 1.0)
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.get("max_grad_norm", 1.0)
                )
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Calculate averages
        avg_loss = total_loss / len(dataloader)
        avg_policy_loss = policy_loss / len(dataloader)
        avg_value_loss = value_loss / len(dataloader)
        avg_difficulty_loss = difficulty_loss / len(dataloader)
        accuracy = correct_predictions / total_samples
        
        # Per-size accuracies
        size_accuracies = {}
        for size in self.board_sizes:
            if size_total[size] > 0:
                size_accuracies[size] = size_correct[size] / size_total[size]
        
        return avg_loss, avg_policy_loss, avg_value_loss, avg_difficulty_loss, accuracy, size_accuracies

    def run_curriculum(self):
        """Run the complete optimized curriculum training"""
        print(f"\n🚀 Starting Optimized Phase 3 Curriculum Training for sizes: {self.board_sizes}")
        print("=" * 70)
        
        # Step 1: Generate training data
        print("📊 Step 1: Generating enhanced training data...")
        if not self.generate_training_data():
            print("❌ Failed to generate training data.")
            return None
        
        # Step 2: Load training data
        print("📥 Step 2: Loading training data...")
        data_files = [f'puzzle_data/{size}x{size}_training_data.pkl' for size in self.board_sizes]
        
        try:
            dataset = OptimizedCurriculumDataset(data_files, self.board_sizes)
            
            # Create optimized data loader
            train_loader = DataLoader(
                dataset, 
                batch_size=self.config["batch_size"], 
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
            
            # Setup optimizer and scheduler
            steps_per_epoch = len(train_loader)
            self.optimizer, self.scheduler = self._setup_optimizer_scheduler(steps_per_epoch)
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            return None
        
        # Step 3: Enhanced curriculum training
        print("🎯 Step 3: Starting enhanced curriculum training...")
        print(f"Training samples: {len(dataset)}")
        print(f"Total epochs: {sum(phase['iterations'] for phase in self.curriculum_schedule)}")
        print(f"Batch size: {self.config['batch_size']}")
        print("-" * 70)
        
        start_time = time.time()
        current_global_epoch = 0
        
        # Curriculum phases
        for phase_idx, phase_config in enumerate(self.curriculum_schedule):
            current_size = phase_config["size"]
            phase_epochs = phase_config["iterations"]
            
            print(f"\n🔷 Curriculum Phase {phase_idx + 1}: {current_size}x{current_size}")
            print(f"   Epochs: {phase_epochs}")
            print("-" * 50)
            
            for phase_epoch in range(phase_epochs):
                epoch_start = time.time()
                
                # Train one epoch
                avg_loss, policy_loss, value_loss, difficulty_loss, accuracy, size_accuracies = self.train_epoch(
                    train_loader, current_global_epoch, current_size
                )
                epoch_time = time.time() - epoch_start
                
                # Update history
                self.history['train_loss'].append(avg_loss)
                self.history['policy_loss'].append(policy_loss)
                self.history['value_loss'].append(value_loss)
                self.history['difficulty_loss'].append(difficulty_loss)
                self.history['train_accuracy'].append(accuracy)
                self.history['epoch_times'].append(epoch_time)
                self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
                self.history['curriculum_phase'].append(phase_idx)
                self.history['current_size'].append(current_size)
                
                # Update size metrics
                for size, acc in size_accuracies.items():
                    self.size_metrics[size]['accuracy'].append(acc)
                
                # Print epoch summary
                print(f"Phase{phase_idx+1} Epoch {phase_epoch+1:2d}/{phase_epochs} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Acc: {accuracy*100:.2f}% | "
                      f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                      f"Time: {epoch_time:.2f}s")
                
                # Save best model
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.save_model(f"numpuz_phase3_best.pth")
                    print(f"🎯 New best accuracy: {accuracy*100:.2f}%")
                
                current_global_epoch += 1
            
            # Phase completion
            self.save_model(f"numpuz_phase3_{current_size}x{current_size}_checkpoint.pth")
        
        total_time = time.time() - start_time
        print(f"\n✅ Curriculum training completed in {total_time/3600:.2f} hours")
        print(f"🏆 Best accuracy: {self.best_accuracy*100:.2f}%")
        
        # Save final model
        self.save_model(f"numpuz_phase3_final.pth")
        self.save_training_history()
        
        return self.history
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'history': dict(self.history),
            'size_metrics': dict(self.size_metrics),
            'best_accuracy': self.best_accuracy,
            'timestamp': time.time()
        }
        
        torch.save(checkpoint, f"models/{filename}")
        print(f"💾 Model saved: models/{filename}")
    
    def save_training_history(self):
        """Save training history and plots"""
        history_file = f"models/phase3_training_history.json"
        
        with open(history_file, 'w') as f:
            json.dump({
                'history': dict(self.history),
                'size_metrics': {str(k): dict(v) for k, v in self.size_metrics.items()}
            }, f, indent=2)
        
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """Enhanced training visualization"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Total Loss', linewidth=2)
        ax1.plot(self.history['policy_loss'], label='Policy Loss', alpha=0.7)
        ax1.plot(self.history['value_loss'], label='Value Loss', alpha=0.7)
        
        # Mark curriculum phases
        phases = []
        current_phase = -1
        for i, phase in enumerate(self.history['curriculum_phase']):
            if phase != current_phase:
                phases.append(i)
                current_phase = phase
        
        for phase_start in phases[1:]:
            ax1.axvline(x=phase_start, color='red', linestyle='--', alpha=0.5, label='Phase Change' if phase_start == phases[1] else "")
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss with Curriculum Phases')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(self.history['train_accuracy'], linewidth=2, color='green', label='Overall')
        
        # Size-specific accuracies
        colors = ['blue', 'orange', 'purple', 'brown']
        for idx, size in enumerate(self.board_sizes):
            if size in self.size_metrics and self.size_metrics[size]['accuracy']:
                size_acc = self.size_metrics[size]['accuracy']
                if len(size_acc) == len(self.history['train_accuracy']):
                    ax2.plot(size_acc, linewidth=1, color=colors[idx % len(colors)], 
                            alpha=0.7, label=f'{size}x{size}')
        
        ax2.axhline(y=self.best_accuracy, color='red', linestyle='--', 
                   label=f'Best: {self.best_accuracy*100:.1f}%')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Training Accuracy by Board Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Learning rate
        ax3.plot(self.history['learning_rates'], linewidth=2, color='purple')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Epoch times
        ax4.plot(self.history['epoch_times'], linewidth=2, color='orange')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Time (s)')
        ax4.set_title('Epoch Training Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'models/phase3_training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def run_optimized_phase3():
    """Run optimized Phase 3 training pipeline"""
    
    # Optimized configuration
    config_phase3 = {
        "board_sizes": [5, 6],
        "training_samples": 10000,  # Giảm để test nhanh
        "epochs_per_size": 150,
        "batch_size": 512,
        "learning_rate": 0.0003,
        "weight_decay": 1e-4,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "model_size": "large",
        "hidden_layers": [1024, 512, 256],
        "loss_weights": {
            "policy": 1.0,
            "value": 0.3,
            "difficulty": 0.1
        },
        "curriculum_schedule": [
            {"size": 5, "difficulty": [15, 40], "iterations": 8},
            {"size": 6, "difficulty": [20, 60], "iterations": 12}
        ],
        "pretrained_path": "models/numpuz_4x4_final.pth"  # Direct path to Phase 2 model
    }
    
    print("=" * 80)
    print("🎯 OPTIMIZED PHASE 3: INTERMEDIATE CURRICULUM TRAINING (5x5-6x6)")
    print("=" * 80)
    
    # Auto-detect Phase 2 model
    if not os.path.exists(config_phase3["pretrained_path"]):
        # Try to find any 4x4 model
        potential_paths = [
            "models/numpuz_4x4_final.pth",
            "models/numpuz_4x4_best.pth", 
            "numpuz_4x4_final.pth",
            "numpuz_4x4_best.pth"
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                config_phase3["pretrained_path"] = path
                print(f"🔄 Found Phase 2 model: {path}")
                break
        else:
            print("❌ No Phase 2 model found. Please run Phase 2 first.")
            return None
    
    print(f"🔄 Using Phase 2 model: {config_phase3['pretrained_path']}")
    
    print("\n📋 Enhanced Pipeline:")
    print("1. ✅ Generate enhanced 5x5 and 6x6 training data")
    print("2. ✅ Load Phase 2 model for optimized transfer learning") 
    print("3. ✅ Train with advanced curriculum and residual networks")
    print("4. ✅ Save final adaptive model with improved accuracy")
    print("=" * 80)
    
    try:
        trainer = OptimizedPhase3Trainer(config_phase3)
        history = trainer.run_curriculum()
        
        if history is not None:
            print("\n" + "=" * 80)
            print("🎉 OPTIMIZED PHASE 3 COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            print(f"📊 Final model: models/numpuz_phase3_final.pth")
            print(f"🏆 Best accuracy: {trainer.best_accuracy*100:.2f}%")
            print(f"📈 Trained on sizes: {config_phase3['board_sizes']}")
            print("=" * 80)
        
        return history
        
    except Exception as e:
        print(f"❌ Error during Phase 3: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    run_optimized_phase3()