#!/usr/bin/env python3
"""
🎯 PHASE 1: FOUNDATION TRAINING (3x3)
Optimized version with enhanced logging, performance improvements, and FLOW.md compliance
"""

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
import logging
from pathlib import Path
import sys
from typing import List, Tuple, Optional, Dict, Any
import heapq
import yaml
import zipfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_phase1.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedPuzzleGenerator:
    """Enhanced puzzle generator with A* solver and FLOW.md compliance"""
    
    def __init__(self, board_size: int = 3):
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
        
        logger.info(f"Initialized EnhancedPuzzleGenerator for {board_size}x{board_size}")

    def is_solvable(self, puzzle: List[List[int]]) -> bool:
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

    def get_blank_position(self, puzzle: List[List[int]]) -> Tuple[int, int]:
        """Find the position of blank tile (0)"""
        for i in range(self.board_size):
            for j in range(self.board_size):
                if puzzle[i][j] == 0:
                    return i, j
        return -1, -1

    def move_tile(self, puzzle: List[List[int]], direction: Tuple[int, int]) -> Optional[List[List[int]]]:
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

    def manhattan_distance(self, state: List[List[int]]) -> int:
        """Calculate Manhattan distance heuristic"""
        distance = 0
        for i in range(self.board_size):
            for j in range(self.board_size):
                tile = state[i][j]
                if tile != 0:
                    target_i, target_j = self.target_positions[tile]
                    distance += abs(i - target_i) + abs(j - target_j)
        return distance

    def a_star_solve(self, start_state: List[List[int]], max_nodes: int = 10000) -> Optional[List[int]]:
        """Solve puzzle using A* algorithm with Manhattan distance heuristic"""
        target_state = self.create_target_state()
        
        if start_state == target_state:
            return []
        
        # Priority queue: (f_score, g_score, state, path)
        open_set = []
        heapq.heappush(open_set, (self.manhattan_distance(start_state), 0, start_state, []))
        
        g_scores = {self.state_to_tuple(start_state): 0}
        closed_set = set()
        
        nodes_expanded = 0
        
        while open_set and nodes_expanded < max_nodes:
            f_score, g_score, current_state, path = heapq.heappop(open_set)
            current_tuple = self.state_to_tuple(current_state)
            
            if current_tuple in closed_set:
                continue
                
            if current_state == target_state:
                logger.debug(f"A* solved puzzle with {len(path)} moves, expanded {nodes_expanded} nodes")
                return path
            
            closed_set.add(current_tuple)
            nodes_expanded += 1
            
            # Try all possible moves
            for move_idx, move in enumerate(self.moves):
                new_state = self.move_tile(current_state, move)
                if new_state:
                    new_tuple = self.state_to_tuple(new_state)
                    if new_tuple not in closed_set:
                        new_g_score = g_score + 1
                        new_f_score = new_g_score + self.manhattan_distance(new_state)
                        
                        if new_tuple not in g_scores or new_g_score < g_scores[new_tuple]:
                            g_scores[new_tuple] = new_g_score
                            heapq.heappush(open_set, (new_f_score, new_g_score, new_state, path + [move_idx]))
        
        logger.warning(f"A* failed to solve puzzle within {max_nodes} nodes")
        return None

    def create_target_state(self) -> List[List[int]]:
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

    def state_to_tuple(self, state: List[List[int]]) -> Tuple:
        """Convert state to tuple for hashing"""
        return tuple(tuple(row) for row in state)

    def generate_training_sample(self, min_moves: int = 5, max_moves: int = 20) -> Optional[Tuple]:
        """Generate one training sample with A* solved path (FLOW.md compliant)"""
        # Start from solved state and make k random moves (k ∈ [5,20])
        k = random.randint(min_moves, max_moves)
        state = self.create_target_state()
        valid_moves_count = 0
        
        for move_num in range(k):
            valid_moves = []
            for move_idx, move in enumerate(self.moves):
                if self.move_tile(state, move) is not None:
                    valid_moves.append((move_idx, move))
            
            if not valid_moves:
                break
                
            move_idx, move = random.choice(valid_moves)
            state = self.move_tile(state, move)
            valid_moves_count += 1
        
        # Ensure puzzle is solvable
        if not self.is_solvable(state):
            logger.debug("Generated unsolvable puzzle, regenerating...")
            return None
        
        # Solve using A* to get optimal path
        solution = self.a_star_solve(state)
        
        if solution and len(solution) > 0:
            # Convert state to enhanced neural network input format (one-hot + features)
            state_encoded = self.state_to_enhanced_encoding(state)
            
            # Create action probabilities (one-hot for first optimal move)
            action_probs = [0.0] * 4  # 4 possible moves
            first_move = solution[0]
            action_probs[first_move] = 1.0
            
            # Value: estimate of solvability (1.0 = easy, 0.1 = hard)
            optimal_moves = len(solution)
            value = max(0.1, 1.0 - (optimal_moves / 30.0))  # Normalize based on max expected moves
            
            # Difficulty based on optimal solution length
            if optimal_moves <= 8:
                difficulty_class = 0  # Easy
            elif optimal_moves <= 15:
                difficulty_class = 1  # Medium
            elif optimal_moves <= 22:
                difficulty_class = 2  # Hard
            else:
                difficulty_class = 3  # Expert
            
            logger.debug(f"Generated sample: {optimal_moves} optimal moves, difficulty {difficulty_class}")
            return state_encoded, action_probs, value, difficulty_class, optimal_moves
        
        return None

    def state_to_enhanced_encoding(self, state: List[List[int]]) -> List[float]:
        """
        Convert puzzle state to enhanced neural network input (FLOW.md compliant)
        Returns: one-hot encoding + positional features
        """
        encoding = []
        
        for i in range(self.board_size):
            for j in range(self.board_size):
                tile = state[i][j]
                
                # One-hot encoding for tile value (10 channels for 3x3)
                one_hot = [0.0] * (self.total_tiles + 1)
                one_hot[tile] = 1.0
                encoding.extend(one_hot)
                
                # Positional encoding
                encoding.append(i / (self.board_size - 1))  # Normalized row
                encoding.append(j / (self.board_size - 1))  # Normalized column
                
                # Manhattan distance to target position
                if tile != 0:
                    target_i, target_j = self.target_positions[tile]
                    manhattan_dist = abs(i - target_i) + abs(j - target_j)
                    encoding.append(manhattan_dist / (2 * (self.board_size - 1)))  # Normalized
                else:
                    encoding.append(0.0)  # Blank tile
        
        return encoding

    def apply_augmentation(self, state: List[List[int]], action_probs: List[float], 
                          augmentation_type: str) -> Tuple[List[List[int]], List[float]]:
        """Apply data augmentation (rotation/reflection) and adjust actions accordingly"""
        state_array = np.array(state)
        new_action_probs = action_probs.copy()
        
        if augmentation_type == 'rot90':
            state_aug = np.rot90(state_array).tolist()
            # Adjust actions: up->right, right->down, down->left, left->up
            new_action_probs = [action_probs[3], action_probs[0], action_probs[1], action_probs[2]]
            
        elif augmentation_type == 'rot180':
            state_aug = np.rot90(state_array, 2).tolist()
            # Adjust actions: up->down, down->up, left->right, right->left
            new_action_probs = [action_probs[1], action_probs[0], action_probs[3], action_probs[2]]
            
        elif augmentation_type == 'rot270':
            state_aug = np.rot90(state_array, 3).tolist()
            # Adjust actions: up->left, left->down, down->right, right->up
            new_action_probs = [action_probs[2], action_probs[3], action_probs[0], action_probs[1]]
            
        elif augmentation_type == 'flip_h':
            state_aug = np.fliplr(state_array).tolist()
            # Adjust actions: left<->right
            new_action_probs = [action_probs[0], action_probs[1], action_probs[3], action_probs[2]]
            
        elif augmentation_type == 'flip_v':
            state_aug = np.flipud(state_array).tolist()
            # Adjust actions: up<->down
            new_action_probs = [action_probs[1], action_probs[0], action_probs[2], action_probs[3]]
            
        else:
            state_aug = state
            new_action_probs = action_probs
        
        return state_aug, new_action_probs

    def generate_dataset(self, num_samples: int = 50000, 
                        output_file: str = 'puzzle_data/3x3_training_data.pkl',
                        enable_augmentation: bool = True) -> List[Tuple]:
        """Generate complete training dataset with optional augmentation"""
        logger.info(f"🚀 Generating {num_samples} training samples for {self.board_size}x{self.board_size}")
        logger.info(f"Augmentation: {'ENABLED' if enable_augmentation else 'DISABLED'}")
        
        Path('puzzle_data').mkdir(exist_ok=True)
        
        base_samples_needed = num_samples // 8 if enable_augmentation else num_samples
        dataset = []
        success_count = 0
        fail_count = 0
        
        # Progress bar for sample generation
        with tqdm(total=base_samples_needed, desc="🎲 Generating base samples", 
                 bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            
            while len(dataset) < base_samples_needed:
                sample = self.generate_training_sample(min_moves=5, max_moves=20)
                
                if sample is not None:
                    state_encoded, action_probs, value, difficulty_class, optimal_moves = sample
                    
                    # Convert state back to 2D for augmentation
                    state_2d = self.encoding_to_state_2d(state_encoded)
                    
                    # Add base sample
                    dataset.append((
                        state_encoded, action_probs, value, difficulty_class, 
                        self.board_size, optimal_moves
                    ))
                    
                    # Apply augmentations
                    if enable_augmentation:
                        augmentations = ['rot90', 'rot180', 'rot270', 'flip_h', 'flip_v']
                        
                        for aug_type in augmentations:
                            state_aug, action_probs_aug = self.apply_augmentation(state_2d, action_probs, aug_type)
                            state_encoded_aug = self.state_to_enhanced_encoding(state_aug)
                            
                            dataset.append((
                                state_encoded_aug, action_probs_aug, value, difficulty_class,
                                self.board_size, optimal_moves
                            ))
                    
                    success_count += 1
                    pbar.update(1)
                    pbar.set_postfix({
                        'success_rate': f'{success_count/(success_count+fail_count)*100:.1f}%',
                        'augmented': f'{len(dataset)}'
                    })
                else:
                    fail_count += 1
        
        # Save dataset
        with open(output_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        logger.info(f"✅ Dataset saved to {output_file}")
        logger.info(f"📊 Generated {len(dataset)} samples ({success_count} base + augmentations)")
        logger.info(f"🎯 Success rate: {success_count/(success_count+fail_count)*100:.1f}%")
        
        return dataset

    def encoding_to_state_2d(self, encoding: List[float]) -> List[List[int]]:
        """Convert enhanced encoding back to 2D state (for augmentation)"""
        state = [[0] * self.board_size for _ in range(self.board_size)]
        encoding_per_tile = (self.total_tiles + 1) + 3  # one-hot + 3 features
        
        for idx in range(self.total_tiles):
            i = idx // self.board_size
            j = idx % self.board_size
            start_idx = idx * encoding_per_tile
            
            # Extract tile value from one-hot encoding
            one_hot = encoding[start_idx:start_idx + self.total_tiles + 1]
            tile_value = one_hot.index(max(one_hot))
            state[i][j] = tile_value
        
        return state

class OptimizedPuzzleDataset(Dataset):
    """Optimized Dataset with caching and efficient data loading"""
    
    def __init__(self, data_file: str, board_size: int = 3):
        self.board_size = board_size
        self.total_tiles = board_size * board_size
        
        logger.info(f"📥 Loading training data from {data_file}")
        
        # Load training data with progress bar
        with open(data_file, 'rb') as f:
            raw_data = pickle.load(f)
        
        # Cache pre-processed tensors
        self.cached_data = []
        
        with tqdm(total=len(raw_data), desc="🔄 Processing data", 
                 bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            for item in raw_data:
                if len(item) == 6:  # New format with optimal_moves
                    state, action_logits, value, difficulty, size, optimal_moves = item
                else:  # Legacy format
                    state, action_logits, value, difficulty, size = item
                    optimal_moves = 0
                
                if size == board_size:
                    # Pre-convert to tensors for faster loading
                    self.cached_data.append((
                        torch.FloatTensor(state),
                        torch.FloatTensor(action_logits),
                        torch.FloatTensor([value]),
                        torch.LongTensor([difficulty]),
                        torch.LongTensor([optimal_moves])
                    ))
                pbar.update(1)
        
        logger.info(f"✅ Loaded {len(self.cached_data)} training samples for {self.board_size}x{self.board_size}")
    
    def __len__(self):
        return len(self.cached_data)
    
    def __getitem__(self, idx):
        return self.cached_data[idx]

class EnhancedNumpuzNetwork(nn.Module):
    """Enhanced neural network architecture (FLOW.md compliant)"""
    
    def __init__(self, board_size: int = 3, hidden_layers: List[int] = [256, 128, 64]):
        super(EnhancedNumpuzNetwork, self).__init__()
        self.board_size = board_size
        self.total_tiles = board_size * board_size
        
        # Input size: (one-hot + features) for each tile
        # For 3x3: 9 tiles * (10 one-hot + 3 features) = 117
        self.encoding_size = self.total_tiles * ((self.total_tiles + 1) + 3)
        self.input_size = self.encoding_size
        
        logger.info(f"🧠 Building network: input_size={self.input_size}, hidden_layers={hidden_layers}")
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(self.input_size)
        
        # Build encoder layers (FLOW.md: encoder with dense layers)
        encoder_layers = []
        prev_size = self.input_size
        
        for i, hidden_size in enumerate(hidden_layers):
            encoder_layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            ])
            prev_size = hidden_size
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Policy head (predict next optimal move)
        self.policy_head = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 4),  # 4 possible moves
            nn.Softmax(dim=1)
        )
        
        # Value head (estimate steps-to-solve / solvability)
        self.value_head = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        
        # Difficulty head (optional)
        self.difficulty_head = nn.Sequential(
            nn.Linear(prev_size, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 4)  # 4 difficulty classes
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
        
        # Input normalization
        x = self.input_bn(x)
        
        # Encoder
        x = self.encoder(x)
        
        # Output heads
        policy = self.policy_head(x)
        value = self.value_head(x)
        difficulty = self.difficulty_head(x)
        
        return policy, value, difficulty

class OptimizedPhase1Trainer:
    """Optimized Phase 1 trainer with enhanced logging and monitoring"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.board_size = config["board_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🚀 Initializing Phase1Trainer for {self.board_size}x{self.board_size}")
        logger.info(f"📊 Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"🎯 GPU: {torch.cuda.get_device_name()}")
        
        # Initialize model
        self.model = EnhancedNumpuzNetwork(
            board_size=config["board_size"],
            hidden_layers=config["hidden_layers"]
        ).to(self.device)
        
        # Print model summary
        self._print_model_summary()
        
        # Optimizer and loss functions (FLOW.md compliant)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config.get("weight_decay", 0.0)
        )
        
        # Enhanced loss functions
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        self.difficulty_criterion = nn.CrossEntropyLoss()
        
        # Learning rate scheduler - FIXED: removed verbose parameter
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Training history with comprehensive metrics
        self.history = {
            'epoch': [],
            'train_loss': [], 'policy_loss': [], 'value_loss': [], 'difficulty_loss': [],
            'train_accuracy': [], 'learning_rate': [], 'epoch_time': [],
            'train_moves_accuracy': [], 'value_mae': []
        }
        
        self.best_accuracy = 0.0
        self.start_time = None
        
        # Create output directories
        Path('models').mkdir(exist_ok=True)
        Path('puzzle_data').mkdir(exist_ok=True)
        Path('training_plots').mkdir(exist_ok=True)
    
    def _print_model_summary(self):
        """Print model architecture summary"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info("🧮 Model Architecture Summary:")
        logger.info(f"   • Input size: {self.model.input_size}")
        logger.info(f"   • Encoder layers: {self.config['hidden_layers']}")
        logger.info(f"   • Total parameters: {total_params:,}")
        logger.info(f"   • Trainable parameters: {trainable_params:,}")
        
        # Print layer details
        for name, module in self.model.named_children():
            if hasattr(module, 'weight'):
                logger.info(f"   • {name}: {tuple(module.weight.shape)}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Enhanced training for one epoch with comprehensive metrics"""
        self.model.train()
        
        metrics = {
            'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 'difficulty_loss': 0.0,
            'correct_predictions': 0, 'total_samples': 0, 'value_errors': 0.0
        }
        
        # Progress bar with enhanced formatting
        pbar = tqdm(dataloader, desc=f"📚 Epoch {epoch+1:3d}", 
                   bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for batch_idx, (states, action_targets, value_targets, difficulty_targets, optimal_moves) in enumerate(pbar):
            # Move to device with non-blocking transfers
            states = states.to(self.device, non_blocking=True)
            action_targets = action_targets.to(self.device, non_blocking=True)
            value_targets = value_targets.to(self.device, non_blocking=True)
            difficulty_targets = difficulty_targets.squeeze().to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            policy_pred, value_pred, difficulty_pred = self.model(states)
            
            # Calculate losses (FLOW.md: policy_loss + value_loss)
            _, action_indices = torch.max(action_targets, 1)
            policy_loss = self.policy_criterion(policy_pred, action_indices)
            value_loss = self.value_criterion(value_pred.squeeze(), value_targets.squeeze())
            difficulty_loss = self.difficulty_criterion(difficulty_pred, difficulty_targets)
            
            # Combined loss with weights (FLOW.md: lambda_policy=1.0, lambda_value=0.5)
            loss_weights = self.config.get("loss_weights", {"policy": 1.0, "value": 0.5, "difficulty": 0.2})
            total_loss = (loss_weights["policy"] * policy_loss + 
                         loss_weights["value"] * value_loss + 
                         loss_weights["difficulty"] * difficulty_loss)
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            batch_size = states.size(0)
            metrics['total_loss'] += total_loss.item() * batch_size
            metrics['policy_loss'] += policy_loss.item() * batch_size
            metrics['value_loss'] += value_loss.item() * batch_size
            metrics['difficulty_loss'] += difficulty_loss.item() * batch_size
            
            # Accuracy calculations
            _, predicted = torch.max(policy_pred, 1)
            metrics['correct_predictions'] += (predicted == action_indices).sum().item()
            metrics['total_samples'] += batch_size
            
            # Value prediction accuracy
            metrics['value_errors'] += torch.abs(value_pred.squeeze() - value_targets.squeeze()).sum().item()
            
            # Update progress bar
            current_acc = metrics['correct_predictions'] / metrics['total_samples']
            pbar.set_postfix({
                'Loss': f'{total_loss.item():.4f}',
                'Acc': f'{current_acc*100:.2f}%',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })
        
        # Calculate epoch averages
        avg_metrics = {}
        avg_metrics['total_loss'] = metrics['total_loss'] / metrics['total_samples']
        avg_metrics['policy_loss'] = metrics['policy_loss'] / metrics['total_samples']
        avg_metrics['value_loss'] = metrics['value_loss'] / metrics['total_samples']
        avg_metrics['difficulty_loss'] = metrics['difficulty_loss'] / metrics['total_samples']
        avg_metrics['accuracy'] = metrics['correct_predictions'] / metrics['total_samples']
        avg_metrics['value_mae'] = metrics['value_errors'] / metrics['total_samples']
        
        return avg_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None) -> Dict[str, List]:
        """Enhanced main training loop with comprehensive logging"""
        logger.info(f"\n🎯 Starting Phase 1 Training for {self.board_size}x{self.board_size}")
        logger.info(f"📊 Training samples: {len(train_loader.dataset):,}")
        logger.info(f"⏰ Epochs: {self.config['epochs']}")
        logger.info(f"📦 Batch size: {self.config['batch_size']}")
        logger.info(f"📈 Learning rate: {self.config['learning_rate']}")
        logger.info("─" * 60)
        
        self.start_time = time.time()
        best_accuracy = 0.0
        
        for epoch in range(self.config["epochs"]):
            epoch_start = time.time()
            
            # Train one epoch
            epoch_metrics = self.train_epoch(train_loader, epoch)
            epoch_time = time.time() - epoch_start
            
            # Update learning rate scheduler
            self.scheduler.step(epoch_metrics['total_loss'])
            
            # Update history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(epoch_metrics['total_loss'])
            self.history['policy_loss'].append(epoch_metrics['policy_loss'])
            self.history['value_loss'].append(epoch_metrics['value_loss'])
            self.history['difficulty_loss'].append(epoch_metrics['difficulty_loss'])
            self.history['train_accuracy'].append(epoch_metrics['accuracy'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_time'].append(epoch_time)
            self.history['value_mae'].append(epoch_metrics['value_mae'])
            
            # Print epoch summary
            logger.info(
                f"📅 Epoch {epoch+1:3d}/{self.config['epochs']} | "
                f"Loss: {epoch_metrics['total_loss']:.4f} | "
                f"Policy: {epoch_metrics['policy_loss']:.4f} | "
                f"Value: {epoch_metrics['value_loss']:.4f} | "
                f"Acc: {epoch_metrics['accuracy']*100:6.2f}% | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.2e} | "
                f"Time: {epoch_time:5.2f}s"
            )
            
            # Save best model
            if epoch_metrics['accuracy'] > best_accuracy:
                best_accuracy = epoch_metrics['accuracy']
                self.save_checkpoint(f"numpuz_{self.board_size}x{self.board_size}_best.pth")
                logger.info(f"🎯 New best accuracy: {best_accuracy*100:.2f}%")
            
            # Save periodic checkpoint
            if (epoch + 1) % self.config.get("checkpoint_interval", 10) == 0:
                self.save_checkpoint(f"numpuz_{self.board_size}x{self.board_size}_epoch_{epoch+1}.pth")
        
        total_time = time.time() - self.start_time
        logger.info("─" * 60)
        logger.info(f"✅ Training completed in {total_time/3600:.2f} hours")
        logger.info(f"🏆 Best accuracy: {best_accuracy*100:.2f}%")
        
        # Save final model and artifacts
        self.save_final_artifacts()
        
        return self.history
    
    def display_training_summary(self):
        """Display comprehensive training summary with visualizations"""
        
        print("\n" + "="*80)
        print("📊 TRAINING SUMMARY - PHASE 1 (3x3)")
        print("="*80)
        
        # 1. Performance Metrics
        print("\n🎯 PERFORMANCE METRICS:")
        print("-" * 80)
        
        final_acc = self.history['train_accuracy'][-1] * 100
        best_acc = max(self.history['train_accuracy']) * 100
        best_epoch = np.argmax(self.history['train_accuracy']) + 1
        final_loss = self.history['train_loss'][-1]
        initial_loss = self.history['train_loss'][0]
        
        print(f"Final Accuracy:         {final_acc:>6.2f}%")
        print(f"Best Accuracy:          {best_acc:>6.2f}% (at epoch {best_epoch})")
        print(f"Accuracy Improvement:   {final_acc - self.history['train_accuracy'][0]*100:>+6.2f}%")
        print(f"Final Loss:             {final_loss:>6.4f}")
        print(f"Initial Loss:           {initial_loss:>6.4f}")
        print(f"Loss Reduction:         {(1 - final_loss/initial_loss)*100:>6.1f}%")
        
        # 2. Loss Breakdown
        print("\n📉 LOSS BREAKDOWN (Last Epoch):")
        print("-" * 80)
        print(f"Total Loss:             {self.history['train_loss'][-1]:>6.4f}")
        print(f"  ├─ Policy Loss:       {self.history['policy_loss'][-1]:>6.4f}")
        print(f"  ├─ Value Loss:        {self.history['value_loss'][-1]:>6.4f}")
        print(f"  └─ Difficulty Loss:   {self.history['difficulty_loss'][-1]:>6.4f}")
        
        # 3. Training Duration
        print("\n⏱️  TRAINING DURATION:")
        print("-" * 80)
        total_time = sum(self.history['epoch_time'])
        avg_time = np.mean(self.history['epoch_time'])
        min_time = min(self.history['epoch_time'])
        max_time = max(self.history['epoch_time'])
        
        print(f"Total Training Time:    {total_time/3600:>6.2f} hours ({total_time/60:>6.1f} minutes)")
        print(f"Average Epoch Time:     {avg_time:>6.2f} seconds")
        print(f"Fastest Epoch:          {min_time:>6.2f} seconds")
        print(f"Slowest Epoch:          {max_time:>6.2f} seconds")
        
        # 4. Learning Rate Info
        print("\n📈 LEARNING RATE:")
        print("-" * 80)
        initial_lr = self.history['learning_rate'][0]
        final_lr = self.history['learning_rate'][-1]
        print(f"Initial LR:             {initial_lr:.2e}")
        print(f"Final LR:               {final_lr:.2e}")
        print(f"LR Reduction Factor:    {initial_lr/final_lr:.2f}x")
        
        # 5. Model Architecture
        print("\n🧠 MODEL ARCHITECTURE:")
        print("-" * 80)
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Input Size:             {self.model.input_size}")
        print(f"Hidden Layers:          {self.config['hidden_layers']}")
        print(f"Total Parameters:       {total_params:,}")
        print(f"Trainable Parameters:   {trainable_params:,}")
        print(f"Model Size (approx):    {total_params * 4 / (1024**2):.2f} MB")
        
        # 6. Dataset Info
        print("\n📦 DATASET INFORMATION:")
        print("-" * 80)
        print(f"Training Samples:       {self.config['training_samples']:,}")
        print(f"Batch Size:             {self.config['batch_size']}")
        print(f"Total Batches/Epoch:    {self.config['training_samples'] // self.config['batch_size']}")
        print(f"Augmentation:           {'ENABLED (8x)' if self.config['enable_augmentation'] else 'DISABLED'}")
        
        # 7. Training Plot
        print("\n📸 TRAINING VISUALIZATION:")
        print("-" * 80)
        plot_file = f'training_plots/training_curves_{self.board_size}x{self.board_size}.png'
        
        if Path(plot_file).exists():
            print(f"Plot saved at: {plot_file}")
            
            # Try to display image inline (if in Jupyter or compatible environment)
            try:
                from IPython.display import Image, display
                display(Image(filename=plot_file))
                print("✓ Training plot displayed above")
            except:
                print("✓ Training plot saved (open file to view)")
                print(f"\n  Preview: [ASCII representation not available]")
                print(f"  Please open: {Path(plot_file).absolute()}")
        
        # 8. Quick Stats Table
        print("\n📋 EPOCH-BY-EPOCH PROGRESS (Last 10 Epochs):")
        print("-" * 80)
        print(f"{'Epoch':<8} {'Loss':<10} {'Accuracy':<12} {'LR':<12} {'Time':<10}")
        print("-" * 80)
        
        start_idx = max(0, len(self.history['epoch']) - 10)
        for i in range(start_idx, len(self.history['epoch'])):
            epoch = self.history['epoch'][i]
            loss = self.history['train_loss'][i]
            acc = self.history['train_accuracy'][i] * 100
            lr = self.history['learning_rate'][i]
            time_taken = self.history['epoch_time'][i]
            
            print(f"{epoch:<8} {loss:<10.4f} {acc:<12.2f}% {lr:<12.2e} {time_taken:<10.2f}s")
        
        # 9. Saved Artifacts
        print("\n💾 SAVED ARTIFACTS:")
        print("-" * 80)
        artifacts = [
            f"models/numpuz_{self.board_size}x{self.board_size}_foundation.pth",
            f"models/numpuz_{self.board_size}x{self.board_size}_best.pth",
            f"models/training_history_{self.board_size}x{self.board_size}.json",
            f"models/train_config_{self.board_size}x{self.board_size}.yaml",
            f"models/model_config_{self.board_size}x{self.board_size}.json",
            plot_file
        ]
        
        for artifact in artifacts:
            if Path(artifact).exists():
                size_mb = Path(artifact).stat().st_size / (1024**2)
                print(f"  ✓ {artifact:<70} {size_mb:>8.2f} MB")
            else:
                print(f"  ✗ {artifact:<70} (not found)")
        
        print("\n" + "="*80)
        print("✅ Training summary complete!")
        print("="*80 + "\n")
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint with comprehensive metadata"""
        checkpoint = {
            'epoch': len(self.history['epoch']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_accuracy': self.best_accuracy,
            'timestamp': datetime.now().isoformat(),
            'total_training_time': time.time() - self.start_time if self.start_time else 0
        }
        
        torch.save(checkpoint, f"models/{filename}")
        logger.info(f"💾 Checkpoint saved: models/{filename}")
    
    def save_final_artifacts(self):
        """Save all final artifacts (FLOW.md compliant) + create ZIP archive"""
        
        # Save final model
        self.save_checkpoint(f"numpuz_{self.board_size}x{self.board_size}_foundation.pth")
        
        # Save training history
        history_file = f"models/training_history_{self.board_size}x{self.board_size}.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"📊 Training history saved: {history_file}")
        
        # Generate and save plots
        self.generate_training_plots()
        
        # Save configuration
        config_file = f"models/train_config_{self.board_size}x{self.board_size}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"⚙️  Training config saved: {config_file}")
        
        # Save model architecture
        model_config = {
            'board_size': self.board_size,
            'input_size': self.model.input_size,
            'hidden_layers': self.config['hidden_layers'],
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        }
        model_config_file = f"models/model_config_{self.board_size}x{self.board_size}.json"
        with open(model_config_file, 'w') as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"🧠 Model config saved: {model_config_file}")
        
        # Create comprehensive ZIP archive
        try:
            logger.info("Starting ZIP archive creation...")
            zip_result = self._create_zip_archive()
            if zip_result:
                logger.info(f"✅ ZIP archive created: {zip_result}")
            else:
                logger.error("❌ ZIP archive creation failed (returned None)")
        except Exception as e:
            logger.error(f"❌ ZIP archive creation exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
        
        # Display comprehensive training summary
        self.display_training_summary()

    def _create_zip_archive(self):
        """Create ZIP archive with all Phase 1 outputs and comprehensive logging"""
        
        zip_filename = f"phase1_output_{self.board_size}x{self.board_size}.zip"
        
        print(f"\n{'='*80}")
        print("📦 ZIP ARCHIVE CREATION INITIATED")
        print(f"{'='*80}")
        print(f"Generated: {datetime.now().isoformat()}")
        print(f"{'='*80}\n")
        
        try:
            temp_output_dir = Path('phase1_output')
            temp_output_dir.mkdir(exist_ok=True)
            
            # Dictionary to track file statistics
            file_stats = {
                'total_files': 0,
                'total_size': 0,
                'files_by_category': {}
            }
            
            logger.info("Starting ZIP compression...\n")
            
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                
                # 1. Add model files
                logger.info("📁 CATEGORY 1: MODEL FILES")
                logger.info("-" * 80)
                models_dir = Path('models')
                if models_dir.exists():
                    model_files = list(models_dir.glob('*'))
                    logger.info(f"Found {len(model_files)} model files\n")
                    
                    file_stats['files_by_category']['models'] = {
                        'count': 0,
                        'size': 0,
                        'files': []
                    }
                    
                    for file in model_files:
                        if file.is_file():
                            arcname = f"phase1_output/models/{file.name}"
                            file_size = file.stat().st_size
                            size_mb = file_size / (1024 ** 2)
                            
                            zipf.write(file, arcname)
                            
                            file_stats['files_by_category']['models']['count'] += 1
                            file_stats['files_by_category']['models']['size'] += file_size
                            file_stats['files_by_category']['models']['files'].append({
                                'name': file.name,
                                'size_bytes': file_size,
                                'size_mb': round(size_mb, 2)
                            })
                            file_stats['total_files'] += 1
                            file_stats['total_size'] += file_size
                            
                            logger.info(f"  ✓ {file.name:<50} {size_mb:>10.2f} MB")
                else:
                    logger.warning("  ⚠ Models directory not found")
                
                logger.info("")
                
                # 2. Add training plots
                logger.info("📁 CATEGORY 2: TRAINING PLOTS")
                logger.info("-" * 80)
                plots_dir = Path('training_plots')
                if plots_dir.exists():
                    plot_files = list(plots_dir.glob('*'))
                    logger.info(f"Found {len(plot_files)} plot files\n")
                    
                    file_stats['files_by_category']['plots'] = {
                        'count': 0,
                        'size': 0,
                        'files': []
                    }
                    
                    for file in plot_files:
                        if file.is_file():
                            arcname = f"phase1_output/training_plots/{file.name}"
                            file_size = file.stat().st_size
                            size_mb = file_size / (1024 ** 2)
                            
                            zipf.write(file, arcname)
                            
                            file_stats['files_by_category']['plots']['count'] += 1
                            file_stats['files_by_category']['plots']['size'] += file_size
                            file_stats['files_by_category']['plots']['files'].append({
                                'name': file.name,
                                'size_bytes': file_size,
                                'size_mb': round(size_mb, 2)
                            })
                            file_stats['total_files'] += 1
                            file_stats['total_size'] += file_size
                            
                            logger.info(f"  ✓ {file.name:<50} {size_mb:>10.2f} MB")
                else:
                    logger.warning("  ⚠ Plots directory not found")
                
                logger.info("")
                
                # 3. Add dataset info
                logger.info("📁 CATEGORY 3: DATASET INFORMATION")
                logger.info("-" * 80)
                dataset_file = 'puzzle_data/3x3_training_data.pkl'
                
                file_stats['files_by_category']['dataset'] = {
                    'count': 0,
                    'size': 0,
                    'files': []
                }
                
                if Path(dataset_file).exists():
                    dataset_size = Path(dataset_file).stat().st_size / (1024 ** 2)
                    logger.info(f"Dataset file found: {dataset_file}")
                    logger.info(f"  Size: {dataset_size:.2f} MB\n")
                    logger.info("  Note: Full dataset NOT included in ZIP (too large)")
                    logger.info("  Creating metadata file instead...\n")
                    
                    dataset_info = {
                        'dataset_file': '3x3_training_data.pkl',
                        'file_size_mb': round(dataset_size, 2),
                        'location': str(Path(dataset_file).absolute()),
                        'note': 'Full dataset stored separately (too large for ZIP)',
                        'last_updated': datetime.now().isoformat()
                    }
                    dataset_info_file = temp_output_dir / 'dataset_info.json'
                    with open(dataset_info_file, 'w') as f:
                        json.dump(dataset_info, f, indent=2)
                    
                    zipf.write(dataset_info_file, 'phase1_output/dataset_info.json')
                    
                    info_size = dataset_info_file.stat().st_size
                    file_stats['files_by_category']['dataset']['count'] += 1
                    file_stats['files_by_category']['dataset']['size'] += info_size
                    file_stats['files_by_category']['dataset']['files'].append({
                        'name': 'dataset_info.json',
                        'size_bytes': info_size,
                        'size_mb': round(info_size / (1024 ** 2), 2)
                    })
                    file_stats['total_files'] += 1
                    file_stats['total_size'] += info_size
                    
                    logger.info(f"  ✓ dataset_info.json{' ':<38} {round(info_size / (1024 ** 2), 2):>10.2f} MB")
                    
                    dataset_info_file.unlink()
                else:
                    logger.warning(f"  ⚠ Dataset file not found: {dataset_file}")
                
                logger.info("")
                
                # 4. Add training logs
                logger.info("📁 CATEGORY 4: TRAINING LOGS")
                logger.info("-" * 80)
                log_files = list(Path('.').glob('training_phase1_*.log'))
                logger.info(f"Found {len(log_files)} log files\n")
                
                file_stats['files_by_category']['logs'] = {
                    'count': 0,
                    'size': 0,
                    'files': []
                }
                
                for log_file in log_files:
                    arcname = f"phase1_output/logs/{log_file.name}"
                    file_size = log_file.stat().st_size
                    size_mb = file_size / (1024 ** 2)
                    
                    zipf.write(log_file, arcname)
                    
                    file_stats['files_by_category']['logs']['count'] += 1
                    file_stats['files_by_category']['logs']['size'] += file_size
                    file_stats['files_by_category']['logs']['files'].append({
                        'name': log_file.name,
                        'size_bytes': file_size,
                        'size_mb': round(size_mb, 2)
                    })
                    file_stats['total_files'] += 1
                    file_stats['total_size'] += file_size
                    
                    logger.info(f"  ✓ {log_file.name:<50} {size_mb:>10.2f} MB")
                
                logger.info("")
                
                # 5. Add README
                logger.info("📁 CATEGORY 5: DOCUMENTATION")
                logger.info("-" * 80)
                readme_content = self._generate_readme()
                readme_file = temp_output_dir / 'README.md'
                with open(readme_file, 'w') as f:
                    f.write(readme_content)
                
                zipf.write(readme_file, 'phase1_output/README.md')
                
                readme_size = readme_file.stat().st_size
                logger.info(f"Created README.md\n")
                logger.info(f"  ✓ README.md{' ':<48} {round(readme_size / (1024 ** 2), 2):>10.2f} MB")
                
                file_stats['files_by_category']['docs'] = {
                    'count': 1,
                    'size': readme_size,
                    'files': [{'name': 'README.md', 'size_bytes': readme_size, 'size_mb': round(readme_size / (1024 ** 2), 2)}]
                }
                file_stats['total_files'] += 1
                file_stats['total_size'] += readme_size
                
                readme_file.unlink()
                
                logger.info("")
                
                # 6. Add training metrics
                logger.info("📁 CATEGORY 6: TRAINING METRICS")
                logger.info("-" * 80)
                metrics_data = {
                    'board_size': self.board_size,
                    'total_epochs_trained': len(self.history['epoch']),
                    'final_accuracy': float(self.history['train_accuracy'][-1]) if self.history['train_accuracy'] else 0.0,
                    'final_loss': float(self.history['train_loss'][-1]) if self.history['train_loss'] else 0.0,
                    'best_accuracy': float(max(self.history['train_accuracy'])) if self.history['train_accuracy'] else 0.0,
                    'average_epoch_time_seconds': float(np.mean(self.history['epoch_time'])) if self.history['epoch_time'] else 0.0,
                    'total_training_time_seconds': float(sum(self.history['epoch_time'])) if self.history['epoch_time'] else 0.0,
                    'last_updated': datetime.now().isoformat()
                }
                metrics_file = temp_output_dir / 'training_metrics.json'
                with open(metrics_file, 'w') as f:
                    json.dump(metrics_data, f, indent=2)
                
                zipf.write(metrics_file, 'phase1_output/training_metrics.json')
                
                metrics_size = metrics_file.stat().st_size
                logger.info(f"Created training_metrics.json\n")
                logger.info(f"  ✓ training_metrics.json{' ':<41} {round(metrics_size / (1024 ** 2), 2):>10.2f} MB")
                
                file_stats['files_by_category']['metrics'] = {
                    'count': 1,
                    'size': metrics_size,
                    'files': [{'name': 'training_metrics.json', 'size_bytes': metrics_size, 'size_mb': round(metrics_size / (1024 ** 2), 2)}]
                }
                file_stats['total_files'] += 1
                file_stats['total_size'] += metrics_size
                
                metrics_file.unlink()
            
            # Get final ZIP file size
            zip_size_bytes = Path(zip_filename).stat().st_size
            zip_size_mb = zip_size_bytes / (1024 ** 2)
            
            # Log summary
            logger.info(f"{'='*80}")
            logger.info("✅ ZIP ARCHIVE CREATED SUCCESSFULLY")
            logger.info(f"{'='*80}\n")
            
            logger.info("📊 ZIP FILE INFORMATION:")
            logger.info("-" * 80)
            logger.info(f"Filename:       {zip_filename}")
            logger.info(f"Location:       {Path(zip_filename).absolute()}")
            logger.info(f"File Size:      {zip_size_mb:.2f} MB ({zip_size_bytes:,} bytes)")
            logger.info(f"Compression:    DEFLATE")
            logger.info(f"Created:        {datetime.now().isoformat()}")
            logger.info("")
            
            logger.info("📈 ARCHIVE STATISTICS:")
            logger.info("-" * 80)
            logger.info(f"Total Files:    {file_stats['total_files']}")
            logger.info(f"Total Size:     {file_stats['total_size'] / (1024 ** 2):.2f} MB")
            logger.info(f"Compression Ratio: {(1 - zip_size_bytes / file_stats['total_size']) * 100:.1f}%\n")
            
            logger.info("📂 BREAKDOWN BY CATEGORY:")
            logger.info("-" * 80)
            
            for category, stats in file_stats['files_by_category'].items():
                if stats['count'] > 0:
                    category_size_mb = stats['size'] / (1024 ** 2)
                    logger.info(f"\n{category.upper()}")
                    logger.info(f"  Files: {stats['count']}")
                    logger.info(f"  Size:  {category_size_mb:.2f} MB")
                    
                    if stats['files']:
                        logger.info(f"  Files list:")
                        for file_info in stats['files']:
                            logger.info(f"    - {file_info['name']:<50} {file_info['size_mb']:>8.2f} MB")
            
            logger.info(f"\n{'='*80}")
            logger.info("📦 ARCHIVE STRUCTURE (Tree View):")
            logger.info("-" * 80)
            
            # Build tree structure
            with zipfile.ZipFile(zip_filename, 'r') as zipf:
                file_tree = {}
                for info in zipf.filelist:
                    parts = info.filename.split('/')
                    current = file_tree
                    for part in parts[:-1]:
                        if part not in current:
                            current[part] = {}
                        current = current[part]
                    # Store file info in tree
                    if parts[-1]:  # Not empty (not a directory)
                        file_size_mb = info.file_size / (1024 ** 2)
                        compressed_size_mb = info.compress_size / (1024 ** 2)
                        compression_ratio = (1 - info.compress_size / info.file_size) * 100 if info.file_size > 0 else 0
                        current[parts[-1]] = {
                            'size': file_size_mb,
                            'compressed': compressed_size_mb,
                            'ratio': compression_ratio
                        }
                
                # Print tree recursively
                def print_tree(node, prefix="", is_last=True):
                    items = list(node.items())
                    for i, (name, value) in enumerate(items):
                        is_last_item = (i == len(items) - 1)
                        connector = "└── " if is_last_item else "├── "
                        
                        if isinstance(value, dict) and 'size' in value:
                            # It's a file
                            logger.info(
                                f"{prefix}{connector}{name} "
                                f"({value['size']:.2f} MB → {value['compressed']:.2f} MB, "
                                f"{value['ratio']:.1f}% compressed)"
                            )
                        else:
                            # It's a directory
                            logger.info(f"{prefix}{connector}{name}/")
                            extension = "    " if is_last_item else "│   "
                            print_tree(value, prefix + extension, is_last_item)
                
                logger.info("")
                print_tree(file_tree)
            
            logger.info(f"\n{'='*80}\n")
            
            # Dọn dẹp
            import shutil
            shutil.rmtree(temp_output_dir, ignore_errors=True)
            
            return zip_filename
            
        except Exception as e:
            logger.error(f"\n❌ FAILED TO CREATE ZIP ARCHIVE")
            logger.error(f"{'='*80}")
            logger.error(f"Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.error(f"{'='*80}\n")
            return None

    def _generate_readme(self) -> str:
        """Generate README for ZIP archive"""
        
        total_time_seconds = sum(self.history['epoch_time']) if self.history['epoch_time'] else 0
        hours = total_time_seconds / 3600
        
        best_epoch = np.argmax(self.history['train_accuracy']) + 1 if self.history['train_accuracy'] else 0
        
        readme = f"""# Phase 1 Training Output - NumpuzAI (3x3)

## Execution Summary

- **Board Size**: {self.board_size}x{self.board_size}
- **Last Training**: {datetime.now().isoformat()}
- **Total Epochs Trained**: {len(self.history['epoch'])}
- **Training Status**: {'COMPLETED' if len(self.history['epoch']) >= self.config['epochs'] else 'INTERRUPTED'}

## Performance Metrics

- **Final Accuracy**: {self.history['train_accuracy'][-1]*100:.2f}%
- **Best Accuracy**: {max(self.history['train_accuracy'])*100:.2f}% (at epoch {best_epoch})
- **Final Loss**: {self.history['train_loss'][-1]:.4f}
- **Initial Loss**: {self.history['train_loss'][0]:.4f}

## Training Duration

- **Total Time**: {hours:.2f} hours
- **Average Epoch Time**: {np.mean(self.history['epoch_time']):.2f} seconds
- **Fastest Epoch**: {min(self.history['epoch_time']):.2f} seconds
- **Slowest Epoch**: {max(self.history['epoch_time']):.2f} seconds

## Training Configuration

### Dataset
- Training Samples: {self.config['training_samples']:,}
- Augmentation: {'Enabled (8x per sample)' if self.config['enable_augmentation'] else 'Disabled'}
- Move Range: {self.config['move_range']}

### Model Architecture
- Input Size: {self.model.input_size}
- Hidden Layers: {self.config['hidden_layers']}
- Total Parameters: {sum(p.numel() for p in self.model.parameters()):,}

### Hyperparameters
- Batch Size: {self.config['batch_size']}
- Learning Rate: {self.config['learning_rate']}
- Optimizer: Adam
- Epochs: {self.config['epochs']}
- Loss Weights: {self.config.get('loss_weights', {})}

## Output Files

### Models
- `numpuz_3x3_foundation.pth` - Final trained model
- `numpuz_3x3_best.pth` - Best model by accuracy
- `model_config_3x3.json` - Model architecture

### Training Data
- `training_history_3x3.json` - Detailed training metrics
- `train_config_3x3.yaml` - Training configuration

### Visualizations
- `training_curves_3x3.png` - Training plots

### Logs
- `training_phase1.log` - Training logs

## Loss Breakdown (Last Epoch)

- **Policy Loss**: {self.history['policy_loss'][-1]:.4f}
- **Value Loss**: {self.history['value_loss'][-1]:.4f}
- **Difficulty Loss**: {self.history['difficulty_loss'][-1]:.4f}

## How to Load Model

```python
import torch
from phase1_3x3 import EnhancedNumpuzNetwork

model = EnhancedNumpuzNetwork(board_size=3, hidden_layers=[256, 128, 64])
checkpoint = torch.load('numpuz_3x3_foundation.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Inference
input_state = torch.randn(1, 117)
policy, value, difficulty = model(input_state)
```

---
**Generated**: {datetime.now().isoformat()}
**Device**: {'CUDA' if self.device.type == 'cuda' else 'CPU'}
"""
        
        return readme

    def generate_training_plots(self):
        """Generate comprehensive training plots"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'NumpuzAI Training Progress - {self.board_size}x{self.board_size}', fontsize=16)
        
        # Loss plot
        ax = axes[0, 0]
        ax.plot(self.history['epoch'], self.history['train_loss'], label='Total Loss', linewidth=2)
        ax.plot(self.history['epoch'], self.history['policy_loss'], label='Policy Loss', alpha=0.7)
        ax.plot(self.history['epoch'], self.history['value_loss'], label='Value Loss', alpha=0.7)
        ax.plot(self.history['epoch'], self.history['difficulty_loss'], label='Difficulty Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax = axes[0, 1]
        ax.plot(self.history['epoch'], self.history['train_accuracy'], linewidth=2, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Training Accuracy')
        ax.grid(True, alpha=0.3)
        
        # Learning rate plot
        ax = axes[0, 2]
        ax.plot(self.history['epoch'], self.history['learning_rate'], linewidth=2, color='purple')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Value MAE plot
        ax = axes[1, 0]
        ax.plot(self.history['epoch'], self.history['value_mae'], linewidth=2, color='orange')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value MAE')
        ax.set_title('Value Prediction MAE')
        ax.grid(True, alpha=0.3)
        
        # Time plot
        ax = axes[1, 1]
        ax.plot(self.history['epoch'], self.history['epoch_time'], linewidth=2, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (s)')
        ax.set_title('Epoch Training Time')
        ax.grid(True, alpha=0.3)
        
        # Combined metrics
        ax = axes[1, 2]
        ax2 = ax.twinx()
        ax.plot(self.history['epoch'], self.history['train_accuracy'], 'g-', label='Accuracy', linewidth=2)
        ax2.plot(self.history['epoch'], self.history['train_loss'], 'b-', label='Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy', color='g')
        ax2.set_ylabel('Loss', color='b')
        ax.set_title('Accuracy vs Loss')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = f'training_plots/training_curves_{self.board_size}x{self.board_size}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Training plots saved: {plot_file}")

def run_optimized_phase1():
    """Run optimized Phase 1 training pipeline"""
    
    # FLOW.md compliant configuration
    config_3x3 = {
        "board_size": 3,
        "training_samples": 50000,
        "epochs": 200,
        "batch_size": 128,
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "hidden_layers": [256, 128, 64],
        "loss_weights": {
            "policy": 1.0,
            "value": 0.5,
            "difficulty": 0.2
        },
        "checkpoint_interval": 10,
        "enable_augmentation": True,
        "move_range": [5, 20]  # FLOW.md: k ∈ [5,20]
    }
    
    print("=" * 80)
    print("🎯 PHASE 1: FOUNDATION TRAINING (3x3) - OPTIMIZED")
    print("=" * 80)
    print("📋 This pipeline will:")
    print("   • Generate 50k 3x3 puzzles with A* solved paths")
    print("   • Apply data augmentation (8 variants per puzzle)")
    print("   • Train foundation model with supervised learning")
    print("   • Save comprehensive artifacts and metrics")
    print("=" * 80)
    
    logger.info("Starting optimized Phase 1 training pipeline")
    
    try:
        # Step 1: Generate or load training data
        data_file = 'puzzle_data/3x3_training_data.pkl'
        if not os.path.exists(data_file):
            logger.info("📊 Step 1: Generating training dataset with A* solver...")
            generator = EnhancedPuzzleGenerator(board_size=3)
            dataset = generator.generate_dataset(
                num_samples=config_3x3["training_samples"],
                output_file=data_file,
                enable_augmentation=config_3x3["enable_augmentation"]
            )
            logger.info("✅ Dataset generation completed!")
        else:
            logger.info(f"📥 Found existing dataset: {data_file}")
        
        # Step 2: Load training data
        logger.info("📥 Step 2: Loading and preprocessing training data...")
        dataset = OptimizedPuzzleDataset(data_file, board_size=3)
        
        # Create optimized data loader
        train_loader = DataLoader(
            dataset,
            batch_size=config_3x3["batch_size"],
            shuffle=True,
            num_workers=0,  # Reduced to avoid multiprocessing issues
            pin_memory=True
        )
        
        logger.info(f"✅ Loaded {len(dataset)} training samples")
        
        # Step 3: Initialize and run training
        logger.info("🎯 Step 3: Starting foundation model training...")
        trainer = OptimizedPhase1Trainer(config_3x3)
        history = trainer.train(train_loader)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PHASE 1 COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("📁 Generated artifacts:")
        logger.info("   • models/numpuz_3x3_foundation.pth - Final trained model")
        logger.info("   • models/numpuz_3x3_best.pth - Best accuracy model")
        logger.info("   • models/training_history_3x3_*.json - Training metrics")
        logger.info("   • training_plots/training_curves_3x3_*.png - Training plots")
        logger.info("   • models/train_config_3x3_*.yaml - Training configuration")
        logger.info("   • models/model_config_3x3_*.json - Model architecture")
        logger.info("=" * 80)
        
        return history
        
    except Exception as e:
        logger.error(f"❌ Error in Phase 1 pipeline: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Run optimized Phase 1
    history = run_optimized_phase1()
    
    if history is not None:
        print("\n🎊 Phase 1 completed successfully! Ready for Phase 2.")
    else:
        print("\n💥 Phase 1 failed. Check logs for details.")