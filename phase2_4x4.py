#!/usr/bin/env python3
"""
🎯 PHASE 2: SCALING TRAINING (4x4) - Transfer Learning from 3x3
Optimized version with transfer learning, curriculum learning, and FLOW.md compliance
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
        logging.FileHandler('training_phase2.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class EnhancedPuzzleGenerator4x4:
    """Enhanced puzzle generator for 4x4 with advanced heuristics and FLOW.md compliance"""
    
    def __init__(self, board_size: int = 4):
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
        
        # Pattern Database partitions (FLOW.md: 6-6-3)
        self.pdb_partitions = self._create_pattern_databases()
        
        logger.info(f"Initialized EnhancedPuzzleGenerator for {board_size}x{board_size}")

    def _create_pattern_databases(self) -> Dict:
        """Create Pattern Databases for 4x4 puzzles (FLOW.md: PDB partition 6-6-3)"""
        # Simplified PDB implementation - in practice would precompute all patterns
        pdb = {
            'partition1': {'size': 6, 'patterns': {}},  # Tiles 1-6
            'partition2': {'size': 6, 'patterns': {}},  # Tiles 7-12  
            'partition3': {'size': 3, 'patterns': {}},  # Tiles 13-15
        }
        logger.info("Created Pattern Database partitions: 6-6-3")
        return pdb

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

    def linear_conflict(self, state: List[List[int]]) -> int:
        """Calculate linear conflict heuristic (enhances Manhattan distance)"""
        conflict = 0
        
        # Check rows
        for i in range(self.board_size):
            row = state[i]
            for j1 in range(self.board_size):
                for j2 in range(j1 + 1, self.board_size):
                    tile1, tile2 = row[j1], row[j2]
                    if tile1 != 0 and tile2 != 0:
                        target_i1, target_j1 = self.target_positions[tile1]
                        target_i2, target_j2 = self.target_positions[tile2]
                        if target_i1 == i and target_i2 == i and target_j1 > target_j2:
                            conflict += 2
        
        # Check columns
        for j in range(self.board_size):
            col = [state[i][j] for i in range(self.board_size)]
            for i1 in range(self.board_size):
                for i2 in range(i1 + 1, self.board_size):
                    tile1, tile2 = col[i1], col[i2]
                    if tile1 != 0 and tile2 != 0:
                        target_i1, target_j1 = self.target_positions[tile1]
                        target_i2, target_j2 = self.target_positions[tile2]
                        if target_j1 == j and target_j2 == j and target_i1 > target_i2:
                            conflict += 2
        
        return conflict

    def enhanced_heuristic(self, state: List[List[int]]) -> int:
        """Enhanced heuristic: max(Manhattan, Manhattan + Linear Conflict)"""
        manhattan = self.manhattan_distance(state)
        conflict = self.linear_conflict(state)
        return max(manhattan, manhattan + conflict)

    def ida_star_solve(self, start_state: List[List[int]], max_nodes: int = 50000) -> Optional[List[int]]:
        """Solve puzzle using IDA* algorithm with enhanced heuristic (FLOW.md compliant)"""
        target_state = self.create_target_state()
        
        if start_state == target_state:
            return []
        
        threshold = self.enhanced_heuristic(start_state)
        path = []
        
        nodes_expanded = [0]

        while nodes_expanded[0] < max_nodes:
            result = self._ida_star_search(start_state, 0, threshold, path, nodes_expanded, max_nodes)
            
            if isinstance(result, list):
                logger.debug(f"IDA* solved puzzle with {len(result)} moves, expanded {nodes_expanded[0]} nodes")
                return result
            
            if result == float('inf'):
                logger.warning(f"IDA* failed to solve puzzle within {max_nodes} nodes")
                return None
            
            threshold = result
        
        logger.warning(f"IDA* reached node limit: {max_nodes} (expanded {nodes_expanded[0]} nodes)")
        return None

    def _ida_star_search(self, state: List[List[int]], g: int, threshold: int, 
                        path: List[int], nodes_expanded: int, max_nodes: int) -> int:
        """Recursive search for IDA*"""
        h = self.enhanced_heuristic(state)
        f = g + h
        
        if f > threshold:
            return f
        
        if h == 0:  # Solved
            return path
        
        min_cost = float('inf')
        
        # Try all possible moves
        for move_idx, move in enumerate(self.moves):
            new_state = self.move_tile(state, move)
            if new_state and nodes_expanded[0] < max_nodes:
                nodes_expanded[0] += 1
                
                # Avoid going back to previous state
                if len(path) > 0 and move_idx == (path[-1] ^ 1):  # Opposite move
                    continue
                
                path.append(move_idx)
                result = self._ida_star_search(new_state, g + 1, threshold, path, nodes_expanded, max_nodes)
                
                if isinstance(result, list):
                    return result
                
                if result < min_cost:
                    min_cost = result
                
                path.pop()
        
        return min_cost

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

    def generate_training_sample(self, min_moves: int = 10, max_moves: int = 50) -> Optional[Tuple]:
        """Generate one training sample with IDA* solved path (FLOW.md compliant)"""
        # Start from solved state and make k random moves (k ∈ [10,50])
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
        
        # Solve using IDA* to get optimal path
        solution = self.ida_star_solve(state)
        
        if solution and len(solution) > 0:
            # Convert state to enhanced neural network input format (one-hot + features)
            state_encoded = self.state_to_enhanced_encoding(state)
            
            # Create action probabilities (one-hot for first optimal move)
            action_probs = [0.0] * 4  # 4 possible moves
            first_move = solution[0]
            action_probs[first_move] = 1.0
            
            # Value: estimate of solvability (1.0 = easy, 0.1 = hard)
            optimal_moves = len(solution)
            value = max(0.1, 1.0 - (optimal_moves / 100.0))  # Normalize based on max expected moves
            
            # Difficulty based on optimal solution length (Curriculum learning stages)
            if optimal_moves <= 25:
                difficulty_class = 0  # Easy (Stage 1)
            elif optimal_moves <= 40:
                difficulty_class = 1  # Medium (Stage 2)
            else:
                difficulty_class = 2  # Hard (Stage 3)
            
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
                
                # One-hot encoding for tile value (17 channels for 4x4: 0-15 + empty)
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

    def generate_dataset(self, num_samples: int = 100000, 
                        output_file: str = 'puzzle_data/4x4_training_data.pkl',
                        enable_augmentation: bool = True,
                        curriculum_stage: str = "all") -> List[Tuple]:
        """Generate complete training dataset with optional augmentation and curriculum stages"""
        logger.info(f"🚀 Generating {num_samples} training samples for {self.board_size}x{self.board_size}")
        logger.info(f"Augmentation: {'ENABLED' if enable_augmentation else 'DISABLED'}")
        logger.info(f"Curriculum Stage: {curriculum_stage}")
        
        Path('puzzle_data').mkdir(exist_ok=True)
        
        # Define move ranges for curriculum stages
        stage_move_ranges = {
            "easy": [10, 25],
            "medium": [25, 40], 
            "hard": [40, 50],
            "all": [10, 50]
        }
        
        move_range = stage_move_ranges[curriculum_stage]
        base_samples_needed = num_samples // 8 if enable_augmentation else num_samples
        dataset = []
        success_count = 0
        fail_count = 0
        
        # Progress bar for sample generation
        with tqdm(total=base_samples_needed, desc=f"🎲 Generating {curriculum_stage} samples", 
                 bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}') as pbar:
            
            while len(dataset) < base_samples_needed:
                sample = self.generate_training_sample(min_moves=move_range[0], max_moves=move_range[1])
                
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
        logger.info(f"📈 Move range: {move_range}, Avg moves: {np.mean([x[5] for x in dataset]):.1f}")
        
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

class OptimizedPuzzleDataset4x4(Dataset):
    """Optimized Dataset for 4x4 with caching and efficient data loading"""
    
    def __init__(self, data_file: str, board_size: int = 4):
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

class EnhancedNumpuzNetwork4x4(nn.Module):
    """Enhanced neural network architecture for 4x4 with transfer learning support"""
    
    def __init__(self, board_size: int = 4, hidden_layers: List[int] = [512, 256, 128], 
                 transfer_config: Dict = None):
        super(EnhancedNumpuzNetwork4x4, self).__init__()
        self.board_size = board_size
        self.total_tiles = board_size * board_size
        
        # Input size: (one-hot + features) for each tile
        # For 4x4: 16 tiles * (17 one-hot + 3 features) = 320
        self.encoding_size = self.total_tiles * ((self.total_tiles + 1) + 3)
        self.input_size = self.encoding_size
        
        logger.info(f"🧠 Building 4x4 network: input_size={self.input_size}, hidden_layers={hidden_layers}")
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(self.input_size)
        
        # Build encoder layers with transfer learning support
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
            nn.Linear(prev_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 4),  # 4 possible moves
            nn.Softmax(dim=1)
        )
        
        # Value head (estimate steps-to-solve / solvability)
        self.value_head = nn.Sequential(
            nn.Linear(prev_size, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1),
            nn.Tanh()
        )
        
        # Curriculum head (for stage-based training)
        self.curriculum_head = nn.Sequential(
            nn.Linear(prev_size, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)  # 3 curriculum stages: easy, medium, hard
        )
        
        # Transfer learning setup
        self.transfer_config = transfer_config
        self.frozen_layers = []
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def transfer_from_3x3(self, model_3x3_path: str):
        """Transfer learning from 3x3 model (FLOW.md: mapping encoder weights)"""
        if not os.path.exists(model_3x3_path):
            logger.warning(f"❌ 3x3 model not found: {model_3x3_path}")
            return False
        
        try:
            # Load 3x3 model checkpoint
            checkpoint = torch.load(model_3x3_path, map_location='cpu')
            model_3x3_state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            # Get current state dict
            current_state_dict = self.state_dict()
            
            # Mapping strategy: match layer names and adapt sizes
            transferred_count = 0
            total_count = 0
            
            for name, param in model_3x3_state_dict.items():
                if name in current_state_dict:
                    current_param = current_state_dict[name]
                    
                    # Handle size mismatches
                    if param.shape == current_param.shape:
                        # Direct transfer
                        current_state_dict[name] = param
                        transferred_count += 1
                    elif len(param.shape) == 2 and len(current_param.shape) == 2:
                        # Linear layer with different sizes - partial transfer
                        min_rows = min(param.shape[0], current_param.shape[0])
                        min_cols = min(param.shape[1], current_param.shape[1])
                        current_state_dict[name][:min_rows, :min_cols] = param[:min_rows, :min_cols]
                        transferred_count += 1
                        logger.debug(f"Partially transferred {name}: {param.shape} -> {current_param.shape}")
                    
                    total_count += 1
            
            # Load the modified state dict
            self.load_state_dict(current_state_dict)
            
            logger.info(f"✅ Transferred {transferred_count}/{total_count} layers from 3x3 model")
            return True
            
        except Exception as e:
            logger.error(f"❌ Transfer learning failed: {e}")
            return False
    
    def freeze_layers(self, layer_names: List[str]):
        """Freeze specified layers for transfer learning"""
        for name, param in self.named_parameters():
            for layer_name in layer_names:
                if layer_name in name:
                    param.requires_grad = False
                    self.frozen_layers.append(name)
                    break
        
        logger.info(f"❄️ Frozen {len(self.frozen_layers)} layers: {self.frozen_layers}")
    
    def unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.parameters():
            param.requires_grad = True
        self.frozen_layers = []
        logger.info("🔓 Unfroze all layers")
    
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
        curriculum = self.curriculum_head(x)
        
        return policy, value, curriculum

class OptimizedPhase2Trainer:
    """Optimized Phase 2 trainer with transfer learning and curriculum learning"""
    
    def __init__(self, config: Dict[str, Any], transfer_model_path: str = None):
        self.config = config
        self.board_size = config["board_size"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"🚀 Initializing Phase2Trainer for {self.board_size}x{self.board_size}")
        logger.info(f"📊 Using device: {self.device}")
        if self.device.type == 'cuda':
            logger.info(f"🎯 GPU: {torch.cuda.get_device_name()}")
        
        # CRITICAL: Find Phase 1 model (required for Phase 2)
        self.transfer_model_path = self._find_phase1_model(transfer_model_path)
        
        if not self.transfer_model_path:
            logger.error("=" * 80)
            logger.error("❌ PHASE 2 REQUIRES TRANSFER LEARNING FROM PHASE 1!")
            logger.error("=" * 80)
            logger.error("Could not find numpuz_3x3_best.pth in any location.")
            logger.error("Please ensure Phase 1 training is completed first.")
            logger.error("=" * 80)
            raise FileNotFoundError("Phase 1 model (numpuz_3x3_best.pth) is required but not found")
        
        # Validate and load Phase 1 inputs
        self.phase1_inputs = self._validate_phase1_inputs(self.transfer_model_path)
        
        # Initialize model with transfer learning
        self.model = EnhancedNumpuzNetwork4x4(
            board_size=config["board_size"],
            hidden_layers=config["hidden_layers"]
        ).to(self.device)
        
        # Apply transfer learning (mandatory for Phase 2)
        logger.info(f"🔄 Applying transfer learning from: {self.transfer_model_path}")
        success = self.model.transfer_from_3x3(self.transfer_model_path)
        
        if not success:
            logger.error("❌ Transfer learning failed! Cannot proceed with Phase 2.")
            raise RuntimeError("Failed to load Phase 1 model weights")
        
        # Freeze initial layers as per FLOW.md
        if config.get("freeze_layers"):
            self.model.freeze_layers(config["freeze_layers"])
        
        # Print model summary
        self._print_model_summary()
        
        # Optimizer with different learning rates for frozen/unfrozen layers
        if self.model.frozen_layers:
            # Separate parameters for frozen and unfrozen layers
            frozen_params = []
            unfrozen_params = []
            
            for name, param in self.model.named_parameters():
                if any(frozen_name in name for frozen_name in self.model.frozen_layers):
                    frozen_params.append(param)
                else:
                    unfrozen_params.append(param)
            
            self.optimizer = optim.Adam([
                {'params': frozen_params, 'lr': config.get("frozen_lr", config["learning_rate"] * 0.1)},
                {'params': unfrozen_params, 'lr': config["learning_rate"]}
            ], weight_decay=config.get("weight_decay", 1e-4))
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=config["learning_rate"],
                weight_decay=config.get("weight_decay", 1e-4)
            )
        
        # Enhanced loss functions
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        self.curriculum_criterion = nn.CrossEntropyLoss()
        
        # Cosine annealing scheduler (FLOW.md compliant)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["epochs"]
        )
        
        # Curriculum learning state
        self.curriculum_stage = config.get("curriculum_start_stage", "easy")
        self.stage_epochs = config.get("stage_epochs", {})
        self.current_stage_epoch = 0
        
        # Training history with comprehensive metrics
        self.history = {
            'epoch': [],
            'train_loss': [], 'policy_loss': [], 'value_loss': [], 'curriculum_loss': [],
            'train_accuracy': [], 'learning_rate': [], 'epoch_time': [],
            'train_moves_accuracy': [], 'value_mae': [], 'curriculum_stage': []
        }
        
        self.best_accuracy = 0.0
        self.start_time = None
        
        # Create output directories
        Path('models').mkdir(exist_ok=True)
        Path('puzzle_data').mkdir(exist_ok=True)
        Path('training_plots').mkdir(exist_ok=True)
        Path('phase2_output').mkdir(exist_ok=True)
        
    def _find_phase1_model(self, config_path: Optional[str] = None) -> Optional[str]:
        """
        Tìm kiếm model Phase 1 trong nhiều vị trí (Colab-optimized)
        
        Search priority:
        1. Config-specified path
        2. /content/ (Colab working directory)
        3. /content/drive/MyDrive/ (Google Drive - common locations)
        4. Current directory và subdirectories
        5. Recursive search trong toàn bộ /content/
        
        Returns:
            Path to Phase 1 model hoặc None nếu không tìm thấy
        """
        logger.info("🔍 Searching for Phase 1 model (numpuz_3x3_best.pth)...")
        
        # Define search patterns với priority
        search_locations = []
        
        # 1. Config-specified path (highest priority)
        if config_path:
            search_locations.append(config_path)
        
        # 2. Colab /content/ directory (common working dir)
        search_locations.extend([
            "/content/numpuz_3x3_best.pth",
            "/content/models/numpuz_3x3_best.pth",
            "/content/phase1_output/models/numpuz_3x3_best.pth",
            "/content/phase1_output/numpuz_3x3_best.pth",
        ])
        
        # 3. Google Drive locations (if mounted)
        if os.path.exists("/content/drive/MyDrive"):
            search_locations.extend([
                "/content/drive/MyDrive/numpuz_3x3_best.pth",
                "/content/drive/MyDrive/NumpuzAI/numpuz_3x3_best.pth",
                "/content/drive/MyDrive/NumpuzAI/models/numpuz_3x3_best.pth",
                "/content/drive/MyDrive/NumpuzAI/phase1_output/models/numpuz_3x3_best.pth",
                "/content/drive/MyDrive/AI_Projects/numpuz_3x3_best.pth",
            ])
        
        # 4. Current directory và subdirectories
        search_locations.extend([
            "numpuz_3x3_best.pth",
            "models/numpuz_3x3_best.pth",
            "phase1_output/models/numpuz_3x3_best.pth",
            "phase1_output/numpuz_3x3_best.pth",
            "../numpuz_3x3_best.pth",
            "../models/numpuz_3x3_best.pth",
        ])
        
        # Try exact paths first (fast)
        logger.info(f"   📂 Checking {len(search_locations)} known locations...")
        for i, path in enumerate(search_locations, 1):
            if path and os.path.exists(path):
                logger.info(f"   ✅ Found at location #{i}: {path}")
                return os.path.abspath(path)
        
        # 5. Recursive search in /content/ (slower but thorough)
        logger.info("   🔍 Performing recursive search in /content/...")
        search_dirs = ['/content'] if os.path.exists('/content') else ['.']
        
        for search_dir in search_dirs:
            for root, dirs, files in os.walk(search_dir):
                # Skip common large directories
                dirs[:] = [d for d in dirs if d not in [
                    '.git', '__pycache__', 'venv', 'env', 'node_modules',
                    '.ipynb_checkpoints', 'sample_data'
                ]]
                
                for file in files:
                    if file == 'numpuz_3x3_best.pth':
                        found_path = os.path.join(root, file)
                        logger.info(f"   ✅ Found via recursive search: {found_path}")
                        return os.path.abspath(found_path)
        
        # Not found anywhere
        logger.warning("   ❌ Phase 1 model not found in any location!")
        logger.warning("   Searched locations:")
        for loc in search_locations[:10]:  # Show first 10
            logger.warning(f"      • {loc}")
        logger.warning("      • ... and recursive search in /content/")
        
        return None
    
    def _validate_phase1_inputs(self, model_path: str) -> Dict[str, Optional[str]]:
        """
        Validate và tìm kiếm các files bổ sung từ Phase 1
        
        FLOW.md requires:
        - ✅ numpuz_3x3_best.pth (weights) - MANDATORY
        - ⚠️ model_config_3x3.json (architecture) - OPTIONAL but recommended
        - ⚠️ train_config_3x3.yaml (hyperparams) - OPTIONAL
        
        Returns:
            Dict với keys: 'model', 'model_config', 'train_config'
        """
        logger.info("📋 Validating Phase 1 inputs...")
        
        inputs = {
            'model': model_path,
            'model_config': None,
            'train_config': None
        }
        
        # Get model directory for searching related files
        model_dir = os.path.dirname(model_path)
        parent_dir = os.path.dirname(model_dir) if model_dir else '.'
        
        # Search for model_config_3x3.json
        config_search_paths = [
            os.path.join(parent_dir, "model_config_3x3.json"),
            os.path.join(model_dir, "model_config_3x3.json"),
            os.path.join(parent_dir, "models", "model_config_3x3.json"),
            "/content/phase1_output/model_config_3x3.json",
            "/content/phase1_output/models/model_config_3x3.json",
            "phase1_output/model_config_3x3.json",
        ]
        
        for config_path in config_search_paths:
            if os.path.exists(config_path):
                inputs['model_config'] = config_path
                logger.info(f"   ✅ Found model config: {config_path}")
                break
        
        if not inputs['model_config']:
            logger.warning("   ⚠️ model_config_3x3.json not found - will infer from checkpoint")
        
        # Search for train_config_3x3.yaml
        train_config_search_paths = [
            os.path.join(parent_dir, "train_config_3x3.yaml"),
            os.path.join(model_dir, "train_config_3x3.yaml"),
            os.path.join(parent_dir, "models", "train_config_3x3.yaml"),
            "/content/phase1_output/train_config_3x3.yaml",
            "/content/phase1_output/models/train_config_3x3.yaml",
            "phase1_output/train_config_3x3.yaml",
        ]
        
        for config_path in train_config_search_paths:
            if os.path.exists(config_path):
                inputs['train_config'] = config_path
                logger.info(f"   ✅ Found train config: {config_path}")
                break
        
        if not inputs['train_config']:
            logger.warning("   ⚠️ train_config_3x3.yaml not found - using Phase 2 defaults")
        
        # Summary
        logger.info("📊 Phase 1 inputs summary:")
        logger.info(f"   • Model weights: ✅ {inputs['model']}")
        logger.info(f"   • Model config:  {'✅' if inputs['model_config'] else '⚠️ Missing'} {inputs['model_config'] or 'N/A'}")
        logger.info(f"   • Train config:  {'✅' if inputs['train_config'] else '⚠️ Missing'} {inputs['train_config'] or 'N/A'}")
        
        return inputs
    
    def _print_model_summary(self):
        """Print model architecture summary"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        
        logger.info("🧮 Model Architecture Summary:")
        logger.info(f"   • Input size: {self.model.input_size}")
        logger.info(f"   • Encoder layers: {self.config['hidden_layers']}")
        logger.info(f"   • Total parameters: {total_params:,}")
        logger.info(f"   • Trainable parameters: {trainable_params:,}")
        logger.info(f"   • Frozen parameters: {frozen_params:,}")
        logger.info(f"   • Transfer learning: {'APPLIED' if self.transfer_model_path else 'NONE'}")
        
        # Print layer details
        for name, module in self.model.named_children():
            if hasattr(module, 'weight'):
                logger.info(f"   • {name}: {tuple(module.weight.shape)}")
    
    def update_curriculum_stage(self, epoch: int):
        """Update curriculum learning stage based on epoch progress"""
        stages = ["easy", "medium", "hard"]
        current_idx = stages.index(self.curriculum_stage)
        
        # Check if we should advance to next stage
        stage_config = self.config.get("curriculum_stages", {})
        stage_epochs = stage_config.get(self.curriculum_stage, 100)
        
        self.current_stage_epoch += 1
        
        if (self.current_stage_epoch >= stage_epochs and 
            current_idx < len(stages) - 1 and
            epoch >= stage_config.get(f"{self.curriculum_stage}_min_epoch", 0)):
            
            self.curriculum_stage = stages[current_idx + 1]
            self.current_stage_epoch = 0
            logger.info(f"📚 Advancing to curriculum stage: {self.curriculum_stage}")
            
            # Unfreeze layers if specified
            if self.curriculum_stage == "hard" and self.config.get("unfreeze_at_hard", False):
                self.model.unfreeze_all()
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Enhanced training for one epoch with curriculum learning"""
        self.model.train()
        
        metrics = {
            'total_loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0, 'curriculum_loss': 0.0,
            'correct_predictions': 0, 'total_samples': 0, 'value_errors': 0.0
        }
        
        # Progress bar with enhanced formatting
        pbar = tqdm(dataloader, desc=f"📚 Epoch {epoch+1:3d} [{self.curriculum_stage:6}]", 
                   bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for batch_idx, (states, action_targets, value_targets, curriculum_targets, optimal_moves) in enumerate(pbar):
            # Move to device with non-blocking transfers
            states = states.to(self.device, non_blocking=True)
            action_targets = action_targets.to(self.device, non_blocking=True)
            value_targets = value_targets.to(self.device, non_blocking=True)
            curriculum_targets = curriculum_targets.squeeze().to(self.device, non_blocking=True)
            
            # Forward pass
            self.optimizer.zero_grad()
            policy_pred, value_pred, curriculum_pred = self.model(states)
            
            # Calculate losses
            _, action_indices = torch.max(action_targets, 1)
            policy_loss = self.policy_criterion(policy_pred, action_indices)
            value_loss = self.value_criterion(value_pred.squeeze(), value_targets.squeeze())
            curriculum_loss = self.curriculum_criterion(curriculum_pred, curriculum_targets)
            
            # Combined loss with weights (FLOW.md: lambda_policy=1.0, lambda_value=0.5)
            loss_weights = self.config.get("loss_weights", {"policy": 1.0, "value": 0.5, "curriculum": 0.3})
            total_loss = (loss_weights["policy"] * policy_loss + 
                         loss_weights["value"] * value_loss + 
                         loss_weights["curriculum"] * curriculum_loss)
            
            # Backward pass with gradient clipping
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            batch_size = states.size(0)
            metrics['total_loss'] += total_loss.item() * batch_size
            metrics['policy_loss'] += policy_loss.item() * batch_size
            metrics['value_loss'] += value_loss.item() * batch_size
            metrics['curriculum_loss'] += curriculum_loss.item() * batch_size
            
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
        avg_metrics['curriculum_loss'] = metrics['curriculum_loss'] / metrics['total_samples']
        avg_metrics['accuracy'] = metrics['correct_predictions'] / metrics['total_samples']
        avg_metrics['value_mae'] = metrics['value_errors'] / metrics['total_samples']
        
        return avg_metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader = None) -> Dict[str, List]:
        """Enhanced main training loop with curriculum learning"""
        logger.info(f"\n🎯 Starting Phase 2 Training for {self.board_size}x{self.board_size}")
        logger.info(f"📊 Training samples: {len(train_loader.dataset):,}")
        logger.info(f"⏰ Epochs: {self.config['epochs']}")
        logger.info(f"📦 Batch size: {self.config['batch_size']}")
        logger.info(f"📈 Learning rate: {self.config['learning_rate']}")
        logger.info(f"📚 Curriculum: {self.curriculum_stage} -> hard")
        logger.info(f"🔄 Transfer learning: {'APPLIED' if self.transfer_model_path else 'NONE'}")
        logger.info("─" * 60)
        
        self.start_time = time.time()
        best_accuracy = 0.0
        
        for epoch in range(self.config["epochs"]):
            epoch_start = time.time()
            
            # Update curriculum stage
            self.update_curriculum_stage(epoch)
            
            # Train one epoch
            epoch_metrics = self.train_epoch(train_loader, epoch)
            epoch_time = time.time() - epoch_start
            
            # Update learning rate scheduler
            self.scheduler.step()
            
            # Update history
            self.history['epoch'].append(epoch + 1)
            self.history['train_loss'].append(epoch_metrics['total_loss'])
            self.history['policy_loss'].append(epoch_metrics['policy_loss'])
            self.history['value_loss'].append(epoch_metrics['value_loss'])
            self.history['curriculum_loss'].append(epoch_metrics['curriculum_loss'])
            self.history['train_accuracy'].append(epoch_metrics['accuracy'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_time'].append(epoch_time)
            self.history['value_mae'].append(epoch_metrics['value_mae'])
            self.history['curriculum_stage'].append(self.curriculum_stage)
            
            # Print epoch summary
            logger.info(
                f"📅 Epoch {epoch+1:3d}/{self.config['epochs']} | "
                f"Stage: {self.curriculum_stage:6} | "
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
            if (epoch + 1) % self.config.get("checkpoint_interval", 5) == 0:
                self.save_checkpoint(f"numpuz_{self.board_size}x{self.board_size}_epoch_{epoch+1}.pth")
        
        total_time = time.time() - self.start_time
        logger.info("─" * 60)
        logger.info(f"✅ Training completed in {total_time/3600:.2f} hours")
        logger.info(f"🏆 Best accuracy: {best_accuracy*100:.2f}%")
        
        # Save final model and artifacts
        self.save_final_artifacts()
        
        return self.history
    
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
            'curriculum_stage': self.curriculum_stage,
            'transfer_model': self.transfer_model_path,
            'timestamp': datetime.now().isoformat(),
            'total_training_time': time.time() - self.start_time if self.start_time else 0
        }
        
        torch.save(checkpoint, f"models/{filename}")
        logger.info(f"💾 Checkpoint saved: models/{filename}")
    
    def save_final_artifacts(self):
        """Save all final artifacts for Phase 2"""
        
        # Save final model
        self.save_checkpoint(f"numpuz_{self.board_size}x{self.board_size}_phase2.pth")
        
        # Save training history
        history_file = f"phase2_output/training_history_{self.board_size}x{self.board_size}.json"
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        logger.info(f"📊 Training history saved: {history_file}")
        
        # Generate and save plots
        self.generate_training_plots()
        
        # Save configuration
        config_file = f"phase2_output/train_config_{self.board_size}x{self.board_size}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"⚙️ Training config saved: {config_file}")
        
        # Save model architecture
        model_config = {
            'board_size': self.board_size,
            'input_size': self.model.input_size,
            'hidden_layers': self.config['hidden_layers'],
            'total_parameters': sum(p.numel() for p in self.model.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            'transfer_learning_applied': bool(self.transfer_model_path),
            'frozen_layers': self.model.frozen_layers
        }
        model_config_file = f"phase2_output/model_config_{self.board_size}x{self.board_size}.json"
        with open(model_config_file, 'w') as f:
            json.dump(model_config, f, indent=2)
        logger.info(f"🧠 Model config saved: {model_config_file}")
        
        # Create curriculum progress report
        curriculum_report = self._generate_curriculum_report()
        curriculum_file = f"phase2_output/curriculum_progress_{self.board_size}x{self.board_size}.json"
        with open(curriculum_file, 'w') as f:
            json.dump(curriculum_report, f, indent=2)
        logger.info(f"📚 Curriculum progress saved: {curriculum_file}")
        
        # Create comprehensive ZIP archive
        self._create_phase2_zip_archive()
        
        # Display training summary
        self.display_training_summary()

    def _generate_curriculum_report(self) -> Dict:
        """Generate curriculum learning progress report"""
        stages = ["easy", "medium", "hard"]
        report = {
            'total_epochs': len(self.history['epoch']),
            'stage_performance': {},
            'final_metrics': {}
        }
        
        for stage in stages:
            stage_indices = [i for i, s in enumerate(self.history['curriculum_stage']) if s == stage]
            if stage_indices:
                report['stage_performance'][stage] = {
                    'epochs': len(stage_indices),
                    'start_epoch': min(stage_indices) + 1,
                    'end_epoch': max(stage_indices) + 1,
                    'start_accuracy': self.history['train_accuracy'][min(stage_indices)] * 100,
                    'end_accuracy': self.history['train_accuracy'][max(stage_indices)] * 100,
                    'accuracy_improvement': (self.history['train_accuracy'][max(stage_indices)] - 
                                           self.history['train_accuracy'][min(stage_indices)]) * 100,
                    'avg_loss': np.mean([self.history['train_loss'][i] for i in stage_indices])
                }
        
        report['final_metrics'] = {
            'final_accuracy': self.history['train_accuracy'][-1] * 100,
            'final_loss': self.history['train_loss'][-1],
            'best_accuracy': max(self.history['train_accuracy']) * 100,
            'training_duration_hours': sum(self.history['epoch_time']) / 3600
        }
        
        return report

    def generate_training_plots(self):
        """Generate comprehensive training plots for Phase 2"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'NumpuzAI Phase 2 Training Progress - {self.board_size}x{self.board_size}', fontsize=16)
        
        # Color coding for curriculum stages
        stage_colors = {'easy': 'green', 'medium': 'orange', 'hard': 'red'}
        
        # Loss plot with curriculum stages
        ax = axes[0, 0]
        for i, stage in enumerate(self.history['curriculum_stage']):
            color = stage_colors.get(stage, 'blue')
            ax.scatter(i, self.history['train_loss'][i], color=color, alpha=0.6, s=20)
        ax.plot(self.history['epoch'], self.history['train_loss'], 'b-', alpha=0.3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss (Curriculum Stages)')
        ax.grid(True, alpha=0.3)
        
        # Add legend for curriculum stages
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                markersize=8, label=stage) 
                          for stage, color in stage_colors.items()]
        ax.legend(handles=legend_elements)
        
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
        ax.set_title('Cosine Annealing Schedule')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Loss breakdown
        ax = axes[1, 0]
        ax.plot(self.history['epoch'], self.history['policy_loss'], label='Policy Loss', alpha=0.7)
        ax.plot(self.history['epoch'], self.history['value_loss'], label='Value Loss', alpha=0.7)
        ax.plot(self.history['epoch'], self.history['curriculum_loss'], label='Curriculum Loss', alpha=0.7)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Time plot
        ax = axes[1, 1]
        ax.plot(self.history['epoch'], self.history['epoch_time'], linewidth=2, color='red')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Time (s)')
        ax.set_title('Epoch Training Time')
        ax.grid(True, alpha=0.3)
        
        # Stage transition plot
        ax = axes[1, 2]
        stage_epochs = {}
        for i, stage in enumerate(self.history['curriculum_stage']):
            if stage not in stage_epochs:
                stage_epochs[stage] = []
            stage_epochs[stage].append(i)
        
        for stage, epochs in stage_epochs.items():
            color = stage_colors.get(stage, 'blue')
            ax.scatter(epochs, [self.history['train_accuracy'][e] for e in epochs], 
                      color=color, label=stage, alpha=0.7)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy by Curriculum Stage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = f'phase2_output/training_curves_{self.board_size}x{self.board_size}.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📈 Training plots saved: {plot_file}")

    def _create_phase2_zip_archive(self):
        """Create ZIP archive with all Phase 2 outputs"""
        zip_filename = f"phase2_output_{self.board_size}x{self.board_size}.zip"
        
        logger.info(f"📦 Creating Phase 2 ZIP archive: {zip_filename}")
        
        try:
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Add all files from phase2_output directory
                for file_path in Path('phase2_output').rglob('*'):
                    if file_path.is_file():
                        arcname = f"phase2_output/{file_path.relative_to('phase2_output')}"
                        zipf.write(file_path, arcname)
                
                # Add best model
                model_file = f"models/numpuz_{self.board_size}x{self.board_size}_best.pth"
                if Path(model_file).exists():
                    zipf.write(model_file, f"phase2_output/models/numpuz_{self.board_size}x{self.board_size}_best.pth")
                
                # Add dataset info
                dataset_file = 'puzzle_data/4x4_training_data.pkl'
                if Path(dataset_file).exists():
                    # Create dataset info file instead of including full dataset
                    dataset_info = {
                        'dataset_file': '4x4_training_data.pkl',
                        'file_size_mb': round(Path(dataset_file).stat().st_size / (1024 ** 2), 2),
                        'location': str(Path(dataset_file).absolute()),
                        'note': 'Full dataset stored separately',
                        'last_updated': datetime.now().isoformat()
                    }
                    dataset_info_file = 'phase2_output/dataset_info.json'
                    with open(dataset_info_file, 'w') as f:
                        json.dump(dataset_info, f, indent=2)
                    zipf.write(dataset_info_file, 'phase2_output/dataset_info.json')
                    Path(dataset_info_file).unlink()
            
            zip_size = Path(zip_filename).stat().st_size / (1024 ** 2)
            logger.info(f"✅ Phase 2 ZIP archive created: {zip_filename} ({zip_size:.2f} MB)")
            
        except Exception as e:
            logger.error(f"❌ Failed to create ZIP archive: {e}")

    def display_training_summary(self):
        """Display comprehensive training summary for Phase 2"""
        
        print("\n" + "="*80)
        print("📊 PHASE 2 TRAINING SUMMARY (4x4)")
        print("="*80)
        
        # Curriculum stage analysis
        stages = ["easy", "medium", "hard"]
        stage_stats = {}
        
        for stage in stages:
            stage_indices = [i for i, s in enumerate(self.history['curriculum_stage']) if s == stage]
            if stage_indices:
                stage_stats[stage] = {
                    'epochs': len(stage_indices),
                    'start_acc': self.history['train_accuracy'][min(stage_indices)] * 100,
                    'end_acc': self.history['train_accuracy'][max(stage_indices)] * 100,
                    'improvement': (self.history['train_accuracy'][max(stage_indices)] - 
                                  self.history['train_accuracy'][min(stage_indices)]) * 100
                }
        
        # 1. Performance Metrics
        print("\n🎯 PERFORMANCE METRICS:")
        print("-" * 80)
        
        final_acc = self.history['train_accuracy'][-1] * 100
        best_acc = max(self.history['train_accuracy']) * 100
        final_loss = self.history['train_loss'][-1]
        
        print(f"Final Accuracy:         {final_acc:>6.2f}%")
        print(f"Best Accuracy:          {best_acc:>6.2f}%")
        print(f"Final Loss:             {final_loss:>6.4f}")
        
        # 2. Curriculum Learning Progress
        print("\n📚 CURRICULUM LEARNING PROGRESS:")
        print("-" * 80)
        for stage, stats in stage_stats.items():
            print(f"{stage.upper():<8}: {stats['epochs']:>3} epochs | "
                  f"Start: {stats['start_acc']:>5.1f}% | "
                  f"End: {stats['end_acc']:>5.1f}% | "
                  f"Improvement: {stats['improvement']:>+5.1f}%")
        
        # 3. Transfer Learning Info
        print("\n🔄 TRANSFER LEARNING:")
        print("-" * 80)
        print(f"Applied:           {'YES' if self.transfer_model_path else 'NO'}")
        if self.transfer_model_path:
            print(f"Source Model:       {Path(self.transfer_model_path).name}")
            print(f"Frozen Layers:      {len(self.model.frozen_layers)}")
        
        # 4. Training Duration
        print("\n⏱️  TRAINING DURATION:")
        print("-" * 80)
        total_time = sum(self.history['epoch_time'])
        print(f"Total Time:         {total_time/3600:>6.2f} hours")
        print(f"Average Epoch Time: {np.mean(self.history['epoch_time']):>6.2f} seconds")
        
        print("\n" + "="*80)
        print("✅ Phase 2 training summary complete!")
        print("="*80 + "\n")

def run_optimized_phase2():
    """Run optimized Phase 2 training pipeline with transfer learning"""
    
    # FLOW.md compliant configuration for Phase 2
    config_4x4 = {
        "board_size": 4,
        "training_samples": 100000,
        "epochs": 300,
        "batch_size": 256,
        "learning_rate": 0.0005,
        "weight_decay": 1e-4,
        "hidden_layers": [512, 256, 128],
        "loss_weights": {
            "policy": 1.0,
            "value": 0.5,
            "curriculum": 0.3
        },
        "checkpoint_interval": 5,
        "enable_augmentation": True,
        "move_range": [10, 50],  # FLOW.md: 10-50 moves
        "freeze_layers": ["encoder.0", "encoder.1"],  # First two encoder layers
        "frozen_lr": 0.00005,  # Lower LR for frozen layers
        "curriculum_start_stage": "easy",
        "curriculum_stages": {
            "easy": 100,
            "medium": 100,
            "hard": 100
        },
        "unfreeze_at_hard": True,
        "transfer_model": None  # Will be auto-detected
    }
    
    print("=" * 80)
    print("🎯 PHASE 2: SCALING TRAINING (4x4) - TRANSFER LEARNING")
    print("=" * 80)
    print("📋 This pipeline will:")
    print("   • Generate 100k 4x4 puzzles with IDA* solved paths")
    print("   • Apply transfer learning from 3x3 model (REQUIRED)")
    print("   • Use curriculum learning (easy → medium → hard)")
    print("   • Apply advanced heuristics (Manhattan + Linear Conflict)")
    print("   • Save comprehensive artifacts and metrics")
    print("=" * 80)
    
    logger.info("Starting optimized Phase 2 training pipeline")
    
    try:
        # Step 1: Generate or load training data
        data_file = 'puzzle_data/4x4_training_data.pkl'
        if not os.path.exists(data_file):
            logger.info("📊 Step 1: Generating 4x4 training dataset with IDA* solver...")
            generator = EnhancedPuzzleGenerator4x4(board_size=4)
            dataset = generator.generate_dataset(
                num_samples=config_4x4["training_samples"],
                output_file=data_file,
                enable_augmentation=config_4x4["enable_augmentation"],
                curriculum_stage="all"
            )
            logger.info("✅ Dataset generation completed!")
        else:
            logger.info(f"📥 Found existing dataset: {data_file}")
        
        # Step 2: Load training data
        logger.info("📥 Step 2: Loading and preprocessing training data...")
        dataset = OptimizedPuzzleDataset4x4(data_file, board_size=4)
        
        # Create optimized data loader
        train_loader = DataLoader(
            dataset,
            batch_size=config_4x4["batch_size"],
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        logger.info(f"✅ Loaded {len(dataset)} training samples")
        
        # Step 3: Initialize trainer (will auto-find Phase 1 model)
        logger.info("🎯 Step 3: Initializing Phase 2 trainer with transfer learning...")
        logger.info("   (Trainer will automatically search for Phase 1 model)")
        
        trainer = OptimizedPhase2Trainer(
            config_4x4, 
            transfer_model_path=config_4x4.get("transfer_model")
        )
        
        # Step 4: Run training
        logger.info("🚀 Step 4: Starting Phase 2 training...")
        history = trainer.train(train_loader)
        
        logger.info("\n" + "=" * 80)
        logger.info("🎉 PHASE 2 COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info("📁 Generated artifacts:")
        logger.info("   • phase2_output/ - All training outputs")
        logger.info("   • models/numpuz_4x4_phase2.pth - Final trained model")
        logger.info("   • models/numpuz_4x4_best.pth - Best accuracy model")
        logger.info("   • phase2_output_4x4.zip - Complete archive")
        logger.info("=" * 80)
        
        return history
        
    except FileNotFoundError as e:
        logger.error("\n" + "=" * 80)
        logger.error("❌ PHASE 2 INITIALIZATION FAILED")
        logger.error("=" * 80)
        logger.error(f"Error: {e}")
        logger.error("\n💡 SOLUTION:")
        logger.error("   1. Run Phase 1 training first to generate numpuz_3x3_best.pth")
        logger.error("   2. Or manually place the model file in one of these locations:")
        logger.error("      • /content/numpuz_3x3_best.pth")
        logger.error("      • /content/models/numpuz_3x3_best.pth")
        logger.error("      • /content/phase1_output/models/numpuz_3x3_best.pth")
        logger.error("      • /content/drive/MyDrive/numpuz_3x3_best.pth")
        logger.error("=" * 80)
        return None
        
    except Exception as e:
        logger.error(f"❌ Error in Phase 2 pipeline: {e}")
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
    
    # Run optimized Phase 2
    history = run_optimized_phase2()
    
    if history is not None:
        print("\n🎊 Phase 2 completed successfully! Ready for Phase 3.")
    else:
        print("\n💥 Phase 2 failed. Check logs for details.")