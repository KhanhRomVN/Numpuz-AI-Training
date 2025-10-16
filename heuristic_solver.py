# heuristic_solver.py
# Phase 2: Heuristic Solver & Expert Demonstrations for Sliding Number Puzzles, Generate expert demonstrations for 3×3, 4×4, and 5×5 puzzles
# =============================================================================

import numpy as np
import json
import pickle
from typing import List, Tuple, Dict, Optional, Set
from collections import deque
import heapq
import time
from tqdm import tqdm
from datetime import datetime
import os
from multiprocessing import Pool, cpu_count
import argparse

# =============================================================================
# CORE CLASSES
# =============================================================================

class SlidingPuzzleEnv:
    """Sliding number puzzle environment"""
    
    def __init__(self, size=4):
        self.size = size
        self.total_tiles = size * size
        self.goal_state = self._create_goal_state()
        self.reset()
    
    def _create_goal_state(self) -> np.ndarray:
        """Create goal state for sliding puzzle"""
        state = np.arange(1, self.total_tiles + 1, dtype=np.int32)
        state[-1] = 0  # Last tile is empty
        return state
    
    def reset(self, shuffle_range: Tuple[int, int] = None) -> np.ndarray:
        """Reset with configurable shuffle range"""
        if shuffle_range is None:
            # Default shuffle ranges by size
            ranges = {3: (20, 50), 4: (50, 150), 5: (100, 250)}
            shuffle_range = ranges.get(self.size, (50, 150))
        
        state = self.goal_state.copy()
        num_shuffles = np.random.randint(*shuffle_range)
        
        for _ in range(num_shuffles):
            valid_moves = self._get_valid_moves(state)
            if valid_moves:
                move = valid_moves[np.random.randint(len(valid_moves))]
                state = self._apply_move(state, move)
        
        self.current_state = state.copy()
        return state
    
    def _get_valid_moves(self, state: np.ndarray) -> List[int]:
        """Get list of tiles that can move to empty position"""
        empty_pos = np.where(state == 0)[0][0]
        empty_row, empty_col = divmod(empty_pos, self.size)
        valid_moves = []
        
        # Check 4 directions: up, down, left, right
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for dr, dc in directions:
            new_row, new_col = empty_row + dr, empty_col + dc
            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                move_pos = new_row * self.size + new_col
                valid_moves.append(move_pos)
        
        return valid_moves
    
    def _apply_move(self, state: np.ndarray, tile_pos: int) -> np.ndarray:
        """Move tile from tile_pos to empty position"""
        new_state = state.copy()
        empty_pos = np.where(new_state == 0)[0][0]
        new_state[empty_pos], new_state[tile_pos] = new_state[tile_pos], new_state[empty_pos]
        return new_state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute one move"""
        valid_moves = self._get_valid_moves(self.current_state)
        
        if action not in valid_moves:
            raise ValueError(f"Invalid move: {action}. Valid moves: {valid_moves}")
        
        self.current_state = self._apply_move(self.current_state, action)
        done = self.is_solved()
        reward = 10.0 if done else -0.1
        
        return self.current_state, reward, done, {}

    def get_state(self) -> np.ndarray:
        return self.current_state.copy()
    
    def is_solved(self) -> bool:
        return np.array_equal(self.current_state, self.goal_state)


class SolvabilityChecker:
    """Check puzzle solvability"""
    
    def __init__(self, size=4):
        self.size = size
    
    def is_solvable(self, state: np.ndarray) -> bool:
        state_no_empty = state[state != 0]
        inversions = self._count_inversions(state_no_empty)
        empty_pos = np.where(state == 0)[0][0]
        empty_row = self.size - (empty_pos // self.size)
        
        if self.size % 2 == 1:
            return inversions % 2 == 0
        else:
            return (inversions + empty_row) % 2 == 1
    
    def _count_inversions(self, arr: np.ndarray) -> int:
        inversions = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    inversions += 1
        return inversions


class DifficultyClassifier:
    """Classify puzzle difficulty"""
    
    @staticmethod
    def classify_by_moves(num_moves: int, size: int) -> str:
        """Difficulty varies by puzzle size"""
        thresholds = {
            3: {'easy': 15, 'medium': 22, 'hard': 31},
            4: {'easy': 20, 'medium': 35, 'hard': 50},
            5: {'easy': 50, 'medium': 100, 'hard': 150}
        }
        
        t = thresholds.get(size, thresholds[4])
        
        if num_moves <= t['easy']:
            return 'easy'
        elif num_moves <= t['medium']:
            return 'medium'
        elif num_moves <= t['hard']:
            return 'hard'
        else:
            return 'expert'

# =============================================================================
# ADVANCED HEURISTICS
# =============================================================================

class AdvancedHeuristics:
    """Advanced heuristic functions for A* and IDA*"""
    
    def __init__(self, size=4):
        self.size = size
        self.total_tiles = size * size
        self._precompute_goal_positions()
    
    def _precompute_goal_positions(self):
        """Precompute goal positions"""
        self.goal_positions = {}
        for tile in range(1, self.total_tiles):
            self.goal_positions[tile] = divmod(tile - 1, self.size)
        self.goal_positions[0] = divmod(self.total_tiles - 1, self.size)
    
    def manhattan_distance(self, state: np.ndarray) -> float:
        """Manhattan distance heuristic"""
        distance = 0.0
        for pos, tile in enumerate(state):
            if tile == 0:
                continue
            goal_row, goal_col = self.goal_positions[tile]
            curr_row, curr_col = divmod(pos, self.size)
            distance += abs(curr_row - goal_row) + abs(curr_col - goal_col)
        return distance
    
    def linear_conflict(self, state: np.ndarray) -> float:
        """Linear conflict heuristic"""
        conflicts = 0
        
        # Check rows
        for row in range(self.size):
            row_tiles = state[row * self.size:(row + 1) * self.size]
            for i in range(len(row_tiles)):
                if row_tiles[i] == 0:
                    continue
                goal_row_i = self.goal_positions[row_tiles[i]][0]
                
                if goal_row_i == row:
                    for j in range(i + 1, len(row_tiles)):
                        if row_tiles[j] == 0:
                            continue
                        goal_row_j = self.goal_positions[row_tiles[j]][0]
                        
                        if goal_row_j == row:
                            goal_col_i = self.goal_positions[row_tiles[i]][1]
                            goal_col_j = self.goal_positions[row_tiles[j]][1]
                            
                            if goal_col_i > goal_col_j:
                                conflicts += 1
        
        # Check columns
        for col in range(self.size):
            col_tiles = state[col::self.size]
            for i in range(len(col_tiles)):
                if col_tiles[i] == 0:
                    continue
                goal_col_i = self.goal_positions[col_tiles[i]][1]
                
                if goal_col_i == col:
                    for j in range(i + 1, len(col_tiles)):
                        if col_tiles[j] == 0:
                            continue
                        goal_col_j = self.goal_positions[col_tiles[j]][1]
                        
                        if goal_col_j == col:
                            goal_row_i = self.goal_positions[col_tiles[i]][0]
                            goal_row_j = self.goal_positions[col_tiles[j]][0]
                            
                            if goal_row_i > goal_row_j:
                                conflicts += 1
        
        return float(conflicts) * 2.0
    
    def weighted_heuristic(self, state: np.ndarray) -> float:
        """Weighted combination: manhattan + linear conflict"""
        h_manhattan = self.manhattan_distance(state)
        h_conflict = self.linear_conflict(state)
        return h_manhattan + h_conflict

# =============================================================================
# SOLVERS
# =============================================================================

class AStarSolver:
    """A* Search with multiple heuristics"""
    
    def __init__(self, size=4, timeout=30):
        self.size = size
        self.timeout = timeout
        self.env = SlidingPuzzleEnv(size)
        self.heuristics = AdvancedHeuristics(size)
        self.goal_state = self.env.goal_state
    
    def solve(self, initial_state: np.ndarray) -> Optional[Tuple[List[int], int, float]]:
        """Solve puzzle using A* search"""
        start_time = time.time()
        
        initial_tuple = tuple(initial_state)
        h_initial = self.heuristics.weighted_heuristic(initial_state)
        
        open_set = [(h_initial, 0, initial_tuple, [])]
        closed_set: Set[Tuple] = set()
        
        nodes_explored = 0
        
        while open_set:
            if time.time() - start_time > self.timeout:
                return None
            
            f_score, g_score, current_tuple, path = heapq.heappop(open_set)
            current_state = np.array(current_tuple, dtype=np.int32)
            
            if np.array_equal(current_state, self.goal_state):
                solve_time = time.time() - start_time
                return path, len(path), solve_time
            
            if current_tuple in closed_set:
                continue
            
            closed_set.add(current_tuple)
            nodes_explored += 1
            
            valid_moves = self.env._get_valid_moves(current_state)
            
            for move_pos in valid_moves:
                next_state = self.env._apply_move(current_state, move_pos)
                next_tuple = tuple(next_state)
                
                if next_tuple in closed_set:
                    continue
                
                g_next = g_score + 1
                h_next = self.heuristics.weighted_heuristic(next_state)
                f_next = g_next + h_next
                
                new_path = path + [move_pos]
                heapq.heappush(open_set, (f_next, g_next, next_tuple, new_path))
        
        return None


class WeightedAStarSolver:
    """Weighted A* - trades optimality for speed"""
    
    def __init__(self, size=4, weight=1.5, timeout=60):
        self.size = size
        self.weight = weight  # w > 1 means faster but suboptimal
        self.timeout = timeout
        self.env = SlidingPuzzleEnv(size)
        self.heuristics = AdvancedHeuristics(size)
        self.goal_state = self.env.goal_state
    
    def solve(self, initial_state: np.ndarray) -> Optional[Tuple[List[int], int, float]]:
        """Solve using WA*: f(n) = g(n) + w*h(n)"""
        start_time = time.time()
        
        initial_tuple = tuple(initial_state)
        h_initial = self.heuristics.weighted_heuristic(initial_state)
        f_initial = 0 + self.weight * h_initial
        
        open_set = [(f_initial, 0, initial_tuple, [])]
        closed_set: Set[Tuple] = set()
        
        while open_set:
            if time.time() - start_time > self.timeout:
                return None
            
            f_score, g_score, current_tuple, path = heapq.heappop(open_set)
            current_state = np.array(current_tuple, dtype=np.int32)
            
            if np.array_equal(current_state, self.goal_state):
                solve_time = time.time() - start_time
                return path, len(path), solve_time
            
            if current_tuple in closed_set:
                continue
            
            closed_set.add(current_tuple)
            
            valid_moves = self.env._get_valid_moves(current_state)
            
            for move_pos in valid_moves:
                next_state = self.env._apply_move(current_state, move_pos)
                next_tuple = tuple(next_state)
                
                if next_tuple in closed_set:
                    continue
                
                g_next = g_score + 1
                h_next = self.heuristics.weighted_heuristic(next_state)
                f_next = g_next + self.weight * h_next
                
                new_path = path + [move_pos]
                heapq.heappush(open_set, (f_next, g_next, next_tuple, new_path))
        
        return None


class IDAStarSolver:
    """Iterative Deepening A* - memory efficient"""
    
    def __init__(self, size=4, max_depth=100):
        self.size = size
        self.max_depth = max_depth
        self.env = SlidingPuzzleEnv(size)
        self.heuristics = AdvancedHeuristics(size)
        self.goal_state = self.env.goal_state
    
    def solve(self, initial_state: np.ndarray) -> Optional[Tuple[List[int], int, float]]:
        """Solve using IDA*"""
        start_time = time.time()
        
        threshold = self.heuristics.weighted_heuristic(initial_state)
        path = []
        
        while threshold <= self.max_depth:
            result = self._search(initial_state, 0, threshold, path, set())
            
            if isinstance(result, list):
                solve_time = time.time() - start_time
                return result, len(result), solve_time
            
            if result == float('inf'):
                return None
            
            threshold = result
        
        return None
    
    def _search(self, state: np.ndarray, g: int, threshold: float, 
                path: List[int], visited: Set[Tuple]) -> any:
        """Recursive DFS with f-cost threshold"""
        
        h = self.heuristics.weighted_heuristic(state)
        f = g + h
        
        if f > threshold:
            return f
        
        if np.array_equal(state, self.goal_state):
            return path
        
        state_tuple = tuple(state)
        if state_tuple in visited:
            return float('inf')
        
        visited.add(state_tuple)
        
        min_threshold = float('inf')
        valid_moves = self.env._get_valid_moves(state)
        
        for move_pos in valid_moves:
            next_state = self.env._apply_move(state, move_pos)
            new_path = path + [move_pos]
            
            result = self._search(next_state, g + 1, threshold, new_path, visited.copy())
            
            if isinstance(result, list):
                return result
            
            if result < min_threshold:
                min_threshold = result
        
        return min_threshold


class GreedySolver:
    """Greedy Best-First Search - fast but suboptimal"""
    
    def __init__(self, size=4, max_moves=300):
        self.size = size
        self.max_moves = max_moves
        self.env = SlidingPuzzleEnv(size)
        self.heuristics = AdvancedHeuristics(size)
        self.goal_state = self.env.goal_state
    
    def solve(self, initial_state: np.ndarray) -> Optional[Tuple[List[int], int, float]]:
        """Solve using greedy best-first"""
        start_time = time.time()
        
        path = []
        current_state = initial_state.copy()
        visited = set()
        
        for move_count in range(self.max_moves):
            if np.array_equal(current_state, self.goal_state):
                solve_time = time.time() - start_time
                return path, len(path), solve_time
            
            current_tuple = tuple(current_state)
            visited.add(current_tuple)
            
            valid_moves = self.env._get_valid_moves(current_state)
            best_move = None
            best_h = float('inf')
            best_next_state = None
            
            for move_pos in valid_moves:
                next_state = self.env._apply_move(current_state, move_pos)
                next_tuple = tuple(next_state)
                
                if next_tuple in visited:
                    continue
                
                h = self.heuristics.weighted_heuristic(next_state)
                
                if h < best_h:
                    best_h = h
                    best_move = move_pos
                    best_next_state = next_state
            
            if best_move is None:
                if valid_moves:
                    best_move = valid_moves[0]
                    best_next_state = self.env._apply_move(current_state, best_move)
                else:
                    return None
            
            path.append(best_move)
            current_state = best_next_state
        
        return None

# =============================================================================
# SOLVER MANAGER
# =============================================================================

class MultiSizeSolverManager:
    """Manages solvers with size-specific strategies"""
    
    def __init__(self, size=4):
        self.size = size
        
        # Size-specific configurations
        if size == 3:
            # 3×3: Fast with A*
            self.astar = AStarSolver(size, timeout=10)
            self.wastar = None
            self.idastar = None
            self.greedy = GreedySolver(size, max_moves=100)
        elif size == 4:
            # 4×4: Optimized for speed - use Weighted A* as primary
            self.astar = None  # Too slow for large batches
            self.wastar = WeightedAStarSolver(size, weight=1.8, timeout=15)  # Faster with weight=1.8
            self.idastar = IDAStarSolver(size, max_depth=60)  # Reduced depth
            self.greedy = GreedySolver(size, max_moves=150)  # Reduced moves
        elif size == 5:
            # 5×5: Use WA* and accept suboptimal
            self.astar = None
            self.wastar = WeightedAStarSolver(size, weight=2.0, timeout=120)
            self.idastar = IDAStarSolver(size, max_depth=150)
            self.greedy = GreedySolver(size, max_moves=400)
        else:
            # Default: 4×4 strategy
            self.astar = AStarSolver(size, timeout=30)
            self.wastar = None
            self.idastar = IDAStarSolver(size, max_depth=80)
            self.greedy = GreedySolver(size, max_moves=200)
    
    def solve(self, initial_state: np.ndarray) -> Optional[Dict]:
        """Solve with size-specific fallback strategy"""
        
        # 3×3: A* → Greedy
        if self.size == 3:
            result = self.astar.solve(initial_state)
            if result:
                path, num_moves, solve_time = result
                return {
                    'path': path,
                    'num_moves': num_moves,
                    'solve_time': solve_time,
                    'solver_used': 'astar',
                    'optimal': True
                }
            
            result = self.greedy.solve(initial_state)
            if result:
                path, num_moves, solve_time = result
                return {
                    'path': path,
                    'num_moves': num_moves,
                    'solve_time': solve_time,
                    'solver_used': 'greedy',
                    'optimal': False
                }
        
        # 4×4: WA* → Greedy → IDA* (Optimized for speed)
        elif self.size == 4:
            # Try Weighted A* first (fastest)
            result = self.wastar.solve(initial_state)
            if result:
                path, num_moves, solve_time = result
                return {
                    'path': path,
                    'num_moves': num_moves,
                    'solve_time': solve_time,
                    'solver_used': 'wastar',
                    'optimal': False  # WA* is suboptimal but much faster
                }
            
            # Fallback to Greedy (very fast)
            result = self.greedy.solve(initial_state)
            if result:
                path, num_moves, solve_time = result
                return {
                    'path': path,
                    'num_moves': num_moves,
                    'solve_time': solve_time,
                    'solver_used': 'greedy',
                    'optimal': False
                }
            
            # Last resort: IDA* (slower but more thorough)
            result = self.idastar.solve(initial_state)
            if result:
                path, num_moves, solve_time = result
                return {
                    'path': path,
                    'num_moves': num_moves,
                    'solve_time': solve_time,
                    'solver_used': 'idastar',
                    'optimal': True
                }
        
        # 5×5: WA* → IDA* → Greedy
        elif self.size == 5:
            result = self.wastar.solve(initial_state)
            if result:
                path, num_moves, solve_time = result
                return {
                    'path': path,
                    'num_moves': num_moves,
                    'solve_time': solve_time,
                    'solver_used': 'wastar',
                    'optimal': False  # WA* is suboptimal
                }
            
            result = self.idastar.solve(initial_state)
            if result:
                path, num_moves, solve_time = result
                return {
                    'path': path,
                    'num_moves': num_moves,
                    'solve_time': solve_time,
                    'solver_used': 'idastar',
                    'optimal': True
                }
            
            result = self.greedy.solve(initial_state)
            if result:
                path, num_moves, solve_time = result
                return {
                    'path': path,
                    'num_moves': num_moves,
                    'solve_time': solve_time,
                    'solver_used': 'greedy',
                    'optimal': False
                }
        
        return None

# =============================================================================
# DATA GENERATION
# =============================================================================

class ExpertDemonstrationGenerator:
    """Generate expert demonstrations for multiple puzzle sizes"""
    
    def __init__(self, size=4):
        self.size = size
        self.env = SlidingPuzzleEnv(size)
        self.solver = MultiSizeSolverManager(size)
        self.solvability_checker = SolvabilityChecker(size)
        self.difficulty_classifier = DifficultyClassifier()
    
    def generate_demonstrations(self, num_games=5000):
        """Generate expert demonstrations"""
        
        print(f"\n{'='*70}")
        print(f"Generating {num_games} expert demonstrations ({self.size}×{self.size})...")
        print(f"{'='*70}\n")
        
        demonstrations = []
        stats = {
            'total_attempts': 0,
            'solved': 0,
            'failed': 0,
            'solver_usage': {'astar': 0, 'wastar': 0, 'idastar': 0, 'greedy': 0},
            'difficulty_dist': {'easy': 0, 'medium': 0, 'hard': 0, 'expert': 0},
            'avg_moves': 0.0,
            'avg_solve_time': 0.0
        }
        
        pbar = tqdm(range(num_games), desc=f"{self.size}×{self.size} Puzzles")
        
        for game_idx in pbar:
            stats['total_attempts'] += 1
            
            # Generate solvable puzzle
            max_attempts = 10
            for attempt in range(max_attempts):
                initial_state = self.env.reset()
                if self.solvability_checker.is_solvable(initial_state):
                    break
            else:
                stats['failed'] += 1
                continue
            
            # Solve puzzle
            solution = self.solver.solve(initial_state)
            
            if solution is None:
                stats['failed'] += 1
                pbar.set_postfix({
                    'solved': f"{stats['solved']}/{stats['total_attempts']}",
                    'status': 'failed'
                })
                continue
            
            stats['solved'] += 1
            stats['solver_usage'][solution['solver_used']] += 1
            
            # Extract trajectory
            trajectory = self._extract_trajectory(initial_state, solution['path'])
            
            # Classify difficulty
            difficulty = self.difficulty_classifier.classify_by_moves(
                solution['num_moves'], self.size
            )
            stats['difficulty_dist'][difficulty] += 1
            
            # Store demonstration
            for state, action, move_idx in trajectory:
                demonstrations.append({
                    'state': state.copy(),
                    'action': action,
                    'move_number': move_idx,
                    'total_moves': solution['num_moves'],
                    'difficulty': difficulty,
                    'solver': solution['solver_used'],
                    'optimal': solution['optimal'],
                    'size': self.size
                })
            
            stats['avg_moves'] += solution['num_moves']
            stats['avg_solve_time'] += solution['solve_time']
            
            pbar.set_postfix({
                'solved': f"{stats['solved']}/{stats['total_attempts']}",
                'moves': solution['num_moves'],
                'solver': solution['solver_used'][:3]
            })
        
        if stats['solved'] > 0:
            stats['avg_moves'] /= stats['solved']
            stats['avg_solve_time'] /= stats['solved']
        
        return demonstrations, stats
    
    def _extract_trajectory(self, initial_state: np.ndarray, 
                           path: List[int]) -> List[Tuple[np.ndarray, int, int]]:
        """Extract state-action trajectory from solution path"""
        
        trajectory = []
        current_state = initial_state.copy()
        
        for move_idx, action in enumerate(path):
            trajectory.append((current_state.copy(), action, move_idx))
            current_state = self.env._apply_move(current_state, action)
        
        return trajectory
    
    def augment_demonstrations(self, demonstrations: List[Dict]) -> List[Dict]:
        """Augment demonstrations with rotations & reflections"""
        
        print(f"\nAugmenting {self.size}×{self.size} demonstrations...")
        augmented = []
        
        for demo in tqdm(demonstrations, desc="Augmentation"):
            state = demo['state']
            action = demo['action']
            
            # Original
            augmented.append(demo.copy())
            
            # Generate augmented versions
            state_2d = state.reshape(self.size, self.size)
            action_row, action_col = divmod(action, self.size)
            
            # Rotation 90°
            aug_state = np.rot90(state_2d, k=1).flatten()
            aug_action_row = action_col
            aug_action_col = self.size - 1 - action_row
            aug_action = aug_action_row * self.size + aug_action_col
            augmented.append({**demo, 'state': aug_state.copy(), 'action': aug_action})
            
            # Rotation 180°
            aug_state = np.rot90(state_2d, k=2).flatten()
            aug_action_row = self.size - 1 - action_row
            aug_action_col = self.size - 1 - action_col
            aug_action = aug_action_row * self.size + aug_action_col
            augmented.append({**demo, 'state': aug_state.copy(), 'action': aug_action})
            
            # Rotation 270°
            aug_state = np.rot90(state_2d, k=3).flatten()
            aug_action_row = self.size - 1 - action_col
            aug_action_col = action_row
            aug_action = aug_action_row * self.size + aug_action_col
            augmented.append({**demo, 'state': aug_state.copy(), 'action': aug_action})
            
            # Horizontal flip
            aug_state = np.fliplr(state_2d).flatten()
            aug_action_col = self.size - 1 - action_col
            aug_action = action_row * self.size + aug_action_col
            augmented.append({**demo, 'state': aug_state.copy(), 'action': aug_action})
            
            # Vertical flip
            aug_state = np.flipud(state_2d).flatten()
            aug_action_row = self.size - 1 - action_row
            aug_action = aug_action_row * self.size + action_col
            augmented.append({**demo, 'state': aug_state.copy(), 'action': aug_action})
        
        print(f"Augmented: {len(demonstrations)} → {len(augmented)} samples (×{len(augmented)//len(demonstrations)})")
        
        return augmented
    
    def convert_to_training_format(self, demonstrations: List[Dict]) -> List[Tuple]:
        """Convert demonstrations to training format"""
        
        print(f"\nConverting {self.size}×{self.size} to training format...")
        training_data = []
        
        for demo in tqdm(demonstrations, desc="Conversion"):
            state = demo['state']
            action = demo['action']
            
            # Action logits (one-hot for supervised learning)
            action_logits = np.zeros(self.size * self.size, dtype=np.float32)
            action_logits[action] = 1.0
            
            # Value: normalized progress
            value = 1.0 - (demo['move_number'] / demo['total_moves'])
            
            # Difficulty label encoding
            difficulty_encoding = {
                'easy': 0,
                'medium': 1,
                'hard': 2,
                'expert': 3
            }
            difficulty_label = difficulty_encoding[demo['difficulty']]
            
            training_data.append((
                state,
                action_logits,
                value,
                difficulty_label,
                self.size  # Add size info
            ))
        
        return training_data

# =============================================================================
# MULTI-SIZE PIPELINE
# =============================================================================

def run_multi_size_pipeline(config: Dict):
    """Run pipeline for multiple puzzle sizes"""
    
    print("\n" + "="*80)
    print("MULTI-SIZE SLIDING PUZZLE EXPERT DEMONSTRATION GENERATOR")
    print("="*80)
    print(f"Configuration:")
    for size, num_games in config.items():
        print(f"  {size}×{size}: {num_games:,} puzzles")
    print("="*80 + "\n")
    
    all_training_data = []
    all_stats = {}
    
    start_time = time.time()
    
    for size, num_games in config.items():
        print(f"\n{'#'*80}")
        print(f"# Processing {size}×{size} Puzzles")
        print(f"{'#'*80}\n")
        
        # Initialize generator
        generator = ExpertDemonstrationGenerator(size)
        
        # Step 1: Generate demonstrations
        if size >= 4:
            print(f"Using parallel processing for {size}×{size} puzzles...")
            demonstrations, stats = generate_demonstrations_parallel(size, min(num_games, 20000), num_workers=4)
        else:
            demonstrations, stats = generator.generate_demonstrations(num_games)
        
        # Print statistics
        print(f"\n{'-'*70}")
        print(f"Solver Statistics for {size}×{size}:")
        print(f"  Total attempts: {stats['total_attempts']}")
        print(f"  Solved: {stats['solved']} ({stats['solved']/stats['total_attempts']*100:.1f}%)")
        print(f"  Failed: {stats['failed']}")
        print(f"  Average moves: {stats['avg_moves']:.1f}")
        print(f"  Average solve time: {stats['avg_solve_time']:.3f}s")
        
        print(f"\nSolver Usage:")
        for solver, count in stats['solver_usage'].items():
            if count > 0:
                print(f"  {solver:8s}: {count:4d} ({count/stats['solved']*100:.1f}%)")
        
        print(f"\nDifficulty Distribution:")
        for difficulty, count in stats['difficulty_dist'].items():
            if count > 0:
                print(f"  {difficulty:8s}: {count:4d} ({count/stats['solved']*100:.1f}%)")
        print(f"{'-'*70}\n")
        
        # Step 2: Augment demonstrations
        augmented_demos = generator.augment_demonstrations(demonstrations)
        
        # Step 3: Convert to training format
        training_data = generator.convert_to_training_format(augmented_demos)
        
        all_training_data.extend(training_data)
        all_stats[f'{size}x{size}'] = {
            'demonstrations_generated': len(demonstrations),
            'augmented_samples': len(augmented_demos),
            'training_samples': len(training_data),
            'statistics': stats
        }
    
    total_time = time.time() - start_time
    
    # Save combined data
    print("\n" + "="*80)
    print("SAVING COMBINED DATASET")
    print("="*80)
    
    # Create output directory
    os.makedirs('puzzle_data', exist_ok=True)
    
    # Save training data
    with open('puzzle_data/multi_size_training_data.pkl', 'wb') as f:
        pickle.dump(all_training_data, f)
    print(f"✓ Saved: puzzle_data/multi_size_training_data.pkl")
    print(f"  Total samples: {len(all_training_data):,}")
    
    # Save statistics
    report = {
        'timestamp': datetime.now().isoformat(),
        'total_samples': len(all_training_data),
        'total_time_seconds': total_time,
        'total_time_formatted': f"{total_time/3600:.2f} hours",
        'size_breakdown': all_stats
    }
    
    with open('puzzle_data/multi_size_stats.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Saved: puzzle_data/multi_size_stats.json")
    
    # Save separate files per size
    for size in config.keys():
        size_data = [item for item in all_training_data if item[4] == size]
        filename = f'puzzle_data/{size}x{size}_training_data.pkl'
        with open(filename, 'wb') as f:
            pickle.dump(size_data, f)
        print(f"✓ Saved: {filename} ({len(size_data):,} samples)")
    
    # Final summary
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Total samples: {len(all_training_data):,}")
    print(f"\nBreakdown by size:")
    for size_name, size_stats in all_stats.items():
        print(f"  {size_name}: {size_stats['training_samples']:,} samples")
    print("="*80 + "\n")
    
    return all_training_data, all_stats

# =============================================================================
# PARALLEL PROCESSING (OPTIONAL)
# =============================================================================

def solve_single_puzzle(args):
    """Helper function for parallel processing"""
    size, puzzle_idx, initial_state = args
    
    solver = MultiSizeSolverManager(size)
    solution = solver.solve(initial_state)
    
    if solution:
        return (puzzle_idx, initial_state, solution)
    else:
        return None

def generate_demonstrations_parallel(size, num_games, num_workers=None):
    """Generate demonstrations using parallel processing"""
    
    if num_workers is None:
        num_workers = max(1, cpu_count() - 1)
    
    print(f"\nGenerating {num_games} puzzles for {size}×{size} using {num_workers} workers...")
    
    # Generate all initial states first
    env = SlidingPuzzleEnv(size)
    checker = SolvabilityChecker(size)
    
    initial_states = []
    for _ in tqdm(range(num_games), desc="Generating states"):
        while True:
            state = env.reset()
            if checker.is_solvable(state):
                initial_states.append(state)
                break
    
    # Prepare arguments for parallel processing
    args_list = [(size, i, state) for i, state in enumerate(initial_states)]
    
    # Solve in parallel
    demonstrations = []
    stats = {
        'total_attempts': num_games,
        'solved': 0,
        'failed': 0,
        'solver_usage': {'astar': 0, 'wastar': 0, 'idastar': 0, 'greedy': 0},
        'difficulty_dist': {'easy': 0, 'medium': 0, 'hard': 0, 'expert': 0},
        'avg_moves': 0.0,
        'avg_solve_time': 0.0
    }
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(solve_single_puzzle, args_list),
            total=num_games,
            desc=f"Solving {size}×{size}"
        ))
    
    # Process results
    env = SlidingPuzzleEnv(size)
    classifier = DifficultyClassifier()
    
    for result in results:
        if result is None:
            stats['failed'] += 1
            continue
        
        puzzle_idx, initial_state, solution = result
        stats['solved'] += 1
        stats['solver_usage'][solution['solver_used']] += 1
        
        # Extract trajectory
        trajectory = []
        current_state = initial_state.copy()
        for move_idx, action in enumerate(solution['path']):
            trajectory.append((current_state.copy(), action, move_idx))
            current_state = env._apply_move(current_state, action)
        
        # Classify difficulty
        difficulty = classifier.classify_by_moves(solution['num_moves'], size)
        stats['difficulty_dist'][difficulty] += 1
        
        # Store demonstrations
        for state, action, move_idx in trajectory:
            demonstrations.append({
                'state': state.copy(),
                'action': action,
                'move_number': move_idx,
                'total_moves': solution['num_moves'],
                'difficulty': difficulty,
                'solver': solution['solver_used'],
                'optimal': solution['optimal'],
                'size': size
            })
        
        stats['avg_moves'] += solution['num_moves']
        stats['avg_solve_time'] += solution['solve_time']
    
    if stats['solved'] > 0:
        stats['avg_moves'] /= stats['solved']
        stats['avg_solve_time'] /= stats['solved']
    
    return demonstrations, stats

# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def generate_expert_data(size3=50000, size4=20000, size5=5000):
    """
    Hàm chính duy nhất để tạo expert demonstrations
    
    Args:
        size3: Số lượng puzzles 3×3 (mặc định: 50000)
        size4: Số lượng puzzles 4×4 (mặc định: 100000)
        size5: Số lượng puzzles 5×5 (mặc định: 10000)
    
    Returns:
        tuple: (all_training_data, all_stats)
    """
    config = {
        3: size3,
        4: size4,
        5: size5
    }
    
    return run_multi_size_pipeline(config)


def interactive_menu():
    """Menu tương tác cho việc chọn cấu hình"""
    
    print("\n" + "="*80)
    print("SLIDING PUZZLE EXPERT DEMONSTRATION GENERATOR")
    print("="*80)
    print("\nChọn chế độ chạy:\n")
    print("  [1] Chạy với cấu hình mặc định")
    print("      - 3×3: 50,000 puzzles")
    print("      - 4×4: 100,000 puzzles")
    print("      - 5×5: 10,000 puzzles")
    print()
    print("  [2] Chạy với cấu hình tùy chỉnh")
    print("      - Tự nhập số lượng puzzles cho mỗi size")
    print()
    print("="*80)
    
    while True:
        try:
            choice = input("\nNhập lựa chọn của bạn (1 hoặc 2): ").strip()
            
            if choice == '1':
                # Chạy mặc định
                print("\n✓ Đã chọn: Chạy với cấu hình mặc định")
                print("  3×3: 50,000 puzzles")
                print("  4×4: 100,000 puzzles")
                print("  5×5: 10,000 puzzles\n")
                
                confirm = input("Xác nhận chạy? (y/n): ").strip().lower()
                if confirm == 'y':
                    return generate_expert_data()
                else:
                    print("\nĐã hủy. Quay lại menu...\n")
                    continue
            
            elif choice == '2':
                # Chạy custom
                print("\n✓ Đã chọn: Chạy với cấu hình tùy chỉnh\n")
                
                while True:
                    try:
                        size3 = int(input("Nhập số lượng puzzles 3×3: ").strip())
                        size4 = int(input("Nhập số lượng puzzles 4×4: ").strip())
                        size5 = int(input("Nhập số lượng puzzles 5×5: ").strip())
                        
                        if size3 <= 0 or size4 <= 0 or size5 <= 0:
                            print("\n⚠️  Lỗi: Số lượng phải lớn hơn 0. Vui lòng nhập lại.\n")
                            continue
                        
                        print(f"\nCấu hình đã chọn:")
                        print(f"  3×3: {size3:,} puzzles")
                        print(f"  4×4: {size4:,} puzzles")
                        print(f"  5×5: {size5:,} puzzles\n")
                        
                        confirm = input("Xác nhận chạy? (y/n): ").strip().lower()
                        if confirm == 'y':
                            return generate_expert_data(size3=size3, size4=size4, size5=size5)
                        else:
                            print("\nĐã hủy. Quay lại menu...\n")
                            break
                    
                    except ValueError:
                        print("\n⚠️  Lỗi: Vui lòng nhập số nguyên hợp lệ.\n")
                        continue
            
            else:
                print("\n⚠️  Lỗi: Vui lòng chọn 1 hoặc 2.")
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Đã hủy bởi người dùng (Ctrl+C).")
            return None
        except EOFError:
            print("\n\n⚠️  Lỗi đầu vào. Thoát chương trình.")
            return None


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Kiểm tra môi trường Jupyter/Colab
    try:
        get_ipython().__class__.__name__
        is_jupyter = True
    except NameError:
        is_jupyter = False
    
    if is_jupyter:
        # Chạy trong Jupyter/Colab - Vẫn hiển thị menu tương tác
        interactive_menu()
    else:
        # Chạy từ command line
        if len(sys.argv) > 1:
            # Có arguments từ command line
            parser = argparse.ArgumentParser(
                description='Generate expert demonstrations for sliding puzzles (3×3, 4×4, 5×5)'
            )
            
            parser.add_argument('--size3', type=int, default=50000,
                               help='Number of 3×3 puzzles (default: 50000)')
            parser.add_argument('--size4', type=int, default=100000,
                               help='Number of 4×4 puzzles (default: 100000)')
            parser.add_argument('--size5', type=int, default=10000,
                               help='Number of 5×5 puzzles (default: 10000)')
            
            args = parser.parse_args()
            
            # Chạy pipeline
            generate_expert_data(
                size3=args.size3,
                size4=args.size4,
                size5=args.size5
            )
        else:
            # Không có arguments - Hiển thị menu tương tác
            interactive_menu()