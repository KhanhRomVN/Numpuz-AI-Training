# pattern_analysis.py
# Phase 1: Pattern Analysis & Feature Engineering for Sliding Number Puzzles
# !pip install numpy tqdm
# =============================================================================

import numpy as np
import json
import pickle
from typing import List, Tuple, Dict
from collections import deque
import itertools
from tqdm import tqdm
from datetime import datetime

# =============================================================================
# 1. SLIDING PUZZLE ENVIRONMENT
# =============================================================================

class SlidingPuzzleEnv:
    """Sliding number puzzle environment"""
    
    def __init__(self, size=4):
        self.size = size
        self.total_tiles = size * size
        self.goal_state = self._create_goal_state()
        self.reset()
    
    def _create_goal_state(self) -> np.ndarray:
        """Create goal state: [1, 2, 3, ..., 15, 0]"""
        state = np.arange(1, self.total_tiles, dtype=np.int32)
        state = np.append(state, 0)
        return state
    
    def reset(self) -> np.ndarray:
        """Reset to solvable puzzle"""
        state = self.goal_state.copy()
        
        # Generate solvable state via valid moves
        num_shuffles = np.random.randint(50, 150)
        for _ in range(num_shuffles):
            valid_moves = self._get_valid_moves(state)
            if valid_moves:
                move = valid_moves[np.random.randint(len(valid_moves))]
                state = self._apply_move(state, move)
        
        self.current_state = state.copy()
        return state
    
    def _get_valid_moves(self, state: np.ndarray) -> List[int]:
        """Get valid moves (indices we can move to empty space)"""
        empty_pos = np.where(state == 0)[0][0]
        empty_row, empty_col = divmod(empty_pos, self.size)
        
        valid_moves = []
        
        # Up
        if empty_row > 0:
            valid_moves.append(empty_pos - self.size)
        # Down
        if empty_row < self.size - 1:
            valid_moves.append(empty_pos + self.size)
        # Left
        if empty_col > 0:
            valid_moves.append(empty_pos - 1)
        # Right
        if empty_col < self.size - 1:
            valid_moves.append(empty_pos + 1)
        
        return valid_moves
    
    def _apply_move(self, state: np.ndarray, tile_pos: int) -> np.ndarray:
        """Apply move by swapping tile with empty space"""
        new_state = state.copy()
        empty_pos = np.where(new_state == 0)[0][0]
        new_state[empty_pos], new_state[tile_pos] = new_state[tile_pos], new_state[empty_pos]
        return new_state
    
    def step(self, tile_pos: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute move"""
        if tile_pos not in self._get_valid_moves(self.current_state):
            return self.current_state.copy(), -1.0, False, {"error": "Invalid move"}
        
        self.current_state = self._apply_move(self.current_state, tile_pos)
        done = self.is_solved()
        reward = 1.0 if done else -0.01
        
        return self.current_state.copy(), reward, done, {}
    
    def is_solved(self) -> bool:
        """Check if puzzle is solved"""
        return np.array_equal(self.current_state, self.goal_state)
    
    def get_state(self) -> np.ndarray:
        """Get current state"""
        return self.current_state.copy()

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

class FeatureExtractor:
    """Extract heuristic features for sliding puzzles"""
    
    def __init__(self, size=4):
        self.size = size
        self.total_tiles = size * size
        self.goal_state = np.arange(size * size, dtype=np.int32)
        self._precompute_goal_positions()
    
    def _precompute_goal_positions(self):
        """Precompute goal positions for each tile"""
        self.goal_positions = {}
        for tile in range(1, self.total_tiles):
            self.goal_positions[tile] = divmod(tile - 1, self.size)
        self.goal_positions[0] = divmod(self.total_tiles - 1, self.size)
    
    def extract_features(self, state: np.ndarray) -> Dict[str, float]:
        """Extract all features from state"""
        features = {}
        
        # Manhattan distance
        features['manhattan'] = self._manhattan_distance(state)
        
        # Linear conflict
        features['linear_conflict'] = self._linear_conflict(state)
        
        # Inversion count
        features['inversion_count'] = self._inversion_count(state)
        
        # Permutation parity
        features['parity'] = self._permutation_parity(state)
        
        # Tile adjacency score
        features['tile_adjacency'] = self._tile_adjacency(state)
        
        # Weighted heuristic (combined)
        features['weighted_h'] = (
            0.6 * (features['manhattan'] / (self.size * self.size * 2)) +
            0.3 * (features['linear_conflict'] / self.total_tiles) +
            0.1 * (features['inversion_count'] / (self.total_tiles * self.total_tiles))
        )
        
        return features
    
    def _manhattan_distance(self, state: np.ndarray) -> float:
        """Calculate Manhattan distance heuristic"""
        distance = 0.0
        for pos, tile in enumerate(state):
            if tile == 0:
                continue
            goal_row, goal_col = self.goal_positions[tile]
            curr_row, curr_col = divmod(pos, self.size)
            distance += abs(curr_row - goal_row) + abs(curr_col - goal_col)
        return distance
    
    def _linear_conflict(self, state: np.ndarray) -> float:
        """Calculate linear conflict heuristic"""
        conflicts = 0
        
        # Check rows
        for row in range(self.size):
            row_tiles = state[row * self.size:(row + 1) * self.size]
            for i in range(len(row_tiles)):
                if row_tiles[i] == 0:
                    continue
                goal_row_i = self.goal_positions[row_tiles[i]][0]
                
                # Tile should be in this row in goal state
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
    
    def _inversion_count(self, state: np.ndarray) -> int:
        """Count inversions in puzzle state"""
        inversions = 0
        state_no_empty = state[state != 0]
        
        for i in range(len(state_no_empty)):
            for j in range(i + 1, len(state_no_empty)):
                if state_no_empty[i] > state_no_empty[j]:
                    inversions += 1
        
        return inversions
    
    def _permutation_parity(self, state: np.ndarray) -> int:
        """Calculate permutation parity (even/odd)"""
        return self._inversion_count(state) % 2
    
    def _tile_adjacency(self, state: np.ndarray) -> float:
        """Score based on tiles adjacent to their goal position"""
        adjacency_score = 0.0
        
        for pos, tile in enumerate(state):
            if tile == 0:
                continue
            
            goal_row, goal_col = self.goal_positions[tile]
            curr_row, curr_col = divmod(pos, self.size)
            
            # Check adjacency
            adjacent_positions = []
            if goal_row > 0:
                adjacent_positions.append((goal_row - 1, goal_col))
            if goal_row < self.size - 1:
                adjacent_positions.append((goal_row + 1, goal_col))
            if goal_col > 0:
                adjacent_positions.append((goal_row, goal_col - 1))
            if goal_col < self.size - 1:
                adjacent_positions.append((goal_row, goal_col + 1))
            
            is_adjacent = (curr_row, curr_col) in adjacent_positions
            adjacency_score += 1.0 if is_adjacent else 0.0
        
        return adjacency_score

# =============================================================================
# 3. SOLVABILITY CHECKER
# =============================================================================

class SolvabilityChecker:
    """Check if puzzle configuration is solvable"""
    
    def __init__(self, size=4):
        self.size = size
    
    def is_solvable(self, state: np.ndarray) -> bool:
        """
        Check if sliding puzzle is solvable using permutation theory
        """
        # Get inversion count
        state_no_empty = state[state != 0]
        inversions = self._count_inversions(state_no_empty)
        
        # Get empty position from bottom
        empty_pos = np.where(state == 0)[0][0]
        empty_row = self.size - (empty_pos // self.size)
        
        if self.size % 2 == 1:
            # Odd width: solvable if inversions even
            return inversions % 2 == 0
        else:
            # Even width: solvable if (inversions + empty_row) is odd
            return (inversions + empty_row) % 2 == 1
    
    def _count_inversions(self, arr: np.ndarray) -> int:
        """Count inversions in array"""
        inversions = 0
        for i in range(len(arr)):
            for j in range(i + 1, len(arr)):
                if arr[i] > arr[j]:
                    inversions += 1
        return inversions

# =============================================================================
# 4. DIFFICULTY CLASSIFIER
# =============================================================================

class DifficultyClassifier:
    """Classify puzzle difficulty"""
    
    @staticmethod
    def classify_by_moves(num_moves: int) -> str:
        """Classify difficulty by optimal move count"""
        if num_moves <= 10:
            return 'easy'
        elif num_moves <= 30:
            return 'medium'
        elif num_moves <= 50:
            return 'hard'
        else:
            return 'expert'
    
    @staticmethod
    def classify_by_heuristic(heuristic_value: float) -> str:
        """Classify difficulty by heuristic value"""
        if heuristic_value <= 8:
            return 'easy'
        elif heuristic_value <= 20:
            return 'medium'
        elif heuristic_value <= 40:
            return 'hard'
        else:
            return 'expert'

# =============================================================================
# 5. PATTERN ANALYZER
# =============================================================================

class PatternAnalyzer:
    """Analyze patterns in puzzle configurations"""
    
    def __init__(self, size=4):
        self.size = size
        self.env = SlidingPuzzleEnv(size)
        self.feature_extractor = FeatureExtractor(size)
        self.solvability_checker = SolvabilityChecker(size)
        self.difficulty_classifier = DifficultyClassifier()
    
    def generate_analysis_dataset(self, num_puzzles=1000):
        """Generate dataset with pattern analysis"""
        print(f"Analyzing {num_puzzles} puzzles ({self.size}x{self.size})...")
        
        analysis_data = []
        solvable_count = 0
        difficulty_distribution = {'easy': 0, 'medium': 0, 'hard': 0, 'expert': 0}
        
        pbar = tqdm(range(num_puzzles), desc="Pattern Analysis")
        
        for _ in pbar:
            state = self.env.reset()
            
            # Check solvability
            is_solvable = self.solvability_checker.is_solvable(state)
            
            if not is_solvable:
                continue
            
            solvable_count += 1
            
            # Extract features
            features = self.feature_extractor.extract_features(state)
            
            # Classify difficulty
            difficulty = self.difficulty_classifier.classify_by_heuristic(
                features['weighted_h']
            )
            difficulty_distribution[difficulty] += 1
            
            # Store analysis
            analysis_data.append({
                'state': state.copy(),
                'features': features,
                'difficulty': difficulty,
                'solvable': is_solvable
            })
            
            pbar.set_postfix({
                'solvable': f'{solvable_count}/{_ + 1}',
                'difficulty': difficulty
            })
        
        return analysis_data, difficulty_distribution
    
    def analyze_feature_statistics(self, analysis_data: List[Dict]) -> Dict:
        """Compute statistics on features"""
        stats = {
            'manhattan': [],
            'linear_conflict': [],
            'inversion_count': [],
            'tile_adjacency': [],
            'weighted_h': []
        }
        
        for item in analysis_data:
            features = item['features']
            for key in stats:
                stats[key].append(features[key])
        
        statistics = {}
        for key, values in stats.items():
            values = np.array(values)
            statistics[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values))
            }
        
        return statistics
    
    def generate_augmented_puzzles(self, state: np.ndarray) -> List[np.ndarray]:
        """Generate augmented versions (rotations & reflections)"""
        state_2d = state.reshape(self.size, self.size)
        augmented = [state.copy()]
        
        # Rotations: 90, 180, 270
        for k in range(1, 4):
            rotated = np.rot90(state_2d, k)
            augmented.append(rotated.flatten())
        
        # Horizontal flip
        h_flip = np.fliplr(state_2d)
        augmented.append(h_flip.flatten())
        
        # Vertical flip
        v_flip = np.flipud(state_2d)
        augmented.append(v_flip.flatten())
        
        return augmented

# =============================================================================
# 6. MAIN PATTERN ANALYSIS FUNCTION
# =============================================================================

def run_pattern_analysis(size=4, num_puzzles=2000):
    """Main pattern analysis pipeline"""
    print("=" * 70)
    print(f"PHASE 1: PATTERN ANALYSIS & FEATURE ENGINEERING ({size}x{size})")
    print("=" * 70 + "\n")
    
    # Initialize analyzer
    analyzer = PatternAnalyzer(size)
    
    # Step 1: Generate analysis dataset
    print("Step 1: Generating puzzle dataset...")
    analysis_data, difficulty_dist = analyzer.generate_analysis_dataset(num_puzzles)
    
    print(f"\nGenerated {len(analysis_data)} valid puzzles")
    print("Difficulty Distribution:")
    for difficulty, count in difficulty_dist.items():
        if count > 0:
            percentage = count / len(analysis_data) * 100
            print(f"  {difficulty:8s}: {count:4d} ({percentage:5.1f}%)")
    
    # Step 2: Compute feature statistics
    print("\nStep 2: Computing feature statistics...")
    feature_stats = analyzer.analyze_feature_statistics(analysis_data)
    
    print("\nFeature Statistics:")
    for feature, stats in feature_stats.items():
        print(f"\n  {feature}:")
        for stat_name, value in stats.items():
            print(f"    {stat_name:8s}: {value:8.2f}")
    
    # Step 3: Generate augmented data
    print("\nStep 3: Generating augmented puzzles...")
    total_samples = 0
    augmented_data = []
    
    for idx, item in enumerate(tqdm(analysis_data, desc="Augmentation")):
        state = item['state']
        augmented_states = analyzer.generate_augmented_puzzles(state)
        
        for aug_state in augmented_states:
            augmented_data.append({
                'state': aug_state.copy(),
                'features': item['features'],
                'difficulty': item['difficulty'],
                'original_idx': idx
            })
        
        total_samples += len(augmented_states)
    
    print(f"\nGenerated {total_samples} augmented samples (×{total_samples // len(analysis_data)})")
    
    # Step 4: Save data
    print("\nStep 4: Saving analysis data...")
    
    with open('pattern_data.pkl', 'wb') as f:
        pickle.dump(augmented_data, f)
    
    with open('analysis_report.json', 'w') as f:
        report = {
            'timestamp': datetime.now().isoformat(),
            'puzzle_size': size,
            'total_puzzles_generated': num_puzzles,
            'valid_puzzles': len(analysis_data),
            'augmented_samples': total_samples,
            'difficulty_distribution': difficulty_dist,
            'feature_statistics': feature_stats
        }
        json.dump(report, f, indent=2)
    
    print("  ✓ Saved pattern_data.pkl")
    print("  ✓ Saved analysis_report.json")
    
    # Final summary
    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETED")
    print("=" * 70)
    print(f"Outputs:")
    print(f"  - pattern_data.pkl: {total_samples} augmented samples")
    print(f"  - analysis_report.json: Feature statistics & distribution")
    print("=" * 70 + "\n")
    
    return augmented_data, feature_stats

if __name__ == "__main__":
    augmented_data, feature_stats = run_pattern_analysis(size=4, num_puzzles=2000)