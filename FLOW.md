# Flow Training - Numpuz Number Puzzle AI
## 📋 Tổng quan
Project sử dụng **6-phase training pipeline** tối ưu cho Sliding Number Puzzle:
1. **Phase 1 (pattern_analysis.py)**: Phân tích pattern và feature engineering
2. **Phase 2 (heuristic_solver.py)**: Tạo dữ liệu từ heuristic solver (A* + IDA*)
3. **Phase 3 (imitation_learning.py)**: Huấn luyện với Imitation Learning
4. **Phase 4 (self_play.py)**: Tạo dữ liệu self-play với MCTS
5. **Phase 5 (reinforcement_learning.py)**: Huấn luyện với AlphaZero style
6. **Phase 6 (curriculum_learning.py)**: Fine-tune với curriculum learning (3×3 → 4×4 → 5×5)

---
## 📊 PHASE 1: PATTERN ANALYSIS (pattern_analysis.py)
### 1. Pattern Recognition
- **Feature engineering** cho sliding puzzles:
  - Manhattan distance (từng tile đến vị trí goal)
  - Linear conflict detection
  - Inversion count
  - Permutation parity
  - Tile adjacency patterns
  
- **State representation**:
  - Flattened board: `[1, 2, 3, ..., 15, 0]` (4×4 = 16 tiles)
  - Normalized features: `[0, 1]` range
  - Goal state caching

### 2. Solvability Analysis
- **Check puzzle solvability** (permutation-based)
- Generate **solvable puzzle variants**
- Categorize by difficulty:
  - Easy: 5-10 moves
  - Medium: 10-30 moves
  - Hard: 30-100 moves
  - Expert: 100+ moves

### 3. Output
- File: `pattern_data.pkl` (features, solvability info)
- Metrics: `analysis_report.json`

---
## 🔍 PHASE 2: HEURISTIC SOLVER (heuristic_solver.py)
### 4. Multi-Heuristic Solver
- **Thuật toán chính**: A* Search + IDA* (Iterative Deepening A*)
- **Heuristics kết hợp**:
  - Manhattan Distance (primary)
  - Linear Conflict (secondary)
  - Gaschnig's heuristic
  - Weighted combination: `h(n) = 0.6 * manhattan + 0.3 * conflict + 0.1 * gaschnig`

- **Fallback strategy**: Greedy Best-First nếu A* timeout (30s)

### 5. Generate Expert Demonstrations
- **Puzzle sizes**:
  - 3×3: 2,000 games
  - 4×4: 5,000 games (main focus)
  - 5×5: 2,000 games (optional, nếu resource cho phép)

- **Per size - 10 iterations × puzzles/iteration**

### 6. Data Augmentation
- **Board rotations**: 4 views (0°, 90°, 180°, 270°)
- **Reflections**: 2 views (horizontal, vertical)
- **Trajectory transformation**: Multiplier × 8 = tối đa 80,000 samples từ 10,000 games

### 7. Output
- File: `heuristic_data.pkl`
  - Format: `[(state, action_logits, value, difficulty_label), ...]`
  - Size: **~80,000-100,000 samples**
- Metrics: `solver_stats.json` (solve rate, avg moves, time)

---
## 🚀 PHASE 3: IMITATION LEARNING (imitation_learning.py)
### 8. Load Training Data
- Load `heuristic_data.pkl` từ Phase 2
- Split: 80% train, 10% val, 10% test
- Stratify by difficulty level

### 9. Neural Network Architecture
```
Input Layer (16 * 4 values) → 256 ReLU → 256 ReLU → 128 ReLU
├─ Policy Head: → 64 ReLU → 4 softmax (action logits)
└─ Value Head: → 64 ReLU → 1 sigmoid (value [0,1])
```

### 10. Training Process
- **Loss function**:
  - Policy: Cross-entropy loss
  - Value: MSE loss
  - Combined: `L = 0.7 * L_policy + 0.3 * L_value`

- **Training config**:
  - **Epochs**: 150 (10 iter × 15 epochs)
  - **Batch size**: 64
  - **Optimizer**: Adam (lr=0.001, β₁=0.9, β₂=0.999)
  - **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
  - **Weight decay**: 1e-4

### 11. Validation & Testing
- Validate mỗi epoch
- Early stopping nếu val_loss không improve trong 10 epochs
- Test trên holdout test set

### 12. Output
- File: `numpuz_imitation.pth`
- Metrics: `imitation_metrics.json`
  - Win rate: **60-75%** (3×3)
  - Win rate: **30-50%** (4×4)
  - Training time: **20-40 phút**

---
## 🔁 PHASE 4: SELF-PLAY GENERATION (self_play.py)
### 13. Load Pre-trained Model
- Load `numpuz_imitation.pth` từ Phase 3

### 14. Self-Play Configuration
- **Game variants**:
  - 3×3: 200 games per iteration
  - 4×4: 100 games per iteration
  - 5×5: 50 games per iteration (nếu train)

- **MCTS parameters**:
  - Simulations per move: **200** (3×3), **150** (4×4), **100** (5×5)
  - Temperature: τ = 1.0 (exploration)
  - C_puct: 1.0 (UCB constant)

### 15. Trajectory Generation
- Mỗi move: MCTS → action probabilities π(a|s)
- Format: `(state, π, outcome, moves_taken, difficulty)`
- Store vào **replay buffer** (max 20,000 samples)
- Prioritize harder puzzles (weight by moves)

### 16. Self-Play Loop
```
for iteration in range(50):
    for game_idx in range(games_per_iteration):
        initial_state = generate_solvable_puzzle()
        trajectory = []
        while not is_solved(state):
            mcts_probs = mcts.search(state, num_simulations)
            action = sample_from_probs(mcts_probs)
            trajectory.append((state, mcts_probs))
            state = apply_action(state, action)
        
        outcome = compute_outcome(num_moves)  # reward function
        replay_buffer.add(trajectory, outcome)
    
    # Save checkpoint mỗi 5 iterations
    if (iteration + 1) % 5 == 0:
        save_checkpoint(f"self_play_iter_{iteration}.pkl")
```

### 17. Output
- File: `self_play_data.pkl` (50 iterations)
  - Total samples: **~200,000-300,000**
- Checkpoint files: `self_play_iter_*.pkl`
- Metrics: `self_play_stats.json` (win rate, avg steps/iteration)

---
## 🎯 PHASE 5: REINFORCEMENT LEARNING (reinforcement_learning.py)
### 18. Load Self-Play Data
- Load `self_play_data.pkl` từ Phase 4
- Shuffle và stratify samples

### 19. RL Training
- **Loss function** (AlphaZero style):
  ```
  L = L_policy + L_value + L_reg
  L_policy = KL_divergence(π_model, π_mcts)
  L_value = MSE(v_model, outcome)
  L_reg = λ * ||weights||²
  ```

- **Training parameters**:
  - **Epochs**: 10 per iteration
  - **Iterations**: 100
  - **Batch size**: 128
  - **Learning rate**: 0.0005 (nếu fine-tune từ Phase 3)
  - **Weight decay**: 1e-4

### 20. Model Improvement Loop
```
for iteration in range(100):
    # Train on self-play data
    for epoch in range(10):
        sample_batch = replay_buffer.sample(batch_size=128)
        loss = compute_loss(model, batch)
        optimizer.step()
    
    # Evaluate every 5 iterations
    if (iteration + 1) % 5 == 0:
        win_rate = evaluate(model, num_test_games=100)
        if win_rate > best_win_rate:
            save_model(f"numpuz_model_iter_{iteration}.pth")
            best_win_rate = win_rate
```

### 21. Evaluation Metrics
- Win rate trên test puzzles
- Average moves to solution
- Efficiency score: `moves_optimal / moves_used`
- Success rate by difficulty

### 22. Output
- File: `numpuz_model.pth` (best model)
- Checkpoints: `numpuz_model_iter_*.pth`
- Training log: `rl_training_log.json`
- Metrics: `rl_metrics.json`
  - Win rate: **85-95%** (4×4)
  - Training time: **15-25 giờ**

---
## 📈 PHASE 6: CURRICULUM LEARNING (curriculum_learning.py)
### 23. Curriculum Strategy
- **Stage 1 (5 iterations)**: 3×3 puzzles (easy → medium)
- **Stage 2 (10 iterations)**: 4×4 puzzles (easy → hard)
- **Stage 3 (5 iterations)**: Mixed 3×3 + 4×4 (consolidation)
- **Stage 4 (5 iterations)**: 5×5 puzzles (nếu needed)

### 24. Progressive Difficulty
```python
difficulty_schedule = {
    "iter_0-4":   [5-10 moves],      # Very easy (3×3)
    "iter_5-14":  [10-30 moves],     # Easy (4×4)
    "iter_15-19": [30-50 moves],     # Medium (4×4)
    "iter_20-24": [mixed + 5×5],     # Hard
}
```

### 25. Training Process
- Load best model từ Phase 5
- Fine-tune với curriculum data
- Lower learning rate: **0.0001**
- Epochs: **15 per iteration** (more stability)

### 26. Transfer Learning
- Freeze early layers (representation)
- Fine-tune policy & value heads
- Monitor overfitting

### 27. Final Evaluation
- Test on diverse puzzle sets
- Benchmark vs reference solutions
- Solvability guarantee check

### 28. Output
- File: `numpuz_curriculum.pth` (final model)
- Report: `curriculum_report.json`
  - Final win rate: **90-98%**
  - Performance by size: `3×3: 98% | 4×4: 95% | 5×5: 85-90%`
  - Total training time: **40-60 giờ**

---
## 📊 Training Timeline & Resources
| Phase | Component | Time | Samples | Output |
|-------|-----------|------|---------|--------|
| 1 | Pattern Analysis | 30 min | - | pattern_data.pkl |
| 2 | Heuristic Solver | 2-3 h | ~100k | heuristic_data.pkl |
| 3 | Imitation | 30-40 min | 100k | numpuz_imitation.pth |
| 4 | Self-Play | 8-12 h | ~250k | self_play_data.pkl |
| 5 | RL Training | 15-25 h | 250k | numpuz_model.pth |
| 6 | Curriculum | 10-15 h | Mixed | numpuz_curriculum.pth |
| **Total** | **All Phases** | **36-56 h** | **~400k** | **Final Model** |

---
## 🎯 Expected Results
| Metric | 3×3 | 4×4 | 5×5 |
|--------|-----|-----|-----|
| Win Rate | 95-99% | 90-95% | 80-90% |
| Avg Moves (vs optimal) | 1.1× | 1.2× | 1.3× |
| Solve Time | <100ms | <200ms | <500ms |

---
## 🔧 Key Implementation Details
### State Representation
- **Board encoding**: Flattened array + normalized features
- **Action space**: 4 (up, down, left, right) based on empty tile position
- **Feature normalization**: Min-max scaling to [0, 1]

### Heuristic Functions
- **Manhattan Distance**: Sum of |x_current - x_goal| + |y_current - y_goal|
- **Linear Conflict**: Tiles in same row/col but wrong order
- **Weighted combo**: Improves search efficiency × 5-10

### Memory Optimization
- Use numpy arrays (C contiguity)
- Batch processing
- Lazy evaluation of features
- Pruning in MCTS

### Hyperparameter Tuning
- Grid search over: learning rates, batch sizes, hidden sizes
- Validation set: track win rate + loss curves
- Adjust based on overfitting signals

---
**Author**: KhanhRomVN  
**GitHub**: [NumpuzPuzzleBot](https://github.com/KhanhRomVN/NumpuzAI)  
**Updated**: Oct 2025