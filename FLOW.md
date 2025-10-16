# Flow Training - Numpuz Number Puzzle AI (3x3 to 15x15)

## 📋 Tổng quan
Project sử dụng **8-phase training pipeline** tối ưu cho Sliding Number Puzzle từ 3x3 đến 15x15:

1. **Phase 1 (Foundation)**: Huấn luyện nền tảng trên 3x3 với expert demonstrations
2. **Phase 2 (Scaling)**: Mở rộng lên 4x4 với transfer learning
3. **Phase 3 (Intermediate)**: Huấn luyện trên 5x5-6x6 với curriculum learning
4. **Phase 4 (Advanced)**: Self-play trên 7x7-8x8 với MCTS enhancement
5. **Phase 5 (Expert)**: Reinforcement learning trên 9x9-10x10
6. **Phase 6 (Master)**: Huấn luyện trên 11x11-12x12 với distributed training
7. **Phase 7 (Grand Master)**: Fine-tune trên 13x13-14x14
8. **Phase 8 (Champion)**: Final training trên 15x15 với ensemble methods

---
## 🧩 PHASE 1: FOUNDATION TRAINING (3x3)

### 1. Input
- **Khởi tạo**: Neural network ngẫu nhiên
- **Data**: Puzzle generators cho 3x3

### 2. Configuration
```python
config_3x3 = {
    "board_size": 3,
    "training_samples": 50_000,
    "epochs": 200,
    "batch_size": 128,
    "learning_rate": 0.001,
    "difficulty_range": [5, 20],  # moves từ optimal solution
    "model_size": "small",
    "hidden_layers": [256, 128, 64]
}
```

### 3. Algorithms
- **Expert Solver**: A* với Manhattan + Linear Conflict
- **Training**: Supervised learning từ expert moves
- **Data Augmentation**: 8x rotation/reflection

### 4. Output
- **Model**: `numpuz_3x3_foundation.pth`
- **Metrics**: Win rate >98%, avg moves 1.1× optimal
- **Training Time**: 2-3 giờ

---
## 🔄 PHASE 2: SCALING TRAINING (4x4)

### 1. Input
- **Model**: `numpuz_3x3_foundation.pth`
- **Data**: 4x4 puzzle generators

### 2. Configuration
```python
config_4x4 = {
    "board_size": 4,
    "training_samples": 100_000,
    "epochs": 300,
    "batch_size": 256,
    "learning_rate": 0.0005,  # giảm learning rate
    "difficulty_range": [10, 50],
    "model_size": "medium",
    "hidden_layers": [512, 256, 128],
    "transfer_learning": {
        "frozen_layers": ["conv1", "conv2"],  # giữ feature extractor
        "fine_tune_layers": ["policy_head", "value_head"]
    }
}
```

### 3. Algorithms
- **Expert Solver**: IDA* với pattern databases
- **Training**: Transfer learning + fine-tuning
- **Progressive Difficulty**: Tăng độ khó theo iterations

### 4. Output
- **Model**: `numpuz_4x4_scaling.pth`
- **Metrics**: Win rate >95%, avg moves 1.2× optimal
- **Training Time**: 4-6 giờ

---
## 📈 PHASE 3: INTERMEDIATE TRAINING (5x5-6x6)

### 1. Input
- **Model**: `numpuz_4x4_scaling.pth`
- **Data**: 5x5 và 6x6 puzzle generators

### 2. Configuration
```python
config_5x5_6x6 = {
    "board_sizes": [5, 6],
    "training_samples": 200_000,  # 100k mỗi size
    "epochs_per_size": 150,
    "batch_size": 512,
    "learning_rate": 0.0003,
    "curriculum_schedule": [
        {"size": 5, "difficulty": [15, 40], "iterations": 8},
        {"size": 6, "difficulty": [20, 60], "iterations": 12}
    ],
    "model_size": "large",
    "hidden_layers": [1024, 512, 256]
}
```

### 3. Algorithms
- **Hybrid Solver**: Limited A* + Neural heuristic
- **Training**: Curriculum learning với progressive sizes
- **Evaluation**: Cross-size validation

### 4. Output
- **Model**: `numpuz_6x6_intermediate.pth`
- **Metrics**: 5x5: win rate >92%, 6x6: win rate >88%
- **Training Time**: 8-12 giờ

---
## 🔁 PHASE 4: ADVANCED SELF-PLAY (7x7-8x8)

### 1. Input
- **Model**: `numpuz_6x6_intermediate.pth`
- **Data**: Self-play generators

### 2. Configuration
```python
config_7x7_8x8 = {
    "board_sizes": [7, 8],
    "self_play_iterations": 100,
    "games_per_iteration": 500,  # 250 mỗi size
    "mcts_simulations": 400,  # tăng simulations
    "training_epochs": 10,
    "batch_size": 512,
    "learning_rate": 0.0002,
    "replay_buffer_size": 500_000,
    "temperature_schedule": {
        "initial": 1.0,
        "final": 0.1,
        "decay_steps": 50
    }
}
```

### 3. Algorithms
- **Self-Play**: MCTS với neural network guide
- **Training**: AlphaZero-style reinforcement learning
- **Evaluation**: Elo rating system

### 4. Output
- **Model**: `numpuz_8x8_advanced.pth`
- **Metrics**: 7x7: win rate >85%, 8x8: win rate >80%
- **Training Time**: 15-20 giờ

---
## 🎯 PHASE 5: EXPERT REINFORCEMENT (9x9-10x10)

### 1. Input
- **Model**: `numpuz_8x8_advanced.pth`
- **Data**: Self-play + expert demonstrations

### 2. Configuration
```python
config_9x9_10x10 = {
    "board_sizes": [9, 10],
    "training_iterations": 150,
    "games_per_iteration": 400,
    "mcts_simulations": 600,
    "batch_size": 1024,  # tăng batch size
    "learning_rate": 0.0001,
    "rl_algorithm": "AlphaZero",
    "reward_shaping": True,
    "priority_replay": True,
    "difficulty_schedule": [
        {"iterations": 50, "max_moves": 100},
        {"iterations": 50, "max_moves": 200},
        {"iterations": 50, "max_moves": 300}
    ]
}
```

### 3. Algorithms
- **Enhanced MCTS**: RAVE, progressive widening
- **Reward Shaping**: Step-based rewards
- **Priority Replay**: Focus trên hard puzzles

### 4. Output
- **Model**: `numpuz_10x10_expert.pth`
- **Metrics**: 9x9: win rate >78%, 10x10: win rate >75%
- **Training Time**: 25-30 giờ

---
## 🌐 PHASE 6: MASTER DISTRIBUTED TRAINING (11x11-12x12)

### 1. Input
- **Model**: `numpuz_10x10_expert.pth`
- **Infrastructure**: Multi-GPU, distributed training

### 2. Configuration
```python
config_11x11_12x12 = {
    "board_sizes": [11, 12],
    "distributed_workers": 8,
    "games_per_worker": 100,
    "training_iterations": 200,
    "mcts_simulations": 800,
    "batch_size": 2048,
    "learning_rate": 0.00005,
    "model_parallelism": True,
    "gradient_accumulation": 4,
    "mixed_precision": True,
    "model_size": "xlarge",
    "hidden_layers": [2048, 1024, 512, 256]
}
```

### 3. Algorithms
- **Distributed Self-Play**: Parallel game generation
- **Model Parallelism**: Split network across GPUs
- **Advanced Optimizations**: Mixed precision, gradient accumulation

### 4. Output
- **Model**: `numpuz_12x12_master.pth`
- **Metrics**: 11x11: win rate >70%, 12x12: win rate >65%
- **Training Time**: 40-50 giờ

---
## ⚡ PHASE 7: GRAND MASTER FINE-TUNING (13x13-14x14)

### 1. Input
- **Model**: `numpuz_12x12_master.pth`
- **Data**: Hard puzzle dataset

### 2. Configuration
```python
config_13x13_14x14 = {
    "board_sizes": [13, 14],
    "training_iterations": 100,
    "fine_tune_epochs": 20,
    "batch_size": 1024,
    "learning_rate": 0.00002,
    "specialized_datasets": {
        "corner_cases": 10_000,
        "long_sequences": 10_000,
        "pattern_breaks": 10_000
    },
    "regularization": {
        "weight_decay": 1e-5,
        "dropout": 0.1,
        "label_smoothing": 0.1
    }
}
```

### 3. Algorithms
- **Specialized Training**: Corner cases, long sequences
- **Advanced Regularization**: Prevent overfitting
- **Ensemble Methods**: Model averaging

### 4. Output
- **Model**: `numpuz_14x14_grandmaster.pth`
- **Metrics**: 13x13: win rate >60%, 14x14: win rate >55%
- **Training Time**: 20-25 giờ

---
## 🏆 PHASE 8: CHAMPION ENSEMBLE TRAINING (15x15)

### 1. Input
- **Model**: `numpuz_14x14_grandmaster.pth`
- **Ensemble Models**: Các model từ previous phases

### 2. Configuration
```python
config_15x15 = {
    "board_size": 15,
    "ensemble_size": 5,
    "training_iterations": 50,
    "self_play_games": 1000,
    "mcts_simulations": 1000,
    "batch_size": 1024,
    "learning_rate": 0.00001,
    "ensemble_methods": [
        "model_averaging",
        "mcts_ensemble",
        "policy_fusion"
    ],
    "final_tuning": {
        "adversarial_training": True,
        "test_time_augmentation": True,
        "progressive_hardening": True
    }
}
```

### 3. Algorithms
- **Model Ensemble**: Multiple model fusion
- **Adversarial Training**: Hardest puzzles
- **Test-time Augmentation**: Multiple evaluations

### 4. Output
- **Final Model**: `numpuz_15x15_champion.pth`
- **Metrics**: Win rate >50%, competitive với human experts
- **Training Time**: 15-20 giờ

---
## 📊 TỔNG QUAN TÀI NGUYÊN & THỜI GIAN

| Phase | Board Sizes | Thời Gian | Samples | Model Size | Hardware |
|-------|-------------|-----------|---------|------------|----------|
| 1 | 3x3 | 2-3h | 50k | Small | 1x GPU |
| 2 | 4x4 | 4-6h | 100k | Medium | 1x GPU |
| 3 | 5x5-6x6 | 8-12h | 200k | Large | 1x GPU |
| 4 | 7x7-8x8 | 15-20h | 500k | Large | 2x GPU |
| 5 | 9x9-10x10 | 25-30h | 600k | XLarge | 2x GPU |
| 6 | 11x11-12x12 | 40-50h | 800k | XLarge | 4x GPU |
| 7 | 13x13-14x14 | 20-25h | 300k | XLarge | 4x GPU |
| 8 | 15x15 | 15-20h | 500k | Ensemble | 4x GPU |
| **Total** | **3-15** | **129-166h** | **~3M** | **-** | **-** |

---
## 🎯 KẾT QUẢ KỲ VỌNG

| Board Size | Win Rate | Avg Moves Multiplier | Solve Time | Human Comparable |
|------------|----------|---------------------|-------------|------------------|
| 3x3 | >98% | 1.05× | <50ms | Superhuman |
| 4x4 | >95% | 1.1× | <100ms | Superhuman |
| 5x5 | >92% | 1.15× | <200ms | Superhuman |
| 6x6 | >88% | 1.2× | <500ms | Expert |
| 7x7 | >85% | 1.3× | <1s | Expert |
| 8x8 | >80% | 1.4× | <2s | Advanced |
| 9x9 | >78% | 1.5× | <3s | Advanced |
| 10x10 | >75% | 1.6× | <5s | Intermediate |
| 11x11 | >70% | 1.7× | <8s | Intermediate |
| 12x12 | >65% | 1.8× | <12s | Intermediate |
| 13x13 | >60% | 1.9× | <18s | Beginner |
| 14x14 | >55% | 2.0× | <25s | Beginner |
| 15x15 | >50% | 2.1× | <35s | Novice |

---
## 🔧 KIẾN TRÚC & TỐI ƯU HÓA

### Neural Network Architecture
```
Input: (board_size × board_size × 4)  # one-hot encoding + features
↓
Conv2D(128, 3×3) + BatchNorm + ReLU
↓
ResBlock(128) × 4
↓
Conv2D(256, 3×3) + BatchNorm + ReLU
↓
ResBlock(256) × 4
↓
Global Average Pooling
↓
Dense(512) → Policy Head (4 units)
↓
Dense(512) → Value Head (1 unit, tanh)
```

### MCTS Enhancements
- **Progressive Widening**: Dynamic branching factor
- **RAVE**: Rapid Action Value Estimation
- **Virtual Loss**: Parallel simulations
- **Domain Knowledge**: Puzzle-specific priors

### Training Optimizations
- **Gradient Checkpointing**: Memory efficiency
- **Mixed Precision**: Faster computation
- **Model Parallelism**: Large model support
- **Data Pipeline**: Async data loading

---
## 🚀 DEPLOYMENT & INFERENCE

### Model Compression
- **Quantization**: FP16/INT8 inference
- **Pruning**: Remove redundant weights
- **Knowledge Distillation**: Smaller student models

### Inference Optimization
- **Caching**: Position evaluation cache
- **Early Stopping**: Confidence-based moves
- **Parallel Evaluation**: Batch position evaluation

---
**Author**: KhanhRomVN  
**GitHub**: [NumpuzAI](https://github.com/KhanhRomVN/NumpuzAI)  
**Updated**: Oct 2025  
**Total Training Time**: ~5.5-7 ngày  
**Total Samples**: ~3 triệu training samples  
**Final Coverage**: 3x3 to 15x15 puzzles