# 🎯 PHASE 1: FOUNDATION (3×3) - `phrase1_3x3.py`

## 📥 Input

```yaml
input_config:
  base_model: null  # Train from scratch (Phrase 1 only)
  train_dataset: "auto_generated"
  puzzle_size: 3
  load_pretrained: false
  device: "cuda" if available else "cpu"
  
validation_config:  # QUAN TRỌNG: Thêm validation
  validation_split: 0.2  # 20% for validation
  stratified_split: true  # Đảm bảo balance difficulty classes
  cross_validation: false  # Không cần K-fold cho phrase 1
  
initialization:
  seed: 42
  weight_init: "xavier_uniform"
  custom_embeddings: false
```

---

## 📊 Dataset Structure

**Chủ đề**: Giải sliding puzzle 3×3 cơ bản  
**Số lượng**: 50,000 puzzles → Split: 40k train / 10k validation
**Augmentation**: 4× (thay vì 8×) → 160k train / 40k validation  
**Dataset**: `dataset_3x3_train.json` và `dataset_3x3_val.json`
**Lý do giảm augmentation**: Tránh overfitting trên biến thể tương tự

### Dataset Generation Algorithm

```python
def generate_3x3_dataset():
    """
    1. Bắt đầu từ solved state: [[1,2,3], [4,5,6], [7,8,0]]
    2. Random shuffle: 5-20 moves hợp lệ
    3. Check solvability: inversion count
    4. Solve với A* + Manhattan distance
    5. Record optimal path
    6. Augment: 8 biến thể (rot + flip)
    """
```

### Dataset Schema

```json
{
  "version": "1.0",
  "phrase": 1,
  "puzzle_size": 3,
  "data": [
    {
      "id": "p1_001",
      "initial_state": [[7,2,4], [5,0,6], [8,3,1]],
      "solved_state": [[1,2,3], [4,5,6], [7,8,0]],
      "optimal_path": ["up", "left", "down", "right", "up"],
      "num_moves": 5,
      "difficulty_class": 0,
      "metadata": {
        "manhattan_distance": 12,
        "linear_conflict": 2,
        "solvable": true,
        "generation_method": "random_walk",
        "solver": "A*"
      }
    }
  ]
}
```

---

## 🏗️ Model Architecture

### Network Structure

```yaml
model_architecture:
  input_shape: [3, 3, 13]  # (rows, cols, channels)
  # Channels: 9 one-hot (tiles 1-9) + 4 features (empty_pos_x, empty_pos_y, parity, normalized_state)
  
  input_layer:
    - Flatten: (3, 3, 13) → 117
    - BatchNorm1d: 117
  
  encoder_layers:
    - name: "encoder.0"
      type: Linear
      in_features: 117
      out_features: 256
      activation: ReLU
      dropout: 0.0
      
    - name: "encoder.1"
      type: Linear
      in_features: 256
      out_features: 128
      activation: ReLU
      dropout: 0.0
      
    - name: "encoder.2"
      type: Linear
      in_features: 128
      out_features: 64
      activation: ReLU
      dropout: 0.0
  
  heads:
    policy_head:
      - name: "policy.0"
        type: Linear
        in_features: 64
        out_features: 128  # Tăng capacity
        activation: ReLU
        dropout: 0.2  # Tăng dropout
        batch_norm: true  # Thêm BatchNorm
        
      - name: "policy.1"
        type: Linear
        in_features: 128
        out_features: 4
        activation: Softmax
        note: "4 directions: up, down, left, right"
        temperature_scaling: true  # Thêm temperature để control confidence
    
    value_head:  # OPTIONAL: Chỉ cần cho puzzle 8x8+
      - name: "value.0"
        type: Linear
        in_features: 64
        out_features: 32  # Giảm complexity
        activation: ReLU
        
      - name: "value.1"
        type: Linear
        in_features: 32
        out_features: 1
        activation: Linear  # Thay Tanh = Linear + MSE loss
        note: "Predict steps-to-solve (chỉ train từ phrase 4+)"
```

---

## 🎓 Training Configuration

```yaml
training_config:
  algorithm: "Supervised Learning"
  
  hyperparameters:
    learning_rate: 0.0005  # Giảm từ 0.001
    batch_size: 256  # Tăng từ 128 (A100 80GB đủ)
    gradient_accumulation_steps: 1
    num_epochs: 100  # Giảm từ 200
    max_length: null  # Not applicable for puzzle
    warmup_steps: 500  # Thêm warmup
    warmup_method: "linear"
    weight_decay: 1e-4  # Thêm regularization
    max_grad_norm: 1.0
    label_smoothing: 0.1  # Thêm label smoothing cho policy
    
  optimizer:
    type: "Adam"
    betas: [0.9, 0.999]
    eps: 1e-8
    
  scheduler:
    type: "CosineAnnealingWarmRestarts"  # Thay ReduceLROnPlateau
    T_0: 10  # Restart every 10 epochs
    T_mult: 2  # Double period after each restart
    eta_min: 1e-6
    
  early_stopping:  # QUAN TRỌNG: Thêm early stopping
    enabled: true
    monitor: "val_loss"
    patience: 15
    min_delta: 0.001
    restore_best_weights: true
    
  losses:
    policy_loss:
      type: "CrossEntropyLoss"
      weight: 1.0
      label_smoothing: 0.1  # Thêm smoothing
      note: "Predict next optimal move"
      
    value_loss:
      type: "MSELoss"
      weight: 0.3  # Giảm từ 0.5
      note: "Predict steps-to-solve (optional cho 3x3)"
      enabled_from_phrase: 4  # Chỉ enable từ phrase 4 (6x6)
      
    # REMOVED: difficulty_loss - Không cần thiết
  
  total_loss: "lambda_policy * policy_loss + lambda_value * value_loss + lambda_difficulty * difficulty_loss"
  
  gradient_clipping:
    enabled: true
    max_norm: 1.0
    
  seed: 42
  deterministic: true
```

### Data Augmentation

```yaml
augmentation:
  enabled: true
  multiplier: 4x  # 40k → 160k samples (giảm từ 8x)
  apply_to_validation: false  # QUAN TRỌNG: Không augment validation set
  
  methods:  # Chỉ giữ 3 augmentations chính
    - rotation_90: "Rotate puzzle 90° clockwise + transform actions"
    - rotation_180: "Rotate puzzle 180°"
    - flip_horizontal: "Mirror horizontally + swap left/right actions"
    # REMOVED: Các combinations phức tạp → Dễ overfit
  
  synchronization: "Actions transformed consistently with state"
  
  additional_techniques:
    random_noise: 
      enabled: false  # Không áp dụng cho puzzle (discrete state)
    mixup:
      enabled: false  # Không phù hợp với classification
```

### Validation Strategy

```yaml
validation:
  split: 0.2  # QUAN TRỌNG: 20% validation (thay vì 0.0)
  stratified: true  # Balance difficulty distribution
  
  evaluation:
    frequency: "every_epoch"  # Validate sau mỗi epoch
    method: "solve_validation_puzzles"
    validation_set_size: 10000  # Full validation set
    
    metrics:
      primary:
        - val_policy_accuracy: "Accuracy of next move prediction"
        - val_solve_rate: "Percentage of puzzles solved"
        - val_optimal_path_rate: "Percentage matching optimal solution"
        
      secondary:
        - val_value_mae: "MAE on steps-to-solve prediction"
        - val_loss: "Combined validation loss"
        - train_val_gap: "Overfitting indicator"
        
      monitoring:
        - solve_rate_threshold: 0.95  # Stop if < 95%
        - overfit_threshold: 0.15  # Stop if train-val gap > 15%
        
  test_set:  # Thêm separate test set
    enabled: true
    size: 5000  # Hold-out test set
    evaluate_at: "end_of_phrase"
    final_benchmark: true
```

---

## 📤 Output Structure (Extended)

```
phrase1_output_3x3.zip/phrase1_output/
├── models/                              # Model checkpoints chính
│   ├── numpuz_3x3_best.pth              # ⭐ QUAN TRỌNG - Best model (dùng cho Phrase 2)
│   ├── model_architecture.json          # Kiến trúc model chi tiết
│   └── optimizer_state.pth              # Optimizer state (optional backup)
│
├── gguf_models/                         # GGUF quantized models (deployment-ready)
│   ├── numpuz_3x3_q4_0.gguf             # 4-bit quantization (nhẹ nhất, mục tiêu cuối)
│   ├── numpuz_3x3_q5_1.gguf             # 5-bit quantization (cân bằng)
│   ├── numpuz_3x3_q8_0.gguf             # 8-bit quantization (chất lượng cao)
│   ├── numpuz_3x3_f16.gguf              # Full 16-bit (chất lượng tốt nhất)
│   └── gguf_conversion_log.txt          # Log quá trình convert sang GGUF
│
├── visualizations/                      # PNG charts để xem trực quan
│   ├── loss_curves_3x3.png             # Training loss (policy + value + difficulty)
│   ├── accuracy_curves_3x3.png          # Policy accuracy, difficulty accuracy
│   ├── value_mae_curve_3x3.png          # Value prediction MAE over epochs
│   ├── learning_rate_schedule_3x3.png   # Learning rate changes
│   ├── gradient_norm_3x3.png            # Gradient norm tracking
│   ├── solve_rate_evolution_3x3.png     # Solve rate improvement over training
│   ├── difficulty_distribution_3x3.png  # Distribution of puzzle difficulties
│   ├── moves_histogram_3x3.png          # Histogram of solution lengths
│   ├── training_speed_3x3.png           # Samples/second over epochs
│   └── gpu_memory_usage_3x3.png         # VRAM usage tracking
│
├── training_logs/                       # Logs chi tiết
│   └── training_history_3x3.json        # Metrics qua tất cả epochs
│
├── evaluation/                          # Kết quả đánh giá
│   ├── test_results_3x3.json            # Kết quả test trên 1000 puzzles
│   ├── solve_rate_breakdown.json        # Solve rate theo difficulty class
│   ├── optimal_path_comparison.json     # So sánh với A* optimal
│   ├── value_prediction_errors.json     # Chi tiết lỗi value prediction
│   ├── confusion_matrix_difficulty.png  # Confusion matrix difficulty classification
│   ├── policy_heatmap.png               # Heatmap policy distribution
│   └── benchmark_results.json           # Benchmark với các baselines
│
├── exports/                             # Export formats khác
│   ├── numpuz_3x3.onnx                  # ONNX format (cross-platform)
│   ├── numpuz_3x3_torchscript.pt        # TorchScript format
│   ├── numpuz_3x3_openvino/             # OpenVINO IR format
│   └── export_config.json               # Export configuration
│
└── configs/                             # Configuration backups
    ├── train_config_3x3.yaml            # Full training config
    ├── model_config_3x3.json            # Model architecture config
    ├── dataset_config_3x3.json          # Dataset generation config
    └── augmentation_config_3x3.json     # Augmentation settings