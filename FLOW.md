# 🎯 FLOW TRAINING - NUMPUZ AI (3x3 → 15x15)

## 📌 Tổng quan kiến trúc

Mỗi độ khó game NxN được xử lý trong một **phrase** riêng.  
Mỗi phrase là file Python độc lập `phrase<number>_nxn.py` — tự động tạo dataset, huấn luyện, xuất artifacts.  

**Luồng I/O giữa các phrase:**
- Phrase nhận model (weights + config) từ phrase trước để transfer learning.
- Phrase sinh ra: dataset, model, metrics, logs, và ảnh trực quan.
- Phrase tiếp theo dùng model đầu ra làm input.

---

## 🎮 PHRASE 1: FOUNDATION (3x3) — `phrase1_3x3.py`

### 🔹 Input
- Không có input từ phrase trước (train từ đầu).

### 🔹 Thuật toán (Dataset generation & Supervised training)
1. **Sinh puzzle (50k):**
   - Bắt đầu từ trạng thái solved.
   - Random shuffle bằng chuỗi hợp lệ với số bước `k ∈ [5,20]`.
   - Kiểm tra solvability bằng inversion count.
2. **Ground-truth path:**
   - Dùng **A\*** với heuristic **Manhattan distance** để tìm đường giải tối ưu cho mỗi puzzle.
   - Lưu toàn bộ sequence state → action (optimal moves) làm label.
3. **One-hot + features:**
   - Mỗi state mã hóa one-hot cho mỗi tile (kênh) + kênh đặc trưng (empty pos, parity).
4. **Data augmentation (deterministic):**
   - 8 biến thể: rotations (90/180/270) + reflections (H/V) — đồng bộ với path (transform action).
5. **Training paradigm:**
   - **Supervised learning**: học policy head (predict next optimal move) + value head (ước lượng steps-to-solve / solvability score).
   - Không dùng RL ở phase này — mục tiêu teach model basic local patterns & moves.

### 🔹 Cấu hình chi tiết (full)
```yaml
dataset:
  n_samples: 50000
  move_range: [5, 20]
  augmentation: 8x (rot90, rot180, rot270, flipH, flipV, combos)

model_architecture:
  input_shape: [3, 3, 4]          # one-hot + features
  encoder:
    - Dense: 256, activation: ReLU, dropout: 0.0
    - Dense: 128, activation: ReLU, dropout: 0.0
    - Dense: 64, activation: ReLU, dropout: 0.0
  heads:
    policy_head:
      - Dense: 64, activation: ReLU, dropout: 0.1
      - Dense: 4, activation: Softmax   # up/down/left/right
    value_head:
      - Dense: 64, activation: ReLU
      - Dense: 1, activation: Tanh
    difficulty_head:
      - Dense: 32, activation: ReLU
      - Dense: 4, activation: None  # 4 difficulty classes

training:
  epochs: 200
  batch_size: 128
  optimizer:
    type: Adam
    lr: 0.001
    weight_decay: 0.0
  losses:
    policy_loss: CrossEntropyLoss
    value_loss: MSELoss
    difficulty_loss: CrossEntropyLoss
    lambda_policy: 1.0
    lambda_value: 0.5
    lambda_difficulty: 0.2
  lr_schedule: ReduceLROnPlateau (patience: 10, factor: 0.5, mode: min)
  gradient_clipping: max_norm: 1.0
  seed: 42

checkpointing:
  save_interval_epochs: 10
  save_best_by: train_accuracy
  keep_best: true
  keep_final: false  # foundation.pth bị xóa sau cleanup
```

### 🔹 Output

```
phase1_output_3x3.zip/phase1_output/
├── dataset_info.json
├── models/
│   └── numpuz_3x3_best.pth          # ⭐ QUAN TRỌNG - dùng cho Phase 2
├── README.md
├── training_history_3x3.json
├── train_config_3x3.yaml
├── model_config_3x3.json
├── training_metrics.json
├── training_curves_3x3.png
└── logs/
    └── training_phase1.log
```

**⚠️ Lưu ý Cleanup:**
- File `numpuz_3x3_foundation.pth` bị xóa sau cleanup (chỉ giữ `best.pth`)
- Các file timestamped duplicates bị xóa
- Intermediate epoch checkpoints bị xóa
- Chỉ giữ 1 log file mới nhất

---

## 🚀 PHRASE 2: SCALING (4x4) — `phrase2_4x4.py`

### 🔹 Input

```
📥 phase1_output/ (hoặc auto-search trong nhiều vị trí)
├─ numpuz_3x3_best.pth          # ⭐ BẮT BUỘC
├─ model_config_3x3.json         # ⚠️ Tùy chọn (có thì tốt)
└─ train_config_3x3.yaml         # ⚠️ Tùy chọn (có thì tốt)
```

**Auto-Search Locations (theo priority):**
1. Config-specified path
2. `/content/` (Colab working directory)
3. `/content/drive/MyDrive/` (Google Drive)
4. Current directory và subdirectories
5. Recursive search trong toàn bộ `/content/`

### 🔹 Thuật toán (Dataset generation, Heuristic & Transfer)

1. **Dataset 4×4 (100k samples):**
   - Sinh puzzle bằng random moves từ solved state
   - Độ khó: `10–50` moves (curriculum stages)
   - Đảm bảo solvable qua inversion count check
   - Data augmentation: 8x (rotation + flip) → 800k total samples

2. **Heuristic nâng cao:**
   - Sử dụng **IDA\*** (Iterative Deepening A*) để tìm optimal path
   - Enhanced heuristic = **max(Manhattan distance, Manhattan + Linear Conflict)**
   - Pattern Database (PDB) partition 6-6-3 (precomputed patterns)
   - Node limit: 50,000 nodes per puzzle để tránh timeout

3. **Transfer learning:**
   - **Auto-detect** Phase 1 model trong nhiều vị trí
   - Khởi tạo model 4×4 by **mapping encoder weights** từ `numpuz_3x3_best.pth`
   - Partial transfer cho layers có size khác biệt (min_rows, min_cols)
   - **Freeze** các encoder layers đầu tiên (`encoder.0`, `encoder.1`) trong giai đoạn đầu
   - Fine-tune phần policy/value/curriculum heads và các layer mở rộng
   - Progressive unfreezing khi vào curriculum stage "hard"

4. **Curriculum learning:**
   - **Stage 1 (easy):** 10–25 moves, 100 epochs
   - **Stage 2 (medium):** 25–40 moves, 100 epochs
   - **Stage 3 (hard):** 40–50 moves, 100 epochs
   - Auto-advance dựa trên epoch progress
   - Unfreeze all layers khi vào hard stage

### 🔹 Cấu hình chi tiết
```yaml
dataset:
  n_samples: 100000
  move_range: [10, 50]
  partition_pdb: [6, 6, 3]
  augmentation:
    enabled: true
    methods: [rot90, rot180, rot270, flip_h, flip_v]
    multiplier: 8x  # 100k → 800k samples
  ida_star:
    max_nodes: 50000
    heuristic: max(manhattan, manhattan + linear_conflict)

model_architecture:
  base: transfer_from_3x3
  input_size: 320  # 16 tiles × (17 one-hot + 3 features)
  input_norm: BatchNorm1d
  encoder_layers:
    - Linear: 512, activation: ReLU, dropout: 0.1
    - Linear: 256, activation: ReLU, dropout: 0.1
    - Linear: 128, activation: ReLU, dropout: 0.1
  heads:
    policy_head:
      layers: [128, 4]
      activation: [ReLU, Softmax]
      dropout: 0.1
    value_head:
      layers: [128, 1]
      activation: [ReLU, Tanh]
    curriculum_head:
      layers: [64, 3]
      activation: [ReLU, None]
      note: "3 outputs for easy/medium/hard classification"

transfer_learning:
  source_model: numpuz_3x3_best.pth
  auto_search: true  # Tự động tìm kiếm trong nhiều vị trí
  search_locations:
    - /content/numpuz_3x3_best.pth
    - /content/models/numpuz_3x3_best.pth
    - /content/phase1_output/models/numpuz_3x3_best.pth
    - /content/drive/MyDrive/numpuz_3x3_best.pth
    - (recursive search in /content/)
  strategy: partial_weight_mapping
  freeze:
    - layer_names: ["encoder.0", "encoder.1"]
    - frozen_lr_multiplier: 0.1  # 10% of base LR
  unfreeze_trigger:
    - condition: curriculum_stage == "hard"
    - method: unfreeze_all()

training:
  epochs: 300
  batch_size: 256
  optimizer:
    type: Adam
    lr: 0.0005
    frozen_lr: 0.00005  # For frozen layers
    weight_decay: 1e-4
  losses:
    policy_loss: CrossEntropyLoss
    value_loss: MSELoss
    curriculum_loss: CrossEntropyLoss
    weights:
      lambda_policy: 1.0
      lambda_value: 0.5
      lambda_curriculum: 0.3
  lr_schedule:
    type: CosineAnnealingWarmRestarts
    T_0: 100  # Restart every 100 epochs
    T_mult: 1
    eta_min: 1e-7
  gradient_clipping:
    max_norm: 1.0
  dataloader:
    num_workers: 4
    pin_memory: true
    prefetch_factor: 2
    persistent_workers: true
  seed: 42

curriculum:
  start_stage: easy
  stages:
    easy:
      epochs: 100
      move_range: [10, 25]
      difficulty_class: 0
    medium:
      epochs: 100
      move_range: [25, 40]
      difficulty_class: 1
    hard:
      epochs: 100
      move_range: [40, 50]
      difficulty_class: 2
  progression:
    method: auto_advance
    based_on: epoch_count

evaluation:
  val_split: 0.0  # No validation split (use full dataset for training)
  metrics:
    - policy_accuracy
    - value_mae
    - curriculum_accuracy
    - loss_components

checkpointing:
  save_interval_epochs: 5
  save_best_by: train_accuracy
  keep_checkpoints:
    - best: numpuz_4x4_best.pth
    - phase2: numpuz_4x4_phase2.pth (bị xóa sau cleanup)
    - periodic: numpuz_4x4_epoch_X.pth (bị xóa sau cleanup)

cleanup:
  after_training: true
  remove:
    - timestamped_duplicates
    - intermediate_checkpoints (epoch_*.pth)
    - numpuz_4x4_phase2.pth
    - old_logs (giữ 1 log mới nhất)
  keep:
    - best_model (numpuz_4x4_best.pth)
    - training_history
    - training_curves
    - configs
    - final_zip_archive
```

### 🔹 Output

```
phase2_output_4x4.zip/phase2_output/
├── curriculum_progress_4x4.json
├── dataset_info.json
├── logs/
│   └── training_phase2.log
├── model_config_4x4.json
├── models/
│   └── numpuz_4x4_best.pth          # ⭐ QUAN TRỌNG - dùng cho Phase 3
├── README.md
├── train_config_4x4.yaml
├── training_curves_4x4.png
├── training_history_4x4.json
└── training_metrics.json
```

**⚠️ Lưu ý Cleanup:**
- File `numpuz_4x4_phase2.pth` bị xóa sau cleanup (chỉ giữ `best.pth`)
- Các file timestamped duplicates bị xóa
- Intermediate epoch checkpoints bị xóa
- Chỉ giữ 1 log file mới nhất

---