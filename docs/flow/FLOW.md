# 🎯 FLOW TRAINING - NUMPUZ AI (3x3 → 15x15)

## 📌 Tổng quan kiến trúc

Dự án training AI giải sliding puzzle từ **3×3** đến **15×15** sử dụng **Supervised Learning + Transfer Learning** với curriculum-based approach. Mỗi phrase xử lý một độ khó NxN riêng biệt, xây dựng dần từ foundation đến mastery.

### 🔧 Đặc điểm kỹ thuật tổng quan

```yaml
project_config:
  task: "Sliding Puzzle Solving"
  puzzle_range: "3x3 → 15x15"
  total_phrases: 13
  algorithm: "Supervised Learning + Transfer Learning"
  output_format: "PyTorch (.pth) + ONNX + GGUF Q5-Q8"  # Tránh Q4 quá thấp
  final_deployment: "GGUF Q5_1 quantization"  # Cân bằng giữa size và accuracys
  
hardware_requirements:
  gpu_memory: "40-80GB VRAM (A100/H100 or 2x RTX 4090)"
  ram: "128GB+ (256GB recommended for 15x15)"
  storage: "1TB SSD (2TB recommended với intermediate checkpoints)"
  
software_stack:
  framework: "PyTorch 2.0+"
  optimization: "GGUF Q5-Q8 quantization via llama.cpp"  # Q4 có thể mất accuracy
  deployment: "Ollama / llama.cpp / ONNX Runtime"
  validation: "Stratified K-Fold (k=5) cho mỗi phrase"  # QUAN TRỌNG: Thêm validation
```

---

## 📋 DANH SÁCH TOÀN BỘ 13 PHRASES

### 🎭 **TIER 1: FOUNDATION (Phrases 1-3)**
Xây dựng nền tảng cho puzzle nhỏ

| Phrase |  Size   | Dataset (Train/Val) | Augmentation   | Training Time | Checkpoint Output   |
|--------|---------|---------------------|----------------|---------------|---------------------|
| **1**  | **3×3** | 40k / 10k           | 4× → 160k/40k  | 3-4 hours     | numpuz_3x3_best.pth |
| **2**  | **4×4** | 80k / 20k           | 4× → 320k/80k  | 6-8 hours     | numpuz_4x4_best.pth |
| **3**  | **5×5** | 160k / 40k          | 4× → 640k/160k | 12-16 hours   | numpuz_5x5_best.pth |

### 💫 **TIER 2: INTERMEDIATE (Phrases 4-6)**
Mở rộng khả năng cho puzzle trung bình

| Phrase | Size | Dataset | Training Time | Checkpoint Output |
|--------|------|---------|---------------|-------------------|
| **4** | **6×6** | 300k samples | 12-15 hours | numpuz_6x6_best.pth |
| **5** | **7×7** | 400k samples | 18-22 hours | numpuz_7x7_best.pth |
| **6** | **8×8** | 500k samples | 24-30 hours | numpuz_8x8_best.pth |

### 🎨 **TIER 3: ADVANCED (Phrases 7-9)**
Xử lý puzzle phức tạp

| Phrase | Size | Dataset | Training Time | Checkpoint Output |
|--------|------|---------|---------------|-------------------|
| **7** | **9×9** | 600k samples | 30-36 hours | numpuz_9x9_best.pth |
| **8** | **10×10** | 700k samples | 36-44 hours | numpuz_10x10_best.pth |
| **9** | **11×11** | 800k samples | 44-52 hours | numpuz_11x11_best.pth |

### 🌟 **TIER 4: EXPERT (Phrases 10-13)**
Đạt đến mastery với puzzle lớn

| Phrase | Size | Dataset | Training Time | Checkpoint Output |
|--------|------|---------|---------------|-------------------|
| **10** | **12×12** | 900k samples | 52-62 hours | numpuz_12x12_best.pth |
| **11** | **13×13** | 1M samples | 62-72 hours | numpuz_13x13_best.pth |
| **12** | **14×14** | 1.1M samples | 72-84 hours | numpuz_14x14_best.pth |
| **13** | **15×15** | 1.2M samples | 84-96 hours | numpuz_15x15_best.pth |

---

## 📊 Thống kê tổng quan

```yaml
total_statistics:
  total_phrases: 13
  total_samples: 8,350,000
  estimated_total_duration: 450-550 hours (~19-23 days)
  average_per_phrase: 35-42 hours
  
  tier_breakdown:
    tier_1_foundation: 
      phrases: 3
      samples: 350,000
      focus: "Nền tảng cơ bản"
    
    tier_2_intermediate:
      phrases: 3
      samples: 1,200,000
      focus: "Mở rộng khả năng"
    
    tier_3_advanced:
      phrases: 3
      samples: 2,100,000
      focus: "Xử lý phức tạp"
    
    tier_4_expert:
      phrases: 4
      samples: 4,200,000
      focus: "Đạt mastery"

recommended_training_schedule:
  phrases_per_week: 2-3
  total_weeks: 5-7
  rest_days: "Sau mỗi tier nên nghỉ 1 ngày để đánh giá"
  checkpoints: "Sau phrases 3, 6, 9, 13"
  
quality_assurance:
  validation_after_each_phrase: true
  solve_rate_threshold: ">95%"
  optimal_path_accuracy: ">80%"
  regression_testing: "Test lại tất cả sizes trước đó"
```

---

## 🎯 Phrase Dependencies

```yaml
dependencies:
  phrase_1_3x3: []  # No dependencies (train from scratch)
  phrase_2_4x4: [1]  # Requires 3x3 model
  phrase_3_5x5: [2]  # Requires 4x4 model
  phrase_4_6x6: [3]  # Requires 5x5 model
  phrase_5_7x7: [4]  # Requires 6x6 model
  phrase_6_8x8: [5]  # Requires 7x7 model
  phrase_7_9x9: [6]  # Requires 8x8 model
  phrase_8_10x10: [7]  # Requires 9x9 model
  phrase_9_11x11: [8]  # Requires 10x10 model
  phrase_10_12x12: [9]  # Requires 11x11 model
  phrase_11_13x13: [10]  # Requires 12x12 model
  phrase_12_14x14: [11]  # Requires 13x13 model
  phrase_13_15x15: [12]  # Requires 14x14 model

transfer_learning_strategy:
  method: "partial_weight_mapping"
  
  weight_mapping_rules:
    encoder_layers: "Full transfer (map all encoder weights)"
    policy_head: "Transfer + expand (add new output dimensions)"
    value_head: "Transfer + fine-tune (similar range)"
    
  freeze_strategy: 
    phase: "progressive_unfreezing"
    schedule:
      epochs_1_5: "Freeze encoder, train only heads"
      epochs_6_15: "Unfreeze encoder.2 (last layer)"
      epochs_16_plus: "Unfreeze all layers, reduced LR"
  
  learning_rate_schedule:
    frozen_layers: 0.0
    newly_unfrozen: 1e-5
    heads: 1e-3
    
  curriculum_learning: 
    method: "difficulty-based batching"
    details: "Start with easy puzzles (5-10 moves), gradually increase to 20+ moves"
```

---

## 📈 Success Metrics

### Metrics tracking qua 13 phrases:

```yaml
core_metrics:
  solve_rate: 
    target: ">98%"  # Tăng từ 95% vì có validation
    measured: "Percentage of puzzles solved successfully"
    validation_required: true
  
  optimal_path_accuracy:
    target: ">85%"  # Tăng từ 80% với validation tốt hơn
    measured: "How close to optimal solution path"
    metric: "Levenshtein distance to A* solution"
  
  solution_time:
    target: "<5 seconds"
    measured: "Average time to find solution (inference only)"
    note: "Không tính generation time của GGUF"
  
  value_prediction_accuracy:
    target: "MAE < 2.0 moves"  # Cụ thể hơn ">85%"
    measured: "Mean Absolute Error on steps-to-solve prediction"
    note: "Chỉ quan trọng với puzzle 10x10+"
    
  additional_metrics:
    validation_loss_gap: "<10%"  # Train vs Val loss gap
    policy_entropy: ">0.5"  # Tránh quá deterministic
    gradient_norm_stability: "<2.0"  # Tránh gradient exploding

progressive_targets:
  phrase_1_3: [0.95, 0.85, 1s]  # [solve_rate, optimal_acc, time]
  phrase_4_6: [0.93, 0.82, 2s]
  phrase_7_9: [0.92, 0.80, 3s]
  phrase_10_13: [0.90, 0.78, 5s]
```

---

## 📝 Notes & Best Practices

1. **Không skip phrases**: Mỗi phrase xây dựng trên phrase trước
2. **Auto-search model**: Script tự động tìm model từ phrase trước
3. **Cleanup sau training**: Xóa các file trung gian, giữ best checkpoint
4. **Validate kỹ**: Test solve rate trước khi chuyển phrase tiếp
5. **Backup thường xuyên**: Sau mỗi phrase backup model ra Google Drive
6. **Monitor performance**: Track GPU usage, memory, training speed
7. **Git versioning**: Commit sau mỗi phrase hoàn thành
8. **Documentation**: Ghi chú observations trong training log

---

## 🔗 Chi tiết từng Phrase

- [Phase 1: Foundation 3×3](./PHASE1.md)
- [Phase 2: Scaling 4×4](./PHASE2.md)
- [Phase 3: Mastery 5×5](./PHASE3.md)
- [Phase 4: Expansion 6×6](./PHASE4.md)
- [Phase 5-13: Advanced Sizes](./PHASE5-13.md) *(Coming soon)*

---

## 🚀 Quick Start

```bash
# Clone repository
git clone https://github.com/khanhromvn/Numpuz-AI-Training
cd Numpuz-AI-Training

# Install dependencies
pip install -r requirements.txt

# Start training from Phrase 1
python phrase1_3x3.py

# Training sẽ tự động tiến hành và xuất artifacts
```

---

**Github**: khanhromvn  
**HuggingFace**: khanhromvn/Numpuz-Solver-15x15-GGUF-Q4