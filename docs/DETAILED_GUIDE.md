# OpenVLA Detailed Implementation Guide

## Table of Contents

1. [Environment Setup](#environment-setup)
2. [Model Architecture Deep Dive](#model-architecture-deep-dive)
3. [LIBERO Evaluation Pipeline](#libero-evaluation-pipeline)
4. [Data Processing](#data-processing)
5. [Performance Optimization](#performance-optimization)
6. [Common Issues and Solutions](#common-issues-and-solutions)

## Environment Setup

### Hardware Requirements

- **GPU**: NVIDIA GPU with at least 12GB VRAM (V100, A100, RTX 3090, or higher)
- **RAM**: 32GB+ system memory recommended
- **Storage**: 100GB+ for model weights and datasets

### Software Dependencies

```bash
# Core dependencies
- CUDA 11.8 or 12.1
- Python 3.10+
- PyTorch 2.0+
- Flash Attention 2.5.5
- Transformers 4.35+
- MuJoCo 2.3.0+
```

### Installation Steps

#### 1. CUDA Toolkit Setup

```bash
# Check if CUDA is available
nvidia-smi

# Install CUDA toolkit (if not present)
sudo apt update
sudo apt install nvidia-cuda-toolkit

# Verify NVCC compiler
nvcc --version
```

#### 2. Python Environment

```bash
# Create conda environment
conda create -n openvla python=3.10
conda activate openvla

# Or use mamba for faster installation
mamba create -n openvla python=3.10
mamba activate openvla
```

#### 3. Core Dependencies

```bash
# Install PyTorch with CUDA support
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Install OpenVLA in editable mode
cd openvla
pip install -e .

# Install build tools
pip install packaging ninja

# Verify ninja
ninja --version
echo $?  # Should return 0
```

#### 4. Flash Attention 2

Flash Attention is critical for memory efficiency and speed.

```bash
# Install Flash Attention 2
pip install flash-attn==2.5.5 --no-build-isolation

# If compilation fails, ensure CUDA toolkit is in PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Troubleshooting Flash Attention:**

- Requires CUDA 11.8+
- Needs compatible GPU compute capability (7.0+)
- Compilation takes 5-10 minutes

#### 5. Model Weights Download

```bash
# Install Git LFS
git lfs install

# Download base model
git lfs clone https://huggingface.co/openvla/openvla-7b

# Download fine-tuned variants
git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial
git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-object
git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal
git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-10
```

**Parallel Download with tmux:**

```bash
# Create tmux session
tmux new-session -d -s openvla_download

# Split into 4 panes
tmux split-window -h
tmux split-window -v
tmux select-pane -t 0
tmux split-window -v

# Download in parallel
tmux send-keys -t 0 'git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial' Enter
tmux send-keys -t 1 'git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-object' Enter
tmux send-keys -t 2 'git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal' Enter
tmux send-keys -t 3 'git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-10' Enter

# Attach to session
tmux attach-session -t openvla_download
```

## Model Architecture Deep Dive

### 1. Vision Backbone

OpenVLA supports multiple vision encoders:

- **SigLIP**: Contrastive vision-language pre-training
- **DINOv2**: Self-supervised vision transformer
- **CLIP**: OpenAI's vision-language model

**Dual Encoder Configuration:**

```python
# Example architecture
vision_backbone:
  encoder_1: DINOv2-Base  # Spatial understanding
  encoder_2: CLIP-L/14    # Semantic understanding
  fusion: concatenate

output_shape: [batch, num_patches, 2*vision_dim]
```

**Feature Extraction:**

```python
# Image preprocessing
image = resize(image, (224, 224))
image = normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Vision encoding
patches = vision_backbone(image)  # [B, 256, 768] for DINOv2-Base
```

### 2. Vision-Language Projector

Maps vision features to LLM embedding space.

**Architecture:**

```python
class VisionProjector(nn.Module):
    def __init__(self, vision_dim, llm_dim):
        self.fc1 = nn.Linear(vision_dim, 4 * vision_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(4 * vision_dim, llm_dim)
        self.fc3 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        return x
```

**Example Dimensions:**

- DINOv2-Base: 768 → 3072 → 4096 → 4096 (for LLaMA-7B)
- CLIP-L/14: 768 → 3072 → 4096 → 4096

### 3. Language Model Backbone

**Supported Models:**

- LLaMA-2 (7B, 13B)
- Vicuna (7B, 13B)
- Mistral-7B

**Optimization Techniques:**

1. **Flash Attention 2**: 2-3x faster attention computation
2. **BF16 Precision**: Reduces memory by 50% with minimal accuracy loss
3. **Gradient Checkpointing**: Trade computation for memory

**Input Sequence Structure:**

```
[BOS] <task_description> [IMG_START] <vision_tokens> [IMG_END] [ACTION]
```

### 4. Action Head

Predicts 7-DoF continuous actions.

**Architecture:**

```python
class ActionHead(nn.Module):
    def __init__(self, llm_dim, action_dim=7):
        self.fc1 = nn.Linear(llm_dim, llm_dim // 2)
        self.fc2 = nn.Linear(llm_dim // 2, action_dim)

    def forward(self, x):
        # x: last token embedding from LLM
        x = self.fc1(x)
        x = nn.GELU()(x)
        x = self.fc2(x)
        return x  # [batch, 7]
```

**Action Space:**

```python
action = [
    x,        # End-effector X position
    y,        # End-effector Y position
    z,        # End-effector Z position
    rx,       # Roll rotation
    ry,       # Pitch rotation
    rz,       # Yaw rotation
    gripper   # Gripper state (-1: close, 1: open)
]
```

## LIBERO Evaluation Pipeline

### Task Suites

LIBERO provides 4 task suites:

1. **libero_spatial** (10 tasks): Spatial reasoning (e.g., "place the bowl between the plate and the box")
2. **libero_object** (10 tasks): Object-centric manipulation
3. **libero_goal** (10 tasks): Goal-conditioned tasks
4. **libero_10** (10 tasks): Long-horizon tasks (10+ steps)

### Evaluation Protocol

```python
# Pseudocode for evaluation
for task_id in range(num_tasks):
    for episode in range(num_trials_per_task):  # Default: 50
        # Reset environment
        obs = env.reset()

        # Set initial state (for reproducibility)
        env.set_init_state(initial_states[task_id][episode])

        success = False
        for step in range(max_steps):  # Default: 300
            # Get image observation
            image = obs['agentview_image']  # [224, 224, 3]

            # Construct prompt
            prompt = f"What action should the robot take to {task_descriptions[task_id]}?"

            # Model inference
            with torch.no_grad():
                action_normalized = model.predict_action(prompt, image)

            # De-normalize action
            action = action_normalized * action_std + action_mean

            # Execute action
            obs, reward, done, info = env.step(action)

            if done:
                success = info['success']
                break

        # Save rollout video
        save_video(f"episode_{episode}_success_{success}.mp4")
```

### Running Evaluation

```bash
# Standard evaluation
python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint /path/to/checkpoint \
    --task_suite_name libero_spatial \
    --center_crop True \
    --num_trials_per_task 50

# Quick test (10 episodes)
python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint /path/to/checkpoint \
    --task_suite_name libero_spatial \
    --center_crop True \
    --num_trials_per_task 10
```

## Data Processing

### Image Preprocessing

**Training:**

```python
# Random crop + augmentation
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

**Inference:**

```python
# Center crop (deterministic)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

### Action Normalization

**Per-Task Statistics:**

```python
# Compute statistics from training data
action_mean = np.mean(actions, axis=0)  # [7]
action_std = np.std(actions, axis=0)    # [7]

# Normalize
action_normalized = (action - action_mean) / action_std

# De-normalize during inference
action = action_normalized * action_std + action_mean
```

**Why Normalize?**

- Different action dimensions have vastly different scales
- Position: [-0.5, 0.5], Rotation: [-π, π], Gripper: [-1, 1]
- Normalization improves training stability and convergence

## Performance Optimization

### Memory Optimization

1. **BF16 Precision**

```python
model = model.to(dtype=torch.bfloat16)
```

- Memory: 16GB → 8GB
- Speed: 1.5x faster
- Accuracy: <1% degradation

2. **Gradient Checkpointing**

```python
model.gradient_checkpointing_enable()
```

- Memory: 12GB → 8GB (during training)
- Speed: 1.2x slower

3. **Flash Attention 2**

```python
# Automatically enabled if installed
attention_output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True
)
```

- Memory: 16GB → 10GB
- Speed: 2-3x faster

### Inference Speed

**Optimization Techniques:**

1. **Batching**: Process multiple images simultaneously
2. **KV Caching**: Reuse past key-value pairs (for autoregressive generation)
3. **Compile**: Use `torch.compile()` for JIT optimization

**Expected Throughput:**

- Single image: 5-8 FPS
- Batch size 4: 20-30 FPS
- With TensorRT: 40-60 FPS

## Common Issues and Solutions

### Issue 1: CUDA Out of Memory

**Symptoms:**

```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. Reduce batch size
2. Enable gradient checkpointing
3. Use BF16 precision
4. Clear cache: `torch.cuda.empty_cache()`

### Issue 2: Slow Model Loading

**Symptoms:**

- Model loading takes 5+ minutes

**Solutions:**

```python
# Use fast initialization
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    low_cpu_mem_usage=True
)
```

### Issue 3: Action Prediction Unstable

**Symptoms:**

- Robot moves erratically
- Actions not converging

**Solutions:**

1. Verify action normalization statistics
2. Check that `unnorm_key` matches task suite
3. Ensure center crop is enabled during inference
4. Validate image preprocessing pipeline

### Issue 4: Git LFS Quota Exceeded

**Symptoms:**

```
Error: rate limit exceeded
```

**Solutions:**

1. Use mirrors: `git clone https://hf-mirror.com/openvla/...`
2. Download via browser and extract manually
3. Use torrents (if available)

## Advanced Topics

### Fine-tuning on Custom Tasks

```python
# Load pre-trained model
model = OpenVLAForActionPrediction.from_pretrained("openvla/openvla-7b")

# Prepare dataset
dataset = CustomRobotDataset(
    data_path="path/to/demos",
    action_stats=action_stats
)

# Fine-tune
trainer = Trainer(
    model=model,
    train_dataset=dataset,
    args=TrainingArguments(
        learning_rate=5e-5,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        bf16=True
    )
)
trainer.train()
```

### Real Robot Deployment

1. **Camera Calibration**: Ensure camera intrinsics match training data
2. **Action Scaling**: Tune action normalization for your robot
3. **Safety Checks**: Implement joint limits and collision avoidance
4. **Latency**: Target <50ms inference time for real-time control

## References

- [OpenVLA Paper](https://arxiv.org/abs/2406.09246)
- [LIBERO Benchmark](https://lifelong-robot-learning.cs.utexas.edu/LIBERO.html)
- [Flash Attention](https://arxiv.org/abs/2205.14135)
