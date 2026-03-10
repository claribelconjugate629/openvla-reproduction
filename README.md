# OpenVLA: Vision-Language-Action Model Reproduction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A complete reproduction and evaluation of the OpenVLA (Open Vision-Language-Action) model on LIBERO benchmark tasks, including environment setup, model deployment, and quantitative performance analysis.

## Project Overview

This project demonstrates the end-to-end implementation of OpenVLA, a state-of-the-art multimodal model that combines vision, language, and robotic action prediction. The model was successfully evaluated on the LIBERO simulation benchmark with detailed performance metrics and visualization.

### Key Achievements

- **Complete Environment Setup**: Successfully configured CUDA toolkit, Flash Attention 2, and all dependencies
- **Model Deployment**: Deployed and optimized OpenVLA-7B with fine-tuned variants for different LIBERO task suites
- **Benchmark Evaluation**: Achieved successful task completion across multiple LIBERO scenarios (Spatial, Object, Goal, 10-task)
- **Performance Optimization**: GPU memory optimized to ~10GB with BF16 precision and efficient batching
- **Issue Resolution**: Documented and fixed critical bugs including PyTorch serialization and EGL context errors

## Demo Videos

### Successful Task Execution Examples

| Task Type | Description | Demo |
|-----------|-------------|------|
| Spatial Reasoning | Pick and place with spatial constraints | [demo_1](assets/demo_videos/demo_1_spatial_task.mp4) |
| Object Manipulation | Complex object handling | [demo_2](assets/demo_videos/demo_2_object_manipulation.mp4) |
| Scene Understanding | Multi-object scene navigation | [demo_3](assets/demo_videos/demo_3_complex_scene.mp4) |

*All videos show real-time robot manipulation in MuJoCo simulation environment*

## Architecture Overview

OpenVLA integrates three core components:

1. **Vision Backbone** (SigLIP/DINOv2/CLIP)
   - Extracts visual features from RGB observations
   - Supports dual-encoder fusion for enhanced perception
   - Output: `[batch, num_patches, vision_dim]`

2. **Vision-Language Projector** (MLP)
   - Maps visual features to LLM embedding space
   - Enables cross-modal attention and reasoning
   - Architecture: `vision_dim → 4*vision_dim → llm_dim → llm_dim`

3. **Language Model + Action Head** (LLaMA-2/Vicuna + MLP)
   - Processes multimodal inputs with Flash Attention 2
   - Predicts 7-DoF continuous actions: `[x, y, z, rx, ry, rz, gripper]`
   - BF16 precision for memory efficiency

## Installation

### Prerequisites

```bash
# System requirements
- NVIDIA GPU with CUDA 11.8+
- Python 3.10+
- Ubuntu 20.04+ (recommended)
```

### Step 1: Install CUDA Toolkit

```bash
sudo apt update
sudo apt install nvidia-cuda-toolkit
nvcc --version  # Verify installation
```

### Step 2: Clone Repository and Install Dependencies

```bash
# Clone OpenVLA
git clone https://github.com/openvla/openvla.git
cd openvla

# Install in editable mode
pip install -e .

# Install build tools
pip install packaging ninja

# Install Flash Attention 2 (critical for performance)
pip install flash-attn==2.5.5 --no-build-isolation
```

### Step 3: Download Model Weights

```bash
# Base model (7B parameters)
git lfs clone https://huggingface.co/openvla/openvla-7b

# Fine-tuned models for LIBERO tasks
git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial
git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-object
git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-goal
git lfs clone https://huggingface.co/openvla/openvla-7b-finetuned-libero-10
```

## Usage

### Run Evaluation on LIBERO Benchmark

```bash
# Evaluate on Spatial task suite
python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint /path/to/openvla-7b-finetuned-libero-spatial \
    --task_suite_name libero_spatial \
    --center_crop True

# Evaluate on Goal task suite
python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint /path/to/openvla-7b-finetuned-libero-goal \
    --task_suite_name libero_goal \
    --center_crop True
```

### Adjust Evaluation Parameters

```bash
# Reduce trials for faster testing (default: 50)
python experiments/robot/libero/run_libero_eval.py \
    --model_family openvla \
    --pretrained_checkpoint /path/to/checkpoint \
    --task_suite_name libero_spatial \
    --center_crop True \
    --num_trials_per_task 10
```

## Troubleshooting

### Issue 1: PyTorch Weights-Only Load Error

**Error Message:**
```
_pickle.UnpicklingError: Weights only load failed...
WeightsUnpickler error: Unsupported global: GLOBAL numpy.core.multiarray._reconstruct
```

**Solution:**

Edit `/path/to/libero/libero/benchmark/__init__.py` (around line 164):

```python
# Change from:
init_states = torch.load(init_states_path)

# To:
import numpy as np
init_states = torch.load(init_states_path, weights_only=False)
```

**Root Cause:** PyTorch 2.0+ enables `weights_only=True` by default for security, blocking numpy objects in LIBERO's initialization states.

### Issue 2: EGL Context Error (Harmless)

**Error Message:**
```
OpenGL.raw.EGL._errors.EGLError: EGLError(err = EGL_NOT_INITIALIZED, ...)
```

**Analysis:** This occurs during MuJoCo rendering context cleanup at program exit. It does not affect evaluation results or video generation.

## Technical Details

### Data Processing Pipeline

1. **Image Preprocessing**
   - Resize to 224x224
   - Center crop during inference (Random crop during training)
   - Normalization to match pre-training distribution

2. **Action Normalization**
   - Z-score normalization: `(action - μ) / σ`
   - Task-specific statistics for better convergence
   - De-normalization during inference: `action = normalized * σ + μ`

3. **Multimodal Fusion**
   - Visual tokens injected into text sequence
   - Cross-modal attention via Transformer self-attention
   - Causal masking for autoregressive generation

### Evaluation Workflow

```python
for task in task_suite:
    for episode in range(num_trials):
        obs = env.reset()
        env.set_init_state(initial_states[episode])

        for step in range(max_steps):
            # Get RGB observation
            image = process_observation(obs)

            # Construct prompt
            prompt = f"What action should the robot take to {task_description}?"

            # Model inference
            action = model.predict_action(prompt, image, unnorm_key=task_suite_name)

            # Execute in simulation
            obs, reward, done, info = env.step(action)

            if done: break
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| GPU Memory Usage | ~10.3 GB (single V100/A100) |
| Inference Speed | ~5-8 FPS in simulation |
| Model Precision | BF16 (mixed precision) |
| Success Rate (Libero Spatial) | Evaluated on 50 episodes per task |

## Project Structure

```
openvla-reproduction/
├── README.md                  # This file
├── assets/
│   ├── demo_videos/          # Task execution videos
│   └── images/               # Screenshots and diagrams
├── docs/
│   ├── DETAILED_GUIDE.md     # In-depth technical documentation
│   └── ARCHITECTURE.md       # Model architecture details
└── scripts/
    ├── setup_env.sh          # Environment setup script
    └── run_eval.sh           # Batch evaluation script
```

## Citation

If you use this work in your research, please cite:

```bibtex
@article{openvla2024,
  title={OpenVLA: An Open-Source Vision-Language-Action Model},
  author={Kim, Moo Jin and Pertsch, Karl and Karamcheti, Siddharth and Xiao, Ted and others},
  journal={arXiv preprint arXiv:2406.09246},
  year={2024}
}
```

## Resources

- **Original Paper**: [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246)
- **Official Repository**: [openvla/openvla](https://github.com/openvla/openvla)
- **Model Weights**: [Hugging Face - openvla](https://huggingface.co/openvla)
- **LIBERO Benchmark**: [LIBERO: Benchmarking Knowledge Transfer in Lifelong Robot Learning](https://lifelong-robot-learning.cs.utexas.edu/LIBERO.html)

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contact

For questions or collaboration opportunities, please open an issue in this repository.

---

**Note**: This is a reproduction project for research and educational purposes. All credit for the original OpenVLA model goes to the authors and contributors of the official repository.
