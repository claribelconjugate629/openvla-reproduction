#!/bin/bash

# OpenVLA Environment Setup Script
# This script automates the installation of OpenVLA and its dependencies

set -e  # Exit on error

echo "=========================================="
echo "OpenVLA Environment Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if CUDA is available
echo -e "\n${YELLOW}[1/6] Checking CUDA availability...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ CUDA detected${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo -e "${RED}✗ CUDA not found. Please install NVIDIA drivers and CUDA toolkit.${NC}"
    exit 1
fi

# Check Python version
echo -e "\n${YELLOW}[2/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then
    echo -e "${GREEN}✓ Python $PYTHON_VERSION is compatible${NC}"
else
    echo -e "${RED}✗ Python $PYTHON_VERSION is too old. Please install Python 3.10+${NC}"
    exit 1
fi

# Clone OpenVLA repository
echo -e "\n${YELLOW}[3/6] Cloning OpenVLA repository...${NC}"
if [ ! -d "openvla" ]; then
    git clone https://github.com/openvla/openvla.git
    echo -e "${GREEN}✓ Repository cloned${NC}"
else
    echo -e "${YELLOW}Repository already exists, skipping...${NC}"
fi

# Install dependencies
echo -e "\n${YELLOW}[4/6] Installing Python dependencies...${NC}"
cd openvla
pip install -e .
pip install packaging ninja
echo -e "${GREEN}✓ Dependencies installed${NC}"

# Verify Ninja
echo -e "\n${YELLOW}[5/6] Verifying Ninja installation...${NC}"
if ninja --version &> /dev/null; then
    echo -e "${GREEN}✓ Ninja $(ninja --version) installed${NC}"
else
    echo -e "${RED}✗ Ninja installation failed${NC}"
    exit 1
fi

# Install Flash Attention 2
echo -e "\n${YELLOW}[6/6] Installing Flash Attention 2 (this may take 5-10 minutes)...${NC}"
pip install flash-attn==2.5.5 --no-build-isolation
echo -e "${GREEN}✓ Flash Attention 2 installed${NC}"

echo -e "\n${GREEN}=========================================="
echo "Environment setup completed successfully!"
echo "==========================================${NC}"

echo -e "\nNext steps:"
echo "1. Download model weights: git lfs clone https://huggingface.co/openvla/openvla-7b"
echo "2. Run evaluation: python experiments/robot/libero/run_libero_eval.py --help"
