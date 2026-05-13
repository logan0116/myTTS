#!/usr/bin/env bash
# ------------------------------------------------------------------
# myTTS — 一键环境搭建脚本
#
# 用法:
#   chmod +x setup.sh
#   ./setup.sh
#
# 需要先安装 conda 或使用系统 Python 3.10
# ------------------------------------------------------------------
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$ROOT_DIR/pretrained_models"

echo "========================================"
echo "  myTTS 环境搭建"
echo "========================================"

# ---- 1. Python 环境 ----
echo ""
echo "[1/5] 检查 Python 版本 ..."
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+')
echo "  Python $PYTHON_VERSION"

# ---- 2. 系统依赖 (Ubuntu) ----
echo ""
echo "[2/5] 安装系统依赖 ..."
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq sox libsox-dev ffmpeg
else
    echo "  (非 Ubuntu 系统，请手动安装 sox 和 ffmpeg)"
fi

# ---- 3. Python 依赖 ----
echo ""
echo "[3/5] 安装 Python 依赖 ..."
pip install -r "$ROOT_DIR/requirements.txt" -q

# ---- 4. CosyVoice 模型框架 ----
echo ""
echo "[4/5] 安装 CosyVoice ..."
COSYVOICE_DIR="$ROOT_DIR/CosyVoice"
if [ ! -d "$COSYVOICE_DIR" ]; then
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_DIR"
else
    echo "  CosyVoice 已存在，跳过 clone。"
fi
cd "$COSYVOICE_DIR"
pip install -r requirements.txt -q
cd "$ROOT_DIR"

# ---- 5. 下载模型 ----
echo ""
echo "[5/5] 下载模型文件 ..."
mkdir -p "$MODELS_DIR"

download_model() {
    local repo_id="$1"
    local local_dir="$2"
    if [ ! -d "$local_dir" ] || [ -z "$(ls -A "$local_dir" 2>/dev/null)" ]; then
        echo "  下载 $repo_id -> $local_dir"
        python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$repo_id', local_dir='$local_dir')
"
    else
        echo "  $local_dir 已存在，跳过下载。"
    fi
}

download_model "FunAudioLLM/Fun-CosyVoice3-0.5B-2512" \
    "$MODELS_DIR/Fun-CosyVoice3-0.5B"

download_model "FunAudioLLM/CosyVoice-ttsfrd" \
    "$MODELS_DIR/CosyVoice-ttsfrd"

# ---- 完成 ----
echo ""
echo "========================================"
echo "  ✅ 环境搭建完成！"
echo ""
echo "  启动后端:  python -m uvicorn main:app --reload --port 8000"
echo "  启动前端:  streamlit run app.py"
echo "========================================"
