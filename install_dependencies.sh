#!/bin/bash
# MOIRAI 蒸馏脚本依赖安装脚本

echo "开始安装 MOIRAI 蒸馏所需依赖..."

# 设置镜像源（可选，加速下载）
# pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 1. 先安装基础依赖（使用预编译包）
echo "1. 安装基础依赖..."
uv pip install --upgrade pip setuptools wheel

# 2. 安装 numpy（预编译包）
echo "2. 安装 numpy..."
uv pip install numpy

# 3. 安装 pandas（使用预编译二进制包，避免编译）
echo "3. 安装 pandas（预编译包）..."
uv pip install --only-binary pandas pandas

# 4. 安装其他依赖
echo "4. 安装其他依赖..."
uv pip install lightning>=2.0
uv pip install einops==0.7.*
uv pip install jaxtyping~=0.2.24
uv pip install hydra-core==1.3
uv pip install huggingface_hub>=0.23.0
uv pip install safetensors

# 5. 安装 gluonts（可能需要一些时间）
echo "5. 安装 gluonts..."
uv pip install gluonts~=0.14.3

# 6. 安装 uni2ts
echo "6. 安装 uni2ts..."
cd /root/shengyuan/Distillation/uni2ts
uv pip install -e .

echo ""
echo "=" * 60
echo "依赖安装完成！"
echo "=" * 60
echo ""
echo "验证安装:"
python -c "import sys; sys.path.insert(0, 'src'); from uni2ts.model.moirai import MoiraiForecast; print('✓ MOIRAI 模块导入成功')"

