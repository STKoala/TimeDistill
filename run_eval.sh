#!/bin/bash

# 评估学生模型在多个数据集上的泛化能力
# 将结果保存到 log 目录

# 设置默认参数
MODEL_TYPE="${MODEL_TYPE:-student}"  # student 或 teacher
MODEL_PATH="${MODEL_PATH:-./chronos-2-distilled/final_model}"  # 学生模型路径
TEACHER_MODEL_ID="${TEACHER_MODEL_ID:-amazon/chronos-2}"  # 教师模型 ID
EVAL_DIR="${EVAL_DIR:-datasets/Eval_Data}"
DATASET="${DATASET:-}"  # 如果设置，则只评估该数据集（滑动窗口模式）
# DATASET="ETT-small/ETTm1"  # 默认数据集（已注释，可通过环境变量设置）
CONTEXT_LENGTH="${CONTEXT_LENGTH:-720}"
HORIZON="${HORIZON:-96}"
STRIDE="${STRIDE:-1}"  # 滑动窗口步长（仅在 DATASET 模式下有效）
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${LOG_DIR:-log}"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/eval_${TIMESTAMP}.log"

MODEL_TYPE_NAME="教师模型"
if [ "$MODEL_TYPE" = "student" ]; then
    MODEL_TYPE_NAME="学生模型"
fi

echo "=========================================="
echo "开始评估$MODEL_TYPE_NAME"
echo "=========================================="
echo "模型类型: $MODEL_TYPE"
if [ "$MODEL_TYPE" = "teacher" ]; then
    echo "教师模型 ID: $TEACHER_MODEL_ID"
else
    echo "学生模型路径: $MODEL_PATH"
fi
echo "评估目录: $EVAL_DIR"
if [ -n "$DATASET" ]; then
    echo "数据集: $DATASET (滑动窗口模式)"
    echo "滑动窗口步长: $STRIDE"
else
    echo "模式: 多数据集评估"
fi
echo "上下文长度: $CONTEXT_LENGTH"
echo "预测长度: $HORIZON"
echo "设备: $DEVICE"
echo "日志目录: $LOG_DIR"
echo "日志文件: $LOG_FILE"
echo "=========================================="
echo ""

# 构建命令参数
CMD_ARGS=(
    --model_type "$MODEL_TYPE"
    --eval_dir "$EVAL_DIR"
    --context_length "$CONTEXT_LENGTH"
    --horizon "$HORIZON"
    --device "$DEVICE"
    --log_dir "$LOG_DIR"
)

# 根据模型类型添加相应的参数
if [ "$MODEL_TYPE" = "teacher" ]; then
    CMD_ARGS+=(--teacher_model_id "$TEACHER_MODEL_ID")
else
    CMD_ARGS+=(--model_path "$MODEL_PATH")
fi

# 如果指定了数据集，添加相关参数
if [ -n "$DATASET" ]; then
    CMD_ARGS+=(--dataset "$DATASET")
    CMD_ARGS+=(--stride "$STRIDE")
fi

# 运行评估脚本，同时输出到控制台和日志文件
python eval.py "${CMD_ARGS[@]}" 2>&1 | tee "$LOG_FILE"

# 检查退出状态
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "评估完成！"
    echo "日志文件: $LOG_FILE"
    echo "结果 JSON 文件: $LOG_DIR/eval_results_*.json"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "评估失败，退出代码: $EXIT_CODE"
    echo "请查看日志文件: $LOG_FILE"
    echo "=========================================="
    exit $EXIT_CODE
fi

