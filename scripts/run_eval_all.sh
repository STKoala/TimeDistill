#!/bin/bash

# 批量评估脚本：遍历所有数据集，分别测试教师模型和学生模型
# 使用 run_eval.sh 来执行每个评估任务

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_EVAL_SCRIPT="$SCRIPT_DIR/run_eval.sh"

# 设置默认参数（可以通过环境变量覆盖）
EVAL_DIR="${EVAL_DIR:-data/datasets/Eval_Data}"
MODEL_PATH="${MODEL_PATH:-./chronos-2-distilled/final_model}"
TEACHER_MODEL_ID="${TEACHER_MODEL_ID:-amazon/chronos-2}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-720}"
HORIZON="${HORIZON:-96}"
STRIDE="${STRIDE:-1}"
DEVICE="${DEVICE:-cuda}"
LOG_DIR="${LOG_DIR:-log}"

# 检查 run_eval.sh 是否存在
if [ ! -f "$RUN_EVAL_SCRIPT" ]; then
    echo "错误: 找不到 run_eval.sh 脚本: $RUN_EVAL_SCRIPT"
    exit 1
fi

# 检查评估目录是否存在
if [ ! -d "$EVAL_DIR" ]; then
    echo "错误: 评估目录不存在: $EVAL_DIR"
    exit 1
fi

# 创建日志目录
mkdir -p "$LOG_DIR"

# 生成主日志文件
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG_FILE="$LOG_DIR/eval_all_${TIMESTAMP}.log"

echo "=========================================="
echo "批量评估脚本"
echo "=========================================="
echo "评估目录: $EVAL_DIR"
echo "学生模型路径: $MODEL_PATH"
echo "教师模型 ID: $TEACHER_MODEL_ID"
echo "上下文长度: $CONTEXT_LENGTH"
echo "预测长度: $HORIZON"
echo "滑动窗口步长: $STRIDE"
echo "设备: $DEVICE"
echo "日志目录: $LOG_DIR"
echo "主日志文件: $MAIN_LOG_FILE"
echo "=========================================="
echo ""

# 查找所有数据集
echo "正在查找数据集..."
DATASETS=()

# 遍历评估目录下的所有子目录
for subdir in "$EVAL_DIR"/*; do
    if [ -d "$subdir" ]; then
        subdir_name=$(basename "$subdir")
        # 查找该目录下的所有 CSV 文件
        for csv_file in "$subdir"/*.csv; do
            if [ -f "$csv_file" ]; then
                csv_name=$(basename "$csv_file" .csv)
                dataset_name="${subdir_name}/${csv_name}"
                DATASETS+=("$dataset_name")
                echo "  找到数据集: $dataset_name"
            fi
        done
    fi
done

# 检查是否找到数据集
if [ ${#DATASETS[@]} -eq 0 ]; then
    echo "错误: 未找到任何数据集"
    exit 1
fi

echo ""
echo "共找到 ${#DATASETS[@]} 个数据集"
echo ""

# 统计变量
TOTAL_TASKS=$((${#DATASETS[@]} * 2))  # 每个数据集测试教师和学生
CURRENT_TASK=0
SUCCESS_COUNT=0
FAILED_COUNT=0
FAILED_DATASETS=()

# 开始批量评估
{
    echo "=========================================="
    echo "开始批量评估"
    echo "总任务数: $TOTAL_TASKS (${#DATASETS[@]} 个数据集 × 2 个模型)"
    echo "开始时间: $(date)"
    echo "=========================================="
    echo ""

    # 遍历每个数据集
    for dataset in "${DATASETS[@]}"; do
        echo ""
        echo "##################################################"
        echo "处理数据集: $dataset"
        echo "##################################################"
        echo ""

        # 测试教师模型
        CURRENT_TASK=$((CURRENT_TASK + 1))
        echo "[$CURRENT_TASK/$TOTAL_TASKS] 测试教师模型 - $dataset"
        echo "----------------------------------------"
        
        export MODEL_TYPE="teacher"
        export MODEL_PATH="$MODEL_PATH"
        export TEACHER_MODEL_ID="$TEACHER_MODEL_ID"
        export EVAL_DIR="$EVAL_DIR"
        export DATASET="$dataset"
        export CONTEXT_LENGTH="$CONTEXT_LENGTH"
        export HORIZON="$HORIZON"
        export STRIDE="$STRIDE"
        export DEVICE="$DEVICE"
        export LOG_DIR="$LOG_DIR"
        
        if bash "$RUN_EVAL_SCRIPT"; then
            echo "✓ 教师模型评估成功: $dataset"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "✗ 教师模型评估失败: $dataset"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_DATASETS+=("teacher:$dataset")
        fi

        echo ""

        # 测试学生模型
        CURRENT_TASK=$((CURRENT_TASK + 1))
        echo "[$CURRENT_TASK/$TOTAL_TASKS] 测试学生模型 - $dataset"
        echo "----------------------------------------"
        
        export MODEL_TYPE="student"
        export MODEL_PATH="$MODEL_PATH"
        export TEACHER_MODEL_ID="$TEACHER_MODEL_ID"
        export EVAL_DIR="$EVAL_DIR"
        export DATASET="$dataset"
        export CONTEXT_LENGTH="$CONTEXT_LENGTH"
        export HORIZON="$HORIZON"
        export STRIDE="$STRIDE"
        export DEVICE="$DEVICE"
        export LOG_DIR="$LOG_DIR"
        
        if bash "$RUN_EVAL_SCRIPT"; then
            echo "✓ 学生模型评估成功: $dataset"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            echo "✗ 学生模型评估失败: $dataset"
            FAILED_COUNT=$((FAILED_COUNT + 1))
            FAILED_DATASETS+=("student:$dataset")
        fi

        echo ""
    done

    echo ""
    echo "=========================================="
    echo "批量评估完成"
    echo "=========================================="
    echo "完成时间: $(date)"
    echo "总任务数: $TOTAL_TASKS"
    echo "成功: $SUCCESS_COUNT"
    echo "失败: $FAILED_COUNT"
    
    if [ $FAILED_COUNT -gt 0 ]; then
        echo ""
        echo "失败的任务:"
        for failed in "${FAILED_DATASETS[@]}"; do
            echo "  - $failed"
        done
    fi
    
    echo "=========================================="
} | tee "$MAIN_LOG_FILE"

# 检查最终状态
if [ $FAILED_COUNT -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "所有评估任务完成！"
    echo "主日志文件: $MAIN_LOG_FILE"
    echo "结果 JSON 文件: $LOG_DIR/eval_results_*.json"
    echo "=========================================="
    exit 0
else
    echo ""
    echo "=========================================="
    echo "部分评估任务失败"
    echo "主日志文件: $MAIN_LOG_FILE"
    echo "失败数量: $FAILED_COUNT / $TOTAL_TASKS"
    echo "=========================================="
    exit 1
fi

