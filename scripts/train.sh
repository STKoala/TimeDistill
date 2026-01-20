#!/bin/bash

# 知识蒸馏训练脚本
# 支持 TimesFM 和 Chronos-2 模型的蒸馏训练
# 自动保存日志到指定目录

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 设置默认参数（可以通过环境变量覆盖）
MODEL_TYPE="${MODEL_TYPE:-chronos2}"  # timesfm 或 chronos2
LOG_DIR="${LOG_DIR:-log}"
DEVICE="${DEVICE:-cuda}"
PYTHON="${PYTHON:-python}"

# 解析命令行参数
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --python)
            PYTHON="$2"
            shift 2
            ;;
        --help|-h)
            echo "用法: $0 [选项] [额外参数...]"
            echo ""
            echo "选项:"
            echo "  --model MODEL_TYPE      模型类型: timesfm 或 chronos2 (默认: chronos2)"
            echo "  --log_dir DIR           日志目录 (默认: log)"
            echo "  --device DEVICE         设备: cuda 或 cpu (默认: cuda)"
            echo "  --python PYTHON_CMD     Python 命令 (默认: python)"
            echo "  --help, -h              显示此帮助信息"
            echo ""
            echo "额外参数会传递给训练脚本（仅 Chronos-2 支持）"
            echo ""
            echo "示例:"
            echo "  # 训练 Chronos-2 模型"
            echo "  $0 --model chronos2"
            echo ""
            echo "  # 训练 TimesFM 模型"
            echo "  $0 --model timesfm"
            echo ""
            echo "  # 使用配置文件训练 Chronos-2"
            echo "  $0 --model chronos2 --config configs/chronos-2-distill.yaml"
            echo ""
            echo "  # 指定日志目录和设备"
            echo "  $0 --model timesfm --log_dir ./my_logs --device cuda"
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

# 验证模型类型
if [[ "$MODEL_TYPE" != "timesfm" && "$MODEL_TYPE" != "chronos2" ]]; then
    echo "错误: 无效的模型类型 '$MODEL_TYPE'，必须是 'timesfm' 或 'chronos2'"
    exit 1
fi

# 创建日志目录
mkdir -p "$LOG_DIR"

# 生成带时间戳的日志文件名
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/train_${MODEL_TYPE}_${TIMESTAMP}.log"

# 模型名称显示
MODEL_NAME="Chronos-2"
if [[ "$MODEL_TYPE" == "timesfm" ]]; then
    MODEL_NAME="TimesFM"
fi

echo "=========================================="
echo "知识蒸馏训练脚本"
echo "=========================================="
echo "模型类型: $MODEL_NAME ($MODEL_TYPE)"
echo "设备: $DEVICE"
echo "日志目录: $LOG_DIR"
echo "日志文件: $LOG_FILE"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "额外参数: ${EXTRA_ARGS[*]}"
fi
echo "=========================================="
echo ""

# 记录开始时间
START_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "训练开始时间: $START_TIME" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 构建训练命令
TRAIN_CMD=(
    "$PYTHON" -m TimeDistill.experiments.distill
    --model "$MODEL_TYPE"
)

# 添加额外参数（如果有）
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    TRAIN_CMD+=("${EXTRA_ARGS[@]}")
fi

# 显示执行的命令
echo "执行命令: ${TRAIN_CMD[*]}" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 切换到项目根目录
cd "$PROJECT_ROOT" || {
    echo "错误: 无法切换到项目根目录: $PROJECT_ROOT" | tee -a "$LOG_FILE"
    exit 1
}

# 执行训练，同时输出到控制台和日志文件
# 使用 unbuffered 模式确保实时输出
if command -v python3 &> /dev/null && [[ "$PYTHON" == "python" ]]; then
    PYTHON_CMD="python3 -u"
else
    PYTHON_CMD="$PYTHON -u"
fi

# 设置环境变量
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# 执行训练并捕获退出状态
TRAIN_EXIT_CODE=0
if $PYTHON_CMD -m TimeDistill.experiments.distill --model "$MODEL_TYPE" "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"; then
    TRAIN_EXIT_CODE=0
else
    TRAIN_EXIT_CODE=${PIPESTATUS[0]}
fi

# 记录结束时间
END_TIME=$(date +"%Y-%m-%d %H:%M:%S")
echo "" | tee -a "$LOG_FILE"
echo "训练结束时间: $END_TIME" | tee -a "$LOG_FILE"

# 计算训练时长
if command -v date &> /dev/null; then
    START_EPOCH=$(date -d "$START_TIME" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "$START_TIME" +%s 2>/dev/null || echo "")
    END_EPOCH=$(date -d "$END_TIME" +%s 2>/dev/null || date -j -f "%Y-%m-%d %H:%M:%S" "$END_TIME" +%s 2>/dev/null || echo "")
    
    if [[ -n "$START_EPOCH" && -n "$END_EPOCH" ]]; then
        DURATION=$((END_EPOCH - START_EPOCH))
        HOURS=$((DURATION / 3600))
        MINUTES=$(((DURATION % 3600) / 60))
        SECONDS=$((DURATION % 60))
        echo "训练时长: ${HOURS}小时 ${MINUTES}分钟 ${SECONDS}秒" | tee -a "$LOG_FILE"
    fi
fi

# 根据退出状态显示结果
echo "" | tee -a "$LOG_FILE"
if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "训练成功完成！" | tee -a "$LOG_FILE"
    echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    exit 0
else
    echo "==========================================" | tee -a "$LOG_FILE"
    echo "训练失败，退出代码: $TRAIN_EXIT_CODE" | tee -a "$LOG_FILE"
    echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"
    echo "==========================================" | tee -a "$LOG_FILE"
    exit $TRAIN_EXIT_CODE
fi

