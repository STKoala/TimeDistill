from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 添加项目根目录到路径（如果需要）
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="知识蒸馏训练入口 - 支持 TimesFM 和 Chronos-2 模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 训练 TimesFM 蒸馏模型
  python -m TimeDistill.experiments.distill --model timesfm
  
  # 训练 Chronos-2 蒸馏模型
  python -m TimeDistill.experiments.distill --model chronos2
  
  # 传递额外参数给 Chronos-2 脚本（例如配置文件）
  python -m TimeDistill.experiments.distill --model chronos2 --config configs/chronos-2-distill.yaml
  
  # 传递额外参数（所有 -- 后面的参数会传递给对应的脚本）
  python -m TimeDistill.experiments.distill --model timesfm --batch_size 64 --learning_rate 1e-5
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["timesfm", "chronos2"],
        required=True,
        help="选择要蒸馏的模型类型: 'timesfm' 或 'chronos2'"
    )
    
    # 解析已知参数（只解析 --model）
    args, unknown_args = parser.parse_known_args()
    
    # 根据模型类型调用相应的脚本
    if args.model == "timesfm":
        print("=" * 60)
        print("启动 TimesFM 知识蒸馏训练")
        print("=" * 60)
        
        # 导入 TimesFM 蒸馏脚本
        try:
            # 尝试从项目根目录导入
            from TimeDistill.timesfm_distill_gkd import main as timesfm_main
        except ImportError:
            try:
                # 尝试从当前目录导入
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from timesfm_distill_gkd import main as timesfm_main
            except ImportError as e:
                print(f"错误: 无法导入 TimesFM 蒸馏脚本: {e}")
                print("请确保 timesfm_distill_gkd.py 文件存在且可访问")
                sys.exit(1)
        
        # 如果有未知参数，打印提示（TimesFM 脚本可能不支持命令行参数）
        if unknown_args:
            print(f"注意: TimesFM 脚本当前不支持命令行参数，以下参数将被忽略: {unknown_args}")
            print("如需配置参数，请直接修改 timesfm_distill_gkd.py 中的配置")
        
        # 调用 TimesFM 蒸馏主函数
        timesfm_main()
        
    elif args.model == "chronos2":
        print("=" * 60)
        print("启动 Chronos-2 知识蒸馏训练")
        print("=" * 60)
        
        # 导入 Chronos-2 蒸馏脚本
        try:
            from TimeDistill.scripts.chronos2_distill_feature_dtw import main as chronos2_main
        except ImportError:
            try:
                # 尝试从当前目录导入
                sys.path.insert(0, str(Path(__file__).parent.parent))
                from scripts.chronos2_distill_feature_dtw import main as chronos2_main
            except ImportError as e:
                print(f"错误: 无法导入 Chronos-2 蒸馏脚本: {e}")
                print("请确保 scripts/chronos2_distill_feature_dtw.py 文件存在且可访问")
                sys.exit(1)
        
        # Chronos-2 脚本支持命令行参数，需要重新解析
        # 将 unknown_args 传递给 sys.argv，让 Chronos-2 脚本自己解析
        if unknown_args:
            # 保存原始 sys.argv
            original_argv = sys.argv.copy()
            # 设置新的 sys.argv（保留原始脚本路径，添加所有未知参数）
            # argparse 会正确解析这些参数
            sys.argv = [sys.argv[0]] + unknown_args
            try:
                chronos2_main()
            finally:
                # 恢复原始 sys.argv
                sys.argv = original_argv
        else:
            chronos2_main()
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
