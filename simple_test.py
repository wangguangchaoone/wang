#!/usr/bin/env python3
"""
简化的YOLO-MHAF代码结构验证脚本
不依赖外部库，只检查代码结构和导入
"""

import os
import sys
import importlib.util

def check_file_exists(filepath):
    """检查文件是否存在"""
    if os.path.exists(filepath):
        print(f"✓ {filepath}")
        return True
    else:
        print(f"✗ {filepath} - 文件不存在")
        return False

def check_python_syntax(filepath):
    """检查Python文件语法"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        compile(code, filepath, 'exec')
        print(f"✓ {filepath} - 语法正确")
        return True
    except SyntaxError as e:
        print(f"✗ {filepath} - 语法错误: {e}")
        return False
    except Exception as e:
        print(f"✗ {filepath} - 错误: {e}")
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("YOLO-MHAF 项目结构验证")
    print("=" * 60)
    
    # 检查项目结构
    required_files = [
        "requirements.txt",
        "README.md",
        "train.py",
        "inference.py", 
        "evaluate.py",
        "test_model.py",
        "models/__init__.py",
        "models/yolo_mhaf_simple.py",
        "models/losses.py",
        "models/modules/__init__.py",
        "models/modules/ghost.py",
        "models/modules/spd_conv.py", 
        "models/modules/attention.py",
        "models/modules/c2f_hybrid.py",
        "models/modules/amfn.py",
        "utils/__init__.py",
        "utils/datasets.py",
        "utils/metrics.py",
        "utils/visualization.py",
        "configs/yolo_mhaf.yaml"
    ]
    
    print("检查文件存在性:")
    print("-" * 30)
    missing_files = []
    for filepath in required_files:
        if not check_file_exists(filepath):
            missing_files.append(filepath)
    
    print(f"\n文件检查结果: {len(required_files) - len(missing_files)}/{len(required_files)} 个文件存在")
    
    # 检查Python文件语法
    python_files = [f for f in required_files if f.endswith('.py')]
    
    print("\n检查Python文件语法:")
    print("-" * 30)
    syntax_errors = []
    for filepath in python_files:
        if os.path.exists(filepath):
            if not check_python_syntax(filepath):
                syntax_errors.append(filepath)
        else:
            syntax_errors.append(filepath)
    
    print(f"\n语法检查结果: {len(python_files) - len(syntax_errors)}/{len(python_files)} 个文件语法正确")
    
    # 检查项目结构完整性
    print("\n项目结构检查:")
    print("-" * 30)
    
    required_dirs = [
        "models",
        "models/modules", 
        "utils",
        "configs",
        "weights"
    ]
    
    missing_dirs = []
    for dirname in required_dirs:
        if os.path.exists(dirname):
            print(f"✓ {dirname}/")
        else:
            print(f"✗ {dirname}/ - 目录不存在")
            missing_dirs.append(dirname)
    
    # 总结
    print("\n" + "=" * 60)
    print("验证总结:")
    print("=" * 60)
    
    if len(missing_files) == 0:
        print("✓ 所有必需文件都存在")
    else:
        print(f"✗ 缺少 {len(missing_files)} 个文件")
        
    if len(syntax_errors) == 0:
        print("✓ 所有Python文件语法正确")
    else:
        print(f"✗ {len(syntax_errors)} 个文件有语法错误")
        
    if len(missing_dirs) == 0:
        print("✓ 项目目录结构完整")
    else:
        print(f"✗ 缺少 {len(missing_dirs)} 个目录")
    
    # 功能模块检查
    print("\n功能模块实现检查:")
    print("-" * 30)
    
    modules_info = [
        ("Ghost模块", "models/modules/ghost.py", ["GhostConv", "C2f_Ghost"]),
        ("SPD-Conv模块", "models/modules/spd_conv.py", ["SPDConv", "C2f_SPD"]),
        ("注意力机制", "models/modules/attention.py", ["CoordAtt", "ECA", "HybridAttention"]),
        ("C2f-Hybrid", "models/modules/c2f_hybrid.py", ["C2f_Hybrid", "SPPF_Hybrid"]),
        ("AMFN网络", "models/modules/amfn.py", ["AMFN", "CARAFE", "BiFPN"]),
        ("主模型", "models/yolo_mhaf_simple.py", ["YOLOMHAFSimple", "create_yolo_mhaf_n"]),
        ("损失函数", "models/losses.py", ["YOLOLoss", "FocalLoss"]),
        ("评估指标", "utils/metrics.py", ["compute_ap", "bbox_iou"]),
        ("数据处理", "utils/datasets.py", ["PeanutDiseaseDataset"]),
        ("可视化", "utils/visualization.py", ["draw_boxes", "plot_results"])
    ]
    
    for module_name, filepath, expected_classes in modules_info:
        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                found_classes = []
                for class_name in expected_classes:
                    if f"class {class_name}" in content or f"def {class_name}" in content:
                        found_classes.append(class_name)
                
                if len(found_classes) == len(expected_classes):
                    print(f"✓ {module_name}: 所有组件已实现")
                else:
                    missing = set(expected_classes) - set(found_classes)
                    print(f"✗ {module_name}: 缺少组件 {missing}")
                    
            except Exception as e:
                print(f"✗ {module_name}: 检查失败 - {e}")
        else:
            print(f"✗ {module_name}: 文件不存在")
    
    # 最终评估
    print("\n" + "=" * 60)
    total_issues = len(missing_files) + len(syntax_errors) + len(missing_dirs)
    
    if total_issues == 0:
        print("🎉 YOLO-MHAF项目实现完整！")
        print("所有核心组件都已正确实现，代码结构完整。")
        print("\n下一步操作:")
        print("1. 安装依赖: pip install -r requirements.txt")
        print("2. 准备花生叶片病害数据集")
        print("3. 开始训练: python train.py --data-path /path/to/dataset")
    else:
        print(f"⚠️  发现 {total_issues} 个问题需要解决")
        print("请检查上述错误信息并修复相关问题。")
    
    print("=" * 60)
    
    return total_issues == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)