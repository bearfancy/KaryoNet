#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型测试脚本
测试保存的模型在测试数据集上的性能
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import logging
from datetime import datetime
import argparse
import json

# 添加模型路径
sys.path.append('MODELS')

# 导入模型
from MODELS.resnet_MFIM import resnet50 as ResNet_MFIM
from MODELS.resnet_DAM import resnet50 as ResNet_DAM
from MODELS.resnet_MFIM_DAM import resnet50 as resnet50_MFIM_DAM

def setup_logging():
    """设置日志"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(f'test_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

def improve_predictions_with_pair_relationships(predictions, pair_relationships, chromosome_groups):
    """
    利用配对关系改善核型分类预测
    
    Args:
        predictions: 原始预测结果 [batch_size]
        pair_relationships: 配对关系矩阵 [batch_size, batch_size, 2]
        chromosome_groups: 染色体组别信息 [batch_size]
    
    Returns:
        improved_predictions: 改善后的预测结果
        pair_accuracy: 配对关系准确率
    """
    batch_size = predictions.size(0)
    improved_predictions = predictions.clone()
    
    # 获取配对概率矩阵 (batch_size, batch_size)
    pair_probs = pair_relationships[:, :, 0]  # 取配对概率
    
    # 计算配对关系准确率
    pair_correct = 0
    pair_total = 0
    
    # 对每个染色体，检查其配对关系
    for i in range(batch_size):
        # 找到与当前染色体配对概率最高的其他染色体
        pair_scores = pair_probs[i]
        # 排除自己
        pair_scores[i] = 0
        
        # 找到配对概率最高的染色体
        best_pair_idx = torch.argmax(pair_scores)
        best_pair_prob = pair_scores[best_pair_idx]
        
        # 检查配对关系是否正确
        if chromosome_groups[i] == chromosome_groups[best_pair_idx]:
            pair_correct += 1
        pair_total += 1
        
        # 如果配对概率很高（>0.5），且配对染色体属于同一组
        if best_pair_prob > 0.5 and chromosome_groups[i] == chromosome_groups[best_pair_idx]:
            # 如果配对染色体的预测更可信，可以考虑调整当前预测
            # 这里使用简单的策略：如果配对染色体预测概率更高，则采用其预测
            if predictions[best_pair_idx] != predictions[i]:
                # 可以选择保持原预测，或者采用配对染色体的预测
                # 这里我们采用保守策略：保持原预测，但记录配对信息
                pass
    
    pair_accuracy = pair_correct / pair_total if pair_total > 0 else 0
    
    return improved_predictions, pair_accuracy

def get_chromosome_group(chromosome_id):
    """
    获取染色体组别信息 - 与训练时保持一致
    使用与训练时相同的label2list分组：
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    对应染色体: 1-12, 13-22, X, Y
    """
    # 与训练时的label2list保持一致
    label2list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1]
    
    if 0 <= chromosome_id < len(label2list):
        return label2list[chromosome_id]
    else:
        return -1  # 未知组别

def read_data(rootdir, rootdirval):
    """
    读取数据目录结构
    返回验证集案例名称列表
    """
    valcasenames = []
    
    if os.path.exists(rootdirval):
        valcasenames = [d for d in os.listdir(rootdirval) if os.path.isdir(os.path.join(rootdirval, d))]
        valcasenames.sort()
        logging.info(f"找到 {len(valcasenames)} 个验证案例")
    else:
        logging.error(f"验证集路径不存在: {rootdirval}")
        return [], [], [], [], []
    
    return valcasenames, [], [], [], []

def load_model(model_path, model_type, num_classes=24, device='cuda'):
    """
    加载保存的模型
    """
    logging.info(f"加载模型: {model_path}")
    logging.info(f"模型类型: {model_type}")
    
    # 创建模拟的args对象
    class Args:
        def __init__(self):
            self.num_class = num_classes
            self.enc_layers = 3
            self.dec_layers = 6
            self.pretrained = True
            self.pretrained_model = './model/pretrain/resnet50-19c8e357.pth'
            self.hidden_dim = 256
            self.dim_feedforward = 2048
            self.dropout = 0.0
            self.nheads = 8
            self.num_queries = 100
            self.seed = 42
            self.pre_norm = False
    
    args = Args()
    
    # 创建模型实例
    if 'MFIM_DAM' in model_type:
        from MODELS.resnet_MFIM_DAM import resnet50
        model = resnet50(args, num_classes=num_classes)
    elif 'MFIM' in model_type:
        from MODELS.resnet_MFIM import ResNet_MFIM
        model = ResNet_MFIM(args, num_classes=num_classes)
    elif 'DAM' in model_type:
        from MODELS.resnet_DAM import ResNet_DAM
        model = ResNet_DAM(args, num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")
    
    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info("加载模型状态字典")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
        logging.info("加载模型权重")
    else:
        # 如果checkpoint不是字典，可能是直接的模型对象
        logging.info("加载模型权重（直接模型对象）")
        model.load_state_dict(checkpoint.state_dict())
    
    model = model.to(device)
    model.eval()
    
    return model

def test_model(model, test_data_path, device='cuda'):
    """
    测试模型 - 同时进行核型分类和极性分类
    """
    logging.info(f"开始测试，测试数据路径: {test_data_path}")
    
    # 数据预处理 - 与训练时保持一致
    mytransforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # 转换为灰度图，与训练时一致
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])  # 使用训练时的归一化参数
    ])
    
    # 获取测试案例
    test_cases = [d for d in os.listdir(test_data_path) if os.path.isdir(os.path.join(test_data_path, d))]
    test_cases.sort()
    
    logging.info(f"找到 {len(test_cases)} 个测试案例")
    
    # 核型分类统计
    total_correct = 0
    total_samples = 0
    
    # 极性分类统计
    total_polarity_correct = 0
    total_polarity_samples = 0
    
    # 配对关系统计
    total_pair_correct = 0
    total_pair_samples = 0
    
    case_results = []
    
    with torch.no_grad():
        for case_idx, case in enumerate(test_cases):
            if case_idx % 50 == 0:
                logging.info(f"测试进度: {case_idx+1}/{len(test_cases)} ({100*(case_idx+1)/len(test_cases):.1f}%)")
            
            case_path = os.path.join(test_data_path, case)
            img_list = os.listdir(case_path)
            
            if len(img_list) == 0:
                logging.warning(f"案例 {case} 没有图片文件")
                continue
            
            # 限制每个案例的图片数量，避免内存问题
            max_images_per_case = 92
            if len(img_list) > max_images_per_case:
                img_list = img_list[:max_images_per_case]
                logging.info(f"案例 {case} 图片数量过多，限制为前 {max_images_per_case} 张")
            
            # 准备数据
            labels = []  # 核型标签
            polarity_labels = []  # 极性标签
            img_tensors = torch.empty(len(img_list), 3, 224, 224)
            
            for img_idx, img_name in enumerate(img_list):
                try:
                    # 加载图片
                    img_path = os.path.join(case_path, img_name)
                    data = Image.open(img_path)
                    data = mytransforms(data)
                    img_tensors[img_idx, :, :, :] = data
                    
                    # 解析标签 (格式: ID_核型_角度_细胞名_Karyotype.jpg)
                    parts = img_name.split('_')
                    if len(parts) >= 3:
                        # 核型标签 (1-24 -> 0-23)
                        label = int(parts[1]) - 1
                        labels.append(label)
                        
                        # 极性标签 (角度 -> 极性类别: 0°, 90°, 180°, 270° -> 0, 1, 2, 3)
                        if len(parts) >= 3:
                            try:
                                angle = int(parts[2])
                                polarity_label = angle // 90  # 将角度映射到极性标签
                                polarity_labels.append(polarity_label)
                            except ValueError:
                                # 如果角度解析失败，跳过极性标签
                                polarity_labels.append(-1)  # 标记为无效
                        else:
                            polarity_labels.append(-1)
                    else:
                        logging.warning(f"无法解析文件名: {img_name}")
                        continue
                        
                except Exception as e:
                    logging.error(f"处理图片 {img_name} 时出错: {e}")
                    continue
            
            if len(labels) == 0:
                logging.warning(f"案例 {case} 没有有效的标签")
                continue
            
            # 移动到设备
            images = img_tensors.to(device)
            
            # 模型推理
            try:
                # 所有模型都返回三个值: tag_space, xp, chrompair1
                outputs, polarity_outputs, pair_relationships = model(images)
                
                # 核型分类预测
                _, predicted = torch.max(outputs.data, 1)
                
                # 利用配对关系改善预测
                chromosome_groups = torch.tensor([get_chromosome_group(label) for label in labels]).to(device)
                improved_predictions, pair_accuracy = improve_predictions_with_pair_relationships(
                    predicted, pair_relationships, chromosome_groups
                )
                
                # 极性分类预测
                _, polarity_predicted = torch.max(polarity_outputs.data, 1)
                
                # 计算核型分类准确率
                correct = 0
                for i in range(len(predicted)):
                    if predicted[i].item() == labels[i]:
                        correct += 1
                
                # 计算极性分类准确率
                polarity_correct = 0
                polarity_valid_samples = 0
                for i in range(len(polarity_predicted)):
                    if polarity_labels[i] != -1:  # 只统计有效的极性标签
                        polarity_valid_samples += 1
                        if polarity_predicted[i].item() == polarity_labels[i]:
                            polarity_correct += 1
                
                case_accuracy = correct / len(labels) if len(labels) > 0 else 0
                case_polarity_accuracy = polarity_correct / polarity_valid_samples if polarity_valid_samples > 0 else 0
                
                total_correct += correct
                total_samples += len(labels)
                total_polarity_correct += polarity_correct
                total_polarity_samples += polarity_valid_samples
                
                # 配对关系统计
                pair_correct_count = int(pair_accuracy * len(labels))
                total_pair_correct += pair_correct_count
                total_pair_samples += len(labels)
                
                case_results.append({
                    'case': case,
                    'total_images': len(labels),
                    'karyotype_correct': correct,
                    'karyotype_accuracy': case_accuracy,
                    'polarity_valid_samples': polarity_valid_samples,
                    'polarity_correct': polarity_correct,
                    'polarity_accuracy': case_polarity_accuracy,
                    'pair_accuracy': pair_accuracy
                })
                
                logging.info(f"案例 {case}: 核型分类 {correct}/{len(labels)} 正确, 准确率: {case_accuracy:.4f}")
                logging.info(f"案例 {case}: 极性分类 {polarity_correct}/{polarity_valid_samples} 正确, 准确率: {case_polarity_accuracy:.4f}")
                
            except Exception as e:
                logging.error(f"案例 {case} 推理时出错: {e}")
                continue
    
    # 计算总体准确率
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    overall_polarity_accuracy = total_polarity_correct / total_polarity_samples if total_polarity_samples > 0 else 0
    overall_pair_accuracy = total_pair_correct / total_pair_samples if total_pair_samples > 0 else 0
    
    logging.info(f"测试完成!")
    logging.info(f"=== 核型分类结果 ===")
    logging.info(f"总样本数: {total_samples}")
    logging.info(f"正确预测: {total_correct}")
    logging.info(f"总体准确率: {overall_accuracy:.4f}")
    logging.info(f"=== 极性分类结果 ===")
    logging.info(f"有效样本数: {total_polarity_samples}")
    logging.info(f"正确预测: {total_polarity_correct}")
    logging.info(f"总体准确率: {overall_polarity_accuracy:.4f}")
    logging.info(f"=== 配对关系结果 ===")
    logging.info(f"总样本数: {total_pair_samples}")
    logging.info(f"正确预测: {total_pair_correct}")
    logging.info(f"总体准确率: {overall_pair_accuracy:.4f}")
    
    return {
        'karyotype_accuracy': overall_accuracy,
        'karyotype_total_samples': total_samples,
        'karyotype_total_correct': total_correct,
        'polarity_accuracy': overall_polarity_accuracy,
        'polarity_total_samples': total_polarity_samples,
        'polarity_total_correct': total_polarity_correct,
        'pair_accuracy': overall_pair_accuracy,
        'pair_total_samples': total_pair_samples,
        'pair_total_correct': total_pair_correct,
        'case_results': case_results
    }

def main():
    parser = argparse.ArgumentParser(description='测试保存的模型')
    parser.add_argument("--model_path", default='experiments/model-resnet50_MFIM_DAM_20250922_172521/models/model-resnet50_MFIM_DAM-best.pth', type=str, help='模型文件路径')
    parser.add_argument("--test_data_path", default='dataset/#1466_karyoNet_test250923/test', type=str, help='测试数据路径')
    parser.add_argument("--model_type", default='MFIM_DAM', type=str, help='模型类型')
    parser.add_argument("--num_classes", default=24, type=int, help='类别数量')
    parser.add_argument("--device", default='cuda:0', type=str, help='设备')
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    
    # 检查文件是否存在
    if not os.path.exists(args.model_path):
        logging.error(f"模型文件不存在: {args.model_path}")
        return
    
    if not os.path.exists(args.test_data_path):
        logging.error(f"测试数据路径不存在: {args.test_data_path}")
        return
    
    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logging.info(f"使用设备: {device}")
    
    try:
        # 加载模型
        model = load_model(args.model_path, args.model_type, args.num_classes, device)
        
        # 测试模型
        results = test_model(model, args.test_data_path, device)
        
        # 保存结果
        results_file = f'test_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logging.info(f"测试结果已保存到: {results_file}")
        
        # 打印详细结果
        print("\n" + "="*70)
        print("测试结果摘要")
        print("="*70)
        print("核型分类结果:")
        print(f"  总体准确率: {results['karyotype_accuracy']:.4f}")
        print(f"  总样本数: {results['karyotype_total_samples']}")
        print(f"  正确预测: {results['karyotype_total_correct']}")
        print()
        print("极性分类结果:")
        print(f"  总体准确率: {results['polarity_accuracy']:.4f}")
        print(f"  有效样本数: {results['polarity_total_samples']}")
        print(f"  正确预测: {results['polarity_total_correct']}")
        print()
        print("配对关系结果:")
        print(f"  总体准确率: {results['pair_accuracy']:.4f}")
        print(f"  总样本数: {results['pair_total_samples']}")
        print(f"  正确预测: {results['pair_total_correct']}")
        print()
        print(f"详细结果已保存到: {results_file}")
        print("="*70)
        
    except Exception as e:
        logging.error(f"测试过程中出错: {e}")
        raise

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    main()
