#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GDPR合规性检测模型评估工具（优化版）
支持评估所有预测模型（LLM和符号/智能方法）
"""
import json
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def evaluate_task1(pred_file, gold_file, out_txt):
    """Evaluate Task 1: Output Accuracy@1, @2, @3, @4, @5 for each granularity."""
    gold_data = load_json(gold_file)
    pred_data = load_json(pred_file)
    levels = ['file_level_violations', 'module_level_violations', 'line_level_violations']
    level_keys = {
        'file_level_violations': lambda x: (x['repo_url'], x['Commit_ID'], x['file_path']),
        'module_level_violations': lambda x: (x['repo_url'], x['Commit_ID'], x['file_path'], x['module_name']),
        'line_level_violations': lambda x: (x['repo_url'], x['Commit_ID'], x['file_path'], x['line_spans'])
    }
    
    results = {}
    
    with open(out_txt, 'w', encoding='utf-8') as fout:
        for level in levels:
            gold_map = {}
            for item in gold_data:
                for v in item.get(level, []):
                    key = level_keys[level]({**item, **v})
                    gold_map[key] = list(v.get('violated_articles', []))
            
            pred_map = {}
            for item in pred_data:
                for v in item.get(level, []):
                    key = level_keys[level]({**item, **v})
                    pred_map[key] = list(v.get('violated_articles', []))
            
            acc_at = {k: [] for k in [1, 2, 3, 4, 5]}
            for key, gold_articles in gold_map.items():
                pred_articles = pred_map.get(key, [])
                gold_set = set(gold_articles) - set([0])
                pred_list = [a for a in pred_articles if a != 0]
                if not gold_set:
                    continue
                for k in [1, 2, 3, 4, 5]:
                    hits = len(gold_set.intersection(pred_list[:k])) if pred_list else 0
                    acc_at[k].append(hits / len(gold_set))
            
            fout.write(f'==== {level} ====' + '\n')
            results[level] = {}
            for k in [1, 2, 3, 4, 5]:
                mean_acc = np.mean(acc_at[k]) if acc_at[k] else 0
                results[level][f'Acc@{k}'] = mean_acc
                fout.write(f'Accuracy@{k}: {mean_acc:.4f}\n')
            fout.write('\n')
        
        fout.write('Evaluation finished.\n')
    print(f"Task1 evaluation finished: {out_txt}")
    return results

def evaluate_task2(pred_file, gold_file, out_txt, out_png):
    """Evaluate Task 2: Multi-label classification metrics and normalized confusion matrix."""
    gold_data = load_json(gold_file)
    pred_data = load_json(pred_file)
    
    gold_map = {}
    for item in gold_data:
        key = (item['repo_url'], item['Commit_ID'], item['code_snippet_path'])
        gold_map[key] = set(item.get('violated_articles', []))
    
    pred_map = {}
    for item in pred_data:
        key = (item['repo_url'], item['Commit_ID'], item['code_snippet_path'])
        pred_map[key] = set(item.get('violated_articles', []))
    
    all_keys = set(gold_map.keys()) | set(pred_map.keys())
    all_arts = set()
    y_true = []
    y_pred = []
    
    for key in all_keys:
        gold = gold_map.get(key, set([0]))
        pred = pred_map.get(key, set([0]))
        gold = set([0]) if gold == set() else gold
        pred = set([0]) if pred == set() else pred
        gold_has = sorted(gold - set([0]))
        pred_has = sorted(pred - set([0]))
        n = max(len(gold_has), len(pred_has))
        gold_pad = gold_has + [0] * (n - len(gold_has)) if n > 0 else [0]
        pred_pad = pred_has + [0] * (n - len(pred_has)) if n > 0 else [0]
        y_true.extend(gold_pad)
        y_pred.extend(pred_pad)
        all_arts |= set(gold_pad) | set(pred_pad)
    
    labels = sorted(list(all_arts))
    
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None, labels=labels, zero_division=0)
    
    results = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    with open(out_txt, 'w', encoding='utf-8') as fout:
        fout.write(f'Accuracy: {acc:.4f}\n')
        fout.write(f'Macro-Precision: {precision:.4f}\n')
        fout.write(f'Macro-Recall: {recall:.4f}\n')
        fout.write(f'Macro-F1: {f1:.4f}\n')
        fout.write('\nPer-class metrics:\n')
        for i, c in enumerate(labels):
            fout.write(f'  Article {c}: Precision={precision_class[i]:.4f}, Recall={recall_class[i]:.4f}, F1={f1_class[i]:.4f}\n')
        fout.write('\nEvaluation finished.\n')
    
    if labels:
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Task2 Confusion Matrix (Normalized)')
        plt.tight_layout()
        plt.savefig(out_png)
        plt.close()
        print(f"Task2 confusion matrix saved: {out_png}")
    
    print(f"Task2 evaluation finished: {out_txt}")
    return results

def generate_comparison_table(all_results_task1, all_results_task2, output_dir):
    """生成所有模型的对比表格 - 统一TXT文件"""
    
    # 创建统一的对比表格TXT文件
    unified_txt = os.path.join(output_dir, 'all_models_comparison.txt')
    
    with open(unified_txt, 'w', encoding='utf-8') as f:
        # 文件头
        f.write("╔" + "═" * 118 + "╗\n")
        f.write("║" + " " * 28 + "GDPR Compliance Detection Model Comparison Table" + " " * 28 + "║\n")
        f.write("║" + " " * 38 + "Evaluation Results Summary" + " " * 38 + "║\n")
        f.write("╚" + "═" * 118 + "╝\n\n")
        
        # Task 1 - 3个level的表格
        if all_results_task1:
            level_names = {
                'file_level_violations': 'File-Level Violations',
                'module_level_violations': 'Module-Level Violations',
                'line_level_violations': 'Line-Level Violations'
            }
            
            for level_key, level_name in level_names.items():
                f.write("\n")
                f.write("┌" + "─" * 118 + "┐\n")
                f.write("│ Task 1: " + level_name + " " * (108 - len(level_name)) + "│\n")
                f.write("├" + "─" * 48 + "┬" + "─" * 13 + "┬" + "─" * 13 + "┬" + "─" * 13 + "┬" + "─" * 13 + "┬" + "─" * 13 + "┤\n")
                f.write("│ " + "Model".ljust(46) + " │ " + "Acc@1".center(11) + " │ " + "Acc@2".center(11) + " │ " + "Acc@3".center(11) + " │ " + "Acc@4".center(11) + " │ " + "Acc@5".center(11) + " │\n")
                f.write("├" + "─" * 48 + "┼" + "─" * 13 + "┼" + "─" * 13 + "┼" + "─" * 13 + "┼" + "─" * 13 + "┼" + "─" * 13 + "┤\n")
                
                # 数据行
                for model in sorted(all_results_task1.keys()):
                    results = all_results_task1[model]
                    if level_key in results:
                        acc1 = f"{results[level_key].get('Acc@1', 0):.4f}".center(11)
                        acc2 = f"{results[level_key].get('Acc@2', 0):.4f}".center(11)
                        acc3 = f"{results[level_key].get('Acc@3', 0):.4f}".center(11)
                        acc4 = f"{results[level_key].get('Acc@4', 0):.4f}".center(11)
                        acc5 = f"{results[level_key].get('Acc@5', 0):.4f}".center(11)
                    else:
                        acc1 = acc2 = acc3 = acc4 = acc5 = "0.0000".center(11)
                    
                    f.write("│ " + model.ljust(46) + " │ " + acc1 + " │ " + acc2 + " │ " + acc3 + " │ " + acc4 + " │ " + acc5 + " │\n")
                
                f.write("└" + "─" * 48 + "┴" + "─" * 13 + "┴" + "─" * 13 + "┴" + "─" * 13 + "┴" + "─" * 13 + "┴" + "─" * 13 + "┘\n")
        
        # Task 2表格
        if all_results_task2:
            f.write("\n\n")
            f.write("┌" + "─" * 118 + "┐\n")
            f.write("│ Task 2: Code Snippet Classification Metrics" + " " * 73 + "│\n")
            f.write("├" + "─" * 48 + "┬" + "─" * 16 + "┬" + "─" * 16 + "┬" + "─" * 16 + "┬" + "─" * 16 + "┤\n")
            f.write("│ " + "Model".ljust(46) + " │ " + "Accuracy".center(14) + " │ " + "Precision".center(14) + " │ " + "Recall".center(14) + " │ " + "F1 Score".center(14) + " │\n")
            f.write("├" + "─" * 48 + "┼" + "─" * 16 + "┼" + "─" * 16 + "┼" + "─" * 16 + "┼" + "─" * 16 + "┤\n")
            
            # 数据行
            for model in sorted(all_results_task2.keys()):
                results = all_results_task2[model]
                acc = f"{results.get('accuracy', 0):.4f}".center(14)
                prec = f"{results.get('precision', 0):.4f}".center(14)
                rec = f"{results.get('recall', 0):.4f}".center(14)
                f1 = f"{results.get('f1', 0):.4f}".center(14)
                
                f.write("│ " + model.ljust(46) + " │ " + acc + " │ " + prec + " │ " + rec + " │ " + f1 + " │\n")
            
            f.write("└" + "─" * 48 + "┴" + "─" * 16 + "┴" + "─" * 16 + "┴" + "─" * 16 + "┴" + "─" * 16 + "┘\n")
        
        f.write("\n")
        f.write("═" * 120 + "\n")
        f.write("Notes:\n")
        f.write("  - Task 1 includes three granularity levels: File-Level, Module-Level, and Line-Level\n")
        f.write("  - Acc@K represents the accuracy of hitting true violations in the top-K predicted GDPR articles\n")
        f.write("  - Task 2 uses multi-label classification metrics to evaluate GDPR article identification\n")
        f.write("  - Model categories: LLM models (gpt-4o, claude, gemini, etc.) and Symbolic/Agentic methods (ast-based, react, rag)\n")
        f.write("═" * 120 + "\n")
    
    print(f"✓ Unified comparison table saved: {unified_txt}")
    
    # Also save CSV format for further analysis
    if all_results_task1:
        for level_key in ['file_level_violations', 'module_level_violations', 'line_level_violations']:
            data = []
            for model, results in sorted(all_results_task1.items()):
                row = {'Model': model}
                if level_key in results:
                    for k in range(1, 6):
                        row[f'Acc@{k}'] = results[level_key].get(f'Acc@{k}', 0)
                else:
                    for k in range(1, 6):
                        row[f'Acc@{k}'] = 0
                data.append(row)
            
            if data:
                df = pd.DataFrame(data)
                csv_path = os.path.join(output_dir, f'task1_{level_key}.csv')
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    if all_results_task2:
        task2_data = []
        for model, results in sorted(all_results_task2.items()):
            task2_data.append({
                'Model': model,
                'Accuracy': results.get('accuracy', 0),
                'Precision': results.get('precision', 0),
                'Recall': results.get('recall', 0),
                'F1': results.get('f1', 0)
            })
        
        df_task2 = pd.DataFrame(task2_data)
        csv_path = os.path.join(output_dir, 'task2_metrics.csv')
        df_task2.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✓ CSV files saved to: {output_dir}/")

def generate_summary_report(all_results_task1, all_results_task2, output_dir):
    """Generate summary report in English"""
    report_path = os.path.join(output_dir, 'evaluation_summary.txt')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("GDPR Compliance Detection Model Evaluation Summary\n")
        f.write("=" * 80 + "\n\n")
        
        # Model classification
        llm_models = []
        symbolic_models = []
        for model in set(list(all_results_task1.keys()) + list(all_results_task2.keys())):
            if model in ['ast-based', 'react', 'rag']:
                symbolic_models.append(model)
            else:
                llm_models.append(model)
        
        f.write(f"Total Models Evaluated: {len(llm_models) + len(symbolic_models)}\n")
        f.write(f"  - LLM Models: {len(llm_models)} - {', '.join(sorted(llm_models))}\n")
        f.write(f"  - Symbolic/Agentic Methods: {len(symbolic_models)} - {', '.join(sorted(symbolic_models))}\n\n")
        
        # Task 1 best model
        if all_results_task1:
            f.write("=" * 80 + "\n")
            f.write("Task 1: File-level GDPR Violation Detection\n")
            f.write("=" * 80 + "\n\n")
            
            # Find best model by Line-level Acc@1
            best_model = None
            best_score = 0
            for model, results in all_results_task1.items():
                if 'line_level_violations' in results:
                    score = results['line_level_violations'].get('Acc@1', 0)
                    if score > best_score:
                        best_score = score
                        best_model = model
            
            if best_model:
                f.write(f"Best Model: {best_model}\n")
                f.write(f"  Line-level Acc@1: {best_score:.4f}\n\n")
                
                # Show top 5
                scores = [(model, results.get('line_level_violations', {}).get('Acc@1', 0)) 
                         for model, results in all_results_task1.items()]
                scores.sort(key=lambda x: x[1], reverse=True)
                
                f.write("Top 5 Models:\n")
                for rank, (model, score) in enumerate(scores[:5], 1):
                    f.write(f"  {rank}. {model}: {score:.4f}\n")
                f.write("\n")
        
        # Task 2 best model
        if all_results_task2:
            f.write("=" * 80 + "\n")
            f.write("Task 2: Code Snippet GDPR Article Classification\n")
            f.write("=" * 80 + "\n\n")
            
            # Find best model by F1
            best_model = None
            best_f1 = 0
            for model, results in all_results_task2.items():
                f1 = results.get('f1', 0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_model = model
            
            if best_model:
                res = all_results_task2[best_model]
                f.write(f"Best Model: {best_model}\n")
                f.write(f"  Accuracy: {res.get('accuracy', 0):.4f}\n")
                f.write(f"  Precision: {res.get('precision', 0):.4f}\n")
                f.write(f"  Recall: {res.get('recall', 0):.4f}\n")
                f.write(f"  F1: {res.get('f1', 0):.4f}\n\n")
                
                # Show top 5
                scores = [(model, results.get('f1', 0)) for model, results in all_results_task2.items()]
                scores.sort(key=lambda x: x[1], reverse=True)
                
                f.write("Top 5 Models by F1:\n")
                for rank, (model, f1) in enumerate(scores[:5], 1):
                    f.write(f"  {rank}. {model}: F1={f1:.4f}\n")
                f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("For detailed results, please check individual model evaluation files\n")
        f.write("and the unified comparison table (all_models_comparison.txt)\n")
        f.write("=" * 80 + "\n")
    
    print(f"Summary report saved: {report_path}")

def main():
    print("=" * 80)
    print("GDPR Compliance Detection Model Evaluation Tool")
    print("=" * 80)
    parser = argparse.ArgumentParser(description="Evaluate GDPR model predictions for Task 1 and Task 2.")
    parser.add_argument('--models', type=str, default='all', help='Comma-separated model names to evaluate, or "all" for all models.')
    parser.add_argument('--task', type=str, default='all', choices=['all', '1', '2'], help='Task to evaluate: 1, 2, or all')
    args = parser.parse_args()
    
    task1_pred_dir = "task1_predictions"
    task2_pred_dir = "task2_predictions"
    task1_gold = "task1_dataset.json"
    task2_gold = "task2_dataset.json"
    task1_eval_dir = "task1_eval_results"
    task2_eval_dir = "task2_eval_results"
    comparison_dir = "evaluation_comparison"
    
    ensure_dir(task1_eval_dir)
    ensure_dir(task2_eval_dir)
    ensure_dir(comparison_dir)

    # Define aggregate result files for each task
    all_task1_txt = os.path.join(task1_eval_dir, "all_models_task1_eval.txt")
    all_task2_txt = os.path.join(task2_eval_dir, "all_models_task2_eval.txt")
    # Clear the aggregate files at the start
    open(all_task1_txt, "w", encoding="utf-8").close()
    open(all_task2_txt, "w", encoding="utf-8").close()
    
    # Get available models from prediction files
    model_set = set()
    if args.task in ['all', '1']:
        for fname in os.listdir(task1_pred_dir):
            if fname.endswith('.json'):
                model_set.add(fname.split('_task1_predictions.json')[0])
    if args.task in ['all', '2']:
        for fname in os.listdir(task2_pred_dir):
            if fname.endswith('.json'):
                model_set.add(fname.split('_task2_predictions.json')[0])
    
    if args.models == 'all':
        selected_models = sorted(model_set)
    else:
        selected_models = [m.strip() for m in args.models.split(',') if m.strip() in model_set]
        if not selected_models:
            print(f"No valid models found. Available models: {sorted(model_set)}")
            return
    
    print(f"Selected models: {', '.join(selected_models)}")
    print(f"Total: {len(selected_models)} models\n")
    
    # 存储所有模型的结果
    all_results_task1 = {}
    all_results_task2 = {}
    
    # Evaluate Task 1 for all selected models and aggregate results
    if args.task in ['all', '1']:
        print("\n" + "=" * 80)
        print("Starting Task 1 Evaluation")
        print("=" * 80)
        for idx, model in enumerate(selected_models, 1):
            pred_file = os.path.join(task1_pred_dir, f"{model}_task1_predictions.json")
            if not os.path.exists(pred_file):
                print(f"[{idx}/{len(selected_models)}] [Task1] Prediction file not found: {model}")
                continue
            print(f"[{idx}/{len(selected_models)}] Evaluating: {model}...")
            out_txt = os.path.join(task1_eval_dir, f"{model}_task1_eval.txt")
            results = evaluate_task1(pred_file, task1_gold, out_txt)
            all_results_task1[model] = results
            # Append this model's results to the aggregate file
            with open(out_txt, "r", encoding="utf-8") as fin, open(all_task1_txt, "a", encoding="utf-8") as fout:
                fout.write(f"==== Model: {model} ====" + "\n")
                fout.write(fin.read())
                fout.write("\n\n")
    
    # Evaluate Task 2 for all selected models and aggregate results
    if args.task in ['all', '2']:
        print("\n" + "=" * 80)
        print("Starting Task 2 Evaluation")
        print("=" * 80)
        for idx, model in enumerate(selected_models, 1):
            pred_file = os.path.join(task2_pred_dir, f"{model}_task2_predictions.json")
            if not os.path.exists(pred_file):
                print(f"[{idx}/{len(selected_models)}] [Task2] Prediction file not found: {model}")
                continue
            print(f"[{idx}/{len(selected_models)}] Evaluating: {model}...")
            out_txt = os.path.join(task2_eval_dir, f"{model}_task2_eval.txt")
            out_png = os.path.join(task2_eval_dir, f"{model}_task2_confusion.png")
            results = evaluate_task2(pred_file, task2_gold, out_txt, out_png)
            all_results_task2[model] = results
            # Append this model's results to the aggregate file
            with open(out_txt, "r", encoding="utf-8") as fin, open(all_task2_txt, "a", encoding="utf-8") as fout:
                fout.write(f"==== Model: {model} ====" + "\n")
                fout.write(fin.read())
                fout.write("\n\n")
    
    # Generate comparison tables and visualizations
    print("\n" + "=" * 80)
    print("Generating Comparison Analysis")
    print("=" * 80)
    if all_results_task1 or all_results_task2:
        generate_comparison_table(all_results_task1, all_results_task2, comparison_dir)
        generate_summary_report(all_results_task1, all_results_task2, comparison_dir)
    
    print("\n" + "=" * 80)
    print("All Evaluations Finished!")
    print("=" * 80)
    print(f"\nOutput Directories:")
    print(f"  - Task 1 detailed results: {task1_eval_dir}/")
    print(f"  - Task 2 detailed results: {task2_eval_dir}/")
    print(f"\n  - Comparison Analysis: {comparison_dir}/")
    print(f"    ├─ all_models_comparison.txt  ⭐ [Unified comparison table - Task1 (3 levels) + Task2]")
    print(f"    ├─ evaluation_summary.txt     (Overall summary report)")
    print(f"    └─ CSV files (for data analysis)")
    print(f"        ├─ task1_file_level_violations.csv")
    print(f"        ├─ task1_module_level_violations.csv")
    print(f"        ├─ task1_line_level_violations.csv")
    print(f"        └─ task2_metrics.csv")
    print()

if __name__ == "__main__":
    main() 