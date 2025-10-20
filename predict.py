import json
import time
import requests
import subprocess
import os
import sys
import logging
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import signal

# Global monitoring data structure
progress_lock = threading.Lock()
model_progress = {}
monitor_stop = False
monitor_save_interval = 60  # seconds
monitor_refresh_interval = 2  # seconds

# Global logger dictionary
model_loggers = {}

SYSTEM_PROMPT = "You are a helpful GDPR compliance assistant. Always follow the instructions strictly."

def setup_logging(model_name, task_id=None):
    """Setup logging for a specific model and task"""
    logger_key = f"{model_name}_task{task_id}" if task_id else model_name
    if logger_key in model_loggers:
        return model_loggers[logger_key]
    
    # Create logs directory if it doesn't exist
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
    
    # Create logger with unique name
    logger_name = f"model_{logger_key}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create file handler with task-specific log file
    log_file = f"logs/{logger_key}.log" if task_id else f"logs/{model_name}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    # Store logger
    model_loggers[logger_key] = logger
    
    return logger

def get_logger(model_name, task_id=None):
    """Get logger for a specific model and task"""
    return setup_logging(model_name, task_id)

# Monitoring thread
def monitor_models(selected_models, selected_tasks, total_counts, out_dirs, out_files, disable_clear=False):
    global monitor_stop
    last_save_time = time.time()
    
    # Setup monitoring logger
    monitor_logger = logging.getLogger("monitor")
    monitor_logger.setLevel(logging.INFO)
    if not os.path.exists("logs"):
        os.makedirs("logs", exist_ok=True)
    monitor_handler = logging.FileHandler("logs/monitor.log", encoding='utf-8')
    monitor_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    monitor_logger.addHandler(monitor_handler)
    
    monitor_logger.info(f"Starting monitoring for models: {selected_models}, tasks: {selected_tasks}")
    
    # Write progress to file for external monitoring
    progress_file = f"progress_{os.getpid()}.json"
    
    while True:
        # Export progress to file
        with progress_lock:
            progress_data = {
                'models': selected_models,
                'tasks': selected_tasks,
                'progress': {}
            }
            for task in selected_tasks:
                for model in selected_models:
                    key = (model, task)
                    prog = model_progress.get(key, {})
                    # Use total from progress if available, otherwise from total_counts
                    total = prog.get('total', total_counts.get((model, task), 0))
                    progress_data['progress'][f"{model}_task{task}"] = {
                        'done': prog.get('done', 0),
                        'total': total,
                        'existing_count': prog.get('existing_count', 0),
                        'new_total': prog.get('new_total', 0),
                        'status': prog.get('status', 'idle'),
                        'current': prog.get('current', '')
                    }
            
            # Write to file
            try:
                with open(progress_file, 'w', encoding='utf-8') as pf:
                    json.dump(progress_data, pf, ensure_ascii=False, indent=2)
            except:
                pass
        
        with progress_lock:
            if not disable_clear:
                os.system('cls' if os.name == 'nt' else 'clear')
            print("=" * 100)
            print("Real-time Model Prediction Progress Monitor")
            print("=" * 100)
            for task in selected_tasks:
                print(f"Task {task}:")
                for model in selected_models:
                    key = (model, task)
                    prog = model_progress.get(key, {})
                    done = prog.get('done', 0)
                    existing = prog.get('existing_count', 0)
                    new_total = prog.get('new_total', 0)
                    total = prog.get('total', total_counts.get((model, task), '?'))
                    status = prog.get('status', 'idle')
                    current = prog.get('current', '')
                    err = prog.get('error', '')
                    
                    # Display format: (existing + done) / total (X existing, Y new/Z total)
                    if existing > 0:
                        overall_done = existing + done
                        progress_str = f"{overall_done}/{total} ({existing} existing, {done}/{new_total} new)"
                    else:
                        progress_str = f"{done}/{total}"
                    
                    print(f"  [{model:<25}] Status: {status:<10} Progress: {progress_str} Current: {current[:40]}{' ...' if len(current)>40 else ''} {'[Error]' if err else ''}")
            print("-" * 100)
            print("Output files:")
            for task in selected_tasks:
                for model in selected_models:
                    key = (model, task)
                    out_file = out_files.get(key, None)
                    if out_file and os.path.exists(out_file):
                        try:
                            with open(out_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            print(f"  {out_file}: {len(data)} records")
                        except Exception:
                            print(f"  {out_file}: [File exists, cannot parse]")
            print("-" * 100)
            print("Press Ctrl+C to gracefully exit and auto-save progress.")
            print("=" * 100)
        # Periodic auto-save
        now = time.time()
        if now - last_save_time >= monitor_save_interval:
            monitor_logger.info("Starting periodic auto-save")
            for key, out_file in out_files.items():
                results = model_progress.get(key, {}).get('results', [])
                if results:
                    try:
                        with open(out_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        print(f"[Auto-save] {out_file} saved {len(results)} records.", flush=True)
                        monitor_logger.info(f"Auto-saved {out_file} with {len(results)} records")
                    except Exception as e:
                        print(f"[Auto-save] Failed to save {out_file}: {e}", flush=True)
                        monitor_logger.error(f"Auto-save failed for {out_file}: {e}")
            last_save_time = now
        with progress_lock:
            if monitor_stop:
                monitor_logger.info("Monitoring stopped")
                break
        time.sleep(monitor_refresh_interval)

# Alternative methods handler
def run_alternative_method(method_name, task, out_dir, total_count, out_file, exclude_apps, incremental_mode, logger):
    """
    Run alternative methods (ReAct, RAG, Rule-based, etc.) instead of LLM API calls.
    """
    from methods.method_factory import MethodFactory
    from methods.config import get_method_config
    
    print(f"\n{'='*80}")
    print(f"🚀 Running {method_name} on Task {task}")
    print(f"{'='*80}\n")
    
    logger.info(f"Initializing alternative method: {method_name}")
    
    # Get method configuration
    config = get_method_config(method_name)
    
    # Create method instance
    try:
        print(f"📦 Creating {method_name} method...")
        method = MethodFactory.create(method_name, config)
        logger.info(f"Successfully created method: {method.get_name()}")
        print(f"✅ Method created: {method.get_name()}\n")
    except Exception as e:
        logger.error(f"Failed to create method {method_name}: {e}", exc_info=True)
        print(f"❌ Failed to create method: {e}")
        return
    
    # Load dataset
    if task == "1":
        dataset_file = "task1_dataset.json"
    else:
        dataset_file = "task2_dataset.json"
    
    print(f"📂 Loading dataset: {dataset_file}...")
    with open(dataset_file, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    logger.info(f"Loaded {len(dataset)} items from {dataset_file}")
    print(f"✅ Loaded {len(dataset)} items\n")
    
    # Filter by exclude_apps
    if exclude_apps:
        original_count = len(dataset)
        print(f"🔍 Filtering dataset...")
        print(f"   Excluding apps: {', '.join(exclude_apps[:3])}{'...' if len(exclude_apps)>3 else ''}")
        dataset = [item for item in dataset if item.get('app_name', '') not in exclude_apps]
        logger.info(f"Filtered: {original_count} -> {len(dataset)} (excluded {exclude_apps})")
        print(f"✅ Filtered: {original_count} → {len(dataset)} items")
        remaining_apps = set([item.get('app_name') for item in dataset])
        print(f"   Remaining apps: {remaining_apps}\n")
    
    # Load existing predictions if incremental mode
    existing_predictions = []
    predicted_keys = set()
    if incremental_mode and os.path.exists(out_file):
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                existing_predictions = json.load(f)
            
            # Build predicted keys based on task
            if task == "1":
                for pred in existing_predictions:
                    key = (pred.get('repo_url'), pred.get('app_name'), pred.get('Commit_ID'))
                    predicted_keys.add(key)
            else:  # task == "2"
                for pred in existing_predictions:
                    key = (pred.get('repo_url'), pred.get('app_name'), pred.get('Commit_ID'), pred.get('code_snippet_path'))
                    predicted_keys.add(key)
            
            logger.info(f"Incremental mode: Loaded {len(existing_predictions)} existing predictions")
        except Exception as e:
            logger.warning(f"Failed to load existing predictions: {e}")
    
    # Filter out already predicted items
    if incremental_mode and predicted_keys:
        original_count = len(dataset)
        if task == "1":
            dataset = [item for item in dataset 
                      if (item.get('repo_url'), item.get('app_name'), item.get('Commit_ID')) not in predicted_keys]
        else:
            dataset = [item for item in dataset 
                      if (item.get('repo_url'), item.get('app_name'), item.get('Commit_ID'), item.get('code_snippet_path')) not in predicted_keys]
        logger.info(f"Incremental filter: {original_count} -> {len(dataset)} items")
    
    # Initialize progress tracking
    key = (method_name, task)
    with progress_lock:
        model_progress[key] = {
            'done': 0, 
            'total': len(dataset) + len(existing_predictions),
            'existing_count': len(existing_predictions),
            'new_total': len(dataset),
            'status': 'init', 
            'current': '', 
            'results': existing_predictions.copy(),
            'error': ''
        }
    
    # Process dataset
    results = existing_predictions.copy() if incremental_mode else []
    appname_to_folder = build_appname_to_folder()
    
    print(f"🔬 Starting analysis...")
    print(f"📊 Total items to process: {len(dataset)}\n")
    
    for idx, item in enumerate(dataset):
        logger.info(f"Processing item {idx+1}/{len(dataset)}")
        
        # Update progress
        with progress_lock:
            model_progress[key]['status'] = 'processing'
            model_progress[key]['current'] = f"{item.get('app_name', 'Unknown')} - {item.get('code_snippet_path', 'Unknown')[:40]}"
        
        print(f"[{idx+1}/{len(dataset)}] Processing: {item.get('app_name', 'Unknown')} - {item.get('code_snippet_path', 'Unknown')[:50]}...", end='', flush=True)
        
        if task == "1":
            # Task 1: Multi-granularity detection
            pred = {
                "repo_url": item["repo_url"],
                "app_name": item["app_name"],
                "Commit_ID": item["Commit_ID"],
                "file_level_violations": [],
                "module_level_violations": [],
                "line_level_violations": []
            }
            
            # Process file-level
            for file_item in item.get("file_level_violations", []):
                folder = appname_to_folder.get(item['app_name'], item['app_name'])
                src_path = f"repos/{folder}/{file_item['file_path']}"
                try:
                    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                except:
                    code = ""
                
                articles = method.predict_file_level(file_item['file_path'], code)
                pred["file_level_violations"].append({
                    "file_path": file_item["file_path"],
                    "violated_articles": articles
                })
            
            # Process module-level
            for mod_item in item.get("module_level_violations", []):
                folder = appname_to_folder.get(item['app_name'], item['app_name'])
                src_path = f"repos/{folder}/{mod_item['file_path']}"
                try:
                    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
                        code = f.read()
                except:
                    code = ""
                
                articles = method.predict_module_level(
                    mod_item['file_path'], 
                    mod_item['module_name'], 
                    code
                )
                pred["module_level_violations"].append({
                    "file_path": mod_item["file_path"],
                    "module_name": mod_item["module_name"],
                    "violated_articles": articles
                })
            
            # Process line-level
            for line_item in item.get("line_level_violations", []):
                folder = appname_to_folder.get(item['app_name'], item['app_name'])
                src_path = f"repos/{folder}/{line_item['file_path']}"
                try:
                    with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
                        code_lines = f.readlines()
                    span = line_item["line_spans"]
                    if "-" in span:
                        start, end = [int(x) for x in span.split("-")]
                    else:
                        start = end = int(span)
                    code = "".join(code_lines[max(0, start-1):end])
                except:
                    code = ""
                
                articles = method.predict_line_level(
                    line_item['file_path'],
                    line_item['line_spans'],
                    code,
                    line_item.get('violation_description', '')
                )
                pred["line_level_violations"].append({
                    "file_path": line_item["file_path"],
                    "line_spans": line_item["line_spans"],
                    "violated_articles": articles
                })
            
            results.append(pred)
            
        else:  # task == "2"
            # Task 2: Snippet classification
            code = item.get("code_snippet", "")
            if isinstance(code, list):
                code = "\n".join([line.strip() for line in code if line.strip()])
            
            articles = method.predict_snippet(code, item.get("code_snippet_path", ""))
            
            results.append({
                "repo_url": item["repo_url"],
                "app_name": item["app_name"],
                "Commit_ID": item["Commit_ID"],
                "code_snippet_path": item["code_snippet_path"],
                "violated_articles": articles
            })
            
            print(f" → {articles}")
        
        # Update progress
        with progress_lock:
            model_progress[key]['done'] = idx + 1
            model_progress[key]['results'] = results.copy()
        
        # Periodic save
        if (idx + 1) % 10 == 0:
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved progress: {len(results)} results")
            print(f"\n💾 Auto-saved: {len(results)} results\n")
    
    # Final save
    print(f"\n💾 Saving final results...")
    with open(out_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info(f"Completed: Saved {len(results)} results to {out_file}")
    
    # Update final status
    with progress_lock:
        model_progress[key]['status'] = 'done'
        model_progress[key]['results'] = results.copy()
    
    print(f"\n{'='*80}")
    print(f"✅ {method_name} Task {task} COMPLETED!")
    print(f"{'='*80}")
    print(f"📁 Output file: {out_file}")
    print(f"📊 Total predictions: {len(results)}")
    print(f"{'='*80}\n")
    
    # Cleanup
    method.cleanup()


# Prediction thread wrapper
def run_model_task_realtime(api_url, api_key, model, task, out_dir, total_count, out_file, exclude_apps=[], incremental_mode=False):
    logger = get_logger(model, task)
    logger.info(f"Starting prediction task for model: {model}, task: {task}")
    logger.info(f"Output directory: {out_dir}, Output file: {out_file}")
    logger.info(f"Total items to process: {total_count}")
    logger.info(f"Exclude apps: {exclude_apps}, Incremental mode: {incremental_mode}")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Created output directory: {out_dir}")
    
    # Check if this is an alternative method (not LLM-based)
    alternative_methods_list = ['react', 'rag', 'ast-based']
    if model.lower() in alternative_methods_list:
        logger.info(f"Using alternative method: {model}")
        return run_alternative_method(model, task, out_dir, total_count, out_file, exclude_apps, incremental_mode, logger)
    
    key = (model, task)
    
    # Initialize with existing predictions if incremental mode
    initial_results = []
    if incremental_mode and os.path.exists(out_file):
        try:
            with open(out_file, 'r', encoding='utf-8') as f:
                initial_results = json.load(f)
            logger.info(f"Pre-loaded {len(initial_results)} existing predictions for progress tracking")
        except:
            pass
    
    with progress_lock:
        model_progress[key] = {'done': 0, 'total': total_count, 'status': 'init', 'current': '', 'results': initial_results, 'error': ''}
    
    try:
        if task == "1":
            logger.info("Processing Task 1 (multi-granularity detection)")
            all_articles = get_all_articles_from_task1()
            article_list = ", ".join(all_articles)
            logger.info(f"Found {len(all_articles)} unique articles in task 1 dataset")
            
            with open("task1_dataset.json", "r", encoding="utf-8") as f:
                dataset = json.load(f)
            logger.info(f"Loaded task1 dataset with {len(dataset)} items")
            
            # Filter by exclude_apps
            if exclude_apps:
                original_count = len(dataset)
                dataset = [item for item in dataset if item.get('app_name', '') not in exclude_apps]
                logger.info(f"Filtered dataset: {original_count} -> {len(dataset)} items (excluded {exclude_apps})")
                print(f"[{model}] Task1: Filtered to {len(dataset)} items (excluded {exclude_apps})")
            
            # Load existing predictions if incremental mode
            existing_predictions = []
            predicted_keys = set()
            if incremental_mode and os.path.exists(out_file):
                try:
                    with open(out_file, 'r', encoding='utf-8') as f:
                        existing_predictions = json.load(f)
                    # Build set of already predicted items
                    for pred in existing_predictions:
                        pred_key = (pred.get('repo_url'), pred.get('app_name'), pred.get('Commit_ID'))
                        predicted_keys.add(pred_key)
                    logger.info(f"Incremental mode: Loaded {len(existing_predictions)} existing predictions")
                    print(f"[{model}] Task1: Incremental mode - {len(existing_predictions)} existing predictions")
                except Exception as e:
                    logger.warning(f"Failed to load existing predictions: {e}")
            
            # Filter out already predicted items
            if incremental_mode and predicted_keys:
                original_count = len(dataset)
                dataset = [item for item in dataset 
                          if (item.get('repo_url'), item.get('app_name'), item.get('Commit_ID')) not in predicted_keys]
                logger.info(f"Incremental filter: {original_count} -> {len(dataset)} items (skipped {original_count - len(dataset)} already predicted)")
                print(f"[{model}] Task1: Skipped {original_count - len(dataset)} already predicted items, {len(dataset)} remaining")
            
            # Update total count after filtering
            actual_total = len(dataset) + len(existing_predictions)
            with progress_lock:
                model_progress[key]['total'] = actual_total
                model_progress[key]['existing_count'] = len(existing_predictions)
                model_progress[key]['new_total'] = len(dataset)
            
            appname_to_folder = build_appname_to_folder()
            logger.info(f"Built appname to folder mapping with {len(appname_to_folder)} entries")
            
            results = existing_predictions.copy() if incremental_mode else []
            for idx, item in enumerate(dataset):
                logger.info(f"Processing item {idx+1}/{len(dataset)}: {item.get('app_name', 'Unknown')} - {item.get('Commit_ID', 'Unknown')}")
                
                with progress_lock:
                    model_progress[key]['status'] = 'processing'
                    model_progress[key]['current'] = f"repo: {item.get('repo_url','')} commit: {item.get('Commit_ID','')}"
                
                pred = {
                    "repo_url": item["repo_url"],
                    "app_name": item["app_name"],
                    "Commit_ID": item["Commit_ID"],
                    "file_level_violations": [],
                    "module_level_violations": [],
                    "line_level_violations": []
                }
                
                # File-level violations
                file_violations = item.get("file_level_violations", [])
                logger.info(f"Processing {len(file_violations)} file-level violations")
                
                for file_idx, file_item in enumerate(file_violations):
                    logger.debug(f"Processing file-level violation {file_idx+1}/{len(file_violations)}: {file_item['file_path']}")
                    
                    folder = appname_to_folder.get(item['app_name'], item['app_name'])
                    src_path = f"repos/{folder}/{file_item['file_path']}"
                    logger.debug(f"Source file path: {src_path}")
                    
                    try:
                        with open(src_path, "r", encoding="utf-8", errors="ignore") as fcode:
                            code = fcode.read()
                        logger.debug(f"Successfully read file {src_path}, size: {len(code)} characters")
                    except FileNotFoundError:
                        code = ""
                        logger.warning(f"File not found: {src_path}, using empty content")
                    except Exception as e:
                        code = ""
                        logger.error(f"Error reading file {src_path}: {e}")
                    
                    prompt = f"""
You are a GDPR compliance expert. Your task is to determine which GDPR articles are violated by the following file content.

**GDPR Article Meanings:**
- Article 5: Principles of processing (lawfulness, fairness, transparency)
- Article 6: Lawfulness of processing (legal basis for data processing)
- Article 7: Conditions for consent (valid consent requirements)
- Article 8: Conditions applicable to child's consent
- Article 9: Processing of special categories of personal data
- Article 12: Transparent information and communication
- Article 13: Information to be provided when personal data are collected
- Article 14: Information to be provided when personal data have not been obtained from the data subject
- Article 15: Right of access by the data subject
- Article 16: Right to rectification
- Article 17: Right to erasure ('right to be forgotten')
- Article 18: Right to restriction of processing
- Article 19: Notification obligation regarding rectification or erasure
- Article 21: Right to object
- Article 25: Data protection by design and by default
- Article 30: Records of processing activities
- Article 32: Security of processing
- Article 33: Notification of a personal data breach to the supervisory authority
- Article 35: Data protection impact assessment
- Article 44: General principle for transfers
- Article 46: Transfers subject to appropriate safeguards
- Article 58: Powers of supervisory authorities
- Article 83: General conditions for imposing administrative fines

**Instructions:**
- Carefully analyze the file content for GDPR compliance issues
- If multiple articles are violated, please list all of them
- Only output the violated GDPR article numbers, separated by commas (e.g., 5,6,32)
- If there is no violation, output exactly 0
- Do not output any explanation, text, or extra symbols. Only output numbers as specified
- Output cannot be empty

**IMPORTANT - You MUST provide an answer:**
- DO NOT return blank/empty response
- DO NOT refuse to answer
- You MUST output either article numbers or 0

File content:
{code}
(Full file content)
"""
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                    
                    logger.debug(f"Making API call for file: {file_item['file_path']}")
                    result = ask_llm(api_url, api_key, messages, model)
                    logger.debug(f"API response for file {file_item['file_path']}: {result}")
                    
                    try:
                        articles = [int(s) for s in result.replace("，", ",").replace(" ", "").split(",") if s.isdigit()]
                        logger.info(f"File {file_item['file_path']} -> Articles: {articles}")
                    except Exception as e:
                        articles = []
                        logger.error(f"Error parsing articles for file {file_item['file_path']}: {e}, raw result: {result}")
                    
                    pred["file_level_violations"].append({
                        "file_path": file_item["file_path"],
                        "violated_articles": articles
                    })
                
                # Module-level violations
                module_violations = item.get("module_level_violations", [])
                logger.info(f"Processing {len(module_violations)} module-level violations")
                
                for mod_idx, mod_item in enumerate(module_violations):
                    logger.debug(f"Processing module-level violation {mod_idx+1}/{len(module_violations)}: {mod_item['module_name']}")
                    
                    folder = appname_to_folder.get(item['app_name'], item['app_name'])
                    src_path = f"repos/{folder}/{mod_item['file_path']}"
                    logger.debug(f"Module source file path: {src_path}")
                    
                    try:
                        with open(src_path, "r", encoding="utf-8", errors="ignore") as fcode:
                            code = fcode.read()
                        logger.debug(f"Successfully read module file {src_path}, size: {len(code)} characters")
                    except FileNotFoundError:
                        code = ""
                        logger.warning(f"Module file not found: {src_path}, using empty content")
                    except Exception as e:
                        code = ""
                        logger.error(f"Error reading module file {src_path}: {e}")
                    
                    prompt = f"""
You are a GDPR compliance expert. Your task is to determine which GDPR articles are violated by the following module.

**GDPR Article Meanings:**
- Article 5: Principles of processing (lawfulness, fairness, transparency)
- Article 6: Lawfulness of processing (legal basis for data processing)
- Article 7: Conditions for consent (valid consent requirements)
- Article 8: Conditions applicable to child's consent
- Article 9: Processing of special categories of personal data
- Article 12: Transparent information and communication
- Article 13: Information to be provided when personal data are collected
- Article 14: Information to be provided when personal data have not been obtained from the data subject
- Article 15: Right of access by the data subject
- Article 16: Right to rectification
- Article 17: Right to erasure ('right to be forgotten')
- Article 18: Right to restriction of processing
- Article 19: Notification obligation regarding rectification or erasure
- Article 21: Right to object
- Article 25: Data protection by design and by default
- Article 30: Records of processing activities
- Article 32: Security of processing
- Article 33: Notification of a personal data breach to the supervisory authority
- Article 35: Data protection impact assessment
- Article 44: General principle for transfers
- Article 46: Transfers subject to appropriate safeguards
- Article 58: Powers of supervisory authorities
- Article 83: General conditions for imposing administrative fines

**Instructions:**
- Carefully analyze the module for GDPR compliance issues
- If multiple articles are violated, please list all of them
- Only output the violated GDPR article numbers, separated by commas (e.g., 5,6,32)
- If there is no violation, output exactly 0
- Do not output any explanation, text, or extra symbols. Only output numbers as specified
- Output cannot be empty

**IMPORTANT - You MUST provide an answer:**
- DO NOT return blank/empty response
- DO NOT refuse to answer
- You MUST output either article numbers or 0

Module name: {mod_item['module_name']}
File: {mod_item['file_path']}
Module content (if available):
{code}
"""
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                    
                    logger.debug(f"Making API call for module: {mod_item['module_name']}")
                    result = ask_llm(api_url, api_key, messages, model)
                    logger.debug(f"API response for module {mod_item['module_name']}: {result}")
                    
                    try:
                        articles = [int(s) for s in result.replace("，", ",").replace(" ", "").split(",") if s.isdigit()]
                        logger.info(f"Module {mod_item['module_name']} -> Articles: {articles}")
                    except Exception as e:
                        articles = []
                        logger.error(f"Error parsing articles for module {mod_item['module_name']}: {e}, raw result: {result}")
                    
                    pred["module_level_violations"].append({
                        "file_path": mod_item["file_path"],
                        "module_name": mod_item["module_name"],
                        "violated_articles": articles
                    })
                
                # Line-level violations
                line_violations = item.get("line_level_violations", [])
                logger.info(f"Processing {len(line_violations)} line-level violations")
                
                for line_idx, line_item in enumerate(line_violations):
                    logger.debug(f"Processing line-level violation {line_idx+1}/{len(line_violations)}: {line_item['file_path']} lines {line_item['line_spans']}")
                    
                    folder = appname_to_folder.get(item['app_name'], item['app_name'])
                    src_path = f"repos/{folder}/{line_item['file_path']}"
                    logger.debug(f"Line source file path: {src_path}")
                    
                    try:
                        with open(src_path, "r", encoding="utf-8", errors="ignore") as fcode:
                            code_lines = fcode.readlines()
                        span = line_item["line_spans"]
                        if "-" in span:
                            start, end = [int(x) for x in span.split("-")]
                        else:
                            start = end = int(span)
                        if start < 1:
                            start = 1
                        if end > len(code_lines):
                            end = len(code_lines)
                        code = "".join(code_lines[start-1:end])
                        logger.debug(f"Successfully read lines {start}-{end} from {src_path}, code length: {len(code)}")
                    except Exception as e:
                        code = ""
                        logger.error(f"Error reading lines from {src_path}: {e}")
                    
                    prompt = f"""
You are a GDPR compliance expert. Your task is to determine which GDPR articles are violated by the following code lines.

**GDPR Article Meanings:**
- Article 5: Principles of processing (lawfulness, fairness, transparency)
- Article 6: Lawfulness of processing (legal basis for data processing)
- Article 7: Conditions for consent (valid consent requirements)
- Article 8: Conditions applicable to child's consent
- Article 9: Processing of special categories of personal data
- Article 12: Transparent information and communication
- Article 13: Information to be provided when personal data are collected
- Article 14: Information to be provided when personal data have not been obtained from the data subject
- Article 15: Right of access by the data subject
- Article 16: Right to rectification
- Article 17: Right to erasure ('right to be forgotten')
- Article 18: Right to restriction of processing
- Article 19: Notification obligation regarding rectification or erasure
- Article 21: Right to object
- Article 25: Data protection by design and by default
- Article 30: Records of processing activities
- Article 32: Security of processing
- Article 33: Notification of a personal data breach to the supervisory authority
- Article 35: Data protection impact assessment
- Article 44: General principle for transfers
- Article 46: Transfers subject to appropriate safeguards
- Article 58: Powers of supervisory authorities
- Article 83: General conditions for imposing administrative fines

**Instructions:**
- Carefully analyze the code lines for GDPR compliance issues
- If multiple articles are violated, please list all of them
- Only output the violated GDPR article numbers, separated by commas (e.g., 5,6,32)
- If there is no violation, output exactly 0
- Do not output any explanation, text, or extra symbols. Only output numbers as specified
- Output cannot be empty

**IMPORTANT - You MUST provide an answer:**
- DO NOT return blank/empty response
- DO NOT refuse to answer
- You MUST output either article numbers or 0

File: {line_item['file_path']}
Lines: {line_item['line_spans']}
Description: {line_item['violation_description']}
Code content:
{code}
"""
                    messages = [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt}
                    ]
                    
                    logger.debug(f"Making API call for lines: {line_item['file_path']} {line_item['line_spans']}")
                    result = ask_llm(api_url, api_key, messages, model)
                    logger.debug(f"API response for lines {line_item['file_path']} {line_item['line_spans']}: {result}")
                    
                    try:
                        articles = [int(s) for s in result.replace("，", ",").replace(" ", "").split(",") if s.isdigit()]
                        logger.info(f"Lines {line_item['file_path']} {line_item['line_spans']} -> Articles: {articles}")
                    except Exception as e:
                        articles = []
                        logger.error(f"Error parsing articles for lines {line_item['file_path']} {line_item['line_spans']}: {e}, raw result: {result}")
                    
                    pred["line_level_violations"].append({
                        "file_path": line_item["file_path"],
                        "line_spans": line_item["line_spans"],
                        "violated_articles": articles
                    })
                
                results.append(pred)
                logger.info(f"Completed processing item {idx+1}/{len(dataset)}")
                
                with progress_lock:
                    model_progress[key]['done'] = idx + 1
                    model_progress[key]['results'] = results.copy()
            
            with progress_lock:
                model_progress[key]['status'] = 'done'
                model_progress[key]['results'] = results.copy()
            
            logger.info(f"Saving results to {out_file}")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Task 1 completed successfully. Saved {len(results)} results to {out_file}")
            
        elif task == "2":
            logger.info("Processing Task 2 (snippet classification)")
            all_articles = get_all_articles_from_task2()
            article_list = ", ".join(all_articles)
            logger.info(f"Found {len(all_articles)} unique articles in task 2 dataset")
            
            with open("task2_dataset.json", "r", encoding="utf-8") as f:
                dataset = json.load(f)
            logger.info(f"Loaded task2 dataset with {len(dataset)} items")
            
            # Filter by exclude_apps
            if exclude_apps:
                original_count = len(dataset)
                dataset = [item for item in dataset if item.get('app_name', '') not in exclude_apps]
                logger.info(f"Filtered dataset: {original_count} -> {len(dataset)} items (excluded {exclude_apps})")
                print(f"[{model}] Task2: Filtered to {len(dataset)} items (excluded {exclude_apps})")
            
            # Load existing predictions if incremental mode
            existing_predictions = []
            predicted_keys = set()
            if incremental_mode and os.path.exists(out_file):
                try:
                    with open(out_file, 'r', encoding='utf-8') as f:
                        existing_predictions = json.load(f)
                    # Build set of already predicted items
                    for pred in existing_predictions:
                        pred_key = (pred.get('repo_url'), pred.get('app_name'), pred.get('Commit_ID'), pred.get('code_snippet_path'))
                        predicted_keys.add(pred_key)
                    logger.info(f"Incremental mode: Loaded {len(existing_predictions)} existing predictions")
                    print(f"[{model}] Task2: Incremental mode - {len(existing_predictions)} existing predictions")
                except Exception as e:
                    logger.warning(f"Failed to load existing predictions: {e}")
            
            # Filter out already predicted items
            if incremental_mode and predicted_keys:
                original_count = len(dataset)
                dataset = [item for item in dataset 
                          if (item.get('repo_url'), item.get('app_name'), item.get('Commit_ID'), item.get('code_snippet_path')) not in predicted_keys]
                logger.info(f"Incremental filter: {original_count} -> {len(dataset)} items (skipped {original_count - len(dataset)} already predicted)")
                print(f"[{model}] Task2: Skipped {original_count - len(dataset)} already predicted items, {len(dataset)} remaining")
            
            # Update total count after filtering
            actual_total = len(dataset) + len(existing_predictions)
            with progress_lock:
                model_progress[key]['total'] = actual_total
                model_progress[key]['existing_count'] = len(existing_predictions)
                model_progress[key]['new_total'] = len(dataset)
            
            results = existing_predictions.copy() if incremental_mode else []
            for idx, item in enumerate(dataset):
                logger.info(f"Processing item {idx+1}/{len(dataset)}: {item.get('code_snippet_path', 'Unknown')}")
                
                with progress_lock:
                    model_progress[key]['status'] = 'processing'
                    model_progress[key]['current'] = f"{item.get('code_snippet_path','')}"
                
                code = item.get("code_snippet", "")
                if isinstance(code, list):
                    code_lines = [line.strip() for line in code if line.strip()]
                    code = "\n".join(code_lines)
                    logger.debug(f"Converted list code snippet to string, {len(code_lines)} lines")
                else:
                    code = code.strip()
                
                logger.debug(f"Code snippet length: {len(code)} characters")
                
                prompt = f"""
You are a GDPR compliance expert. Your task is to determine which GDPR articles are violated by the following code snippet.

The following GDPR articles are commonly relevant in this context: {article_list}

**GDPR Article Meanings:**
- Article 5: Principles of processing (lawfulness, fairness, transparency)
- Article 6: Lawfulness of processing (legal basis for data processing)
- Article 7: Conditions for consent (valid consent requirements)
- Article 8: Conditions applicable to child's consent
- Article 9: Processing of special categories of personal data
- Article 12: Transparent information and communication
- Article 13: Information to be provided when personal data are collected
- Article 14: Information to be provided when personal data have not been obtained from the data subject
- Article 15: Right of access by the data subject
- Article 16: Right to rectification
- Article 17: Right to erasure ('right to be forgotten')
- Article 18: Right to restriction of processing
- Article 19: Notification obligation regarding rectification or erasure
- Article 21: Right to object
- Article 25: Data protection by design and by default
- Article 30: Records of processing activities
- Article 32: Security of processing
- Article 33: Notification of a personal data breach to the supervisory authority
- Article 35: Data protection impact assessment
- Article 44: General principle for transfers
- Article 46: Transfers subject to appropriate safeguards
- Article 58: Powers of supervisory authorities
- Article 83: General conditions for imposing administrative fines

**Instructions:**
- Carefully analyze the code snippet for GDPR compliance issues
- If multiple articles are violated, please list all of them
- Only output the violated GDPR article numbers from the list above, separated by commas (e.g., 5,6,32)
- If there is no violation, output exactly 0
- Do not output any explanation, text, or extra symbols. Only output numbers as specified
- Output cannot be empty

**IMPORTANT - You MUST provide an answer:**
- DO NOT return blank/empty response
- DO NOT refuse to answer
- You MUST output either article numbers or 0

**Examples:**
- Code that collects personal data without consent: 6,7
- Code that lacks security measures: 25,32
- Code that doesn't provide privacy notice: 12,13
- Code with no data access controls: 15,16,17

Code snippet:
{code}
"""
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ]
                
                logger.debug(f"Making API call for snippet: {item['code_snippet_path']}")
                result = ask_llm(api_url, api_key, messages, model)
                logger.debug(f"API response for snippet {item['code_snippet_path']}: {result}")
                
                try:
                    articles = [int(s) for s in result.replace("，", ",").replace(" ", "").split(",") if s.isdigit()]
                    logger.info(f"Snippet {item['code_snippet_path']} -> Articles: {articles}")
                except Exception as e:
                    articles = []
                    logger.error(f"Error parsing articles for snippet {item['code_snippet_path']}: {e}, raw result: {result}")
                
                results.append({
                    "repo_url": item["repo_url"],
                    "app_name": item["app_name"],
                    "Commit_ID": item["Commit_ID"],
                    "code_snippet_path": item["code_snippet_path"],
                    "violated_articles": articles
                })
                
                with progress_lock:
                    model_progress[key]['done'] = idx + 1
                    model_progress[key]['results'] = results.copy()
            
            with progress_lock:
                model_progress[key]['status'] = 'done'
                model_progress[key]['results'] = results.copy()
            
            logger.info(f"Saving results to {out_file}")
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Task 2 completed successfully. Saved {len(results)} results to {out_file}")
    
    except Exception as e:
        logger.error(f"Error in prediction task: {e}", exc_info=True)
        with progress_lock:
            model_progress[key]['status'] = 'error'
            model_progress[key]['error'] = str(e)

def get_all_articles_from_task1():
    logger = logging.getLogger("helper")
    logger.info("Extracting all articles from task1 dataset")
    with open("task1_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    articles = set()
    for item in data:
        for level in ["file_level_violations", "module_level_violations", "line_level_violations"]:
            for v in item.get(level, []):
                for a in v.get("violated_articles", []):
                    articles.add(str(a))
    result = sorted(articles, key=int)
    logger.info(f"Found {len(result)} unique articles in task1: {result}")
    return result

def get_all_articles_from_task2():
    logger = logging.getLogger("helper")
    logger.info("Extracting all articles from task2 dataset")
    with open("task2_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    articles = set()
    for item in data:
        a = item.get("violated_articles", None)
        if isinstance(a, list):
            for aa in a:
                articles.add(str(aa))
        elif a is not None:
            articles.add(str(a))
    result = sorted(articles, key=int)
    logger.info(f"Found {len(result)} unique articles in task2: {result}")
    return result

def build_appname_to_folder():
    logger = logging.getLogger("helper")
    logger.info("Building appname to folder mapping")
    repo_root = "repos"
    if not os.path.exists(repo_root):
        logger.warning(f"Repository root directory {repo_root} does not exist")
        return {}
    folders = [f for f in os.listdir(repo_root) if os.path.isdir(os.path.join(repo_root, f))]
    mapping = {}
    for folder in folders:
        mapping[folder] = folder
    logger.info(f"Built mapping with {len(mapping)} entries: {list(mapping.keys())}")
    return mapping

def ask_llm(api_url, api_key, messages, model, max_tokens=1024, temperature=0.0):
    logger = logging.getLogger("api")
    logger.debug(f"Making API call to {api_url} with model {model}")
    logger.debug(f"Messages: {messages}")
    
    # o1 model needs more tokens for internal reasoning
    if model == "o1":
        max_tokens = 4096  # Allow o1 to complete its thinking process
        logger.info(f"o1 model detected: using max_tokens={max_tokens} for complete reasoning")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    # Unified timeout: 300 seconds
    timeout = 300
    
    for attempt in range(3):
        resp = None
        try:
            logger.debug(f"API call attempt {attempt + 1}/3 (timeout={timeout}s)")
            resp = requests.post(api_url, headers=headers, json=data, timeout=timeout)  
            resp.raise_for_status()
            result = resp.json()["choices"][0]["message"]["content"]
            logger.debug(f"API call successful, response: {result}")
            return result
        except Exception as e:
            logger.warning(f"API call attempt {attempt + 1} failed: {e}")
            if resp is not None:
                logger.warning(f"API response: {resp.text}")
            if attempt < 2:  # Don't sleep on last attempt
                logger.info("Retrying in 2 seconds...")
                time.sleep(2)
    
    logger.error("All API call attempts failed")
    return ""

def predict_task1(api_url, api_key, model, out_dir):
    logger = get_logger(model)
    logger.info(f"Starting predict_task1 for model: {model}")
    all_articles = get_all_articles_from_task1()
    article_list = ", ".join(all_articles)
    with open("task1_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    appname_to_folder = build_appname_to_folder()
    predictions = []
    logger.info(f"Processing {len(dataset)} items for task 1")
    for item_idx, item in enumerate(dataset):
        logger.info(f"Processing item {item_idx+1}/{len(dataset)}: {item.get('app_name', 'Unknown')} - {item.get('Commit_ID', 'Unknown')}")
        pred = {
            "repo_url": item["repo_url"],
            "app_name": item["app_name"],
            "Commit_ID": item["Commit_ID"],
            "file_level_violations": [],
            "module_level_violations": [],
            "line_level_violations": []
        }
        # Collect all file/module/line targets for this file
        file_targets = item.get("file_level_violations", [])
        module_targets = item.get("module_level_violations", [])
        line_targets = item.get("line_level_violations", [])
        # For each file-level target, build a unified prompt for that file
        for file_item in file_targets:
            file_path = file_item["file_path"]
            folder = appname_to_folder.get(item['app_name'], item['app_name'])
            src_path = f"repos/{folder}/{file_path}"
            try:
                with open(src_path, "r", encoding="utf-8", errors="ignore") as fcode:
                    file_content = fcode.read()
                logger.debug(f"Successfully read file {src_path}, size: {len(file_content)} characters")
            except FileNotFoundError:
                file_content = ""
                logger.warning(f"File not found: {src_path}, using empty content for prediction.")
            except Exception as e:
                file_content = ""
                logger.error(f"Error reading file {src_path}: {e}")
            # Gather module and line targets for this file
            related_modules = [m for m in module_targets if m["file_path"] == file_path]
            related_lines = [l for l in line_targets if l["file_path"] == file_path]
            # Build module target descriptions
            module_descs = []
            for m in related_modules:
                # If line range info is available, include it; else just module name
                module_desc = f"Module: {m['module_name']}"
                module_descs.append(module_desc)
            # Build line target descriptions
            line_descs = []
            for l in related_lines:
                desc = l.get("violation_description", "")
                line_desc = f"{l['file_path']} lines {l['line_spans']}: {desc}"
                line_descs.append(line_desc)
            # Build the unified prompt
            prompt = f"""
You are a GDPR compliance expert. Your task is to determine whether the following code file and its components violate any GDPR articles.

The following GDPR articles are commonly relevant in this context: {article_list}. However, it is also possible that none of these articles are violated.

**GDPR Article Meanings:**
- Article 5: Principles of processing (lawfulness, fairness, transparency)
- Article 6: Lawfulness of processing (legal basis for data processing)
- Article 7: Conditions for consent (valid consent requirements)
- Article 8: Conditions applicable to child's consent
- Article 9: Processing of special categories of personal data
- Article 12: Transparent information and communication
- Article 13: Information to be provided when personal data are collected
- Article 14: Information to be provided when personal data have not been obtained from the data subject
- Article 15: Right of access by the data subject
- Article 16: Right to rectification
- Article 17: Right to erasure ('right to be forgotten')
- Article 18: Right to restriction of processing
- Article 19: Notification obligation regarding rectification or erasure
- Article 21: Right to object
- Article 25: Data protection by design and by default
- Article 30: Records of processing activities
- Article 32: Security of processing
- Article 33: Notification of a personal data breach to the supervisory authority
- Article 35: Data protection impact assessment
- Article 44: General principle for transfers
- Article 46: Transfers subject to appropriate safeguards
- Article 58: Powers of supervisory authorities
- Article 83: General conditions for imposing administrative fines

[File Content]
{file_content}

[Detection Targets]
1. File-level: the entire file.
2. Module-level:
"""
            if module_descs:
                prompt += "   - " + "\n   - ".join(module_descs) + "\n"
            else:
                prompt += "   (none)\n"
            prompt += "3. Line-level:\n"
            if line_descs:
                prompt += "   - " + "\n   - ".join(line_descs) + "\n"
            else:
                prompt += "   (none)\n"
            prompt += """
[Instructions]
- If multiple articles are violated, please list all of them
- For each target, output only the violated GDPR article numbers from the list above, separated by commas (e.g., 5,6,32). If there is no violation, output exactly 0.
- Do not add any explanation or extra text.
File: <file_path>: <articles>
Module: <module_name>: <articles>
Line: <file_path> <line_span>: <articles>
..."""
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ]
            logger.debug(f"Making unified API call for file: {file_path}")
            result = ask_llm(api_url, api_key, messages, model)
            logger.debug(f"Unified API response for file {file_path}: {result}")
            # Parse the output
            file_articles = []
            module_articles = {m["module_name"]: [] for m in related_modules}
            line_articles = {(l["file_path"], l["line_spans"]): [] for l in related_lines}
            for line in result.splitlines():
                line = line.strip()
                if line.startswith("File:"):
                    # File: <file_path>: <articles>
                    parts = line[len("File:"):].strip().split(":")
                    if len(parts) == 2:
                        articles = [int(s) for s in parts[1].replace("，", ",").replace(" ", "").split(",") if s.isdigit()]
                        file_articles = articles
                elif line.startswith("Module:"):
                    # Module: <module_name>: <articles>
                    parts = line[len("Module:"):].strip().split(":")
                    if len(parts) == 2:
                        module_name = parts[0].strip()
                        articles = [int(s) for s in parts[1].replace("，", ",").replace(" ", "").split(",") if s.isdigit()]
                        module_articles[module_name] = articles
                elif line.startswith("Line:"):
                    # Line: <file_path> <line_span>: <articles>
                    parts = line[len("Line:"):].strip().split(":")
                    if len(parts) == 2:
                        left = parts[0].strip()
                        articles = [int(s) for s in parts[1].replace("，", ",").replace(" ", "").split(",") if s.isdigit()]
                        # left: <file_path> <line_span>
                        if " " in left:
                            fpath, lspan = left.split(" ", 1)
                            line_articles[(fpath, lspan)] = articles
            # Fill results
            pred["file_level_violations"].append({
                "file_path": file_path,
                "violated_articles": file_articles
            })
            for m in related_modules:
                pred["module_level_violations"].append({
                    "file_path": m["file_path"],
                    "module_name": m["module_name"],
                    "violated_articles": module_articles.get(m["module_name"], [])
                })
            for l in related_lines:
                pred["line_level_violations"].append({
                    "file_path": l["file_path"],
                    "line_spans": l["line_spans"],
                    "violated_articles": line_articles.get((l["file_path"], l["line_spans"]), [])
                })
            print(f"Unified-Task1-File: {file_path} -> File: {file_articles}, Modules: {module_articles}, Lines: {line_articles}")
            time.sleep(1)
        predictions.append(pred)
        logger.info(f"Completed processing item {item_idx+1}/{len(dataset)}")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
        logger.info(f"Created output directory: {out_dir}")
    out_file = os.path.join(out_dir, f"{model}_task1_predictions.json")
    logger.info(f"Saving results to {out_file}")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    logger.info(f"Task1 prediction completed, results saved to {out_file}")
    print(f"Task1 prediction completed, results saved to {out_file}")
    return out_file

def evaluate_task1(pred_file):
    logger = logging.getLogger("evaluation")
    logger.info(f"Starting evaluation for task1 prediction file: {pred_file}")
    print("\nStart evaluation...\n")
    
    cmd = ["python", "evaluate_model.py"]
    std_pred_file = "task1_predictions.json"
    
    if os.path.exists(std_pred_file+".bak"):
        os.remove(std_pred_file+".bak")
        logger.debug("Removed existing backup file")
    
    if os.path.exists(std_pred_file):
        os.rename(std_pred_file, std_pred_file+".bak")
        logger.debug("Backed up existing prediction file")
    
    try:
        with open(pred_file, "r", encoding="utf-8") as src, open(std_pred_file, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        logger.info(f"Copied prediction file {pred_file} to {std_pred_file}")
    except Exception as e:
        logger.error(f"Error copying prediction file: {e}")
        return
    
    logger.info("Running evaluation script")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"Evaluation completed with return code: {result.returncode}")
        if result.stdout:
            logger.info(f"Evaluation stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Evaluation stderr: {result.stderr}")
    except Exception as e:
        logger.error(f"Error running evaluation script: {e}")
    
    if os.path.exists(std_pred_file+".bak"):
        if os.path.exists(std_pred_file):
            os.remove(std_pred_file)
        os.rename(std_pred_file+".bak", std_pred_file)
        logger.debug("Restored original prediction file")

def predict_task2(api_url, api_key, model, out_dir):
    logger = get_logger(model)
    logger.info(f"Starting predict_task2 for model: {model}")
    
    all_articles = get_all_articles_from_task2()
    article_list = ", ".join(all_articles)
    with open("task2_dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    predictions = []
    
    logger.info(f"Processing {len(dataset)} items for task 2")
    
    for item_idx, item in enumerate(dataset):
        logger.info(f"Processing item {item_idx+1}/{len(dataset)}: {item.get('code_snippet_path', 'Unknown')}")
        
        code = item.get("code_snippet", "")
        if isinstance(code, list):
            code_lines = [line.strip() for line in code if line.strip()]
            code = "\n".join(code_lines)
            logger.debug(f"Converted list code snippet to string, {len(code_lines)} lines")
        else:
            code = code.strip()
        
        logger.debug(f"Code snippet length: {len(code)} characters")
        
        prompt = f"""
You are a GDPR compliance expert. Your task is to determine which GDPR articles are violated by the following code snippet.

The following GDPR articles are commonly relevant in this context: {article_list}

**GDPR Article Meanings:**
- Article 5: Principles of processing (lawfulness, fairness, transparency)
- Article 6: Lawfulness of processing (legal basis for data processing)
- Article 7: Conditions for consent (valid consent requirements)
- Article 8: Conditions applicable to child's consent
- Article 9: Processing of special categories of personal data
- Article 12: Transparent information and communication
- Article 13: Information to be provided when personal data are collected
- Article 14: Information to be provided when personal data have not been obtained from the data subject
- Article 15: Right of access by the data subject
- Article 16: Right to rectification
- Article 17: Right to erasure ('right to be forgotten')
- Article 18: Right to restriction of processing
- Article 19: Notification obligation regarding rectification or erasure
- Article 21: Right to object
- Article 25: Data protection by design and by default
- Article 30: Records of processing activities
- Article 32: Security of processing
- Article 33: Notification of a personal data breach to the supervisory authority
- Article 35: Data protection impact assessment
- Article 44: General principle for transfers
- Article 46: Transfers subject to appropriate safeguards
- Article 58: Powers of supervisory authorities
- Article 83: General conditions for imposing administrative fines

**Instructions:**
- Carefully analyze the code snippet for GDPR compliance issues
- If multiple articles are violated, please list all of them
- Only output the violated GDPR article numbers from the list above, separated by commas (e.g., 5,6,32)
- If there is no violation, output exactly 0
- Do not output any explanation, text, or extra symbols. Only output numbers as specified
- Output cannot be empty

**IMPORTANT - You MUST provide an answer:**
- DO NOT return blank/empty response
- DO NOT refuse to answer
- You MUST output either article numbers or 0

**Examples:**
- Code that collects personal data without consent: 6,7
- Code that lacks security measures: 25,32
- Code that doesn't provide privacy notice: 12,13
- Code with no data access controls: 15,16,17

Code snippet:
{code}
"""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        
        logger.debug(f"Making API call for snippet: {item['code_snippet_path']}")
        result = ask_llm(api_url, api_key, messages, model)
        logger.debug(f"API response for snippet {item['code_snippet_path']}: {result}")
        
        try:
            articles = [int(s) for s in result.replace("，", ",").replace(" ", "").split(",") if s.isdigit()]
            logger.info(f"Snippet {item['code_snippet_path']} -> Articles: {articles}")
        except Exception as e:
            articles = []
            logger.error(f"Error parsing articles for snippet {item['code_snippet_path']}: {e}, raw result: {result}")
        
        predictions.append({
            "repo_url": item["repo_url"],
            "app_name": item["app_name"],
            "Commit_ID": item["Commit_ID"],
            "code_snippet_path": item["code_snippet_path"],
            "violated_articles": articles
        })
        print(f"Task2: {item['code_snippet_path']} -> {articles}")
        time.sleep(1)
        
        logger.info(f"Completed processing item {item_idx+1}/{len(dataset)}")
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        logger.info(f"Created output directory: {out_dir}")
    
    out_file = os.path.join(out_dir, f"{model}_task2_predictions.json")
    logger.info(f"Saving results to {out_file}")
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Task2 prediction completed, results saved to {out_file}")
    print(f"Task2 prediction completed, results saved to {out_file}")
    return out_file

def evaluate_task2(pred_file):
    logger = logging.getLogger("evaluation")
    logger.info(f"Starting evaluation for task2 prediction file: {pred_file}")
    print("\nStart evaluation...\n")
    
    cmd = ["python", "evaluate_model.py"]
    std_pred_file = "task2_predictions.json"
    
    if os.path.exists(std_pred_file+".bak"):
        os.remove(std_pred_file+".bak")
        logger.debug("Removed existing backup file")
    
    if os.path.exists(std_pred_file):
        os.rename(std_pred_file, std_pred_file+".bak")
        logger.debug("Backed up existing prediction file")
    
    try:
        with open(pred_file, "r", encoding="utf-8") as src, open(std_pred_file, "w", encoding="utf-8") as dst:
            dst.write(src.read())
        logger.info(f"Copied prediction file {pred_file} to {std_pred_file}")
    except Exception as e:
        logger.error(f"Error copying prediction file: {e}")
        return
    
    logger.info("Running evaluation script")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        logger.info(f"Evaluation completed with return code: {result.returncode}")
        if result.stdout:
            logger.info(f"Evaluation stdout: {result.stdout}")
        if result.stderr:
            logger.warning(f"Evaluation stderr: {result.stderr}")
    except Exception as e:
        logger.error(f"Error running evaluation script: {e}")
    
    if os.path.exists(std_pred_file+".bak"):
        if os.path.exists(std_pred_file):
            os.remove(std_pred_file)
        os.rename(std_pred_file+".bak", std_pred_file)
        logger.debug("Restored original prediction file")

def main():
    try:
        print("🔧 Starting GDPR Prediction Program...")
        
        # Setup main logger
        main_logger = logging.getLogger("main")
        main_logger.setLevel(logging.INFO)
        if not os.path.exists("logs"):
            os.makedirs("logs", exist_ok=True)
        main_handler = logging.FileHandler("logs/main.log", encoding='utf-8')
        main_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        main_logger.addHandler(main_handler)
        
        print(f"📋 Arguments: {sys.argv}\n")
        main_logger.info("Starting GDPR prediction program")
        main_logger.info(f"Program arguments: {sys.argv}")
    except Exception as e:
        print(f"❌ ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")
        return
    
    # Parse additional arguments
    exclude_apps = []
    incremental_mode = False
    
    for arg in sys.argv[1:]:
        if arg.startswith("--exclude-apps="):
            exclude_apps = arg.split("=")[1].split(",")
            exclude_apps = [app.strip() for app in exclude_apps if app.strip()]
            main_logger.info(f"Exclude apps: {exclude_apps}")
            print(f"🚫 Excluded apps: {exclude_apps}")
        elif arg == "--incremental":
            incremental_mode = True
            main_logger.info("Incremental mode enabled")
            print(f"📥 Incremental mode: Will append to existing predictions")
    
    # API Configuration
    API_URL = os.environ.get('OPENAI_API_BASE', 'https://api.openai.com/v1') + '/chat/completions'
    
    # LLM-based models
    llm_model_list = [
        "gpt-4o",
        "o1",
        "claude-sonnet-4-5-20250929",
        "claude-3-7-sonnet-20250219",
        "gemini-2.5-pro-thinking",
        "gemini-2.5-pro-preview-06-05-thinking",
        "qwen2.5-72b-instruct",
        "deepseek-r1"
    ]
    
    # Alternative methods (Agentic, Rule-based)
    alternative_methods = [
        "react",           # ReAct Agent with GDPR tools
        "rag",             # RAG-based analysis  
        "ast-based",       # AST-based static analysis
    ]
    
    print(f"\n✅ Available alternative methods: {', '.join(alternative_methods)}")
    print("   - Agentic: react, rag")
    print("   - Static Analysis: ast-based\n")
    
    # Combined model list
    model_list = llm_model_list + alternative_methods
    
    # Load API keys from environment variables
    # You can set multiple keys for different models/tasks using:
    # OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY, etc.
    default_api_key = os.environ.get('OPENAI_API_KEY', '')
    
    if not default_api_key:
        print("⚠️  Warning: No API key found in environment variables.")
        print("   Please set OPENAI_API_KEY or other model-specific API keys.")
        print("   Example: export OPENAI_API_KEY='your-api-key-here'\n")
    
    # For simplicity, use the same key for all LLM models
    # You can customize this to use different keys for different models
    key_list_task1 = [default_api_key] * len(llm_model_list) + [None] * len(alternative_methods)
    key_list_task2 = [default_api_key] * len(llm_model_list) + [None] * len(alternative_methods)
    
    # Build task-specific mappings
    model_to_key_task1 = dict(zip(model_list, key_list_task1))
    model_to_key_task2 = dict(zip(model_list, key_list_task2))
    main_logger.info(f"Configured {len(model_list)} models: {model_list}")
    
    args = sys.argv[1:]
    
    print("\n" + "="*60)
    print("GDPR Prediction Program")
    print("="*60)
    print("\nModel list:")
    for i, m in enumerate(model_list):
        print(f"{i+1}. {m}")
    print("all. All models")
    print("\nTask list:")
    print("1. Task 1 (multi-granularity detection)")
    print("2. Task 2 (snippet classification)")
    print("all. Both tasks run")
    print("\nOptional Arguments:")
    print("  --models=model1,model2    Specify models (or use 'all')")
    print("  --tasks=1,2               Specify tasks (or use 'all')")
    print("  --exclude-apps=App1,App2  Exclude specific apps")
    print("  --incremental             Incremental mode (append to existing predictions)")
    print("\nExample:")
    print("  python predict.py --models=gpt-4o,o1 --tasks=1,2 --exclude-apps=Android_Spy_App,Dash --incremental")
    print("="*60 + "\n")
    
    # Parse named arguments (--models, --tasks) and positional arguments
    task_choice = None
    model_choice = None
    positional_args = []
    
    for arg in args:
        if arg.startswith('--models='):
            model_choice = arg.split('=', 1)[1]
        elif arg.startswith('--tasks='):
            task_choice = arg.split('=', 1)[1]
        elif not arg.startswith('--'):
            positional_args.append(arg)
    
    # Support old positional format: task model
    if not task_choice and not model_choice and len(positional_args) >= 2:
        task_choice = positional_args[0]
        model_choice = positional_args[1]
        main_logger.info(f"Using positional arguments: task={task_choice}, model={model_choice}")
    elif task_choice and model_choice:
        main_logger.info(f"Using named arguments: task={task_choice}, model={model_choice}")
    else:
        print("Select task (enter 1, 2, or all, separated by commas):")
        task_choice = input("Enter: ").strip()
        print("Select model (enter index, model name, or all, separated by commas):")
        model_choice = input("Enter: ").strip()
        main_logger.info(f"Using user input: task={task_choice}, model={model_choice}")
    
    if task_choice == "all":
        selected_tasks = ["1", "2"]
    else:
        selected_tasks = [t.strip() for t in task_choice.split(",") if t.strip() in ["1", "2"]]
    if not selected_tasks:
        main_logger.error("Invalid task selection, exiting")
        print("Invalid task selection, exiting.")
        return
    
    if model_choice == "all":
        selected_models = model_list
    else:
        selected_models = []
        for m in model_choice.split(","):
            m = m.strip()
            if m.isdigit() and 1 <= int(m) <= len(model_list):
                selected_models.append(model_list[int(m)-1])
            elif m in model_list:
                selected_models.append(m)
        if not selected_models:
            main_logger.error("Invalid model selection, exiting")
            print("Invalid model selection, exiting.")
            return
    
    main_logger.info(f"Selected tasks: {selected_tasks}")
    main_logger.info(f"Selected models: {selected_models}")
    
    # Count total
    total_counts = {}
    out_dirs = {}
    out_files = {}
    for task in selected_tasks:
        if task == "1":
            with open("task1_dataset.json", "r", encoding="utf-8") as f:
                total = len(json.load(f))
            out_dir = "task1_predictions"
            main_logger.info(f"Task 1 dataset has {total} items")
        else:
            with open("task2_dataset.json", "r", encoding="utf-8") as f:
                total = len(json.load(f))
            out_dir = "task2_predictions"
            main_logger.info(f"Task 2 dataset has {total} items")
        
        if not os.path.exists(out_dir):
            os.makedirs(out_dir, exist_ok=True)
            main_logger.info(f"Created output directory: {out_dir}")
        
        for model in selected_models:
            total_counts[(model, task)] = total
            out_dirs[(model, task)] = out_dir
            out_files[(model, task)] = os.path.join(out_dir, f"{model}_task{task}_predictions.json")
            main_logger.info(f"Configured {model} for task {task}: {total} items -> {out_files[(model, task)]}")
    
    # Start monitoring thread
    # Allow manual disable via command line argument
    disable_clear = '--disable-monitor-clear' in sys.argv
    
    if disable_clear:
        print("💡 Screen clearing disabled - monitor will append new updates\n")
    
    main_logger.info("Starting monitoring thread")
    monitor_thread = threading.Thread(target=monitor_models, args=(selected_models, selected_tasks, total_counts, out_dirs, out_files, disable_clear), daemon=True)
    monitor_thread.start()
    
    # Catch Ctrl+C graceful exit
    def signal_handler(sig, frame):
        global monitor_stop
        main_logger.info("Received interrupt signal, starting graceful shutdown")
        print("\n[Exit] Saving all progress...")
        
        with progress_lock:
            for key, out_file in out_files.items():
                results = model_progress.get(key, {}).get('results', [])
                if results:
                    try:
                        with open(out_file, 'w', encoding='utf-8') as f:
                            json.dump(results, f, ensure_ascii=False, indent=2)
                        main_logger.info(f"Saved progress for {key}: {len(results)} results to {out_file}")
                    except Exception as e:
                        main_logger.error(f"Failed to save progress for {key}: {e}")
        
        monitor_stop = True
        time.sleep(2)
        main_logger.info("Program terminated by user")
        exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Task parallel, model parallel (16 threads total: 8 for Task1 + 8 for Task2)
    main_logger.info("Starting prediction tasks in PARALLEL mode")
    print("\n⚡ 并行模式: Task 1 和 Task 2 同时运行（16线程）\n")
    
    with ThreadPoolExecutor(max_workers=len(selected_models) * len(selected_tasks)) as executor:
        futures = []
        
        for task in selected_tasks:
            main_logger.info(f"Submitting task {task}")
            out_dir = "task1_predictions" if task == "1" else "task2_predictions"
            
            # Select appropriate keys for this task
            model_to_key = model_to_key_task1 if task == "1" else model_to_key_task2
            
            for model in selected_models:
                # Alternative methods don't need API keys
                alternative_methods_list = ['react', 'rag', 'ast-based']
                if model.lower() in alternative_methods_list:
                    api_key = None  # Not used for alternative methods
                else:
                    api_key = model_to_key.get(model, "")
                
                out_file = out_files[(model, task)]
                total_count = total_counts[(model, task)]
                
                main_logger.info(f"Submitting {model} on task {task} (using {'task1' if task=='1' else 'task2'} keys)")
                futures.append(executor.submit(run_model_task_realtime, API_URL, api_key, model, task, out_dir, total_count, out_file, exclude_apps, incremental_mode))
        
        # Wait for all tasks to complete
        for future in as_completed(futures):
            try:
                future.result()
                main_logger.info("Model task completed successfully")
            except Exception as e:
                main_logger.error(f"Model thread exception: {e}", exc_info=True)
                print(f"Model thread exception: {e}")
    
    # End monitoring
    main_logger.info("All tasks completed, stopping monitoring")
    global monitor_stop
    monitor_stop = True
    monitor_thread.join()
    main_logger.info("Program completed successfully")
    print("All tasks completed!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ CRITICAL ERROR")
        print(f"{'='*80}")
        print(f"Error: {e}")
        print(f"\nFull traceback:")
        import traceback
        traceback.print_exc()
        print(f"\n{'='*80}")
        print(f"Please check logs/main.log for details")
        print(f"{'='*80}")
        input("\nPress Enter to exit...") 