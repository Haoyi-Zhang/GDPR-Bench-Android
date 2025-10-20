#!/usr/bin/env python3
import json
import os
import sys
from collections import defaultdict, Counter

def main(input_file='GDPR_dataset.json', output_file='task1_dataset.json'):
    # Read original data - handling multiple JSON objects
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {input_file}")
    
    # Group violations by file (not by repository/commit)
    file_groups = {}
    for entry in data:
        file_path = entry.get('code_snippet_path','').split(':')[0].strip()
        repo_url = entry.get('repo_url','')
        app_name = entry.get('app_name','')
        commit_id = entry.get('Commit_ID','')
        
        # Create unique key for each file across all repositories
        # Use (repo_url, app_name, file_path) to ensure uniqueness
        key = (repo_url, app_name, file_path)
        
        if key not in file_groups:
            file_groups[key] = {
                'repo_url': repo_url,
                'app_name': app_name,
                'commit_id': commit_id,  # Use the first commit_id found for this file
                'file_path': file_path,
                'violations': []
            }
        
        file_groups[key]['violations'].append(entry)
    
    print(f"Grouped into {len(file_groups)} unique files")
    
    # Process data for Task 1 format - one record per file
    task1_data = []
    for (repo_url, app_name, file_path), file_data in file_groups.items():
        violations = file_data['violations']
        
        # Collect articles for file-level violations
        file_articles = set()
        module_violations = defaultdict(set)
        line_violations = []
        
        for violation in violations:
            article = violation.get('violated_article','')
            
            # Add to file-level violations
            file_articles.add(article)
            
            # Extract module name (simplified approach - could be improved)
            code_path_parts = file_path.split('/')
            if len(code_path_parts) > 0:
                file_name = code_path_parts[-1]
                class_name = file_name.split('.')[0]
                # Try to find a more specific module name from the code path or annotation
                module_name = class_name
                if 'annotation_note' in violation and '.' in violation['annotation_note']:
                    method_hints = [word for word in violation['annotation_note'].split() 
                                   if "." in word and class_name.lower() in word.lower()]
                    if method_hints:
                        module_name = method_hints[0].strip('.,():;')
                
                module_violations[module_name].add(article)
            
            # Extract line spans (handle both 'line' and 'lines')
            code_path = violation.get('code_snippet_path','')
            if "line" in code_path:
                # Handle both 'line 45' and 'lines 67-71'
                if "lines" in code_path:
                    line_info = code_path.split('lines')[-1].strip()
                else:
                    line_info = code_path.split('line')[-1].strip()
                line_spans = line_info.replace('–', '-').strip(':')
                
                line_violations.append({
                    "file_path": file_path,
                    "line_spans": line_spans,
                    "violated_articles": [article],
                    "violation_description": violation.get('annotation_note','')
                })
        
        # Create the file entry - one record per file
        file_entry = {
            "repo_url": file_data['repo_url'],
            "app_name": file_data['app_name'],
            "Commit_ID": file_data['commit_id'],
            "file_level_violations": [
                {"file_path": file_path, "violated_articles": sorted(list(file_articles))}
            ],
            "module_level_violations": [
                {
                    "file_path": file_path,
                    "module_name": module_name,
                    "violated_articles": sorted(list(articles))
                }
                for module_name, articles in module_violations.items()
            ],
            "line_level_violations": []
        }
        
        # Merge line violations with the same file path and line spans
        line_map = {}
        for violation in line_violations:
            key = (violation['file_path'], violation['line_spans'])
            if key not in line_map:
                line_map[key] = violation
            else:
                # Merge violated articles
                line_map[key]['violated_articles'] = sorted(list(set(
                    line_map[key]['violated_articles'] + violation['violated_articles']
                )))
        
        file_entry['line_level_violations'] = list(line_map.values())
        task1_data.append(file_entry)
    
    # Write to task1_dataset.json
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(task1_data, f, indent=2, ensure_ascii=False)
    
    print(f"Created {output_file} with {len(task1_data)} files")
    print(f"Each record contains exactly one file with its three violation levels:")
    print(f"- File-level violations")
    print(f"- Module-level violations") 
    print(f"- Line-level violations")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'GDPR_dataset.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'task1_dataset.json'
    main(input_file, output_file)