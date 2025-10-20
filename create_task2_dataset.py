#!/usr/bin/env python3
import json
import os
import random
import sys
from collections import defaultdict, Counter

def main(input_file='GDPR_dataset.json', output_file='task2_dataset.json'):
    # Load data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} entries from {input_file}")
    
    # Group by code_snippet_path (multi-label)
    grouped = defaultdict(lambda: {
        "repo_url": None,
        "app_name": None,
        "Commit_ID": None,
        "code_snippet_path": None,
        "violated_articles": set(),
        "code_snippet": None  # Add code_snippet field
    })
    for entry in data:
        key = entry.get("code_snippet_path", "")
        group = grouped[key]
        # Only keep the first occurrence of meta information
        if group["repo_url"] is None:
            group["repo_url"] = entry.get("repo_url", "")
        if group["app_name"] is None:
            group["app_name"] = entry.get("app_name", "")
        if group["Commit_ID"] is None:
            group["Commit_ID"] = entry.get("Commit_ID", "")
        if group["code_snippet_path"] is None:
            group["code_snippet_path"] = entry.get("code_snippet_path", "")
        if group["code_snippet"] is None:
            group["code_snippet"] = entry.get("code_snippet", "")
        # Collect all violated articles
        violated_article = entry.get("violated_article", None)
        if violated_article is not None and violated_article != "" and violated_article != 0 and violated_article != "0":
            group["violated_articles"].add(violated_article)
    # Assemble multi-label data
    task2_data = []
    for group in grouped.values():
        task2_entry = {
            "repo_url": group["repo_url"],
            "app_name": group["app_name"],
            "Commit_ID": group["Commit_ID"],
            "code_snippet_path": group["code_snippet_path"],
            "violated_articles": sorted(list(group["violated_articles"])) if group["violated_articles"] else [0],
            "code_snippet": group["code_snippet"] if group["code_snippet"] is not None else ""
        }
        task2_data.append(task2_entry)
    # Output
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(task2_data, f, indent=2, ensure_ascii=False)
    # Statistics
    article_counter = Counter()
    for entry in task2_data:
        for art in entry["violated_articles"]:
            article_counter[str(art)] += 1
    print(f"Created {output_file} with {len(task2_data)} entries")
    print("Article distribution:")
    for article, count in sorted(article_counter.items(), key=lambda x: str(x[0])):
        print(f"  Article {article}: {count} entries")

if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else 'GDPR_dataset.json'
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'task2_dataset.json'
    main(input_file, output_file) 