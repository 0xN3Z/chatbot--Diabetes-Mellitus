import csv
from typing import List, Dict
from pathlib import Path

import config


def load_test_set() -> List[Dict]:
    test_file = config.EVAL_DIR / "test_set.csv"
    
    if not test_file.exists():
        create_default_test_set()
    
    test_set = []
    with open(test_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            test_set.append(row)
    
    return test_set


def create_default_test_set():
    test_file = config.EVAL_DIR / "test_set.csv"
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    test_data = [
        {
            "question": "What is the recommended screening for diabetes?",
            "expected": "Screening for prediabetes and type 2 diabetes in adults aged 35 to 70 years with overweight or obesity"
        },
        {
            "question": "What is the target blood pressure for diabetes?",
            "expected": "Blood pressure target should be <=135/85 mmHg"
        }
    ]
    
    with open(test_file, 'w', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "expected"])
        writer.writeheader()
        writer.writerows(test_data)