"""Analyze IV-VQA data coverage - simpler version"""
import pickle
import json
from pathlib import Path

# Load CausalVQA pickle
cv_pickle = Path(r'data\raw\ivvqa\CausalVQA\cv_vqa_generation\train2014coco_counting_id_area_overlap_only_one_considered_at_a_time.pickle')
with open(cv_pickle, 'rb') as f:
    cv_data = pickle.load(f)

print("CausalVQA Pickle:")
print(f"  Total entries: {len(cv_data)}")
print(f"  Unique images: {len(set(item.get('image_id') for item in cv_data))}")
print(f"  Unique questions: {len(set(item.get('question_id') for item in cv_data))}")

# Check for missing IDs
missing_id = sum(1 for item in cv_data if not item.get('id'))
print(f"  Missing ID: {missing_id} ({missing_id/len(cv_data)*100:.1f}%)")

# Area stats
areas = [item.get('area_occupied', 0) for item in cv_data if item.get('area_occupied')]
print(f"  Area range: {min(areas):.3f} - {max(areas):.3f}, mean={sum(areas)/len(areas):.3f}")

# Our pairs
pairs_file = Path(r'data\processed\ivvqa_train_pairs.jsonl')
with open(pairs_file, 'r') as f:
    pairs = [json.loads(line) for line in f]

print(f"\nOur IVVQA Pairs:")
print(f"  Total pairs: {len(pairs)}")
print(f"  Sample keys: {list(pairs[0].keys())}")
print(f"  Has mask info: {'mask' in pairs[0] or 'bbox' in pairs[0]}")
