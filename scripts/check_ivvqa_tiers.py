"""Check IV-VQA COCO pickle hit rate"""
import json
import pickle
from pathlib import Path

# Load CausalVQA pickle
cv_pickle = Path(r'data\raw\ivvqa\CausalVQA\cv_vqa_generation\train2014coco_counting_id_area_overlap_only_one_considered_at_a_time.pickle')
with open(cv_pickle, 'rb') as f:
    cv_data = pickle.load(f)

# Build question_id -> coco_ann_id mapping
qid_to_ann = {item['question_id']: item['id'] for item in cv_data}
print(f"CausalVQA entries: {len(cv_data)}")
print(f"Unique question_ids in pickle: {len(qid_to_ann)}")

# Load our pairs
pairs_file = Path(r'data\processed\ivvqa_train_pairs.jsonl')
with open(pairs_file, 'r') as f:
    pairs = [json.loads(line) for line in f]

print(f"\nOur IVVQA pairs: {len(pairs)}")

# Check hit rate
tier1_hits = 0
tier2_needs_dino = 0
tier3_zero_mask = 0

for pair in pairs:
    # Check positive question
    pos_qid = pair.get('positive', {}).get('question_id')
    neg_qid = pair.get('negative', {}).get('question_id')
    
    # Try to match
    if pos_qid and pos_qid in qid_to_ann:
        tier1_hits += 1
    elif neg_qid and neg_qid in qid_to_ann:
        tier1_hits += 1
    else:
        # Check if it's an existence question (Is there / Are there)
        pos_q = pair.get('positive', {}).get('question', '')
        neg_q = pair.get('negative', {}).get('question', '')
        
        existence_patterns = ['is there', 'are there', 'do you see', 'can you see']
        is_existence = any(p in pos_q.lower() or p in neg_q.lower() for p in existence_patterns)
        
        if is_existence:
            tier3_zero_mask += 1
        else:
            tier2_needs_dino += 1

print("\n=== Tier Distribution ===")
print(f"Tier 1 (COCO Gold): {tier1_hits} ({tier1_hits/len(pairs)*100:.1f}%)")
print(f"Tier 2 (Needs DINO): {tier2_needs_dino} ({tier2_needs_dino/len(pairs)*100:.1f}%)")
print(f"Tier 3 (Zero Mask): {tier3_zero_mask} ({tier3_zero_mask/len(pairs)*100:.1f}%)")
