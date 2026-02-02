"""
Build IV-VQA Tier Classification and Mask Generation

This script:
1. Classifies IV-VQA samples into Tier 1/2/3
2. Generates tier indices files for downstream processing
3. For Tier 2: prepares prompts for Grounding DINO

Tier Classification:
- Tier 1 (COCO Gold): question_id matches CausalVQA pickle
- Tier 2 (Needs DINO): No match, but object likely exists
- Tier 3 (Zero Mask): Existence negation questions (Answer=No + "Is there")
"""

import json
import pickle
import re
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm


def is_existence_negation(question: str, answer: str) -> bool:
    """
    Check if this is an existence negation question.
    
    Returns True if:
    - Question contains existence patterns ("is there", "are there", etc.)
    - Answer is negative ("no", "0", "none", etc.)
    """
    existence_patterns = [
        r'\bis there\b',
        r'\bare there\b',
        r'\bdo you see\b',
        r'\bcan you see\b',
        r'\bdoes .* have\b',
        r'\bdo .* have\b',
        r'\bhow many\b',  # "How many dogs" with answer "0"
    ]
    
    question_lower = question.lower()
    answer_lower = str(answer).lower().strip()
    
    has_existence_pattern = any(
        re.search(p, question_lower) for p in existence_patterns
    )
    
    is_negative_answer = answer_lower in ['no', 'none', '0', 'zero', 'nothing', 'not', 'false']
    
    return has_existence_pattern and is_negative_answer


def extract_object_prompt(question: str) -> str:
    """
    Extract object name from question for Grounding DINO.
    
    Simple heuristic extraction - can be improved with LLM.
    """
    # Common patterns
    patterns = [
        r'(?:is|are) the (.+?) (?:in|on|near|next|behind|above|below)',
        r'(?:is|are) there (?:a|an|any) (.+?)(?:\?|$)',
        r'what color is the (.+?)(?:\?|$)',
        r'how many (.+?)(?:\?|$)',
        r'where is the (.+?)(?:\?|$)',
        r'what is the (.+?) doing',
        r'(?:is|are) the (.+?)\?',
    ]
    
    question_lower = question.lower().strip()
    
    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            obj = match.group(1).strip()
            # Clean up
            obj = re.sub(r'\b(a|an|the)\b', '', obj).strip()
            if obj and len(obj) > 1:
                return obj
    
    # Fallback: extract nouns (simplified)
    # For production, use spaCy or LLM
    words = question_lower.replace('?', '').split()
    # Skip common function words
    stopwords = {'is', 'are', 'the', 'a', 'an', 'in', 'on', 'of', 'to', 'what', 'where', 'how', 'there', 'this', 'that'}
    content_words = [w for w in words if w not in stopwords and len(w) > 2]
    
    if content_words:
        return content_words[-1]  # Often the object is at the end
    
    return 'object'


def classify_ivvqa(
    pairs_file: Path,
    pickle_file: Path,
    output_dir: Path,
) -> Dict[str, int]:
    """
    Classify IV-VQA samples into tiers.
    
    Args:
        pairs_file: Path to ivvqa_train_pairs.jsonl
        pickle_file: Path to CausalVQA pickle
        output_dir: Output directory for tier files
        
    Returns:
        Statistics dict
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CausalVQA pickle
    print("Loading CausalVQA pickle...")
    with open(pickle_file, 'rb') as f:
        cv_data = pickle.load(f)
    
    qid_to_ann = {item['question_id']: item['id'] for item in cv_data}
    print(f"  Loaded {len(qid_to_ann)} mappings")
    
    # Load pairs
    print("Loading IV-VQA pairs...")
    with open(pairs_file, 'r', encoding='utf-8') as f:
        pairs = [json.loads(line) for line in f]
    print(f"  Loaded {len(pairs)} pairs")
    
    # Classify
    tier1_items = []  # COCO Gold
    tier2_items = []  # Needs DINO
    tier3_items = []  # Zero Mask
    
    for idx, pair in enumerate(tqdm(pairs, desc="Classifying")):
        image_id = pair.get('image_id')
        pos = pair.get('positive', {})
        neg = pair.get('negative', {})
        
        pos_qid = pos.get('question_id')
        neg_qid = neg.get('question_id')
        pos_q = pos.get('question', '')
        neg_q = neg.get('question', '')
        pos_a = pos.get('answer', '')
        neg_a = neg.get('answer', '')
        
        # Build image path
        image_path = f"data/raw/ivvqa/train2014/COCO_train2014_{image_id:012d}.jpg"
        
        # Check Tier 1: COCO match
        coco_ann_id = None
        if pos_qid and pos_qid in qid_to_ann:
            coco_ann_id = qid_to_ann[pos_qid]
        elif neg_qid and neg_qid in qid_to_ann:
            coco_ann_id = qid_to_ann[neg_qid]
        
        if coco_ann_id:
            tier1_items.append({
                'pair_idx': idx,
                'image_id': image_id,
                'image_path': image_path,
                'question': pos_q or neg_q,
                'coco_ann_id': coco_ann_id,
                'tier': 't1',
            })
            continue
        
        # Check Tier 3: Existence negation
        if is_existence_negation(pos_q, pos_a) or is_existence_negation(neg_q, neg_a):
            tier3_items.append({
                'pair_idx': idx,
                'image_id': image_id,
                'image_path': image_path,
                'question': pos_q or neg_q,
                'tier': 't3',
            })
            continue
        
        # Tier 2: Needs DINO
        prompt = extract_object_prompt(pos_q or neg_q)
        tier2_items.append({
            'pair_idx': idx,
            'image_id': image_id,
            'image_path': image_path,
            'question': pos_q or neg_q,
            'prompt': prompt,
            'tier': 't2',
        })
    
    # Save tier files
    print("\nSaving tier files...")
    
    with open(output_dir / 'tier1_coco_indices.json', 'w') as f:
        json.dump(tier1_items, f, indent=2)
    print(f"  Tier 1 (COCO Gold): {len(tier1_items)}")
    
    with open(output_dir / 'tier2_dino_indices.json', 'w') as f:
        json.dump(tier2_items, f, indent=2)
    print(f"  Tier 2 (Needs DINO): {len(tier2_items)}")
    
    with open(output_dir / 'tier3_zero_indices.json', 'w') as f:
        json.dump(tier3_items, f, indent=2)
    print(f"  Tier 3 (Zero Mask): {len(tier3_items)}")
    
    stats = {
        'total': len(pairs),
        'tier1': len(tier1_items),
        'tier2': len(tier2_items),
        'tier3': len(tier3_items),
    }
    
    return stats


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Classify IV-VQA into tiers')
    parser.add_argument(
        '--pairs', type=str,
        default='data/processed/ivvqa_train_pairs.jsonl',
        help='Path to IV-VQA pairs file'
    )
    parser.add_argument(
        '--pickle', type=str,
        default='data/raw/ivvqa/CausalVQA/cv_vqa_generation/train2014coco_counting_id_area_overlap_only_one_considered_at_a_time.pickle',
        help='Path to CausalVQA pickle'
    )
    parser.add_argument(
        '--output', type=str,
        default='data/processed/ivvqa_tiers',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    stats = classify_ivvqa(
        pairs_file=Path(args.pairs),
        pickle_file=Path(args.pickle),
        output_dir=Path(args.output),
    )
    
    print("\n" + "="*50)
    print("Classification Complete!")
    print(f"  Total: {stats['total']}")
    print(f"  Tier 1 (COCO): {stats['tier1']} ({stats['tier1']/stats['total']*100:.1f}%)")
    print(f"  Tier 2 (DINO): {stats['tier2']} ({stats['tier2']/stats['total']*100:.1f}%)")
    print(f"  Tier 3 (Zero): {stats['tier3']} ({stats['tier3']/stats['total']*100:.1f}%)")
