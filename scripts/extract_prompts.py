"""
Extract Entity Prompts from IV-VQA Questions for Grounding DINO

This script extracts physical object names from natural language questions
to create prompts suitable for Grounding DINO detection.

Methods:
1. Rule-based (fast, ~70% accuracy)
2. spaCy NLP (medium, ~85% accuracy)
3. LLM API (slow, ~95% accuracy)

Usage:
    python scripts/extract_prompts.py --method spacy  # Default
    python scripts/extract_prompts.py --method llm --api-key YOUR_KEY
"""

import json
import re
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm


def extract_with_rules(question: str) -> str:
    """
    Rule-based entity extraction (fast but less accurate).
    
    Uses regex patterns to find likely object mentions.
    """
    question_lower = question.lower().strip()
    
    # Remove question words and verbs
    remove_patterns = [
        r'^(is|are|does|do|can|what|where|how|which)\s+',
        r'\b(the|a|an)\b',
        r'\b(there|this|that|these|those)\b',
        r'\b(is|are|was|were|be|been|being)\b',
        r'\b(on|in|at|by|with|near|next to|behind|above|below|under)\b',
        r'\b(you|i|we|it)\b',
        r'\b(see|look|have|has|any)\b',
        r'\?$',
    ]
    
    cleaned = question_lower
    for pattern in remove_patterns:
        cleaned = re.sub(pattern, ' ', cleaned)
    
    # Extract remaining content words
    words = [w.strip() for w in cleaned.split() if len(w.strip()) > 2]
    
    # Join with comma for DINO
    if words:
        return ', '.join(words[:3])  # Limit to 3 objects
    return 'object'


def extract_with_spacy(questions: List[str]) -> List[str]:
    """
    spaCy-based entity extraction (better accuracy).
    
    Uses dependency parsing to find noun chunks.
    """
    try:
        import spacy
    except ImportError:
        print("spaCy not installed. Run: pip install spacy")
        print("Then: python -m spacy download en_core_web_sm")
        return [extract_with_rules(q) for q in questions]
    
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")
    
    # Words to filter out
    skip_phrases = {
        'it', 'this', 'that', 'there', 'what', 'where', 'which', 'who', 'whom',
        'what color', 'what kind', 'what type', 'how many', 'how much',
        'his', 'her', 'their', 'its', 'my', 'your', 'our',
        'something', 'anything', 'nothing', 'everything',
    }
    
    results = []
    
    for doc in tqdm(nlp.pipe(questions, batch_size=100), total=len(questions), desc="spaCy processing"):
        nouns = []
        
        for chunk in doc.noun_chunks:
            text = chunk.text.lower().strip()
            
            # Skip if in filter list
            if text in skip_phrases:
                continue
            
            # Remove leading determiners
            text = re.sub(r'^(the|a|an|this|that|these|those)\s+', '', text)
            
            # Skip question words at start
            text = re.sub(r'^(what|which|how|where|who)\s+', '', text)
            
            # Skip if empty or too short
            if text and len(text) > 1 and text not in skip_phrases:
                nouns.append(text)
        
        # Also check for standalone nouns not captured in chunks
        for token in doc:
            if token.pos_ == 'NOUN':
                word = token.text.lower()
                # Check if already captured
                if word not in [n.split()[-1] for n in nouns]:
                    if word not in skip_phrases and len(word) > 2:
                        nouns.append(word)
        
        # Deduplicate while preserving order
        seen = set()
        unique_nouns = []
        for n in nouns:
            if n not in seen:
                seen.add(n)
                unique_nouns.append(n)
        
        if unique_nouns:
            results.append(', '.join(unique_nouns[:3]))
        else:
            results.append('object')
    
    return results


def extract_with_llm(
    questions: List[str],
    api_key: str,
    model: str = "gpt-4o-mini",
    batch_size: int = 20,
) -> List[str]:
    """
    LLM-based entity extraction (highest accuracy).
    
    Uses OpenAI API for precise extraction.
    """
    try:
        from openai import OpenAI
    except ImportError:
        print("OpenAI not installed. Run: pip install openai")
        return [extract_with_rules(q) for q in questions]
    
    client = OpenAI(api_key=api_key)
    results = []
    
    system_prompt = """Extract physical object names from the question that would need to be visually located in an image.
Return ONLY the object names as a comma-separated list, no explanation.
Include adjectives if they describe the object (e.g., "red cup", "large dog").
If multiple objects, list all of them.

Examples:
- "Is the cat on the sofa?" -> "cat, sofa"
- "What color is the large elephant?" -> "large elephant"
- "Is there a red cup on the table?" -> "red cup, table"
- "How many people are standing?" -> "people"
"""
    
    for i in tqdm(range(0, len(questions), batch_size), desc="LLM processing"):
        batch = questions[i:i+batch_size]
        
        for q in batch:
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": q},
                    ],
                    max_tokens=50,
                    temperature=0,
                )
                result = response.choices[0].message.content.strip()
                results.append(result if result else 'object')
            except Exception as e:
                print(f"API error: {e}")
                results.append(extract_with_rules(q))
    
    return results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract DINO prompts from questions')
    parser.add_argument(
        '--input', type=str,
        default='data/processed/ivvqa_tiers/tier2_dino_indices.json',
        help='Input tier2 indices file'
    )
    parser.add_argument(
        '--output', type=str,
        default='data/processed/ivvqa_tiers/tier2_with_prompts.json',
        help='Output file with prompts'
    )
    parser.add_argument(
        '--method', type=str,
        choices=['rules', 'spacy', 'llm'],
        default='spacy',
        help='Extraction method'
    )
    parser.add_argument(
        '--api-key', type=str,
        default=None,
        help='OpenAI API key (required for llm method)'
    )
    parser.add_argument(
        '--limit', type=int,
        default=None,
        help='Limit samples for testing'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if args.limit:
        data = data[:args.limit]
    
    print(f"Processing {len(data)} samples with method: {args.method}")
    
    # Extract questions
    questions = [item.get('question', '') for item in data]
    
    # Run extraction
    if args.method == 'rules':
        prompts = [extract_with_rules(q) for q in tqdm(questions, desc="Rule extraction")]
    elif args.method == 'spacy':
        prompts = extract_with_spacy(questions)
    elif args.method == 'llm':
        if not args.api_key:
            print("Error: --api-key required for llm method")
            return 1
        prompts = extract_with_llm(questions, args.api_key)
    
    # Update data with prompts
    for item, prompt in zip(data, prompts):
        item['dino_prompt'] = prompt
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(data)} items to {output_path}")
    
    # Show examples
    print("\nExamples:")
    for item in data[:5]:
        print(f"  Q: {item['question']}")
        print(f"  -> {item['dino_prompt']}")
        print()
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
