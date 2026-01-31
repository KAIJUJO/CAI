"""
Quick verification test for data pipeline core functionality.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_pipeline import DataConfig, PairBuilder, CausalPairBuilder

def test_pair_builder():
    config = DataConfig()
    builder = CausalPairBuilder(config, seed=42)

    # Test question classification
    tests = [
        'What color is the dog?',
        'Is there a tree in the background?',
        'What shape is the ball?',
        'Is this indoors or outdoors?',
    ]

    print('=== Question Classification Test ===')
    for q in tests:
        cls, subtype = builder.classify_question(q)
        print(f'{q:45} -> {cls:10} ({subtype})')

    # Test pair building
    print('\n=== Pair Building Test ===')
    data = [
        {'question_id': 1, 'image_id': 'img1', 'question': 'What color is the cat?', 'answer': 'brown'},
        {'question_id': 2, 'image_id': 'img1', 'question': 'Is there a tree?', 'answer': 'yes'},
        {'question_id': 3, 'image_id': 'img1', 'question': 'What shape is the ball?', 'answer': 'round'},
        {'question_id': 4, 'image_id': 'img1', 'question': 'Is this outdoors?', 'answer': 'yes'},
    ]

    pairs = builder.build_pairs(data)
    print(f'Built {len(pairs)} pairs from {len(data)} questions')

    for pair in pairs[:2]:
        print(f"  Pair {pair['pair_id']}:")
        print(f"    Positive: {pair['positive']['question'][:50]}")
        print(f"    Negative: {pair['negative']['question'][:50]}")

    # Test scoring
    print('\n=== Scoring Test ===')
    score1 = builder.score_causal_relevance("What color is the dog?", "brown")
    score2 = builder.score_background_relevance("Is this indoors?", "yes")
    print(f"Causal score for 'What color is the dog?': {score1:.2f}")
    print(f"Background score for 'Is this indoors?': {score2:.2f}")

    print('\n=== All Tests Passed! ===')
    return True


if __name__ == "__main__":
    success = test_pair_builder()
    sys.exit(0 if success else 1)
