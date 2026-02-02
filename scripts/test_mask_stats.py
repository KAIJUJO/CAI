"""Quick test for mask generator stats"""
import sys
sys.path.insert(0, '.')
from data_pipeline.config import DataConfig
from data_pipeline.gqa.parser import GQAParser
from data_pipeline.gqa.mask_generator import MaskGenerator

config = DataConfig()
parser = GQAParser(config)
mask_gen = MaskGenerator()

print('Loading scene graphs...')
scene_graphs = parser.parse_scene_graphs(split='train', limit=None)
print(f'Loaded {len(scene_graphs)} scene graphs')

print('Loading questions...')
questions = parser.parse_questions(split='train_balanced', limit=1000)
print(f'Loaded {len(questions)} questions')

# Count stats
stats = {'total': 0, 'valid': 0, 'no_sg': 0, 'no_match': 0, 'match_types': {}}
for q in questions:
    stats['total'] += 1
    img_id = q.get('image_id', '')
    if img_id not in scene_graphs:
        stats['no_sg'] += 1
        continue
    
    result = mask_gen.process_sample(q, scene_graphs[img_id])
    if result:
        stats['valid'] += 1
        for mt in result.get('match_types', {}).values():
            stats['match_types'][mt] = stats['match_types'].get(mt, 0) + 1
    else:
        stats['no_match'] += 1

print()
print('='*50)
print('STATISTICS:')
print(f"Total: {stats['total']}")
print(f"Valid: {stats['valid']} ({stats['valid']/stats['total']:.1%})")
print(f"No scene graph: {stats['no_sg']}")
print(f"No match: {stats['no_match']}")
print(f"Match types: {stats['match_types']}")
