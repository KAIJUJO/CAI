"""Check 150k data"""
import json
from pathlib import Path

p = Path(r'C:\Users\kaiji\OneDrive - University of Birmingham\Desktop\CAI\data\data\processed\gqa_sib_150k.json')
with open(p, 'r') as f:
    data = json.load(f)

meta = data['metadata']
print("=== 150K Data Summary ===")
print(f"Total samples: {meta['total_samples']}")
print(f"Valid ratio: {meta['stats']['valid_ratio']:.1%}")
print(f"Match types: {meta['stats']['match_types']}")
print(f"Coverage mean: {meta['stats']['coverage_mean']:.3f}")
