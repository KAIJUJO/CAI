"""Inspect GQA semantic format"""
import json
from pathlib import Path

q_path = Path(r'C:\Users\kaiji\OneDrive - University of Birmingham\Desktop\CAI\data\raw\gqa\questions\train_balanced_questions.json')
sg_path = Path(r'C:\Users\kaiji\OneDrive - University of Birmingham\Desktop\CAI\data\raw\gqa\sceneGraphs\train_sceneGraphs.json')

print("Loading questions...")
with open(q_path, 'r') as f:
    questions = json.load(f)

print("Loading scene graphs...")
with open(sg_path, 'r') as f:
    scene_graphs = json.load(f)

# Look at first 5 questions in detail
for i, (qid, q) in enumerate(list(questions.items())[:5]):
    print(f"\n{'='*60}")
    print(f"Question {i+1} (ID: {qid})")
    print(f"Image ID: {q.get('imageId')}")
    print(f"Q: {q.get('question')}")
    print(f"A: {q.get('answer')}")
    print(f"semanticStr: {q.get('semanticStr')}")
    print(f"semantic:")
    for step in q.get('semantic', []):
        print(f"  {step}")
    
    # Check scene graph
    img_id = q.get('imageId')
    if img_id in scene_graphs:
        sg = scene_graphs[img_id]
        print(f"\nScene Graph objects:")
        for obj_id, obj in list(sg.get('objects', {}).items())[:5]:
            print(f"  {obj_id}: {obj.get('name')} @ ({obj.get('x')},{obj.get('y')},{obj.get('w')},{obj.get('h')})")
