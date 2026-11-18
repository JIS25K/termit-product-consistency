from pathlib import Path
import json,sys,hashlib
p=Path('experiments/edit_agent/run12');root=Path.cwd();proto=json.loads((p/'frozen/protocol.json').read_text());step=int(sys.argv[1])
for arm in ['baseline','agent']:
 tasks=[]
 for prod,meta in proto['products'].items():
  versions={'v0':{'path':str((p/'images'/prod/'shared/v0.png').resolve()),'created_by_request':'실제 정면을 밝은 카페, 따뜻하고 부드러운 제품 조명으로 광고화'}}
  for i in range(1,step):
   path=(p/'images'/prod/arm/f'v{i}.png').resolve();assert path.exists()
   versions[f'v{i}']={'path':str(path),'created_by_request':proto['track_C']['requests'][i-1]}
  tasks.append({'product':prod,'label':meta['label'],'step':step,'latest_version':f'v{step-1}','initial_request':'실제 정면 제품을 밝은 카페 배경, 따뜻하고 부드러운 제품 조명으로 중앙에 전체가 보이게 광고화','history':proto['track_C']['requests'][:step],'current_request':proto['track_C']['requests'][step-1],'versions':versions,'actual_photos':{'actual_front':meta['front'],'actual_side':meta['side']},'target_side_description':meta['target'],'product_invariants':meta['invariants']})
 (p/'plans'/f'{arm}_step{step}_input.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2))
if step==1:
 common='''You are the planning model for a multi-turn product-image editing comparison. Read only the provided step input, your own instruction file and your own prior step input/output files. Do not read A/B outputs, scores, private files, protocol, other arm outputs/plans/images, or root reports. Prior conversation tasks from other trials are unavailable. Each product is a separate session logically. Inspect relevant images with view_image before deciding. Use the full user history and all available version metadata; choose up to 4 references per output. First reference must be edit_base. Provide a runnable Korean image editing prompt which labels each selected image role and fulfills the latest request while respecting prior unrevoked constraints. Do not generate images yourself. Do not browse or delegate. Save the first answer; do not reroll or revise based on grading. Output JSON {"plans":[{"product":"character|bottle","step":1,"action":"edit","edit_base":"version id","references":[{"asset_id":"version id or actual_front/actual_side","path":"absolute path","role":"purpose"}],"prompt":"Korean executable edit instruction","reason":"brief"}]}. Step number must match current input. Preserve failed decisions too. The executor will append the same user-history block and generic product-preservation suffix to both arms. Max 4 references, max 2500 prompt characters. No extra image-generation opportunity. No automatic retries or revisions. Reply with the output path only when saved.\n'''
 baseline=common+'Baseline mode: act as a normal competent editing assistant using full conversation history. Decide the next edit directly. Do not maintain an external structured state ledger; do not intentionally simplify or weaken the baseline.'
 agent=common+'Agent mode: maintain an explicit per-product state ledger at states/agent_c_ledger.json. Before each edit, trace the intended accepted state, apply only the latest change, list preserved fields, map each changed attribute to evidence/reference versions, then select an action and write its prompt. Keep desired state separate from observed output; do not silently accept accidental output drift as a changed user request. Log your state transition and source references. Do not change user requirements to fit an output.'
 (p/'frozen/C_baseline_instruction.txt').write_text(baseline)
 (p/'frozen/C_agent_instruction.txt').write_text(agent)
 hashes={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in [p/'frozen/C_baseline_instruction.txt',p/'frozen/C_agent_instruction.txt',p/'frozen/C_seed_prompt.txt']}
 (p/'C_execution_sha256.json').write_text(json.dumps(hashes,indent=2))
print('C step input prepared',step)
