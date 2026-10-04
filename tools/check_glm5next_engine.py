"""Exercise the actual pipe engine, cancellation and clean subsequent requests.

Explicit GPU/model integration tool; not part of CPU-only setup tests.
Use --profile and --reference for the real baseline, or --exe/--fixture for CTest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from serve.server import StrataEngine, child_env, engine_args


def check(cfg,ids,expected):
    report={'status':'error','cases':[]}
    engine=StrataEngine(cfg['exe'],engine_args(cfg),cwd=cfg.get('cwd'),env=child_env(cfg),log=cfg.get('log'))
    try:
        assert engine.info['architecture']=='glm5next' and engine.info['gpu_only']==1
        assert engine.can_stop and engine.can_session_id and not engine.can_cache_admin and not engine.can_vram_control
        report['info']=engine.info
        def generate(name,count,sampling=None):
            start=time.monotonic()
            tokens=[t for t in engine.generate(ids,count,sampling or {},threading.Event(),session_id='glm-test') if t is not None]
            assert len(tokens)==count and engine.last['finish']=='length', engine.last
            report['cases'].append({'name':name,'tokens':tokens,'done':dict(engine.last),'wall_seconds':time.monotonic()-start})
            return tokens
        baseline=generate('baseline',len(expected) if expected else 8)
        if expected: assert baseline==expected, 'saved baseline token mismatch'
        assert generate('clean second request',len(baseline))==baseline
        try:
            list(engine.generate([-1],1,{},threading.Event()))
            raise AssertionError('invalid token accepted')
        except ValueError as exc:
            report['cases'].append({'name':'invalid token rejected','error':str(exc)})
        sampled=generate('seeded sampling',8,{'temperature':0.7,'top_p':0.9,'top_k':32,'seed':42,'repetition_penalty':1.1})
        assert generate('seeded sampling repeat',8,{'temperature':0.7,'top_p':0.9,'top_k':32,'seed':42,'repetition_penalty':1.1})==sampled
        for phase in ('prefill','decode'):
            event=threading.Event(); gen=engine.generate(ids,256,{},event)
            emitted=0
            for token in gen:
                if token is not None: emitted+=1
                if (phase=='prefill' and token is None) or (phase=='decode' and emitted>=2):
                    start=time.monotonic(); event.set(); gen.close(); break
            else: raise AssertionError('generation ended before cancellation')
            assert engine.last['finish']=='cancel',engine.last
            report['cases'].append({'name':'cancel '+phase,'tokens_observed':emitted,'cancel_seconds':time.monotonic()-start,'done':dict(engine.last)})
            assert generate('clean after '+phase,min(8,len(baseline)))==baseline[:8]
        report['status']='pass'
    finally:
        process=engine.proc
        engine.close()
        report['process_exit_code']=process.returncode
    assert report['process_exit_code']==0,report
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--profile',type=Path)
    ap.add_argument('--reference',type=Path)
    ap.add_argument('--exe',type=Path)
    ap.add_argument('--fixture',type=Path)
    ap.add_argument('--cuda-dir',type=Path)
    ap.add_argument('--output',type=Path)
    a=ap.parse_args()
    if a.profile:
        if not a.reference or a.exe or a.fixture: ap.error('--profile requires --reference and cannot be combined with --exe/--fixture')
    elif not a.exe or not a.fixture or a.reference:
        ap.error('provide --profile/--reference or --exe/--fixture')
    if a.profile:
        cfg=json.loads(a.profile.read_text(encoding='utf-8'))
        reference=json.loads(a.reference.read_text(encoding='utf-8'))
        ids,expected=reference['prompt_ids'],reference['generated_ids']
    else:
        cfg={'exe':str(a.exe.resolve()),'cwd':str(ROOT),'args':['--native',str(a.fixture.resolve()),'--max-context','512','--batch-size','16'],
             'log':str(a.fixture.with_suffix('.protocol.log').resolve())}
        if a.cuda_dir:
            cfg['lib_dirs']=[str((a.cuda_dir/p).resolve()) for p in ('bin','bin/x64') if (a.cuda_dir/p).is_dir()]
        ids=[(7*i+11)%64 for i in range(33)]; expected=None
    result=check(cfg,ids,expected)
    result.update(binary_sha256=hashlib.sha256(Path(cfg['exe']).read_bytes()).hexdigest(),configuration=cfg)
    text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if a.output: a.output.write_text(text,encoding='utf-8')
    else: print(text)


if __name__=='__main__': main()
