"""Validate real Step shard slices through the CUDA pipeline and cache."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from setup_step35 import inspect_model


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checker',type=Path,required=True)
    parser.add_argument('--gguf',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    inventory=inspect_model(args.gguf)
    first=Path(inventory['first_shard'])
    selected=[]
    for shard in inventory['shard_details']:
        for role in ('gate','up','down'):
            tensors=sorted((t for t in inventory['tensor_details'] if t['shard']==shard['split_no'] and
                            re.fullmatch(r'blk\.\d+\.ffn_'+role+r'_exps\.weight',t['name'])),
                           key=lambda t:int(t['name'].split('.')[1]))
            if not tensors: continue
            for index in sorted({0,len(tensors)//2,len(tensors)-1}):
                t=tensors[index];path=first.with_name(shard['name'])
                for expert in (0,t['shape'][2]//2,t['shape'][2]-1):
                    size=t['expert_bytes']+(512 if expert+1<t['shape'][2] else 0)
                    offset=t['file_offset']+expert*t['expert_bytes']
                    with path.open('rb') as stream:
                        stream.seek(offset);raw=stream.read(size)
                    if len(raw)!=size: raise ValueError('short matrix read')
                    selected.append({'path':str(path),'shard':shard['split_no'],'tensor':t['name'],
                        'expert':expert,'file_offset':offset,'bytes':size,'payload_sha256':hashlib.sha256(raw).hexdigest()})
    args.output_dir.mkdir(parents=True,exist_ok=True)
    output=args.output_dir.resolve()
    outputs=[output/name for name in ('shard-samples.json','shard-report.json')]
    for target in outputs:
        if target.exists() and (target.samefile(args.checker) or
                any(target.samefile(first.with_name(s['name'])) for s in inventory['shard_details'])):
            raise ValueError('output aliases an input')
    if all(p.exists() for p in outputs) and outputs[0].samefile(outputs[1]):
        raise ValueError('report and manifest alias each other')
    manifest={'model':str(first),'model_structural_fingerprint_sha256':inventory['structural_fingerprint_sha256'],
              'payload_shards':sorted({s['shard'] for s in selected}),'samples':selected}
    source=output/'shard-samples.json';source.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    (output/'shard-report.json').write_text('{"status":"error","error":"checker did not produce a report"}\n',encoding='utf8')
    result=subprocess.run([str(args.checker.resolve()),str(source),str(output/'shard-report.json')],
                          timeout=180,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    report=json.loads((output/'shard-report.json').read_text(encoding='utf8'))
    report.update(checker_sha256=hashlib.sha256(args.checker.read_bytes()).hexdigest(),
                  samples_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),exit_code=result.returncode)
    if result.returncode!=0 or report.get('case_count')!=len(selected)*12+1:
        report.update(status='error',error=report.get('error','wrong checker exit/count'))
    (output/'shard-report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(report['status'],report.get('case_count',0),'real-shard byte checks',report.get('error',''),flush=True)
    return int(report['status']!='pass')


if __name__=='__main__':
    raise SystemExit(main())
