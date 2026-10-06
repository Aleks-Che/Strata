"""Make reproducible offline MiMo draft comparisons (greedy, no thinking)."""
import argparse
import json
from pathlib import Path
from .gguf_reader import GGUFFile
from .mimo2_template import renderer
from .mimo2_tokenizer import from_gguf


QUESTIONS = [
    ('count', 'Count from 1 to 20 separated by commas. Start immediately.'),
    ('code', 'Write a Python function that returns the unique elements of a list, preserving their order. Include a short example.'),
    ('ru', 'Объясни простыми словами, почему на Земле сменяются времена года. Ответь по-русски, тремя предложениями.'),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suite', choices=['corpus', 'tune', 'boundary'], default='corpus')
    p.add_argument('--depth', type=int, default=0, choices=range(8))
    p.add_argument('--p-min', type=float, default=0)
    a = p.parse_args()
    assert a.output.suffix == '.json' and not a.output.exists()
    assert 0 <= a.p_min <= 1
    tokenizer = from_gguf(a.model)
    template = renderer(GGUFFile(a.model).metadata['tokenizer.chat_template'])
    questions = QUESTIONS if a.suite == 'corpus' else QUESTIONS[:1]
    if a.suite == 'boundary':
        questions = [('swa128', 'Remember this list: '+', '.join(str(i) for i in range(40))+
                      '. Repeat the numbers from 30 to 39 separated by commas.')]
    requests = []
    for name, text in questions:
        prompt = template.render(messages=[dict(role='user', content=text)], enable_thinking=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, True)
        if a.suite == 'boundary': assert len(ids) > 128
        settings = [(a.depth, a.p_min)] if a.suite != 'tune' else [(1, 0.), (3, 0.), (7, 0.), (7, .7)]
        for depth, pmin in settings:
            for repeat in range(4):
                requests.append(dict(name=f'{name}-d{depth}-p{pmin}-r{repeat}', prompt_text=text,
                    tokens=ids, predict=32, depth=depth, p_min=pmin, warmup=repeat == 0))
    a.output.write_text(json.dumps(requests, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(len(requests), 'requests;', a.output)


if __name__ == '__main__': main()
