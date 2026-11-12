"""查看原始资料摘要；导出分层胜比分析及可选的逐对明细。"""
import argparse
import json
import sys
from pathlib import Path

from .io import load, integer
from .domain import summarize
from .winratio import analyze


def _block_size(text):
    return integer(int(text), 1)


def _write_json(value, path):
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if path is None:
        sys.stdout.write(text + '\n')
    else:
        Path(path).write_text(text + '\n', encoding='utf8')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['inspect', 'analyze'])
    p.add_argument('file')
    p.add_argument('--pairs', metavar='PATH',
                   help='analyze 时把逐对明细以 JSON Lines 流式写入该路径')
    p.add_argument('--block-size', type=_block_size,
                   help='覆盖资料中的配对分块大小；不改变结果')
    p.add_argument('--output', metavar='PATH',
                   help='analyze 时把汇总 JSON 写入该路径而不是标准输出')
    a = p.parse_args()
    value, digest = load(a.file)
    if a.command == 'inspect':
        _write_json({'source_sha256': digest, 'summary': summarize(value)}, None)
        return

    sink = None
    stream = None
    if a.pairs:
        stream = Path(a.pairs).open('w', encoding='utf8')

        def sink(pair):
            stream.write(json.dumps(pair, ensure_ascii=False, allow_nan=False) + '\n')
    try:
        result = analyze(value, block_size=a.block_size, pair_sink=sink)
    finally:
        if stream is not None:
            stream.close()
    _write_json({'source_sha256': digest, 'analysis': result}, a.output)


if __name__ == '__main__':
    main()
