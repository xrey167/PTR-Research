"""Audit raw reader outputs against the typed Pod answer barrier."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.reader_answer_guard import guarded_answer


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--cases', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.evaluation / 'report.json').read_text(encoding='utf-8'))
    cases = {row['id']: row for row in json.loads(args.cases.read_text(encoding='utf-8'))}
    rows = []
    for result in report['rows']:
        row = cases[result['id']]
        guarded = guarded_answer(row, result['text'])
        rows.append({'id': result['id'], 'raw': result['text'], 'guarded': guarded,
                     'raw_exact': result['text'] == row['target'], 'guarded_exact': guarded == row['target']})
    output = {'schema': 'reader-answer-guard-audit:v1', 'evaluation_report_sha256': sha(args.evaluation / 'report.json'),
              'cases_sha256': sha(args.cases), 'rows': rows,
              'raw_exact': sum(x['raw_exact'] for x in rows),
              'guarded_exact': sum(x['guarded_exact'] for x in rows), 'total': len(rows),
              'full_research_goal_complete': False}
    (args.evaluation / 'guard-audit.json').write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: output[k] for k in ('raw_exact', 'guarded_exact', 'total')}))


if __name__ == '__main__':
    main()
