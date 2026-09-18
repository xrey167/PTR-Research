"""Recover literal heredoc bodies from a saved share page, without executing shells.

Fragments retain revisions and append operations separately. They are evidence,
not an assertion of byte identity with the missing original archive.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shlex


HEADER = re.compile(r'cat\s+(>>?)\s*([^\s]+)\s*<<\s*[\x27\x22]?([A-Za-z_][A-Za-z_0-9]*)[\x27\x22]?\s*\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('html', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.html.read_bytes()
    encoded, _ = json.JSONDecoder().raw_decode(source.decode('utf-8').split('streamController.enqueue(', 1)[1])
    data = json.loads(encoded)
    args.output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, command in enumerate(data):
        if not isinstance(command, str):
            continue
        decoded_command = command
        # Parse only the known transport wrapper. shlex never executes commands.
        if command.startswith('bash -lc /bin/bash -lc '):
            try:
                words = shlex.split(command, posix=True)
                if len(words) == 5 and words[:4] == ['bash', '-lc', '/bin/bash', '-lc']:
                    decoded_command = words[4]
            except ValueError:
                pass
        for ordinal, match in enumerate(HEADER.finditer(decoded_command)):
            tail = decoded_command[match.end():]
            end = re.search(r'^' + re.escape(match.group(3)) + r'\s*$', tail, re.MULTILINE)
            row = {'entry':index, 'ordinal':ordinal, 'original_path':match.group(2),
                   'operation':'append' if match.group(1) == '>>' else 'replace',
                   'transport_unquoted':decoded_command != command,
                   'command_sha256':hashlib.sha256(command.encode()).hexdigest()}
            if end is None:
                row['status'] = 'missing_exact_delimiter'
                rows.append(row)
                continue
            body = tail[:end.start()]
            # Never use source paths as local destinations.
            suffix = Path(match.group(2)).suffix
            name = f'entry_{index}_{ordinal}' + (suffix if suffix in ['.py', '.md', '.json', '.txt'] else '.txt')
            (args.output / name).write_text(body, encoding='utf-8')
            row.update(status='literal_fragment', local_file=name, bytes=len(body.encode()),
                       sha256=hashlib.sha256(body.encode()).hexdigest())
            if suffix == '.py':
                try:
                    tree = ast.parse(body)
                    row['syntax'] = 'valid'
                    row['imports'] = sorted({n.module or '' for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} |
                                            {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names})
                except SyntaxError as exc:
                    row['syntax'] = 'invalid'
                    row['syntax_error'] = str(exc)
            rows.append(row)
    manifest = {'source':str(args.html), 'source_sha256':hashlib.sha256(source).hexdigest(),
                'archive_identity_verified':False, 'shell_commands_executed':False, 'fragments':rows}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# Recovered literal fragments', '', 'These are separate historical fragments, not a restored final source tree.',
             'No shell commands were executed. Original archive identity is unverified.', '',
             '| Source entry | Original path | Operation | Local fragment | Syntax |', '| --- | --- | --- | --- | --- |']
    for row in rows:
        local = row.get('local_file')
        lines.append(f"| {row['entry']} | {row['original_path']} | {row['operation']} | " +
                     (f'[{local}]({local})' if local else row['status']) + f" | {row.get('syntax', '')} |")
    (args.output / 'INDEX.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'fragments':len(rows), 'recovered':sum('local_file' in r for r in rows),
                      'python_valid':sum(r.get('syntax') == 'valid' for r in rows),
                      'python_invalid':sum(r.get('syntax') == 'invalid' for r in rows)}))


if __name__ == '__main__': main()
