#!/usr/bin/env python3
"""Write an atomic, bounded supervisor handoff without guessing benchmark success."""
import json
import os
from pathlib import Path
import sys
import time


def main(argv):
    run_dir, status, rc, payload, args, session, project, started = argv
    run = Path(run_dir)
    root = Path(project)
    result = None
    result_error = None
    path = run / 'result.json'
    if path.exists():
        try:
            if path.stat().st_size > 32768:
                raise ValueError('result.json exceeds 32 KiB; retrieve relevant fields explicitly')
            result = json.loads(path.read_text())
            if not isinstance(result, dict):
                raise ValueError('result.json must be an object')
            if len(json.dumps(result).encode()) > 4096:
                result = None
                result_error = 'Result exceeds inline budget; read selected fields from result_file.'
        except (ValueError, OSError) as error:
            result_error = str(error)
    # Copy only new RESULTS bytes, retaining the complete delta separately.
    delta_path = run / 'results-added.md'
    offset = int(os.environ.get('RESULTS_OFFSET', '0'))
    if status != 'prepared' and 'RESULTS_OFFSET' in os.environ and (root / 'RESULTS.md').exists():
        with (root / 'RESULTS.md').open('rb') as source, delta_path.open('wb') as target:
            if source.seek(0, 2) >= offset:
                source.seek(offset)
                while chunk := source.read(65536):
                    target.write(chunk)
    data = {
        'run_id': run.name, 'status': status, 'payload_rc': int(rc),
        'elapsed_seconds': max(0, int(time.time()) - int(started)) if int(started) else None,
        'session_id': session, 'payload': payload, 'payload_args': args,
        'payload_log': str(run / 'payload.log'),
        'results_added': str(delta_path) if delta_path.exists() else None,
        'result_file': str(path) if path.exists() else None,
        'payload_result': result, 'result_error': result_error,
        'validation': 'Review payload_result and run artifacts; exit status alone is not correctness or a speedup.',
        'next_action': 'Review the run result and continue the pending action; resolve failures before another run.',
    }
    temporary = run / 'summary.json.tmp'
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(run / 'summary.json')


if __name__ == '__main__':
    main(sys.argv[1:])
