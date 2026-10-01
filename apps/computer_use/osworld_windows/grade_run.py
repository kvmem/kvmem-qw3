"""Host-only grading entry point for the qualified evaluator environment."""
import argparse
import json
from pathlib import Path

from .scenarios import IMPLEMENTED, grade


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=IMPLEMENTED, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    result = grade(args.task, args.directory)
    with (args.directory / 'independent-grade.json').open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
    print(json.dumps({'task': args.task, 'strict_success': result.get('strict_success')}))
