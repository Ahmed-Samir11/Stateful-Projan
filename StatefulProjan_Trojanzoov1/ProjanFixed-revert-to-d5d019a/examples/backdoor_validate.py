#!/usr/bin/env python3

import sys
import os
# Ensure the repository root is on sys.path so local packages (e.g., IBAU) are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trojanvision
from trojanvision.attacks import BadNet

from trojanvision.utils import summary
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    trojanvision.environ.add_argument(parser)
    trojanvision.datasets.add_argument(parser)
    trojanvision.models.add_argument(parser)
    trojanvision.marks.add_argument(parser)
    trojanvision.attacks.add_argument(parser)
    args = parser.parse_args()

    env = trojanvision.environ.create(**args.__dict__)
    dataset = trojanvision.datasets.create(**args.__dict__)
    model = trojanvision.models.create(dataset=dataset, **args.__dict__)

    # Normalize mark paths so both Windows-style (".\square.png") and Unix-style
    # relative paths resolve correctly (use repo root as fallback).
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _resolve_path(p: str) -> str:
        if p is None:
            return p
        # Replace backslashes with os.sep and strip surrounding quotes/spaces
        p = p.strip().strip('"').strip("'")
        p = p.replace('\\', os.sep)
        # absolute path is fine
        if os.path.isabs(p) and os.path.exists(p):
            return p
        # try as given relative path
        if os.path.exists(p):
            return os.path.abspath(p)
        # try repo root + given path
        cand = os.path.join(repo_root, p.lstrip('./\\'))
        if os.path.exists(cand):
            return cand
        # try repo root + basename
        cand2 = os.path.join(repo_root, os.path.basename(p))
        return cand2

    # Normalize main mark_path in parsed args before creating mark
    if hasattr(args, 'mark_path') and args.mark_path is not None:
        args.mark_path = _resolve_path(args.mark_path)

    mark = trojanvision.marks.create(dataset=dataset, **args.__dict__)

    # Handle extra marks (additional triggers) for multi-trigger attacks like prob/ntoone
    if 'extra_marks' in args and args.extra_marks is not None:
        # Normalize each extra mark's path before creating Watermark objects
        for m in args.extra_marks:
            if 'mark_path' in m and m['mark_path'] is not None:
                m['mark_path'] = _resolve_path(m['mark_path'])
        extra_marks = [trojanvision.marks.create(dataset=dataset, **m) for m in args.extra_marks]
    else:
        extra_marks = []
    marks = [mark] + extra_marks

    # For attacks that expect multiple marks, pass the marks list; otherwise pass single mark
    if args.attack_name in ['prob', 'ntoone']:
        attack: BadNet = trojanvision.attacks.create(dataset=dataset, model=model, marks=marks, **args.__dict__)
    else:
        attack: BadNet = trojanvision.attacks.create(dataset=dataset, model=model, mark=mark, **args.__dict__)

    if env['verbose']:
        summary(env=env, dataset=dataset, model=model, mark=mark, attack=attack)
    attack.load()
    attack.validate_fn()