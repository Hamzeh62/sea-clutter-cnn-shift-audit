#!/usr/bin/env python3
"""Verify release-staging integrity against the included SHA256 manifest."""
import csv,hashlib,sys
from pathlib import Path
r=Path(__file__).resolve().parents[1]
problems=[];checked=0
with (r/'RELEASE_SHA256.csv').open(newline='',encoding='utf-8') as f:
 for row in csv.DictReader(f):
  p=r/row['relative_path']
  if not p.is_file():problems.append('MISSING '+row['relative_path']);continue
  h=hashlib.sha256(p.read_bytes()).hexdigest()
  if h!=row['sha256'] or p.stat().st_size!=int(row['bytes']):
   problems.append('MISMATCH '+row['relative_path'])
  checked+=1
print(f'Release checks: {checked} entries, {len(problems)} mismatches')
for issue in problems:print(issue)
sys.exit(1 if problems else 0)
