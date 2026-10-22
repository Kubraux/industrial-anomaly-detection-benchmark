from pathlib import Path
import argparse,csv,hashlib,json,re
P=Path(__file__).resolve().parent
def rows(name):
 with (P/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def check():
 ap=argparse.ArgumentParser()
 ap.add_argument('--backup-dir',type=Path,metavar='DIRECTORY',help='Original archive to verify; relative paths resolve from the current working directory')
 args=ap.parse_args()
 if args.backup_dir is not None:
  args.backup_dir=args.backup_dir.expanduser().resolve()
  if not args.backup_dir.is_dir():ap.error('--backup-dir must name an existing directory')
 manifest=rows('provenance/files.csv')
 expected={r['path'] for r in manifest}|{'provenance/files.csv'}
 actual={p.relative_to(P).as_posix() for p in P.rglob('*') if p.is_file()}
 assert actual==expected,('File set changed',actual^expected)
 for r in manifest:assert sha(P/r['path'])==r['sha256'],r['path']
 for r in rows('provenance/copy_manifest.csv'):assert sha(P/r['destination'])==r['sha256'],r['destination']
 catalog=rows('SOURCES.csv')
 assert catalog,'SOURCES.csv must contain source records'
 ids=[r['method_id'].strip() for r in catalog]
 assert all(ids) and len(set(ids))==len(ids),'method_id values must be nonempty and unique'
 assert all(r['method'].strip() for r in catalog),'method values must be nonempty'
 record_dirs=[r['record_dir'] for r in catalog]
 assert len(set(record_dirs))==len(record_dirs),'Each source record must have its own record_dir'
 status_dirs={p.parent.relative_to(P).as_posix() for p in (P/'records').rglob('STATUS.md')}
 assert set(record_dirs)==status_dirs,('Catalog/status record mismatch',set(record_dirs)^status_dirs)
 for r in catalog:
  assert r['recorded_commit']=='unknown' or re.fullmatch('[0-9a-f]{40}',r['recorded_commit'])
  for key in ['license_file','record_dir']:
   if r[key]:assert (P/r[key]).exists(),r[key]
  assert (P/r['record_dir']).resolve().is_relative_to(P/'records'),r['record_dir']
  assert (P/r['record_dir']/'STATUS.md').is_file()
  expected_commit_url=r['repository_url']+'/commit/'+r['recorded_commit'] if r['repository_url'] and r['recorded_commit']!='unknown' else ''
  assert r['commit_url']==expected_commit_url,('Inconsistent commit URL',r['method_id'])
  up=P/r['record_dir']/'upstream.json'
  if up.exists():
   old=json.loads(up.read_text(encoding='utf-8'))
   assert r['method_id']==old['method'],('Inconsistent source identity',r['method_id'])
   assert r['repository_url']==old['url'] and r['recorded_commit']==old['commit']
  for key in ['repository_url','commit_url','adaptation_reference']:
   if r[key]:assert r[key].startswith('https://github.com/')
 for p in P.rglob('*.md'):
  for link in re.findall(r'\]\(([^)]+)\)',p.read_text(encoding='utf-8')):
   if '://' in link or link.startswith('#'):continue
   q=(p.parent/link.split('#')[0]).resolve()
   assert q.is_relative_to(P) and q.exists(),(p,link)
 if args.backup_dir is not None:
  a=args.backup_dir;records=rows('provenance/archive_inventory.csv')
  assert {p.relative_to(a).as_posix() for p in a.rglob('*') if p.is_file()}=={r['archive_path'] for r in records}
  for r in records:assert sha(a/r['archive_path'])==r['sha256'],r['archive_path']
 print(json.dumps(dict(passed=True,files=len(actual),source_records=len(catalog),methods=len({r['method'] for r in catalog}),backup_checked=args.backup_dir is not None,research_code_executed=False)))
if __name__=='__main__':check()
