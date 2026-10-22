"""Read-only training observer: never modifies training state or checkpoints."""
import argparse
import csv
import ctypes
from ctypes import wintypes
from datetime import datetime
import io
import json
import os
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'mvtec_official_cpu'
EPOCH = re.compile(r'(\w+) epoch (\d+)/(\d+) metrics=\[([^]]+)\].*?elapsed_s=([\d.]+)')
FIELDS = ['class', 'epoch', 'total_epochs', 'I_AUROC', 'P_AUROC', 'PRO_official', 'epoch_seconds', 'cumulative_seconds']

def read_shared(path):
    # Windows training redirects keep these files open for writing.
    if os.name != 'nt':
        return path.read_text(encoding='utf-8', errors='replace')
    import msvcrt
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(path), 0x80000000, 7, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    with os.fdopen(fd, 'rb') as f:
        data = f.read()
    encoding = 'utf-16' if data.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    return data.decode(encoding, errors='replace')

def alive(pid):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x1000, False, int(pid))
    if not handle:
        return 'unknown' if ctypes.get_last_error() == 5 else 'stopped'
    try:
        code = wintypes.DWORD()
        return ('running' if code.value == 259 else 'stopped') if kernel.GetExitCodeProcess(handle, ctypes.byref(code)) else 'unknown'
    finally:
        kernel.CloseHandle(handle)

def collect_epochs():
    records = {}
    paths = [OUT / 'run.log'] + sorted(OUT.glob('resume_*.stdout.log'), key=lambda p:p.stat().st_mtime)
    for path in paths:
        if not path.exists(): continue
        for m in EPOCH.finditer(read_shared(path)):
            name, epoch, total, metrics, elapsed = m.groups()
            values = json.loads('[' + metrics + ']')
            records[(name, int(epoch))] = dict(zip(FIELDS, [name, int(epoch), int(total), *values, '', float(elapsed)]))
    rows = list(records.values())
    for row in rows:
        previous = records.get((row['class'], row['epoch'] - 1))
        if previous or row['epoch'] == 1:
            row['epoch_seconds'] = round(row['cumulative_seconds'] - (previous['cumulative_seconds'] if previous else 0), 1)
    return rows

def snapshot():
    state = json.loads(read_shared(OUT / 'status.json'))
    rows = collect_epochs()
    logs = sorted(OUT.glob('resume_*.stderr.log'), key=lambda p:p.stat().st_mtime)
    stderr = logs[-1] if logs else OUT / 'stderr.log'
    raw = read_shared(stderr)
    segment = raw.rsplit('Training discriminator...', 1)[-1]
    inner = list(re.finditer(r'epoch:(\d+) loss:([\d.eE+-]+)[^\r\n]*?\|\s*(\d+)/4', segment))
    completed = max((int(m[3]) for m in inner), default=0)
    loss = inner[-1][2] if inner else 'pending'
    phase = 'Training'
    detail = f'Completed inner training rounds: {completed}/4; latest loss: {loss}'
    if 'Inferring...' in segment:
        phase = 'Test inference'
        batches = re.findall(r'Inferring[^\r\n]*?\|\s*(\d+)/(\d+)', segment)
        if batches:
            done, total = batches[-1]
            detail = f'Test batches: {done}/{total}'
            if done == total:
                phase = 'Inference finished; metrics/checkpoint may still be running'
    process = alive(state.get('pid', 0))
    current = [r for r in rows if r['class'] == state.get('class_name')]
    last = current[-1] if current else None
    if last and last['epoch'] >= state.get('epoch', 0):
        phase = 'Epoch recorded; awaiting next status update'
    if state['status'] != 'training': phase = state['status']
    if process == 'stopped' and state['status'] == 'training': phase = 'PROCESS STOPPED - status file is stale'
    label = f"SimpleNet | {state.get('class_name', '-')} | MAIN EPOCH {state.get('epoch', '-')}/{state.get('total_epochs', 40)}"
    lines = [label, '=' * 76, f"Updated: {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"Training process: {process} (PID {state.get('pid')})",
             f"Completed classes: {state.get('completed_classes', 0)}/15", f'Phase: {phase}', detail,
             'Inner rounds (0/4..4/4) are NOT the main epoch number.', '']
    if last:
        lines += [f"Last completed main epoch: {last['epoch']}/{last['total_epochs']}",
                  f"I-AUROC: {last['I_AUROC']:.2%} | P-AUROC: {last['P_AUROC']:.2%} | PRO: {last['PRO_official']:.2%}"]
        if last['epoch_seconds'] != '': lines += [f"Last completed epoch duration (incl. evaluation): {last['epoch_seconds']/60:.1f} min"]
    lines += ['', f"Last raw log write: {datetime.fromtimestamp(stderr.stat().st_mtime):%H:%M:%S}",
              'Raw logs update after inner rounds, not every batch.',
              'Closing this monitor does not stop training.',
              'Files: live_status.txt | epoch_metrics.csv | training_progress.log']
    key = (label, process, phase, detail, len(rows))
    return '\n'.join(lines) + '\n', rows, key, label

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    previous = None
    previous_csv = None
    while True:
        try:
            text, rows, key, label = snapshot()
            buffer = io.StringIO(newline='')
            writer = csv.DictWriter(buffer, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(rows)
            csv_text = buffer.getvalue()
            if csv_text != previous_csv:
                (OUT/'epoch_metrics.csv').write_text(csv_text, encoding='utf-8')
                previous_csv = csv_text
            (OUT/'live_status.txt').write_text(text, encoding='utf-8')
            if key != previous:
                with (OUT/'training_progress.log').open('a', encoding='utf-8') as log:
                    log.write(text + '\n')
                previous = key
            if not args.once:
                ctypes.windll.kernel32.SetConsoleTitleW(label)
                os.system('cls')
            print(text, flush=True)
            if args.once: return
        except (OSError, ValueError) as exc:
            print(f'Monitor read/write retry: {exc}', flush=True)
            if args.once: raise
        time.sleep(5)

if __name__ == '__main__': main()
