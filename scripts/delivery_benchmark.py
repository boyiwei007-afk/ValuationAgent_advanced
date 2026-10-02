"""Synthetic local benchmarks; writes only to the explicitly named output dir."""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path

from fastapi.testclient import TestClient
from openpyxl import Workbook
from valuationagent.api.main import create_app
from valuationagent.core.documents import parse_document


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('var/delivery-benchmark'))
    args = parser.parse_args()
    app = create_app(args.output)
    service, store = app.state.workspaces, app.state.store
    result = {"scope": "合成负载、本机耗时，不是生产SLA", "excel": [], "snapshots": []}
    for n in (100, 400, 800):
        book = Workbook()
        for i in range(n):
            book.active.append([i, f'=A{i+1}*2'])
        buffer = BytesIO()
        book.save(buffer)
        meta = store.save_upload(f'formula-{n}.xlsx', 'historical_financials', 'application/octet-stream', buffer.getvalue())
        durations = []
        for _ in range(3):
            start = time.perf_counter()
            blocks, warnings = parse_document(store.get_file(meta['file_id']))
            durations.append(time.perf_counter() - start)
        result['excel'].append({"rows": n, "median_seconds": round(statistics.median(durations), 4), "blocks": len(blocks)})
    with TestClient(app) as client:
        for n in (25, 250, 1000):
            workspace = service.create()
            sid = workspace.research_session_id
            for _ in range(n):
                store.append_event(sid, type='tool.completed', stage='research', status='completed', summary='synthetic event', payload={'text': 'x' * 20000})
            url = f'/api/workspaces/{workspace.workspace_id}'
            durations = []
            for _ in range(5):
                start = time.perf_counter()
                response = client.get(url)
                response.raise_for_status()
                durations.append((time.perf_counter() - start) * 1000)
            result['snapshots'].append({'events': n, 'bytes': len(response.content), 'median_ms': round(statistics.median(durations), 2)})
        def concurrent(_):
            start = time.perf_counter()
            response = client.get(url)
            return {'status': response.status_code, 'ms': round((time.perf_counter() - start) * 1000, 2)}
        with ThreadPoolExecutor(max_workers=8) as executor:
            result['concurrency_8'] = list(executor.map(concurrent, range(8)))
    (args.output / 'metrics.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
