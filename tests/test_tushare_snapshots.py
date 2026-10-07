import json
from datetime import date

import httpx
import pytest

from valuationagent.market.tushare import TushareApiClient, TushareDataProvider, TushareApiError


def client_for(raw):
    return TushareApiClient("synthetic-test-secret", transport=httpx.MockTransport(lambda request: httpx.Response(200, content=raw)))


def test_snapshot_keeps_exact_response_bytes_and_decimal_precision():
    raw = b'{"code":0,"data":{"fields":["amount","missing"],"items":[[123456789012.123456789,null]]}}'
    snapshot = client_for(raw).query_snapshot("income")
    assert snapshot.raw == raw
    assert snapshot.records == [{"amount": "123456789012.123456789", "missing": None}]
    assert client_for(raw).query("income") == snapshot.records


@pytest.mark.parametrize("fields,items", [(["same", "same"], [[1, 2]]), (["value"], [[1, 2]]), (["value"], [[]]), ([123], [[1]])])
def test_vendor_column_errors_are_not_silently_truncated(fields, items):
    raw = json.dumps({"code": 0, "data": {"fields": fields, "items": items}}).encode()
    with pytest.raises(TushareApiError, match="列名或行宽无效"):
        client_for(raw).query_snapshot("income")


def test_response_containing_credential_is_never_saved_or_echoed():
    raw = b'{"code":0,"msg":"synthetic-test-secret","data":{}}'
    with pytest.raises(ValueError) as failure:
        client_for(raw).query_snapshot("income")
    assert "synthetic-test-secret" not in str(failure.value)


def test_market_snapshot_dates_units_and_original_positions_are_preserved():
    fields = ["ts_code", "trade_date", "total_share", "total_mv", "ps_ttm"]
    rows = [["600123.SH", "20260930", "12345.6789", "90000.1234", "2.1"],
        ["600123.SH", "20261010", "12345", "90000", "2"],
        ["000999.SZ", "20260930", "12345", "90000", "2"]]
    raw = json.dumps({"code": 0, "data": {"fields": fields, "items": rows}}).encode()
    result = TushareDataProvider(client_for(raw)).fetch_history("600123.SH", "statistics", [2025], date(2026, 10, 3))
    assert result.raw == raw and result.accepted_records == 1 and result.excluded_records == 2
    block = result.blocks[0]
    assert "行情日期 2026-09-30" in block["text"] and "2025" not in block["text"]
    assert "总股本 | 单位：万股 | 12345.6789" in block["text"]
    assert "总市值 | 单位：万元 | 90000.1234" in block["text"]
    assert "市销率TTM" in block["text"]
    assert block["location"]["column_pointers"]["total_share"] == "/data/items/0/2"
    assert block["location"]["date_semantics"] == "market_data_as_of_close"
    assert result.catalog["record_columns_path"] == "/data/fields"


def test_financial_catalog_surfaces_only_available_complete_annual_periods():
    fields = ["ts_code", "ann_date", "end_date", "report_type", "revenue"]
    rows = [["600123.SH", "20250401", "20241231", "1", "100"],
        ["600123.SH", "20260401", "20251231", "1", "110"],
        ["600123.SH", "20260801", "20260630", "1", "50"],
        ["600123.SH", "20270401", "20261231", "1", "120"],
        ["600124.SH", "20240401", "20231231", "1", "90"],
        ["600123.SH", "20240401", "20231231", "4", "90"]]
    raw = json.dumps({"code": 0, "data": {"fields": fields, "items": rows}}).encode()
    result = TushareDataProvider(client_for(raw)).fetch_history("600123.SH", "income", [2024], date(2026, 10, 3))
    assert result.raw == raw and result.accepted_records == 1
    assert result.catalog["available_annual_periods"] == ["2025-12-31", "2024-12-31"]
    assert result.catalog["requested_years"] == [2024]
    assert {block["location"]["period_end"] for block in result.blocks} == {"2024-12-31"}
