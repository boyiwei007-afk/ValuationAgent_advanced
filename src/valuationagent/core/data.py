"""Replaceable data boundary. Only explicit demo requests receive synthetic facts."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Protocol
from pydantic import Field
from valuationagent.schemas.models import (
    ApiModel,
    AssumptionInputs,
    CompanyInput,
    EvidenceRef,
    FinancialSnapshot,
    PeerCompany,
    ValuationRequest,
)
from valuationagent.storage.sqlite import SQLiteRunStore


class DataBundle(ApiModel):
    company: CompanyInput | None = None
    financials: FinancialSnapshot
    historical_financials: list[FinancialSnapshot] = Field(default_factory=list)
    peers: list[PeerCompany]
    assumptions: AssumptionInputs
    assumption_evidence: dict[str, list[EvidenceRef]] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class DataProvider(Protocol):
    version: str

    def resolve(
        self, request: ValuationRequest, store: SQLiteRunStore
    ) -> DataBundle: ...


def demo_financials() -> FinancialSnapshot:
    return FinancialSnapshot(
        period_end="2025-12-31",
        revenue="10000000000",
        ebit_margin="0.16",
        tax_rate="0.25",
        depreciation_amortization="400000000",
        capital_expenditure="600000000",
        change_operating_nwc="180000000",
        cash_and_non_operating_assets="1600000000",
        interest_bearing_debt="900000000",
        common_shares="500000000",
        net_income_parent="1050000000",
        ebitda="2000000000",
        source_label="synthetic_demo_data",
    )


def demo_peers() -> list[PeerCompany]:
    return [
        PeerCompany(
            ticker=f"DEMO-{i + 1}",
            name=f"示例同业 {i + 1}",
            pe=str(pe),
            ev_ebitda=str(ev),
            rationale="synthetic demo",
        )
        for i, (pe, ev) in enumerate(
            [(18, 11), (21, 12.5), (24, 14), (27, 15.5), (30, 17)]
        )
    ]


class LocalDataProvider:
    version = "local-0.2"

    @staticmethod
    def read_json(
        store: SQLiteRunStore, ids: list[str], role: str
    ) -> tuple[dict, dict]:
        if len(ids) != 1:
            raise ValueError("当前结构化导入需要且仅需要一个对应角色的 JSON 文件。")
        meta = store.get_file(ids[0])
        if meta["role"] != role:
            raise ValueError("文件角色不匹配，请分别上传历史财务与假设。")
        path = Path(meta["storage_path"])
        if path.suffix.lower() != ".json":
            raise NotImplementedError(
                "PDF/Excel 解析器尚未接入；请复核后补充结构化 JSON。"
            )
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta["sha256"]:
            raise ValueError("原始文件哈希不一致，请重新上传。")
        data = json.loads(raw, parse_float=str)
        if not isinstance(data, dict):
            raise ValueError("JSON 文件顶层必须为对象。")
        return data, meta

    def resolve(self, request: ValuationRequest, store: SQLiteRunStore) -> DataBundle:
        financials = request.financials
        peers = request.peers
        if request.mode == "demo":
            if (
                request.data_source != "structured"
                or request.file_ids
                or request.historical_financials
            ):
                raise ValueError(
                    "demo 只接受合成结构化输入；真实文件/代码请使用 snapshot 或 live。"
                )
            if (
                financials is not None
                and financials.source_label != "synthetic_demo_data"
            ):
                raise ValueError("demo 不能混入真实财务数据，请切换 snapshot 或 live。")
            financials = financials or demo_financials()
            peers = peers or demo_peers()
        elif request.data_source == "ticker":
            raise NotImplementedError(
                "A股在线取数需要配置TUSHARE_TOKEN；也可以改用结构化数据或上传资料。"
            )
        elif request.data_source == "upload":
            payload, meta = self.read_json(
                store, request.file_ids, "historical_financials"
            )
            financials = FinancialSnapshot.model_validate(
                payload.get("financials", payload)
            )
            financials.source_label = meta["original_name"]
            for field in FinancialSnapshot.model_fields:
                if field in (payload.get("financials", payload)):
                    financials.evidence.setdefault(field, []).append(
                        EvidenceRef(
                            evidence_id=f"{meta['file_id']}:{field}",
                            source=meta["original_name"],
                            file_id=meta["file_id"],
                            note=f"JSON path: financials.{field}",
                        )
                    )
            if "peers" in payload:
                if peers:
                    raise ValueError("文件与请求均包含可比公司，请保留一个明确来源。")
                peers = [PeerCompany.model_validate(item) for item in payload["peers"]]
        if financials is None and request.historical_financials:
            eligible = [
                f
                for f in request.historical_financials
                if f.period_end <= request.valuation_date
                and (f.published_at is None or f.published_at <= request.valuation_date)
            ]
            financials = max(eligible, key=lambda f: f.period_end) if eligible else None
        if financials is None:
            raise ValueError("缺少结构化财务快照。请通过复核补充 financials。")
        assumptions = request.assumptions
        evidence = dict(request.assumption_evidence)
        if request.assumption_source == "upload":
            payload, meta = self.read_json(
                store, request.assumption_file_ids, "assumptions"
            )
            if assumptions.model_dump(exclude_none=True, exclude_defaults=True):
                raise ValueError(
                    "文件假设与手工假设不能同时生效，请通过复核选择一个来源。"
                )
            assumptions = AssumptionInputs.model_validate(
                payload.get("assumptions", payload)
            )
            for key in assumptions.model_dump(exclude_none=True, exclude_defaults=True):
                evidence[key] = [
                    EvidenceRef(
                        evidence_id=f"{meta['file_id']}:{key}",
                        source=meta["original_name"],
                        file_id=meta["file_id"],
                        note=f"JSON path: assumptions.{key}",
                    )
                ]
        elif request.assumption_source == "manual" and not assumptions.model_dump(
            exclude_none=True, exclude_defaults=True
        ):
            raise ValueError("已选择手工假设，但尚未填写任何假设。")
        return DataBundle(
            company=request.company,
            financials=financials,
            historical_financials=request.historical_financials,
            peers=peers,
            assumptions=assumptions,
            assumption_evidence=evidence,
        )
