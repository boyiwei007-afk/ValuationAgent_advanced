from __future__ import annotations

import os

from valuationagent.core.data import LocalDataProvider
from valuationagent.market.tushare import TushareApiClient, TushareDataProvider
from valuationagent.market.infoway import InfowayApiClient, InfowayDataProvider


def create_history_provider():
    api_key = os.getenv("INFOWAY_API_KEY", "").strip()
    if api_key:
        return InfowayDataProvider(InfowayApiClient(api_key))
    token = (os.getenv("TUSHARE_TOKEN") or os.getenv("VALUATION_MARKET_DATA_TOKEN") or "").strip()
    return TushareDataProvider(TushareApiClient(token)) if token else None


def create_data_provider():
    token = os.getenv("TUSHARE_TOKEN") or os.getenv("VALUATION_MARKET_DATA_TOKEN")
    if token:
        return TushareDataProvider(TushareApiClient(token))
    return LocalDataProvider()
