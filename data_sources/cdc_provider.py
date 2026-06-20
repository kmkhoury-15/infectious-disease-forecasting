"""CDC open data providers.

This provider uses the Socrata Open Data API exposed by data.cdc.gov. Provide a
CDC dataset/resource ID and an optional Socrata query string.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from .base_provider import BaseDataProvider, ProviderResult


class CDCDataGovProvider(BaseDataProvider):
    """Fetch rows from data.cdc.gov Socrata datasets."""

    source_name = "cdc_data_gov"
    base_url = "https://data.cdc.gov/resource/{resource_id}.json"

    def fetch(
        self,
        resource_id: str,
        select: str | None = None,
        where: str | None = None,
        order: str | None = None,
        limit: int = 50000,
        app_token: str | None = None,
        timeout: int = 60,
        **kwargs: Any,
    ) -> ProviderResult:
        params: dict[str, Any] = {"$limit": int(limit)}
        if select:
            params["$select"] = select
        if where:
            params["$where"] = where
        if order:
            params["$order"] = order
        if app_token:
            params["$$app_token"] = app_token

        url = self.base_url.format(resource_id=resource_id)
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        records = response.json()
        df = pd.DataFrame.from_records(records)
        return ProviderResult(
            data=df,
            source_name=self.source_name,
            metadata={"resource_id": resource_id, "url": response.url, "row_count": len(df)},
        )


class CDCWonderProvider(BaseDataProvider):
    """Placeholder for CDC WONDER XML POST workflows.

    CDC WONDER is feasible but database-specific. The XML request body must be
    configured per WONDER database, group-by fields, measures, and suppression
    rules. This class is intentionally explicit so it is not confused with a
    generic all-purpose infectious disease case-count API.
    """

    source_name = "cdc_wonder"

    def fetch(self, *args, **kwargs) -> ProviderResult:
        raise NotImplementedError(
            "CDC WONDER support requires database-specific XML request templates. "
            "Use CDCDataGovProvider first, or subclass CDCWonderProvider for a specific WONDER database."
        )
