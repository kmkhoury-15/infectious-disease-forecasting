"""ADHS public data portal provider.

ADHS public pages often expose dashboards, reports, or downloadable files rather
than one stable API pattern. This provider supports a direct public CSV/XLSX URL
first, which keeps the forecasting pipeline reproducible and avoids scraping
fragile dashboard markup.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
import requests

from .base_provider import BaseDataProvider, ProviderResult


class ADHSPublicFileProvider(BaseDataProvider):
    """Fetch a public ADHS CSV or Excel file by URL."""

    source_name = "adhs_public_file"

    def fetch(self, url: str, timeout: int = 60, **kwargs: Any) -> ProviderResult:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").lower()
        suffix = url.lower().split("?")[0]

        if "csv" in content_type or suffix.endswith(".csv"):
            df = pd.read_csv(BytesIO(response.content))
        elif any(suffix.endswith(ext) for ext in [".xlsx", ".xls"]):
            df = pd.read_excel(BytesIO(response.content))
        else:
            raise ValueError(
                "URL did not look like a CSV/XLS/XLSX file. Use a direct download link "
                "or add a custom ADHS provider for the target dashboard."
            )

        return ProviderResult(data=df, source_name=self.source_name, metadata={"url": url, "row_count": len(df)})
