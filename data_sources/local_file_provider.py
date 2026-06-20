"""Local file provider for CSV/XLSX/XLS data."""

from __future__ import annotations

from src.data_loader import load_case_data, normalize_column_names

from .base_provider import BaseDataProvider, ProviderResult


class LocalFileProvider(BaseDataProvider):
    """Load local user-provided case data."""

    source_name = "local_file"

    def fetch(self, path: str, **kwargs) -> ProviderResult:
        df = normalize_column_names(load_case_data(path))
        return ProviderResult(data=df, source_name=self.source_name, metadata={"path": path})
