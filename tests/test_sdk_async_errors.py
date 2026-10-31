# QuantJourney Backtester
# Copyright (c) 2026 QuantJourney.
# Licensed under the Apache License 2.0.

"""Async SDK errors must keep the structured metadata the sync client keeps."""

from __future__ import annotations

import asyncio

import pytest

from backtester.sdk.client import APIError, AsyncAPIClient


class _RaisingTransport:
    async def request(self, *_args, **_kwargs):
        raise APIError(
            "forbidden",
            request_id="request-123",
            error_code="feature:backtester",
        )


def test_async_client_preserves_api_error_metadata() -> None:
    client = AsyncAPIClient("https://api.example.invalid", api_key="test")
    try:
        asyncio.run(client.client.aclose())
        client.client = _RaisingTransport()

        with pytest.raises(APIError) as excinfo:
            asyncio.run(client._request_with_method("GET", "/bt/prepare"))

        assert excinfo.value.error_code == "feature:backtester"
        assert excinfo.value.request_id == "request-123"
    finally:
        client.client = None
