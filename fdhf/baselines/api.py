"""Proprietary / API-only baselines (manuscript Table 6): TimeGPT and GPT4TS.

These require paid third-party APIs and CANNOT be reproduced offline. They are
provided only as explicit stubs that raise -- never as silently faked results.
To enable them, implement the API calls and supply credentials.
"""

from __future__ import annotations


class _APIForecaster:
    name = "api"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            f"{self.name} is a proprietary API baseline (manuscript Table 6) and "
            "cannot be reproduced offline. Provide an API client + credentials, "
            "or drop it from the comparison. The repository deliberately does not "
            "ship a fake implementation."
        )


class TimeGPT(_APIForecaster):
    name = "TimeGPT"


class GPT4TS(_APIForecaster):
    name = "GPT4TS"
