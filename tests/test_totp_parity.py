"""`totp=` parity across every construction site in this package.

`ColonyToolkit` accepts `totp=` and forwards it. `ColonyEventPoller` and
`ColonyRetriever` did not — they hardcoded `ColonyClient(api_key=..., base_url=...)`.

That gap is not cosmetic. It took a four-agent rota down for over 24 hours on
2026-07-20: the operator wired a TOTP provider into their toolkit, 2FA was
enrolled, and every `POST /auth/token` from the POLLER started returning
401 "This account has 2FA enabled — supply totp_code". Actions worked; polling
did not. The capability existed in the package and the failing path could not
reach it.

The workaround is to inject a pre-built `client=`, which works and which every
consumer has had to discover independently.
"""

import pytest

from langchain_colony.events import ColonyEventPoller
from langchain_colony.retriever import ColonyRetriever


class _Captor:
    """Stands in for ColonyClient and records its construction kwargs."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def captured(monkeypatch):
    seen = {}

    def factory(**kwargs):
        seen.update(kwargs)
        return _Captor(**kwargs)

    monkeypatch.setattr("langchain_colony.events.ColonyClient", factory)
    monkeypatch.setattr("langchain_colony.retriever.ColonyClient", factory)
    return seen


class TestPollerTotp:
    def test_totp_is_forwarded_to_the_client(self, captured):
        provider = lambda: "123456"  # noqa: E731
        ColonyEventPoller(api_key="col_x", totp=provider)
        assert captured.get("totp") is provider

    def test_absent_totp_is_not_passed_at_all(self, captured):
        """Agents without 2FA must be unaffected — no stray totp=None kwarg."""
        ColonyEventPoller(api_key="col_x")
        assert "totp" not in captured

    def test_a_callable_is_not_invoked_at_construction(self, captured):
        """The SDK calls it per exchange; calling it here would burn a window."""
        calls = []
        ColonyEventPoller(api_key="col_x", totp=lambda: calls.append(1) or "123456")
        assert calls == [], "provider must be handed over uncalled"

    def test_totp_ignored_when_a_client_is_injected(self, captured):
        sentinel = object()
        poller = ColonyEventPoller(client=sentinel, totp=lambda: "123456")
        assert poller.client is sentinel
        assert captured == {}, "must not construct a client when one is given"


class TestRetrieverTotp:
    def test_totp_is_forwarded_to_the_client(self, captured):
        provider = lambda: "123456"  # noqa: E731
        ColonyRetriever(api_key="col_x", totp=provider)
        assert captured.get("totp") is provider

    def test_absent_totp_is_not_passed_at_all(self, captured):
        ColonyRetriever(api_key="col_x")
        assert "totp" not in captured
