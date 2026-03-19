"""Unit tests for provider client resolution."""

import reviewer.client as client


class FakeOpenAI:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def test_get_client_uses_custom_openai_base_url(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example/v1")
    monkeypatch.delenv("REVIEW_PROVIDER", raising=False)
    monkeypatch.setattr(client, "OpenAI", FakeOpenAI)

    sdk_client, provider, prefix = client.get_client(provider="openai")

    assert provider == "openai"
    assert prefix is None
    assert sdk_client.kwargs == {
        "api_key": "openai-key",
        "base_url": "https://openai.example/v1",
    }


def test_get_client_uses_custom_anthropic_base_url_for_model_prefix(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example/v1/")
    monkeypatch.delenv("REVIEW_PROVIDER", raising=False)
    monkeypatch.setattr(client, "OpenAI", FakeOpenAI)

    sdk_client, provider, prefix = client.get_client(model="anthropic/claude-opus-4-6")

    assert provider == "anthropic"
    assert prefix == "anthropic/"
    assert sdk_client.kwargs == {
        "api_key": "anthropic-key",
        "base_url": "https://anthropic.example/v1/",
    }


def test_get_client_falls_back_to_default_anthropic_base_url(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "anthropic-key")
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("REVIEW_PROVIDER", raising=False)
    monkeypatch.setattr(client, "OpenAI", FakeOpenAI)

    sdk_client, provider, prefix = client.get_client(provider="anthropic")

    assert provider == "anthropic"
    assert prefix == "anthropic/"
    assert sdk_client.kwargs == {
        "api_key": "anthropic-key",
        "base_url": "https://api.anthropic.com/v1/",
    }


def test_get_client_does_not_set_openai_base_url_by_default(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("REVIEW_PROVIDER", raising=False)
    monkeypatch.setattr(client, "OpenAI", FakeOpenAI)

    sdk_client, provider, prefix = client.get_client(provider="openai")

    assert provider == "openai"
    assert prefix is None
    assert sdk_client.kwargs == {"api_key": "openai-key"}


class FakeResponse:
    def __init__(self, content="ok"):
        self.usage = None
        self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return FakeResponse()


class FakeChatClient:
    def __init__(self):
        self.chat = type("Chat", (), {"completions": FakeCompletions()})()


def test_chat_uses_max_completion_tokens_for_openai(monkeypatch):
    sdk_client = FakeChatClient()
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(
        client,
        "get_client",
        lambda provider=None, model=None: (sdk_client, "openai", None),
    )

    content, usage = client.chat(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-5-mini",
        max_tokens=321,
        reasoning_effort="medium",
    )

    assert content == "ok"
    assert usage["completion_tokens"] == 0
    request = sdk_client.chat.completions.calls[0]
    assert request["max_completion_tokens"] == 321
    assert "max_tokens" not in request
    assert request["reasoning_effort"] == "medium"


def test_chat_keeps_max_tokens_for_non_openai_providers(monkeypatch):
    sdk_client = FakeChatClient()
    monkeypatch.setattr(
        client,
        "get_client",
        lambda provider=None, model=None: (sdk_client, "openrouter", None),
    )

    client.chat(
        messages=[{"role": "user", "content": "Hello"}],
        model="openai/gpt-5-mini",
        max_tokens=654,
        reasoning_effort="medium",
    )

    request = sdk_client.chat.completions.calls[0]
    assert request["max_tokens"] == 654
    assert "max_completion_tokens" not in request
    assert request["extra_body"] == {"reasoning": {"max_tokens": 1024}}


def test_chat_uses_max_tokens_for_openai_custom_base_url(monkeypatch):
    sdk_client = FakeChatClient()
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example/v1")
    monkeypatch.setattr(
        client,
        "get_client",
        lambda provider=None, model=None: (sdk_client, "openai", None),
    )

    client.chat(
        messages=[{"role": "user", "content": "Hello"}],
        model="gpt-5-mini",
        max_tokens=222,
        reasoning_effort="medium",
    )

    request = sdk_client.chat.completions.calls[0]
    assert request["max_tokens"] == 222
    assert "max_completion_tokens" not in request
    assert request["reasoning_effort"] == "medium"


def test_chat_skips_anthropic_thinking_for_custom_base_url(monkeypatch):
    sdk_client = FakeChatClient()
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example/v1")
    monkeypatch.setattr(
        client,
        "get_client",
        lambda provider=None, model=None: (sdk_client, "anthropic", "anthropic/"),
    )

    client.chat(
        messages=[{"role": "user", "content": "Hello"}],
        model="claude-opus-4-6",
        max_tokens=333,
        reasoning_effort="medium",
    )

    request = sdk_client.chat.completions.calls[0]
    assert request["max_tokens"] == 333
    assert "extra_body" not in request
