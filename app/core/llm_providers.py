"""Unified LLM provider layer — supports multiple vendors + multiple models.

Design goals:
    * One shape for all completions — caller doesn't know if it's Gemini,
      Claude, or OpenAI under the hood.
    * Each configured model has a ROLE tag: ``primary`` (first opinion),
      ``verifier`` (second opinion on low-confidence primaries),
      ``critic`` (adversarial check), ``chat`` (interactive follow-ups).
    * Config stored in ``config.yaml:llm_providers`` as a list — UI (Settings)
      can add/remove/reorder without code changes.

Config schema (config.yaml):
    llm_providers:
      - id: gemini-flash-primary
        vendor: gemini
        model: gemini-2.5-flash
        api_key: AIza…
        roles: [primary]
        max_tokens: 4096
        temperature: 0.4
      - id: gemini-pro-verifier
        vendor: gemini
        model: gemini-2.5-pro
        api_key: AIza…
        roles: [verifier]
      - id: claude-opus-critic
        vendor: anthropic
        model: claude-opus-4-7
        api_key: sk-ant-…
        roles: [critic, chat]

Legacy config (single ``gemini.api_key``) is auto-adapted into the new
list on first read so old installs keep working.

Public API:
    - list_providers()        → [LLMProviderConfig]
    - get_providers_for_role(role) → subset
    - complete(provider_id, prompt, system=None, ...) → LLMResponse
    - complete_many(provider_ids, prompt, ...) → list[LLMResponse]
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Config + response dataclasses
# ──────────────────────────────────────────────────────────────────────

@dataclass
class LLMProviderConfig:
    """One configured model endpoint."""
    id: str                     # user-chosen id (e.g. "gemini-flash-primary")
    vendor: str                 # gemini | anthropic | openai
    model: str                  # vendor-specific model id
    api_key: str = ''
    roles: list[str] = field(default_factory=list)
    max_tokens: int = 4096
    temperature: float = 0.4
    enabled: bool = True

    def is_configured(self) -> bool:
        return bool(self.api_key and self.vendor and self.model)


@dataclass
class LLMResponse:
    """Normalized response across vendors."""
    provider_id: str
    vendor: str
    model: str
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text)


# ──────────────────────────────────────────────────────────────────────
# Config loading — with legacy fallback
# ──────────────────────────────────────────────────────────────────────

def list_providers() -> list[LLMProviderConfig]:
    """Return configured providers. Adapts legacy config if needed."""
    from app.core.database import load_config
    cfg = load_config()
    raw = cfg.get('llm_providers')
    providers: list[LLMProviderConfig] = []

    if isinstance(raw, list) and raw:
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            try:
                providers.append(LLMProviderConfig(
                    id=str(entry.get('id') or f"{entry.get('vendor','?')}-{entry.get('model','?')}"),
                    vendor=str(entry.get('vendor') or 'gemini'),
                    model=str(entry.get('model') or ''),
                    api_key=str(entry.get('api_key') or ''),
                    roles=list(entry.get('roles') or []),
                    max_tokens=int(entry.get('max_tokens') or 4096),
                    temperature=float(entry.get('temperature') or 0.4),
                    enabled=bool(entry.get('enabled', True)),
                ))
            except Exception as e:
                logger.warning("Skipping bad provider config: %s", e)
    else:
        # Legacy single-Gemini fallback
        gem = cfg.get('gemini') or {}
        if gem.get('api_key'):
            providers.append(LLMProviderConfig(
                id='gemini-default',
                vendor='gemini',
                model=gem.get('model') or 'gemini-2.0-flash',
                api_key=gem.get('api_key') or '',
                roles=['primary', 'verifier', 'chat'],
            ))
        anthro_key = (os.environ.get('ANTHROPIC_API_KEY')
                      or cfg.get('anthropic_api_key'))
        if anthro_key:
            providers.append(LLMProviderConfig(
                id='anthropic-default',
                vendor='anthropic',
                model='claude-opus-4-7',
                api_key=str(anthro_key),
                roles=['verifier', 'critic'],
            ))

    return [p for p in providers if p.enabled and p.is_configured()]


def get_provider(provider_id: str) -> Optional[LLMProviderConfig]:
    for p in list_providers():
        if p.id == provider_id:
            return p
    return None


def get_providers_for_role(role: str) -> list[LLMProviderConfig]:
    return [p for p in list_providers() if role in p.roles]


def save_providers(providers: list[LLMProviderConfig]) -> None:
    """Persist providers back to config.yaml.

    Writes under the ``llm_providers`` key. Preserves all other top-level
    config keys.
    """
    from app.core.database import load_config, save_config
    cfg = load_config()
    cfg['llm_providers'] = [asdict(p) for p in providers]
    save_config(cfg)


# ──────────────────────────────────────────────────────────────────────
# Unified completion call — dispatches to vendor SDK
# ──────────────────────────────────────────────────────────────────────

def complete(
    provider_id: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> LLMResponse:
    """Call one configured provider with ``prompt``. Returns normalized response."""
    p = get_provider(provider_id)
    if p is None:
        return LLMResponse(provider_id=provider_id, vendor='?', model='?',
                           text='', error=f'Provider {provider_id!r} not found')
    if not p.is_configured():
        return LLMResponse(provider_id=p.id, vendor=p.vendor, model=p.model,
                           text='', error='Provider not configured (missing API key)')

    max_tokens = max_tokens or p.max_tokens
    temperature = temperature if temperature is not None else p.temperature

    t0 = time.time()
    try:
        if p.vendor == 'gemini':
            text, tin, tout = _call_gemini(p, prompt, system, max_tokens, temperature)
        elif p.vendor == 'anthropic':
            text, tin, tout = _call_anthropic(p, prompt, system, max_tokens, temperature)
        elif p.vendor == 'openai':
            text, tin, tout = _call_openai(p, prompt, system, max_tokens, temperature)
        else:
            return LLMResponse(provider_id=p.id, vendor=p.vendor, model=p.model,
                               text='', error=f'Unsupported vendor {p.vendor!r}')
    except Exception as e:
        logger.exception("LLM call failed (%s)", p.id)
        return LLMResponse(provider_id=p.id, vendor=p.vendor, model=p.model,
                           text='', latency_s=time.time() - t0, error=str(e))

    return LLMResponse(
        provider_id=p.id, vendor=p.vendor, model=p.model,
        text=text, tokens_in=tin, tokens_out=tout,
        latency_s=time.time() - t0,
    )


def complete_many(
    provider_ids: list[str],
    prompt: str,
    *,
    system: Optional[str] = None,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    parallel: bool = True,
) -> list[LLMResponse]:
    """Fan out the same prompt to multiple providers.

    When ``parallel=True`` (default) requests run in a thread pool so a
    2-model council completes in max(latency) rather than sum(latency).
    """
    if not provider_ids:
        return []

    if not parallel:
        return [complete(pid, prompt, system=system,
                         max_tokens=max_tokens, temperature=temperature)
                for pid in provider_ids]

    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(provider_ids)) as ex:
        futures = [ex.submit(complete, pid, prompt,
                             system=system,
                             max_tokens=max_tokens,
                             temperature=temperature)
                   for pid in provider_ids]
        return [f.result() for f in futures]


# ──────────────────────────────────────────────────────────────────────
# Vendor adapters — each returns (text, tokens_in, tokens_out)
# ──────────────────────────────────────────────────────────────────────

def _call_gemini(p: LLMProviderConfig, prompt: str, system: Optional[str],
                 max_tokens: int, temperature: float) -> tuple[str, int, int]:
    from google import genai
    client = genai.Client(api_key=p.api_key)
    full = f"{system}\n\n{prompt}" if system else prompt
    resp = client.models.generate_content(
        model=p.model,
        contents=full,
        config={
            'max_output_tokens': max_tokens,
            'temperature': temperature,
        },
    )
    text = resp.text or ''
    usage = getattr(resp, 'usage_metadata', None)
    tin = int(getattr(usage, 'prompt_token_count', 0) or 0) if usage else 0
    tout = int(getattr(usage, 'candidates_token_count', 0) or 0) if usage else 0
    return text, tin, tout


def _call_anthropic(p: LLMProviderConfig, prompt: str, system: Optional[str],
                    max_tokens: int, temperature: float) -> tuple[str, int, int]:
    import anthropic
    client = anthropic.Anthropic(api_key=p.api_key)
    kwargs = dict(
        model=p.model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{'role': 'user', 'content': prompt}],
    )
    if system:
        kwargs['system'] = system
    msg = client.messages.create(**kwargs)
    # Content is a list of content blocks
    parts = []
    for block in getattr(msg, 'content', []) or []:
        text = getattr(block, 'text', None)
        if text:
            parts.append(text)
    text = '\n'.join(parts)
    usage = getattr(msg, 'usage', None)
    tin = int(getattr(usage, 'input_tokens', 0) or 0) if usage else 0
    tout = int(getattr(usage, 'output_tokens', 0) or 0) if usage else 0
    return text, tin, tout


def _call_openai(p: LLMProviderConfig, prompt: str, system: Optional[str],
                 max_tokens: int, temperature: float) -> tuple[str, int, int]:
    try:
        from openai import OpenAI
    except ImportError:
        raise RuntimeError("openai package not installed — pip install openai")
    client = OpenAI(api_key=p.api_key)
    messages = []
    if system:
        messages.append({'role': 'system', 'content': system})
    messages.append({'role': 'user', 'content': prompt})
    resp = client.chat.completions.create(
        model=p.model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    text = resp.choices[0].message.content or '' if resp.choices else ''
    usage = getattr(resp, 'usage', None)
    tin = int(getattr(usage, 'prompt_tokens', 0) or 0) if usage else 0
    tout = int(getattr(usage, 'completion_tokens', 0) or 0) if usage else 0
    return text, tin, tout
