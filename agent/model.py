import asyncio
import json
import urllib.request
import urllib.error
import os
from urllib.parse import urlparse


class APIError(RuntimeError):
    def __init__(self, status, message, retry_after=None):
        self.status = status
        self.retry_after = retry_after
        super().__init__(f"Model API HTTP {status}. {message}")


def retry_delay(headers):
    from email.utils import parsedate_to_datetime
    import time
    value = headers.get('Retry-After') if headers else None
    if not value:
        return None
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, parsedate_to_datetime(value).timestamp() - time.time())
        except (ValueError, TypeError, OverflowError):
            return None


def provider_error_detail(raw):
    """Extract only message fields, never dump echoed request/input/headers."""
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            return ''
    if not isinstance(raw, dict):
        return ''
    error = raw.get('error', raw)
    if isinstance(error, str):
        return error[:800]
    if not isinstance(error, dict):
        return ''
    message = error.get('message')
    if isinstance(message, str):
        return message[:800]
    detail = error.get('detail')
    if isinstance(detail, str):
        return detail[:800]
    if isinstance(detail, list):
        return '; '.join(str(x['msg']) for x in detail[:3]
                         if isinstance(x, dict) and isinstance(x.get('msg'), str))[:800]
    return ''


def safe_error(error, key):
    import re
    try:
        body = json.loads(error.read(16384).decode("utf-8", errors="replace"))
        value = body.get("error", {})
        message = value.get("message", "") if isinstance(value, dict) else str(value)
        metadata = value.get('metadata') or {} if isinstance(value, dict) else {}
        if not isinstance(metadata, dict):
            metadata = {}
        # Classify provider metadata without printing raw prompts or opaque payloads.
        raw_value = metadata.get('raw', '')
        raw = str(raw_value).lower()
        provider = str(metadata.get('provider_name') or '')
        if provider:
            message += ' | Provider: ' + provider[:100]
        for field in ('error_type', 'provider_code'):
            if isinstance(metadata.get(field), (str, int)):
                message += ' | ' + field + ': ' + str(metadata[field])[:100]
        detail = provider_error_detail(raw_value)
        if detail and detail != message:
            message += ' | Detail: ' + detail
        elif getattr(error, 'code', None) == 400:
            known = [('reasoning', 'Провайдер упоминает формат reasoning.'),
                     ('tool_call', 'Провайдер упоминает формат вызовов инструментов.'),
                     ('image', 'Провайдер упоминает изображения.'),
                     ('context', 'Провайдер упоминает контекст запроса.'),
                     ('token', 'Провайдер упоминает лимит токенов.')]
            category = next((hint for token, hint in known if token in raw), None)
            message += ' | ' + (category or 'Подробное описание HTTP 400 не предоставлено в поддерживаемом формате.')
        if 'per-day' in raw or 'daily' in raw:
            message += ' | Провайдер сообщает о дневном лимите. Дождитесь его сброса.'
        elif 'rate-limit' in raw or 'rate limit' in raw:
            message += ' | Провайдер сообщает об ограничении частоты запросов.'
        elif 'quota' in raw or 'insufficient' in raw:
            message += ' | Провайдер сообщает об исчерпанной квоте или недостатке средств.'
        elif 'unavailable' in raw or 'no endpoints' in raw:
            message += ' | Провайдер сообщает о недоступности модели/маршрута.'
        elif getattr(error, 'code', None) == 429:
            message += ' | Причина лимита не уточнена; проверьте квоту аккаунта и доступность модели.'
    except (ValueError, AttributeError, TypeError):
        message = ""
    delay = retry_delay(getattr(error, 'headers', None))
    if delay is not None:
        message += f' | Retry-After: {delay:.0f} секунд.'
    message = str(message).replace(key, "[KEY HIDDEN]") if key else str(message)
    message = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE HIDDEN]", message)
    message = re.sub(r"(?i)Bearer\s+[^\s,;]+", "Bearer [KEY HIDDEN]", message)
    message = re.sub(r"sk-[A-Za-z0-9_-]+", "[KEY HIDDEN]", message)
    return "".join(c for c in message if c.isprintable() or c in "\n\t")[:1200] or "Проверьте ключ, модель, доступность провайдера и баланс."



class Model:
    """Minimal Anthropic Messages protocol client; no coding-plan assumptions."""

    def __init__(self, key, model, base_url="https://api.anthropic.com"):
        if not base_url.startswith("https://"):
            raise ValueError("Model endpoint must use HTTPS")
        self.key, self.model = key, model
        self.url = base_url.rstrip("/") + "/v1/messages"

    def _request(self, payload):
        req = urllib.request.Request(self.url, json.dumps(payload).encode(), {
            "x-api-key": self.key, "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=getattr(self, "timeout", 90)) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            # Do not log response bodies that could contain sensitive input.
            raise APIError(exc.code, safe_error(exc, self.key), retry_delay(exc.headers)) from exc

    async def complete(self, system, messages, tools=None, max_tokens=2500):
        payload = dict(model=self.model, max_tokens=max_tokens, system=system,
                       messages=messages)
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        for attempt in range(3):
            try:
                return await asyncio.to_thread(self._request, payload)
            except (RuntimeError, urllib.error.URLError, TimeoutError) as exc:
                if attempt == 2 or (isinstance(exc, RuntimeError) and
                   getattr(exc, "status", None) not in (429, 500, 502, 503, 504)):
                    raise
                if getattr(exc, 'status', None) == 429:
                    delay = getattr(exc, 'retry_after', None)
                    # Do not hammer unknown/daily limits or block the chat for minutes.
                    if delay is None or delay > 30:
                        raise
                    await asyncio.sleep(max(1, delay))
                else:
                    await asyncio.sleep(2 ** attempt)


class OpenRouterModel(Model):
    """Chat Completions adapter; the runner keeps its internal block format."""

    def __init__(self, key, model, base_url="https://openrouter.ai/api/v1"):
        from urllib.parse import urlparse
        parsed = urlparse(base_url)
        if parsed.scheme != 'https' or parsed.hostname != 'openrouter.ai' or parsed.path.rstrip('/') not in ('/api', '/api/v1') or parsed.query or parsed.fragment or parsed.username or parsed.port not in (None, 443):
            raise ValueError('OPENROUTER_BASE_URL must be https://openrouter.ai/api/v1')
        self.key, self.model = key, model
        self.url = 'https://openrouter.ai/api/v1/chat/completions'
        self.provider = 'openrouter'

    @staticmethod
    def convert_messages(system, messages):
        result = [{'role': 'system', 'content': system}]
        for message in messages:
            content = message['content']
            if isinstance(content, str):
                result.append(dict(message))
                continue
            if message['role'] == 'assistant':
                converted = {'role': 'assistant', 'content': '\n'.join(b['text'] for b in content if b['type'] == 'text') or None}
                calls = [{'id': b['id'], 'type': 'function', 'function': {'name': b['name'], 'arguments': json.dumps(b['input'], ensure_ascii=False)}} for b in content if b['type'] == 'tool_use']
                for call, block in zip(calls, [b for b in content if b['type'] == 'tool_use']):
                    if 'extra_content' in block:
                        call['extra_content'] = block['extra_content']
                if calls:
                    converted['tool_calls'] = calls
                # Some routed reasoning models require opaque details on subsequent turns.
                details = [b['details'] for b in content if b['type'] == 'provider_reasoning']
                if details:
                    converted['reasoning_details'] = details[0]
                extra = [b['fields'] for b in content if b['type'] == 'provider_fields']
                if extra:
                    converted.update(extra[0])
                result.append(converted)
                continue
            images, text = [], []
            for block in content:
                if block['type'] == 'tool_result':
                    value = block['content']
                    if isinstance(value, list):
                        parts = []
                        for item in value:
                            if item['type'] == 'image':
                                source = item['source']
                                images.append({'type': 'text', 'text': 'Screenshot from tool ' + block['tool_use_id']})
                                images.append({'type': 'image_url', 'image_url': {'url': 'data:' + source['media_type'] + ';base64,' + source['data']}})
                            elif item['type'] == 'text':
                                parts.append(item['text'])
                        value = '\n'.join(parts) or 'Screenshot attached in the following user message.'
                    if block.get('is_error'):
                        value = 'Tool error: ' + value
                    result.append({'role': 'tool', 'tool_call_id': block['tool_use_id'], 'content': value})
                elif block['type'] == 'text':
                    text.append(block)
                elif block['type'] == 'image':
                    source = block['source']
                    images.append({'type': 'image_url', 'image_url': {'url': 'data:' + source['media_type'] + ';base64,' + source['data']}})
                else:
                    raise ValueError('Unsupported user content block')
            if text or images:
                result.append({'role': 'user', 'content': text + images})
        return result

    @staticmethod
    def normalize(data):
        if data.get('error') or not data.get('choices'):
            raise RuntimeError('Model API returned an error or no choices; check model availability and account limits')
        choice = data['choices'][0]
        if choice.get('finish_reason') == 'length':
            return {'content': [], 'stop_reason': 'max_tokens', 'usage': {}}
        message = choice['message']
        blocks = []
        if message.get('content'):
            if not isinstance(message['content'], str):
                raise RuntimeError('Unexpected model text format')
            blocks.append({'type': 'text', 'text': message['content']})
        if message.get('reasoning_content') is not None:
            blocks.append({'type': 'provider_fields', 'fields': {'reasoning_content': message['reasoning_content']}})
        if message.get('reasoning_details'):
            blocks.append({'type': 'provider_reasoning', 'details': message['reasoning_details']})
        for call in message.get('tool_calls') or []:
            try:
                args = json.loads(call['function']['arguments'])
            except (ValueError, TypeError) as exc:
                raise RuntimeError('Model returned invalid tool JSON; no action executed') from exc
            if not isinstance(args, dict):
                raise RuntimeError('Tool arguments must be a JSON object')
            block = {'type': 'tool_use', 'id': call['id'], 'name': call['function']['name'], 'input': args}
            if 'extra_content' in call:
                block['extra_content'] = call['extra_content']
            blocks.append(block)
        usage = data.get('usage') or {}
        return {'content': blocks, 'stop_reason': 'tool_use' if message.get('tool_calls') else 'end_turn',
                'usage': {'input_tokens': usage.get('prompt_tokens', 0), 'output_tokens': usage.get('completion_tokens', 0)}}

    def _request(self, payload):
        converted = {'model': self.model, 'messages': self.convert_messages(payload['system'], payload['messages']),
                     'max_tokens': payload['max_tokens'], 'stream': False}
        if payload.get('tools'):
            converted['tools'] = [{'type': 'function', 'function': {'name': t['name'], 'description': t['description'], 'parameters': t['input_schema']}} for t in payload['tools']]
            converted['tool_choice'] = 'auto'
            if self.provider == 'openrouter':
                converted['provider'] = {'require_parameters': True}
        if self.provider == 'openai':
            converted['max_completion_tokens'] = converted.pop('max_tokens')
        req = urllib.request.Request(self.url, json.dumps(converted).encode(), {
            'Authorization': 'Bearer ' + self.key, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=getattr(self, "timeout", 90)) as response:
                return self.normalize(json.load(response))
        except urllib.error.HTTPError as exc:
            raise APIError(exc.code, safe_error(exc, self.key), retry_delay(exc.headers)) from exc


class CompatibleModel(OpenRouterModel):
    def __init__(self, key, model, base_url, provider):
        parsed = urlparse(base_url)
        local = parsed.hostname in ('localhost', '127.0.0.1', '::1')
        if (parsed.scheme != 'https' and not (parsed.scheme == 'http' and local)) or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('API base URL must be HTTPS (HTTP allowed only for localhost)')
        self.key, self.model, self.provider = key, model, provider
        base = base_url.rstrip('/')
        self.url = base if base.endswith('/chat/completions') else base + '/chat/completions'


PROVIDERS = {
    'anthropic': ('ANTHROPIC_API_KEY', 'https://api.anthropic.com'),
    'openrouter': ('OPENROUTER_API_KEY', 'https://openrouter.ai/api/v1'),
    'openai': ('OPENAI_API_KEY', 'https://api.openai.com/v1'),
    'gemini': ('GEMINI_API_KEY', 'https://generativelanguage.googleapis.com/v1beta/openai'),
    'xai': ('XAI_API_KEY', 'https://api.x.ai/v1'),
    'deepseek': ('DEEPSEEK_API_KEY', 'https://api.deepseek.com'),
    'compatible': ('AGENT_API_KEY', None),
}



def model_from_env():
    provider = os.getenv('AGENT_PROVIDER', 'openrouter').strip().lower()
    if provider not in PROVIDERS:
        raise ValueError('AGENT_PROVIDER: ' + ', '.join(PROVIDERS))
    key_name, default_url = PROVIDERS[provider]
    key = os.getenv(key_name, '').strip()
    model_id = os.getenv('AGENT_MODEL', '').strip()
    if not key or not model_id or key == 'your-key' or model_id.startswith('your-'):
        raise ValueError(f'Заполните {key_name} и AGENT_MODEL в .env; или запустите python -m agent.cli --setup')
    base = os.getenv(provider.upper() + '_BASE_URL', default_url) if provider != 'compatible' else os.getenv('AGENT_BASE_URL')
    if not base:
        raise ValueError('Set AGENT_BASE_URL for compatible provider')
    if provider == 'anthropic':
        model = Model(key, model_id, base)
        model.provider = provider
    elif provider == 'openrouter':
        model = OpenRouterModel(key, model_id, base)
    else:
        model = CompatibleModel(key, model_id, base, provider)
    model.timeout = float(os.getenv('AGENT_TIMEOUT', '90'))
    model.output_budget = int(os.getenv('AGENT_MAX_TOKENS', '4096'))
    if not 5 <= model.timeout <= 300 or not 256 <= model.output_budget <= 32768:
        raise ValueError('AGENT_TIMEOUT: 5..300; AGENT_MAX_TOKENS: 256..32768')
    return model
