"""Two-turn protocol check; never performs browser actions."""
import base64
import struct
import zlib


def test_image():
    def chunk(kind, data):
        return struct.pack('!I', len(data)) + kind + data + struct.pack('!I', zlib.crc32(kind + data) & 0xffffffff)
    png = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('!2I5B', 64, 64, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress((b'\x00' + b'\xff\x00\x00' * 64) * 64)) + chunk(b'IEND', b'')
    return {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': base64.b64encode(png).decode()}}


async def check(model, vision=True):
    tool = {'name': 'connection_probe', 'description': 'Harmless connection check.', 'input_schema': {'type': 'object', 'properties': {'color': {'type': 'string'}}, 'required': ['color'], 'additionalProperties': False}}
    prompt = 'Call connection_probe exactly once. Set color to the dominant image color in English.' if vision else 'Call connection_probe exactly once with color="none".'
    content = [{'type': 'text', 'text': prompt}]
    if vision:
        content.append(test_image())
    messages = [{'role': 'user', 'content': content}]
    budget = getattr(model, 'output_budget', 4096)
    result = await model.complete('Follow the connection test instructions.', messages, [tool], budget)
    calls = [b for b in result['content'] if b['type'] == 'tool_use']
    if len(calls) != 1 or calls[0]['name'] != 'connection_probe':
        raise RuntimeError('Модель не выполнила тестовый tool call. Проверьте поддержку tools и лимит AGENT_MAX_TOKENS.')
    color = str(calls[0]['input'].get('color', '')).lower().strip()
    if color != ('red' if vision else 'none'):
        raise RuntimeError('Модель не прошла проверку изображения/инструкции. Выберите другую модель или отключите AGENT_VISION.')
    messages += [{'role': 'assistant', 'content': result['content']}, {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': calls[0]['id'], 'content': 'Probe succeeded. Reply with a short confirmation; do not call tools.'}]}]
    second = await model.complete('Confirm the successful probe in plain text.', messages, None, budget)
    if second.get('stop_reason') == 'max_tokens' or not any(b['type'] == 'text' and b.get('text', '').strip() for b in second['content']):
        raise RuntimeError('Модель не завершила второй шаг проверки. Проверьте лимит ответа и совместимость API.')
    return 'Проверка пройдена: API, инструменты, ответ после инструмента' + (', изображение.' if vision else '. Режим DOM без изображений.')
