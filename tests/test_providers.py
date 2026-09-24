import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from agent.model import PROVIDERS, CompatibleModel, OpenRouterModel, model_from_env
from agent.doctor import check
from agent.setup import setup


class Providers(unittest.TestCase):
    def test_direct_payloads(self):
        for provider in ('openai', 'gemini', 'xai', 'deepseek', 'compatible'):
            key, endpoint = PROVIDERS[provider]
            with self.subTest(provider=provider), patch.dict(os.environ, {'AGENT_PROVIDER': provider, key: 'test', 'AGENT_MODEL': 'test-model', 'AGENT_BASE_URL': 'http://localhost:11434/v1'}, clear=True):
                model = model_from_env()
                answer = {'choices': [{'message': {'content': 'ok'}}]}
                with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(answer).encode())) as request:
                    model._request({'system': 's', 'messages': [], 'max_tokens': 512, 'tools': [{'name':'probe','description':'test','input_schema':{'type':'object'}}]})
                    req = request.call_args.args[0]
                    body = json.loads(req.data)
                    self.assertNotIn('provider', body)
                    self.assertEqual(req.get_header('Authorization'), 'Bearer test')
                    self.assertEqual(body['max_completion_tokens' if provider == 'openai' else 'max_tokens'], 512)
                    self.assertEqual(req.full_url, (endpoint or 'http://localhost:11434/v1') + '/chat/completions')

    def test_reasoning_and_signature_roundtrip(self):
        call = {'id': 'c1', 'type': 'function', 'function': {'name': 'probe', 'arguments': '{}'}, 'extra_content': {'google': {'thought_signature': 'opaque'}}}
        result = OpenRouterModel.normalize({'choices': [{'message': {'reasoning_content': 'preserved', 'tool_calls': [call]}}]})
        wire = OpenRouterModel.convert_messages('s', [{'role': 'assistant', 'content': result['content']}])[-1]
        self.assertEqual(wire['tool_calls'][0], call)
        self.assertEqual(wire['reasoning_content'], 'preserved')

    def test_unsafe_custom_endpoint(self):
        for url in ('http://example.org/v1', 'https://user:pass@example.org', 'https://example.org?key=test'):
            with self.assertRaises(ValueError):
                CompatibleModel('key','model',url,'compatible')

    def test_setup_preserves_other_settings(self):
        from dotenv import dotenv_values
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / '.env'
            path.write_text('AGENT_TIMEOUT=120\n', encoding='utf8')
            replies = iter(['openai', 'test-model', 'n'])
            setup(path, read=lambda _: next(replies), secret=lambda _: 'test-key')
            values = dotenv_values(path)
            self.assertEqual(values['AGENT_TIMEOUT'], '120')
            self.assertEqual(values['AGENT_VISION'], 'false')
            self.assertEqual(values['OPENAI_API_KEY'], 'test-key')


class Doctor(unittest.IsolatedAsyncioTestCase):
    async def test_real_http_two_turn_protocol(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from threading import Thread
        requests = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(data)
                message = {'content': 'Confirmed'} if len(requests) == 2 else {'tool_calls': [{'id':'p1','type':'function','function':{'name':'connection_probe','arguments':'{"color":"red"}'}}]}
                body = json.dumps({'choices':[{'message':message}]}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def log_message(self, *args):
                pass
        server = ThreadingHTTPServer(('127.0.0.1',0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            model = CompatibleModel('local-test','test',f'http://127.0.0.1:{server.server_port}/v1','compatible')
            self.assertIn('Проверка пройдена', await check(model))
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[0]['messages'][1]['content'][1]['type'], 'image_url')
            self.assertEqual(requests[1]['messages'][-1]['role'], 'tool')
            self.assertEqual(requests[1]['messages'][-1]['tool_call_id'], 'p1')
        finally:
            server.shutdown()
            server.server_close()

    async def test_reject_missing_tools(self):
        class Fake:
            async def complete(self, *args):
                return {'content': [{'type':'text','text':'hi'}]}
        with self.assertRaisesRegex(RuntimeError, 'tool call'):
            await check(Fake(), False)
