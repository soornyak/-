import io
import json
import os
import unittest
from unittest.mock import patch
from agent.model import Model, OpenRouterModel, model_from_env
from agent.tools import TOOLS


class ModelTests(unittest.TestCase):
    def test_provider_config(self):
        with patch.dict(os.environ, {'AGENT_PROVIDER': 'openrouter', 'OPENROUTER_API_KEY': 'test-key', 'AGENT_MODEL': 'vendor/test'}, clear=True):
            self.assertIsInstance(model_from_env(), OpenRouterModel)
        with patch.dict(os.environ, {'AGENT_PROVIDER': 'anthropic', 'ANTHROPIC_API_KEY': 'test-key', 'AGENT_MODEL': 'claude-test'}, clear=True):
            self.assertIsInstance(model_from_env(), Model)
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'OPENROUTER_API_KEY'):
                model_from_env()

    def test_tools_images_and_errors(self):
        messages = [
            {'role': 'user', 'content': 'task'},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 't1', 'name': 'screenshot', 'input': {}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': [{'type': 'image', 'source': {'media_type': 'image/png', 'data': 'YWJj'}}]}]},
        ]
        converted = OpenRouterModel.convert_messages('system', messages)
        self.assertEqual([m['role'] for m in converted], ['system', 'user', 'assistant', 'tool', 'user'])
        self.assertEqual(converted[2]['tool_calls'][0]['function']['arguments'], '{}')
        self.assertEqual(converted[3]['tool_call_id'], 't1')
        self.assertEqual(converted[4]['content'][1]['image_url']['url'], 'data:image/png;base64,YWJj')
        err = OpenRouterModel.convert_messages('s', [{'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 't1', 'content': 'absent', 'is_error': True}]}])
        self.assertEqual(err[-1]['content'], 'Tool error: absent')

    def test_response_and_reasoning_roundtrip(self):
        details = [{'type': 'reasoning.encrypted', 'data': 'opaque'}]
        response = {'choices': [{'finish_reason': 'tool_calls', 'message': {'content': None, 'reasoning_details': details, 'tool_calls': [{'id': 't1', 'function': {'name': 'observe', 'arguments': '{}'}}]}}], 'usage': {'prompt_tokens': 12, 'completion_tokens': 4}}
        result = OpenRouterModel.normalize(response)
        self.assertEqual(result['usage']['input_tokens'], 12)
        self.assertEqual(result['content'][-1]['name'], 'observe')
        converted = OpenRouterModel.convert_messages('s', [{'role': 'assistant', 'content': result['content']}])
        self.assertEqual(converted[-1]['reasoning_details'], details)

    def test_bad_json_and_truncation(self):
        response = {'choices': [{'finish_reason': 'length', 'message': {'tool_calls': [{'function': {'arguments': '{'}}]}}]}
        self.assertEqual(OpenRouterModel.normalize(response)['stop_reason'], 'max_tokens')
        response['choices'][0]['finish_reason'] = 'tool_calls'
        with self.assertRaisesRegex(RuntimeError, 'invalid tool JSON'):
            OpenRouterModel.normalize(response)

    def test_wire_request(self):
        answer = {'choices': [{'finish_reason': 'stop', 'message': {'content': 'ok'}}]}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(answer).encode())) as request:
            model = OpenRouterModel('test-key', 'vendor/test')
            result = model._request({'system': 's', 'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 100, 'tools': TOOLS})
            req = request.call_args.args[0]
            self.assertEqual(req.full_url, 'https://openrouter.ai/api/v1/chat/completions')
            self.assertEqual(req.get_header('Authorization'), 'Bearer test-key')
            body = json.loads(req.data)
            self.assertEqual(body['tools'][0]['type'], 'function')
            self.assertNotIn('parallel_tool_calls', body)
            self.assertTrue(body['provider']['require_parameters'])
            self.assertEqual(result['content'][0]['text'], 'ok')

    def test_url_validation(self):
        for url in ['http://openrouter.ai/api/v1', '[https://openrouter.ai/api](https://openrouter.ai/api)', 'https://other.example/api/v1']:
            with self.assertRaises(ValueError):
                OpenRouterModel('test-key', 'vendor/test', url)
        self.assertEqual(OpenRouterModel('k', 'm', 'https://openrouter.ai/api').url, 'https://openrouter.ai/api/v1/chat/completions')

    def test_http_error_shows_reason_without_key(self):
        from urllib.error import HTTPError
        body = {'error': {'message': 'No endpoints for test-key sk-or-v1-secret', 'metadata': {'raw': 'PRIVATE RAW DATA'}}}
        error = HTTPError('https://openrouter.ai/api/v1/chat/completions', 404, 'Not Found', {}, io.BytesIO(json.dumps(body).encode()))
        with patch('urllib.request.urlopen', side_effect=error):
            with self.assertRaises(RuntimeError) as caught:
                OpenRouterModel('test-key', 'model')._request({'system':'s','messages':[], 'max_tokens':10})
        message = str(caught.exception)
        self.assertIn('No endpoints', message)
        self.assertNotIn('test-key', message)
        self.assertNotIn('sk-or-v1-secret', message)
        self.assertNotIn('PRIVATE RAW DATA', message)

    def test_nested_provider_error_is_extracted_without_input_dump(self):
        from urllib.error import HTTPError
        from agent.model import safe_error
        raw = {'error': {'message': 'Missing tool_call_id for test-key', 'input': 'PRIVATE REQUEST'}}
        for value in (raw, json.dumps(raw)):
            body = {'error': {'message':'Provider returned error', 'metadata': {'provider_name':'ModelRun', 'raw':value, 'error_type':'invalid_request'}}}
            error = HTTPError('https://openrouter.ai',400,'Bad Request',{},io.BytesIO(json.dumps(body).encode()))
            message = safe_error(error,'test-key')
            self.assertIn('Missing tool_call_id', message)
            self.assertIn('invalid_request', message)
            self.assertNotIn('PRIVATE REQUEST',message)
            self.assertNotIn('test-key',message)

    def test_validation_details_do_not_echo_input(self):
        from agent.model import provider_error_detail
        raw = {'detail':[{'msg':'Field required','input':{'secret':'PRIVATE'}}]}
        self.assertEqual(provider_error_detail(raw), 'Field required')

    def test_multiturn_images_preserve_tool_result_pairing(self):
        from copy import deepcopy
        from agent.runner import Runner
        image={'type':'image','source':{'type':'base64','media_type':'image/jpeg','data':'YWJj'}}
        history=[{'role':'user','content':[{'type':'text','text':'task'},deepcopy(image)]}]
        for index in range(4):
            runner=object.__new__(Runner)
            runner.messages=history
            runner.prune_images()
            history.append({'role':'assistant','content':[{'type':'tool_use','id':str(index),'name':'type_text','input':{'element_id':1,'text':'task'}}]})
            history.append({'role':'user','content':[{'type':'tool_result','tool_use_id':str(index),'content':[{'type':'text','text':'observed'},deepcopy(image)]}]})
            wire=OpenRouterModel.convert_messages('system',history)
            for i, message in enumerate(wire):
                if message.get('tool_calls'):
                    self.assertEqual(wire[i+1]['role'],'tool')
                    self.assertEqual(wire[i+1]['tool_call_id'],message['tool_calls'][0]['id'])
            images=sum(1 for m in wire if isinstance(m['content'],list) for b in m['content'] if b.get('type')=='image_url')
            self.assertEqual(images,1)
