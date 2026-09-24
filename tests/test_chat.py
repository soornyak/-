import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from agent.chat import chat_loop
from agent.runner import Runner
from unittest.mock import patch

class Browser:
    page = None
    version = 0
    observed = False
    def invalidate(self):
        self.version += 1
        self.observed = False

class FakeRunner:
    def __init__(self):
        self.messages = []
        self.notes = ''
        self.browser = Browser()
        self.model = SimpleNamespace(model='test-model')
        self.calls = []
    async def run(self, task, continue_session=False):
        self.calls.append((task, continue_session, self.run_dir))
        self.messages.append({'role': 'user', 'content': task})
        return {'status': 'blocked' if len(self.calls)==1 else 'completed'}

class ChatTests(unittest.IsolatedAsyncioTestCase):
    async def test_error_followup_retry_and_new(self):
        commands = iter(['/status','/retry','Уточнение','/new','Новая задача','/exit'])
        async def read(prompt):
            return next(commands)
        args = SimpleNamespace(task='Первая задача', once=False, headless=False, env_file='.env')
        with tempfile.TemporaryDirectory() as d, patch('agent.chat.chat_help'):
            runner = FakeRunner()
            await chat_loop(runner, args, d, read=read, write=lambda *a, **kw: None)
            self.assertEqual(len(runner.calls), 4)
            self.assertEqual([c[1] for c in runner.calls], [False, True, True, False])
            self.assertIn('Не повторяй', runner.calls[1][0])
            self.assertEqual(len({c[2] for c in runner.calls}), 4)

    async def test_once_does_not_prompt_after_failure(self):
        async def read(prompt):
            raise AssertionError('Unexpected prompt')
        args = SimpleNamespace(task='Task', once=True, headless=False)
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(await chat_loop(FakeRunner(), args, d, read=read, write=lambda *a, **kw: None), 2)

    async def test_runner_preserves_conversation(self):
        class Model:
            def __init__(self):
                self.requests = []
            async def complete(self, system, messages, tools, max_tokens):
                self.requests.append(list(messages))
                return {'content': [{'type': 'tool_use', 'id': str(len(self.requests)), 'name': 'reply', 'input': {'text': 'Ответ'}}]}
        with tempfile.TemporaryDirectory() as d:
            model = Model()
            runner = Runner(model, Browser(), d, vision=False, json_logs=True)
            first = await runner.run('Запомни: проект Orbit')
            second = await runner.run('Как называется проект?', continue_session=True)
            self.assertEqual(first['status'], 'answered')
            self.assertEqual(second['status'], 'answered')
            self.assertTrue(any(m['content']=='Запомни: проект Orbit' for m in model.requests[-1]))

    async def test_reload_keeps_browser_and_replaces_model(self):
        commands = iter(['/reload','/exit'])
        async def read(prompt): return next(commands)
        args = SimpleNamespace(task=None, once=False, headless=False, env_file='.env')
        runner = FakeRunner()
        browser = runner.browser
        new = SimpleNamespace(model='new-model')
        with tempfile.TemporaryDirectory() as d, patch('agent.chat.chat_help'), patch('agent.chat.load_dotenv'), patch('agent.chat.model_from_env', return_value=new):
            await chat_loop(runner,args,d,read=read,write=lambda *a, **kw:None)
            self.assertIs(runner.browser,browser)
            self.assertIs(runner.model,new)
