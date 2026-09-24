import json
import tempfile
import unittest
from agent.runner import Runner
from agent.model import OpenRouterModel

IMAGE = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': 'YWJj'}}
class Browser:
    version = 0
    observed = False
    actions = 0
    shots = 0
    async def observe(self, frame=0):
        self.version += 1
        self.observed = True
        return {'version': self.version, 'url': 'https://example.test', 'text': str(self.actions), 'screenshot_reasons': ['interactive_elements_changed']}
    async def screenshot(self):
        self.shots += 1
        return [IMAGE.copy()]
    async def act(self, **kwargs):
        self.actions += 1
        return {'action_dispatched': True}

class VisionTests(unittest.IsolatedAsyncioTestCase):
    async def test_initial_image_precedes_first_model_call(self):
        class Model:
            calls = 0
            async def complete(self, system, messages, tools, max_tokens):
                wire = OpenRouterModel.convert_messages(system, messages)
                assert wire[-1]['content'][-1]['type'] == 'image_url'
                self.calls += 1
                if self.calls == 1:
                    return {'content': [{'type': 'tool_use', 'id': 'v', 'name': 'verify', 'input': {}}]}
                return {'content': [{'type': 'tool_use', 'id': 'f', 'name': 'finish', 'input': {'status': 'completed', 'summary': 'done', 'evidence': 'initial state', 'version': 2}}]}
        with tempfile.TemporaryDirectory() as d:
            browser = Browser()
            result = await Runner(Model(), browser, d).run('task')
            self.assertEqual(result['status'], 'completed')
            self.assertEqual(browser.shots, 2)

    async def test_action_refreshes_state_and_prunes_old_image(self):
        with tempfile.TemporaryDirectory() as d:
            browser = Browser()
            runner = Runner(None, browser, d)
            initial = await runner.dispatch('observe', {})
            runner.messages = [{'role': 'user', 'content': initial}]
            result = await runner.dispatch('act', {'action': 'click', 'version': 1, 'role': 'button', 'name': 'Save'})
            self.assertEqual(json.loads(result[0]['text'])['version'], 2)
            self.assertEqual(browser.actions, 1)
            self.assertEqual(browser.shots, 2)
            self.assertFalse(any(b['type'] == 'image' for b in runner.messages[0]['content']))
            self.assertEqual(result[-1]['type'], 'image')

    async def test_screenshot_failure_keeps_observation(self):
        class Broken(Browser):
            async def screenshot(self):
                raise RuntimeError('closed')
        with tempfile.TemporaryDirectory() as d:
            result = await Runner(None, Broken(), d).dispatch('observe', {})
            self.assertEqual(json.loads(result[0]['text'])['version'], 1)
            self.assertIn('unavailable', result[-1]['text'])

    async def test_observe_failure_does_not_repeat_action(self):
        class Broken(Browser):
            async def observe(self, frame=0):
                raise RuntimeError('dynamic page')
        with tempfile.TemporaryDirectory() as d:
            browser = Broken()
            result = await Runner(None, browser, d).dispatch('act', {'action': 'click', 'version': 1, 'role': 'button', 'name': 'Save'})
            self.assertEqual(browser.actions, 1)
            self.assertTrue(result['action_result']['action_dispatched'])

    async def test_minor_change_skips_image_but_keeps_dom(self):
        class Quiet(Browser):
            async def observe(self, frame=0):
                result = await super().observe(frame)
                result['screenshot_reasons'] = []
                return result
        with tempfile.TemporaryDirectory() as d:
            browser = Quiet()
            result = await Runner(None, browser, d).dispatch('act', {'action': 'fill', 'version': 1, 'role': 'textbox', 'name': 'Title', 'text': 'abc'})
            self.assertEqual(result['text'], '1')
            self.assertEqual(browser.shots, 0)

    async def test_finish_requires_fresh_verification(self):
        with tempfile.TemporaryDirectory() as d:
            browser = Browser()
            runner = Runner(None, browser, d)
            await runner.dispatch('observe', {})
            args = {'status': 'completed', 'summary': 'ok', 'evidence': 'seen', 'version': 1}
            with self.assertRaisesRegex(ValueError, 'verify'):
                await runner.dispatch('finish', args)
            await runner.dispatch('verify', {})
            args['version'] = 2
            self.assertTrue((await runner.dispatch('finish', args))['finished'])
