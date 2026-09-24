import tempfile
import unittest
from agent.runner import Runner as BaseRunner, compactable_size

def Runner(*args, **kwargs):
    kwargs.setdefault("vision", False)
    return BaseRunner(*args, **kwargs)


class FakeBrowser:
    version = 0
    observed = False

    def invalidate(self):
        self.version += 1
        self.observed = False

    async def observe(self):
        self.version += 1
        self.observed = True
        return {"version": self.version, "text": "Added"}


class FakeModel:
    def __init__(self, calls):
        self.calls = iter(calls)
        self.requests = []

    async def complete(self, system, messages, tools=None, max_tokens=2500):
        self.requests.append(list(messages))
        return next(self.calls)


def call(name, args, identifier="t1"):
    return {"content": [{"type": "tool_use", "id": identifier, "name": name, "input": args}],
            "stop_reason": "tool_use"}


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_observation_required_for_completion(self):
        model = FakeModel([
            call("finish", {"status": "completed", "summary": "ok", "evidence": "x", "version": 0}),
            call("verify", {}, "t2"),
            call("finish", {"status": "completed", "summary": "ok", "evidence": "Added"}, "t3"),
        ])
        with tempfile.TemporaryDirectory() as d:
            runner = Runner(model, FakeBrowser(), d)
            result = await runner.run("do task")
            self.assertEqual(result["status"], "completed")
            self.assertTrue(model.requests[1][-1]["content"][0]["is_error"])

    async def test_unknown_tool_and_step_budget(self):
        with tempfile.TemporaryDirectory() as d:
            runner = Runner(FakeModel([call("shell", {"cmd": "bad"})]), FakeBrowser(), d, max_steps=1)
            result = await runner.run("task")
            self.assertEqual(result["status"], "partial")
            self.assertTrue(runner.messages[-1]["content"][0]["is_error"])

    async def test_schema_rejects_excessive_wait(self):
        with tempfile.TemporaryDirectory() as d:
            runner = Runner(None, FakeBrowser(), d)
            with self.assertRaises(Exception):
                await runner.dispatch("wait", {"seconds": 999})

    async def test_compaction_preserves_task_and_notes(self):
        model = FakeModel([{"content": [{"type": "text", "text": "Verified action A"}]}])
        with tempfile.TemporaryDirectory() as d:
            runner = Runner(model, FakeBrowser(), d)
            runner.notes = "Pending B"
            runner.messages = [{"role": "user", "content": "task"}]
            await runner.compact("original task")
            self.assertEqual(runner.messages[0]["content"], "original task")
            self.assertIn("Pending B", runner.messages[1]["content"])
            self.assertFalse(runner.browser.observed)

    async def test_parallel_mutations_are_not_executed(self):
        response = call("observe", {})
        response["content"].append({"type": "tool_use", "id": "t2", "name": "observe", "input": {}})
        with tempfile.TemporaryDirectory() as d:
            browser = FakeBrowser()
            runner = Runner(FakeModel([response]), browser, d, max_steps=1)
            await runner.run("task")
            self.assertEqual(browser.version, 3)
            self.assertEqual(len(runner.messages[-1]["content"]), 2)

    def test_image_size_is_bounded(self):
        self.assertEqual(compactable_size([{"type": "image", "source": {"data": "x" * 100000}}]), 6000)

    async def test_model_failure_saves_blocked_report(self):
        import json
        from pathlib import Path
        class BrokenModel:
            async def complete(self, *args, **kwargs):
                raise RuntimeError('Model API HTTP 429')
        with tempfile.TemporaryDirectory() as d:
            result = await Runner(BrokenModel(), FakeBrowser(), d).run('task')
            self.assertEqual(result['status'], 'blocked')
            self.assertEqual(json.loads((Path(d)/'result.json').read_text())['status'], 'blocked')

    async def test_blocked_can_finish_without_observation(self):
        with tempfile.TemporaryDirectory() as d:
            runner = Runner(None, FakeBrowser(), d)
            result = await runner.dispatch('finish', {'status': 'blocked', 'summary': 'Tab closed', 'evidence': 'No active tab', 'version': 0})
            self.assertEqual(result['status'], 'blocked')

    async def test_failed_actions_are_loop_counted(self):
        class BrokenBrowser(FakeBrowser):
            attempts = 0
            async def act(self, **kwargs):
                self.attempts += 1
                raise ValueError('Target absent')
        actions = []
        for i in range(3):
            actions.extend([call('act', {'action': 'click', 'version': i, 'role': 'button', 'name': 'Save'}, f'a{i}'), call('observe', {}, f'o{i}')])
        with tempfile.TemporaryDirectory() as d:
            browser = BrokenBrowser()
            await Runner(FakeModel(actions), browser, d, max_steps=6).run('task')
            self.assertEqual(browser.attempts, 2)
