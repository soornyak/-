import tempfile
import unittest
from pathlib import Path
from playwright.async_api import async_playwright
from agent.browser import Browser


class BrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.pw = await async_playwright().start()
        self.context = await self.pw.chromium.launch_persistent_context(self.tmp.name, headless=True)
        self.browser = Browser(self.context)
        self.page = self.browser.page
        await self.page.set_content(Path("demo/index.html").read_text(encoding="utf-8"))

    async def asyncTearDown(self):
        await self.context.close()
        await self.pw.stop()
        self.tmp.cleanup()

    async def test_real_form_and_stale_target(self):
        obs = await self.browser.observe()
        await self.browser.act("fill", obs["version"], role="textbox", name="Новая задача", text="Проверить отчёт")
        with self.assertRaisesRegex(ValueError, "Stale"):
            await self.browser.act("click", obs["version"], role="button", name="Добавить")
        obs = await self.browser.observe()
        await self.browser.act("click", obs["version"], role="button", name="Добавить")
        obs = await self.browser.observe()
        self.assertIn("Проверить отчёт", obs["aria"])
        await self.browser.act("check", obs["version"], role="checkbox", name="Проверить отчёт")
        self.assertTrue(await self.page.get_by_role("checkbox").is_checked())
        self.assertEqual((await self.browser.screenshot())[0]["type"], "image")

    async def test_ambiguous_target_and_bad_scheme(self):
        await self.page.set_content('<button>Save</button><button>Save</button>')
        obs = await self.browser.observe()
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            await self.browser.act("click", obs["version"], role="button", name="Save")
        with self.assertRaises(ValueError):
            await self.browser.navigate("file:///etc/passwd")

    async def test_frames_tabs_and_inspection(self):
        await self.page.set_content('<iframe srcdoc="<button>Inside</button>"></iframe>')
        await self.page.frame_locator("iframe").get_by_role("button").wait_for()
        obs = await self.browser.observe(frame=1)
        self.assertIn("Inside", obs["aria"])
        self.assertEqual((await self.browser.inspect("button", frame=1))["total"], 1)
        await self.context.new_page()
        result = await self.browser.switch_tab(0)
        self.assertEqual(len(result["tabs"]), 2)

    async def test_unrelated_heading_does_not_invalidate_target(self):
        obs = await self.browser.observe()
        await self.page.evaluate("document.querySelector('h1').textContent = 'Changed'")
        await self.browser.act("fill", obs["version"], role="textbox", name="Новая задача", text="Still valid")

    async def test_wait_invalidates_observation(self):
        await self.browser.observe()
        await self.browser.wait(0)
        self.assertFalse(self.browser.observed)

    async def test_must_observe_target_frame(self):
        await self.page.set_content('<iframe srcdoc="<button>Inside</button>"></iframe>')
        await self.page.frame_locator("iframe").get_by_role("button").wait_for()
        obs = await self.browser.observe()
        with self.assertRaisesRegex(ValueError, "target frame"):
            await self.browser.act("click", obs["version"], frame=1, role="button", name="Inside")

    async def test_observation_pagination(self):
        await self.page.set_content('<p>' + 'x' * 30000 + '</p>')
        first = await self.browser.observe()
        self.assertEqual(first['next_offset'], 14000)
        last = await self.browser.observe(offset=28000)
        self.assertIsNone(last['next_offset'])
        self.assertLessEqual(len(last['text']), 14000)

    async def test_element_ids_and_browser_context(self):
        obs = await self.browser.observe()
        self.assertIn('title', obs)
        self.assertEqual(obs['browser']['active_tab'], 0)
        textbox = next(e for e in obs['page']['interactive_elements'] if e['role']=='textbox')
        await self.browser.act('fill', obs['version'], element_id=textbox['id'], text='By ID')
        new = await self.browser.observe()
        with self.assertRaisesRegex(ValueError, 'Stale'):
            await self.browser.act('click', obs['version'], element_id=textbox['id'])
        button = next(e for e in new['page']['interactive_elements'] if e['name']=='Добавить')
        await self.browser.act('click', new['version'], element_id=button['id'])
        self.assertIn('By ID', (await self.browser.observe())['text'])

    async def test_removed_id_never_targets_replacement(self):
        await self.page.set_content('<button>Old</button>')
        obs = await self.browser.observe()
        await self.page.set_content('<button>New</button>')
        with self.assertRaisesRegex(ValueError, 'Stale'):
            await self.browser.act('click', obs['version'], element_id=1)

    async def test_cosmetic_mutations_do_not_block_fill(self):
        obs = await self.browser.observe()
        textbox = next(e for e in obs['page']['interactive_elements'] if e['role'] == 'textbox')
        await self.page.evaluate("""() => {
            document.body.dataset.heartbeat = 'tick';
            document.querySelector('input').setAttribute('data-test-internal', 'new');
            const style = document.createElement('style');
            style.textContent = 'body { outline-color: red; }';
            document.head.appendChild(style);
        }""")
        await self.browser.act('fill', obs['version'], element_id=textbox['id'], text='Stable action')
        self.assertEqual(await self.page.get_by_role('textbox', name='Новая задача').input_value(), 'Stable action')

    async def test_same_looking_replacement_is_still_stale(self):
        await self.page.set_content('<button>Save</button>')
        obs = await self.browser.observe()
        await self.page.evaluate("const b=document.querySelector('button'); b.replaceWith(b.cloneNode(true))")
        with self.assertRaisesRegex(ValueError, 'Stale'):
            await self.browser.act('click', obs['version'], element_id=1)

    async def test_property_value_change_is_stale(self):
        obs = await self.browser.observe()
        await self.page.get_by_role('textbox', name='Новая задача').evaluate("e => e.value = 'User edit'")
        with self.assertRaisesRegex(ValueError, 'Stale'):
            await self.browser.act('click', obs['version'], role='button', name='Добавить')

    async def test_dialog_appearance_is_stale(self):
        obs = await self.browser.observe()
        await self.page.evaluate("document.body.insertAdjacentHTML('beforeend','<dialog open>Confirm</dialog>')")
        with self.assertRaisesRegex(ValueError, 'Stale'):
            await self.browser.act('click', obs['version'], role='button', name='Добавить')

    async def test_three_tasks_with_cosmetic_updates_between_steps(self):
        for title in ('подготовить отчёт', 'проверить договор', 'отправить результат'):
            obs = await self.browser.observe()
            field = next(e for e in obs['page']['interactive_elements'] if e['role'] == 'textbox')
            await self.page.evaluate("document.body.dataset.tick = String(Math.random())")
            await self.browser.act('fill', obs['version'], element_id=field['id'], text=title)
            obs = await self.browser.observe()
            button = next(e for e in obs['page']['interactive_elements'] if e['name'] == 'Добавить')
            await self.page.evaluate("document.body.dataset.tick = String(Math.random())")
            await self.browser.act('click', obs['version'], element_id=button['id'])
        obs = await self.browser.observe()
        await self.browser.act('check', obs['version'], role='checkbox', name='подготовить отчёт')
        self.assertEqual(await self.page.get_by_role('checkbox').count(), 3)
        self.assertEqual(await self.page.locator('input[type=checkbox]:checked').count(), 1)
