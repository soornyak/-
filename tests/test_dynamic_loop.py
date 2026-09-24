"""Executor integration: scripted model responses, real Chromium, changing page."""
import json
import tempfile
import unittest
from pathlib import Path
from playwright.async_api import async_playwright
from agent.browser import Browser
from agent.runner import Runner
from agent.tools import TOOLS

class DynamicLoop(unittest.IsolatedAsyncioTestCase):
    async def test_full_runner_three_tasks_while_clock_changes(self):
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            context = await browser.new_context()
            page = await context.new_page()
            await page.set_content(Path('demo/index.html').read_text())
            await page.evaluate("""() => {
                const clock=document.createElement('p');clock.id='clock'; document.body.appendChild(clock);
                window.tick=setInterval(()=>clock.textContent=String(Date.now()),10);
            }""")
            adapter = Browser(context)
            actions=[]
            for title in ('подготовить отчёт','проверить договор','отправить результат'):
                actions += [('type_text','textbox',None,title),('click_element','button','Добавить',None)]
            actions += [('check_element','checkbox','подготовить отчёт',None),('verify',None,None,None),('finish',None,None,None)]
            class Model:
                async def complete(self, system, messages, tools=None, max_tokens=4096):
                    name,role,label,text=actions.pop(0)
                    # Prove actual time passes between observation and action.
                    await page.wait_for_timeout(60)
                    if name=='finish':
                        args={'status':'completed','summary':'3 tasks, 1 checked','evidence':'Verified DOM'}
                    elif name=='verify': args={}
                    else:
                        target=next(e for e in adapter.builder.previous['elements'] if e['role']==role and (label is None or e['name']==label))
                        args={'element_id':target['id']}
                        if text is not None: args['text']=text
                    return {'content':[{'type':'tool_use','id':str(len(actions)),'name':name,'input':args}]}
            try:
                with tempfile.TemporaryDirectory() as tmp:
                    result=await Runner(Model(),adapter,tmp,vision=False,json_logs=True).run('Create exactly three tasks, check first')
                    self.assertEqual(result['status'],'completed')
                    self.assertEqual(await page.get_by_role('checkbox').count(),3)
                    self.assertEqual(await page.locator('input[type=checkbox]:checked').count(),1)
                    logs=[json.loads(x) for x in (Path(tmp)/'events.jsonl').read_text().splitlines()]
                    self.assertFalse(any(x.get('error') for x in logs if x['event']=='tool_result'))
            finally:
                await browser.close()

    async def test_five_failed_actions_stop_despite_successful_observations(self):
        class Browser:
            version=0;observed=True;observed_frame=0
            async def observe(self, **kwargs):
                self.version+=1;self.observed=True
                return {'version':self.version,'url':'test','text':'same'}
            async def act(self, **kwargs):
                raise ValueError('Missing target')
        class Model:
            step=0
            async def complete(self,*args,**kwargs):
                self.step+=1
                name='type_text' if self.step%2 else 'observe'
                params={'element_id':1,'text':str(self.step)} if name=='type_text' else {}
                return {'content':[{'type':'tool_use','id':str(self.step),'name':name,'input':params}]}
        with tempfile.TemporaryDirectory() as tmp:
            model=Model()
            result=await Runner(model,Browser(),tmp,vision=False,json_logs=True).run('task')
            self.assertEqual(result['status'],'blocked')
            self.assertEqual(model.step,9)

    def test_exposed_actions_have_no_version_or_selector(self):
        definitions={t['name']:t for t in TOOLS}
        self.assertNotIn('act',definitions)
        self.assertEqual(set(definitions['type_text']['input_schema']['properties']),{'element_id','text'})
