import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock
from urllib.error import HTTPError
from dotenv import dotenv_values
from agent.model import OpenRouterModel, APIError, safe_error, retry_delay
from agent.openrouter import catalog, setup_openrouter


class OpenRouterSetup(unittest.TestCase):
    def test_catalog_filters(self):
        def item(name, tools=True, image=True, price='0'):
            return {'id':name,'supported_parameters':['tools'] if tools else [],'architecture':{'input_modalities':['text','image'] if image else ['text']},'pricing':{'prompt':price,'completion':price}}
        data = {'data':[item('good'),item('paid',price='0.01'),item('text',image=False),item('no-tools',tools=False)]}
        with patch('urllib.request.urlopen', return_value=io.BytesIO(json.dumps(data).encode())):
            self.assertEqual([m['id'] for m in catalog()], ['good'])

    def test_setup_defaults_to_free_router(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'.env'
            path.write_text('AGENT_PROVIDER=anthropic\nAGENT_TIMEOUT=100\n')
            replies = iter(['y',''])
            with patch('agent.openrouter.catalog', return_value=[]):
                setup_openrouter(path, lambda _:next(replies), lambda _:'fake-key')
            data = dotenv_values(path)
            self.assertEqual(data['AGENT_MODEL'],'openrouter/free')
            self.assertEqual(data['AGENT_PROVIDER'],'openrouter')
            self.assertEqual(data['AGENT_TIMEOUT'],'100')

    def test_error_classification_redacts_raw(self):
        payload={'error':{'message':'Provider returned error','metadata':{'provider_name':'Vendor','raw':'temporarily rate-limited upstream SECRET PAGE TEXT'}}}
        error=HTTPError('https://openrouter.ai',429,'rate',{'Retry-After':'60'},io.BytesIO(json.dumps(payload).encode()))
        message=safe_error(error,'fake-key')
        self.assertIn('ограничении частоты',message)
        self.assertIn('Vendor',message)
        self.assertIn('60 секунд',message)
        self.assertNotIn('SECRET',message)
        self.assertEqual(retry_delay(error.headers),60)


class Retries(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_429_not_repeated(self):
        model=OpenRouterModel('fake-key','test')
        with patch.object(model,'_request',side_effect=APIError(429,'unknown')) as request:
            with self.assertRaises(APIError):
                await model.complete('s',[])
            self.assertEqual(request.call_count,1)

    async def test_retry_after_respected(self):
        model=OpenRouterModel('fake-key','test')
        with patch.object(model,'_request',side_effect=[APIError(429,'busy',7),{'content':[]}]) as request, patch('agent.model.asyncio.sleep',new_callable=AsyncMock) as sleep:
            await model.complete('s',[])
            sleep.assert_awaited_once_with(7)
            self.assertEqual(request.call_count,2)

    async def test_long_wait_returns_to_user(self):
        model=OpenRouterModel('fake-key','test')
        with patch.object(model,'_request',side_effect=APIError(429,'busy',120)) as request, patch('agent.model.asyncio.sleep',new_callable=AsyncMock) as sleep:
            with self.assertRaises(APIError):
                await model.complete('s',[])
            sleep.assert_not_awaited()
            self.assertEqual(request.call_count,1)
