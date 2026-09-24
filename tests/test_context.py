import unittest
from copy import deepcopy
from agent.context import ContextBuilder

class ContextTests(unittest.TestCase):
    def setUp(self):
        self.builder = ContextBuilder()
        self.semantic = {'elements': [{'id': 1, 'role': 'button', 'name': 'Save', 'rect': [0,0,50,20]}], 'total': 1, 'dialogs': [], 'scroll_x': 0, 'scroll_y': 0, 'viewport': [1280,800], 'visual_content': False}
    def observe(self, text='Hello', url='https://example.test'):
        return self.builder.enrich({'url': url, 'text': text, 'version': 1}, deepcopy(self.semantic), 'Page', 0)
    def test_initial_then_minor_edit(self):
        self.assertIn('initial', self.observe()['screenshot_reasons'])
        result = self.observe('Hello!')
        self.assertEqual(result['screenshot_reasons'], [])
        self.assertIn('page_text_changed', result['changes'])
    def test_dialog_and_scroll(self):
        self.observe()
        self.semantic['dialogs'] = ['Confirm']
        self.semantic['scroll_y'] = 300
        result = self.observe()
        self.assertIn('dialog_changed', result['screenshot_reasons'])
        self.assertIn('scroll_changed', result['screenshot_reasons'])
    def test_appeared_element_and_big_text(self):
        self.observe()
        self.semantic['elements'].append({'id': 2, 'role': 'button', 'name': 'Delete', 'rect': [0,30,50,20]})
        result = self.observe('x'*500)
        self.assertIn('interactive_elements_changed', result['screenshot_reasons'])
        self.assertIn('substantial_text_change', result['screenshot_reasons'])
        self.assertTrue(any('Delete' in change for change in result['changes']))
