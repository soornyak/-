"""Compact semantic context and conservative screenshot heuristics."""
from collections import Counter

ELEMENTS = """() => {
 const visible = e => {const r=e.getBoundingClientRect(),s=getComputedStyle(e);return r.width>0&&r.height>0&&s.visibility!=='hidden'&&s.display!=='none'&&r.bottom>0&&r.right>0&&r.top<innerHeight&&r.left<innerWidth;};
 const nodes=[...document.querySelectorAll('button,a[href],input:not([type=hidden]),textarea,select,[role],[contenteditable=true],[tabindex]')].filter(visible);
 window.__agentElements=nodes.slice(0,80);
 const role=e=>e.getAttribute('role')||({BUTTON:'button',A:'link',TEXTAREA:'textbox',SELECT:'combobox'}[e.tagName])||(e.tagName==='INPUT'?(['checkbox','radio'].includes(e.type)?e.type:'textbox'):'element');
 const name=e=>e.getAttribute('aria-label')||(e.getAttribute('aria-labelledby')||'').split(/\\s+/).map(id=>document.getElementById(id)?.textContent||'').join(' ').trim()||[...(e.labels||[])].map(x=>x.innerText).join(' ')||e.innerText||e.getAttribute('placeholder')||e.getAttribute('title')||'';
 return {elements:window.__agentElements.map((e,i)=>{const r=e.getBoundingClientRect();return {id:i+1,role:role(e),name:name(e).trim().slice(0,180),disabled:!!e.disabled,checked:typeof e.checked==='boolean'?e.checked:undefined,rect:[r.x,r.y,r.width,r.height].map(Math.round)};}),total:nodes.length,scroll_x:scrollX,scroll_y:scrollY,viewport:[innerWidth,innerHeight],dialogs:[...document.querySelectorAll('dialog[open],[role=dialog],[aria-modal=true]')].filter(visible).map(e=>e.innerText.slice(0,300)),visual_content:!![...document.querySelectorAll('canvas,video')].find(visible)};
}"""


class ContextBuilder:
    def __init__(self):
        self.previous = None

    def enrich(self, observation, semantic, title, active_tab):
        current = {'url': observation['url'], 'frame': observation.get('frame', 0),
                   'tab': active_tab, 'elements': semantic['elements'],
                   'dialogs': semantic['dialogs'], 'text': observation['text'],
                   'scroll': (semantic['scroll_x'], semantic['scroll_y'])}
        previous = self.previous
        changes, reasons = [], []
        if previous is None:
            reasons.append('initial')
        elif any(current[k] != previous[k] for k in ('url', 'frame', 'tab')):
            reasons.append('page_or_frame_changed')
        else:
            labels = lambda items: Counter((e['role'], e['name']) for e in items)
            old, new = labels(previous['elements']), labels(current['elements'])
            for kind, diff in [('appeared', new-old), ('removed', old-new)]:
                changes.extend(f'{kind}: {role} {name} ({count})' for (role, name), count in diff.items())
            if changes:
                reasons.append('interactive_elements_changed')
            if current['dialogs'] != previous['dialogs']:
                reasons.append('dialog_changed')
            if current['scroll'] != previous['scroll']:
                reasons.append('scroll_changed')
            if current['text'] != previous['text']:
                changes.append('page_text_changed')
                # Short text edits are kept in DOM; larger content changes get vision.
                a, b = current['text'], previous['text']
                delta = sum(x != y for x, y in zip(a, b)) + abs(len(a)-len(b))
                if delta >= 400:
                    reasons.append('substantial_text_change')
            old_geometry = [(e['role'],e['name'],e['rect']) for e in previous['elements']]
            new_geometry = [(e['role'],e['name'],e['rect']) for e in current['elements']]
            if old_geometry != new_geometry and not changes:
                reasons.append('layout_changed')
        if semantic['visual_content']:
            reasons.append('canvas_or_video')
        self.previous = current
        observation.update(title=title, frame=current['frame'],
            page={'interactive_elements': semantic['elements'], 'visible_element_total': semantic['total'],
                  'elements_truncated': semantic['total'] > 80, 'dialogs': semantic['dialogs']},
            browser={'active_tab': active_tab, 'scroll_x': semantic['scroll_x'],
                     'scroll_y': semantic['scroll_y'], 'viewport': semantic['viewport']},
            changes=changes[:20], screenshot_reasons=reasons)
        return observation
