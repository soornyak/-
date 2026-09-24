import asyncio
import base64
from urllib.parse import urlparse
from .context import ContextBuilder, ELEMENTS

# Snapshot target identity and local meaning, not unrelated page text.
INSTALL_GUARD = """() => {
 window.__agentTargetState = e => {
   const attrs = ['role','aria-label','aria-labelledby','aria-describedby','aria-disabled','aria-checked','aria-expanded','aria-selected','href','type','name','placeholder','readonly'];
   const scope = e.closest('form,li,tr,article,[role=row],[role=dialog]');
   const labels = [...(e.labels || [])].map(x=>x.innerText);
   const refs = ['aria-labelledby','aria-describedby'].map(a => (e.getAttribute(a)||'').split(/\\s+/).map(id=>document.getElementById(id)?.textContent||''));
   return JSON.stringify([e.tagName,attrs.map(a=>e.getAttribute(a)),labels,refs,
      e.innerText||'',e.type==='password'?null:e.value,!!e.checked,!!e.disabled,
      scope?.innerText||'', [...(scope?.querySelectorAll('input,textarea,select')||[])].map(x=>[x.type==='password'?null:x.value,!!x.checked,!!x.disabled])]);
 };
 window.__agentDialogs = () => JSON.stringify([...document.querySelectorAll('dialog[open],[role=dialog],[aria-modal=true]')]
   .filter(e=>e.getClientRects().length && getComputedStyle(e).visibility!=='hidden')
   .map(e=>e.innerText));
}"""


class Browser:
    def __init__(self, context):
        self.builder = ContextBuilder()
        self.context = context
        self.page = context.pages[0] if context.pages else None
        self.version = 0
        self.observed = False
        self.observed_frame = None
        context.on("page", self._new_page)
        for page in context.pages:
            self._watch(page)

    def _watch(self, page):
        page.on("dialog", lambda dialog: dialog.dismiss())
        page.on("framenavigated", lambda _: self.invalidate())

    def _new_page(self, page):
        self._watch(page)
        self.page = page
        self.invalidate()

    def invalidate(self):
        self.version += 1
        self.observed = False

    def frame(self, index=0):
        if self.page is None or self.page.is_closed():
            raise ValueError("Active tab closed; switch_tab or navigate first")
        return self.page.frames[index]

    async def observe(self, frame=0, offset=0):
        f = self.frame(frame)
        await f.evaluate(INSTALL_GUARD)
        text = await f.locator("body").inner_text(timeout=8000)
        aria = await f.locator("body").aria_snapshot(timeout=8000)
        # Capture IDs and their guard in one JS evaluation; do not remap before acting.
        semantic = await f.evaluate("""() => {
          const semantic = (""" + ELEMENTS + """)();
          const nodes = window.__agentElements;
          window.__agentObservedState = {url:location.href, dialogs:window.__agentDialogs(),
            targets: new Map(nodes.map(e=>[e,window.__agentTargetState(e)]))};
          return semantic;
        }""")
        title = await self.page.title()
        self.version += 1
        self.observed = True
        self.observed_frame = frame
        observation = {"version": self.version, "url": self.page.url, "frame": frame,
                "tabs": [{"index": i, "url": p.url} for i, p in enumerate(self.context.pages)],
                "frames": [{"index": i, "url": x.url} for i, x in enumerate(self.page.frames)],
                "text": text[offset:offset + 14000], "text_length": len(text),
                "aria": aria[offset:offset + 14000], "aria_length": len(aria),
                "offset": offset, "next_offset": offset + 14000 if offset + 14000 < max(len(text), len(aria)) else None}
        return self.builder.enrich(observation, semantic, title, self.context.pages.index(self.page))

    async def navigate(self, url):
        if urlparse(url).scheme not in ("https", "http"):
            raise ValueError("Only HTTP(S) navigation is supported")
        self.invalidate()
        if self.page is None or self.page.is_closed():
            self.page = await self.context.new_page()
        await self.page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return await self.observe()

    async def inspect(self, selector, frame=0):
        locator = self.frame(frame).locator(selector)
        return await locator.evaluate_all("""els => ({total: els.length, elements: els.slice(0,40).map(e => ({
            tag: e.tagName, text: (e.innerText || '').slice(0,600),
            attributes: Object.fromEntries([...e.attributes].filter(a =>
              ['id','class','role','aria-label','name','type','href','placeholder','title'].includes(a.name)).map(a=>[a.name,a.value.slice(0,600)]))
        }))})""")

    async def act(self, action, version, frame=0, selector=None, role=None,
                  name=None, text=None, index=None, element_id=None):
        if not self.observed or version != self.version:
            raise ValueError("Stale observation: call observe before acting")
        if frame != self.observed_frame:
            raise ValueError("Observe the target frame before acting")
        if action not in {"click", "fill", "press", "select", "check", "uncheck"}:
            raise ValueError("Unsupported action")
        if sum((bool(selector), bool(role), element_id is not None)) != 1:
            raise ValueError("Choose element_id OR selector OR role+name")
        if element_id is not None and index is not None:
            raise ValueError("index cannot be combined with element_id")
        f = self.frame(frame)
        if element_id is not None:
            handle = await f.evaluate_handle("id => window.__agentElements?.[id-1]", element_id)
            loc = handle.as_element()
            if loc is None or not await loc.evaluate("e => e.isConnected"):
                await handle.dispose()
                raise ValueError("Stale target: element missing or detached; observe again")
        else:
            loc = f.locator(selector) if selector else f.get_by_role(role, name=name, exact=True)
            if index is not None:
                loc = loc.nth(index)
            if await loc.count() != 1:
                raise ValueError("Target is absent or ambiguous; inspect and refine")
        try:
            valid = await loc.evaluate("""e => {
              const old = window.__agentObservedState;
              return !!old && e.isConnected && old.url===location.href &&
                old.dialogs===window.__agentDialogs() && old.targets.has(e) &&
                old.targets.get(e)===window.__agentTargetState(e);
            }""")
            if not valid:
                self.invalidate()
                raise ValueError('Stale DOM: selected target or its context changed; observe again')
            if action in ("fill", "press", "select") and text is None:
                raise ValueError("text is required for this action")
            if action == "fill" and await loc.get_attribute("type") == "password":
                raise ValueError("Ask the user to enter passwords directly in the browser")
            self.invalidate()
            if action == "click":
                await loc.click(timeout=8000)
            elif action == "fill":
                await loc.fill(text, timeout=8000)
            elif action == "press":
                await loc.press(text, timeout=8000)
            elif action == "select":
                await loc.select_option(label=text, timeout=8000)
            elif action == "check":
                if not await loc.is_checked():
                    await loc.click(timeout=8000)
            elif action == "uncheck":
                if await loc.is_checked():
                    await loc.click(timeout=8000)
            return {"action_dispatched": True, "next": "observe and verify the actual outcome"}
        finally:
            if element_id is not None:
                await handle.dispose()

    async def scroll(self, pixels):
        self.invalidate()
        await self.page.mouse.wheel(0, pixels)
        return {"scrolled": pixels}

    async def switch_tab(self, index):
        self.page = self.context.pages[index]
        await self.page.bring_to_front()
        self.invalidate()
        return await self.observe()

    async def screenshot(self):
        data = await self.page.screenshot(type="jpeg", quality=75)
        return [{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                  "data": base64.b64encode(data).decode()}}]

    async def wait(self, seconds):
        self.invalidate()
        await asyncio.sleep(seconds)
        return {"waited": seconds, "next": "observe"}
