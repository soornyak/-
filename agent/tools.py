def tool(name, description, properties=None, required=None):
    return {"name": name, "description": description, "input_schema": {
        "type": "object", "properties": properties or {},
        "required": required or [], "additionalProperties": False}}


S = {"type": "string"}
I = {"type": "integer", "minimum": 0}
TOOLS = [
    tool("reply", "Answer a conversational question or greeting. Does NOT mark browser work completed. For browser results use finish.", {"text": S}, ["text"]),
    tool("observe", "Read live page text, tabs, frames and accessible structure.",
         {"frame": I, "offset": I}),
    tool("navigate", "Open an HTTP(S) URL discovered from the task or page.", {"url": S}, ["url"]),
    tool("inspect", "Inspect matching live DOM elements and attributes; use only selectors derived from observed content.",
         {"selector": S, "frame": I}, ["selector"]),
    tool("act", "Interact with an observed element. Prefer element_id from latest observation (version-scoped), or role+name; selector may come from live DOM. Exactly one target mode. Snapshot version is mandatory. Mutations invalidate observations, even on timeout. Never blindly retry a submission.",
         {"action": {"enum": ["click", "fill", "press", "select", "check", "uncheck"]},
          "element_id": {"type": "integer", "minimum": 1}, "version": I, "frame": I, "selector": S, "role": S, "name": S,
          "text": S, "index": I}, ["action", "version"]),
    tool("scroll", "Scroll the current page vertically.",
         {"pixels": {"type": "integer", "minimum": -3000, "maximum": 3000}}, ["pixels"]),
    tool("switch_tab", "Select a tab from the latest observation.", {"index": I}, ["index"]),
    tool("screenshot", "See the current browser viewport."),
    tool("wait", "Wait briefly for asynchronous UI changes.",
         {"seconds": {"type": "number", "minimum": 0, "maximum": 5}}, ["seconds"]),
    tool("ask_user", "Ask only for missing facts, login/captcha, or unapproved consequential actions.",
         {"question": S}, ["question"]),
    tool("remember", "Replace durable progress notes: verified facts, completed actions, remaining work, uncertainties. These survive compaction.",
         {"notes": {"type": "string", "maxLength": 6000}}, ["notes"]),
    tool("verify", "Get fresh DOM and screenshot for final verification. Analyze the returned state before finish.", {"frame": I}),
    tool("finish", "Finish only after verifying the result in the latest observation, or report a blocked/partial task honestly.",
         {"status": {"enum": ["completed", "partial", "blocked"]}, "summary": S,
          "evidence": S, "version": I}, ["status", "summary", "evidence", "version"]),
]
SCHEMAS = {t["name"]: t["input_schema"] for t in TOOLS}

# Keep legacy act schema for old records/tests, but expose simple commands to LLM.
LEGACY_TOOLS = TOOLS
TARGET_ACTIONS = {'click_element':'click', 'type_text':'fill', 'press_key':'press',
                  'select_option':'select', 'check_element':'check', 'uncheck_element':'uncheck'}
TOOLS = [t for t in TOOLS if t['name'] != 'act']
for name, action in TARGET_ACTIONS.items():
    props = {'element_id': {'type':'integer', 'minimum':1}}
    required = ['element_id']
    if action in ('fill','press','select'):
        props['text'] = S
        required.append('text')
    entry = tool(name, 'Perform '+action+' on an element from the latest observation. One action per turn; fresh state is returned automatically.', props, required)
    TOOLS.append(entry)
    SCHEMAS[name] = entry['input_schema']
# Versions remain internal; finish still requires verified evidence.
for entry in TOOLS:
    if entry['name'] == 'finish':
        entry['input_schema']['required'].remove('version')
