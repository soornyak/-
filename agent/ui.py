"""Human-readable terminal output; full events remain in JSONL."""
import json
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

console = Console()

def banner(model, profile):
    console.print(Panel(Text(f"BROWSER AGENT / 0.9.1\nАвтономные задачи в браузере\n\nМодель: {model}\nПрофиль: {profile}"), border_style="cyan"))

def display_event(record):
    event = record["event"]
    if event == "tool_call":
        console.print(Text(f"{record['step']:02d}  → {record['name']}", style="bold cyan"))
        console.print(Text(json.dumps(record["arguments"], ensure_ascii=False)))
    elif event == "assistant":
        console.print(Text(record["text"]))
    elif event == "tool_result":
        if record["error"]:
            console.print(Text(str(record["result"]), style="yellow"))
        else:
            value = record["result"]
            detail = value.get("url", "") if isinstance(value, dict) else ""
            console.print(Text("    ✓ " + record["name"] + " " + detail, style="green"))
    elif event == "finished":
        color = "green" if record.get("status") in ("completed", "answered") else "yellow"
        console.print(Panel(Text(f"{record.get('status', '')}\n{record.get('summary', '')}\n\n{record.get('evidence', '')}"), title="Результат", border_style=color))
    elif event == "initial_observation":
        console.print(Text("Осматриваю страницу…", style="dim"))
    elif event == "run_error":
        console.print(Text("Запрос не выполнен. Подробности ниже.", style="yellow"))
    elif event == "context_compacted":
        console.print(Text("История сокращена, важные заметки сохранены.", style="dim"))
    else:
        console.print(Text(event, style="dim"))


def message(text, error=False):
    console.print(Text(text, style="yellow" if error else "dim"))


def chat_help():
    console.print(Panel(Text(
        "Пишите задачу обычным текстом. После ответа можно продолжить разговор.\n"
        "Откройте нужный сайт в окне агента и опишите цель.\n\n"
        "/check   — проверить API (два запроса)\n"
        "/status  — модель и текущая страница\n"
        "/reload  — перечитать .env без закрытия браузера\n"
        "/retry   — продолжить последнюю задачу после проверки состояния\n"
        "/new     — очистить память диалога, сохранить браузер\n"
        "/help    — показать команды\n"
        "/exit    — закрыть сессию"
    ), title="Чат с агентом", border_style="cyan"))
