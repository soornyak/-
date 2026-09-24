import argparse
import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from playwright.async_api import async_playwright
from .browser import Browser
from .model import model_from_env
from .runner import Runner
from .ui import banner
from .chat import chat_loop
from dotenv import load_dotenv


async def run(args):
    load_dotenv(args.env_file, override=True)
    if getattr(args, "setup", False):
        from .openrouter import setup_openrouter
        setup_openrouter(args.env_file)
        return 0
    vision = not args.no_vision and os.getenv("AGENT_VISION", "true").lower() not in ("false", "0", "no")
    if getattr(args, "models", False):
        from .openrouter import show_models
        await asyncio.to_thread(show_models, vision)
        return 0
    if getattr(args, "check", False):
        from .doctor import check
        print("Проверка подключения: два запроса к выбранному API…")
        print(await check(model_from_env(), vision))
        return 0
    run_dir = Path("runs") / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True)
    model = None if args.login else model_from_env()
    model_id = model.model if model else None
    if not args.json_logs:
        banner(model_id or "Ручной вход", args.profile)
    async with async_playwright() as pw:
        options = dict(headless=args.headless, viewport={"width": 1280, "height": 800})
        if args.record:
            options["record_video_dir"] = str(run_dir / "video")
        context = await pw.chromium.launch_persistent_context(str(Path(args.profile).resolve()), **options)
        browser = Browser(context)
        try:
            if browser.page is None:
                browser.page = await context.new_page()
            if args.url:
                await browser.navigate(args.url)
            if args.login:
                await asyncio.to_thread(input, "Войдите в нужные сервисы в браузере. Затем нажмите Enter здесь. ")
                return
            runner = Runner(model, browser, run_dir, args.max_steps,
                            json_logs=args.json_logs, vision=vision)
            return await chat_loop(runner, args, run_dir)

        finally:
            await context.close()


def main():
    p = argparse.ArgumentParser(description="Autonomous browser agent")
    p.add_argument("--models", action="store_true", help="Бесплатные OpenRouter модели с tools и изображениями")
    p.add_argument("--setup", action="store_true", help="Мастер настройки подключения")
    p.add_argument("--check", action="store_true", help="Проверить API, tools и изображения (2 запроса)")
    p.add_argument("--env-file", default=str(Path(__file__).resolve().parent.parent / ".env"), help="Файл настроек")
    p.add_argument("--no-vision", action="store_true", help="Disable automatic screenshots")
    p.add_argument("--json-logs", action="store_true", help="JSON вместо красивого вывода событий")
    p.add_argument("--once", action="store_true", help="Выполнить одну задачу и выйти")
    p.add_argument("--task", help="Первое сообщение чата")
    p.add_argument("--url", help="Optional starting URL supplied by the user")
    p.add_argument("--profile", default=".profile")
    p.add_argument("--login", action="store_true", help="Manual login; no model API needed")
    p.add_argument("--record", action="store_true", help="Record browser only, not terminal")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--max-steps", type=int, default=80)
    args = p.parse_args()
    if args.max_steps < 1:
        p.error("--max-steps must be positive")
    if args.login and args.headless:
        p.error("--login requires a visible browser")
    try:
        code = asyncio.run(run(args))
        if code:
            raise SystemExit(code)
    except KeyboardInterrupt:
        print("Остановлено пользователем.")
    except Exception as exc:
        p.exit(1, f"Ошибка: {exc}\n")


if __name__ == "__main__":
    main()
