"""Persistent terminal session; no browser lifecycle hidden in this module."""
import asyncio
import os
from pathlib import Path
from .ui import console, message, chat_help
from .model import model_from_env
from dotenv import load_dotenv


async def read_input(prompt):
    return await asyncio.to_thread(console.input, prompt)


async def chat_loop(runner, args, session_dir, read=read_input, write=message):
    pending = args.task
    last_task = None
    task_number = 0
    once = args.once or args.headless
    if not once:
        chat_help()
    while True:
        if pending is None:
            try:
                pending = await read('\nВы › ')
            except EOFError:
                return 0
        task = pending.strip()
        pending = None
        if not task:
            continue
        if task in ('/exit', '/quit'):
            return 0
        if task == '/help':
            chat_help()
            continue
        if task == '/new':
            runner.messages = []
            runner.notes = ''
            runner.browser.invalidate()
            last_task = None
            write('Память диалога очищена. Страница и авторизация сохранены.')
            continue
        if task == '/status':
            page = runner.browser.page
            url = page.url if page and not page.is_closed() else 'Нет открытой вкладки'
            write(f'Модель: {runner.model.model}\nСтраница: {url}\nЗадач в сессии: {task_number}')
            continue
        if task == '/check':
            from .doctor import check
            try:
                write(await check(runner.model, runner.vision))
            except Exception as exc:
                write(str(exc), error=True)
            continue
        if task == '/reload':
            try:
                load_dotenv(args.env_file, override=True)
                candidate = model_from_env()
                if (getattr(candidate, 'provider', None), candidate.model) != (getattr(runner.model, 'provider', None), runner.model.model):
                    runner.messages = []
                    runner.notes = ''
                    write('Модель изменена: история очищена, состояние браузера сохранено.')
                runner.model = candidate
                runner.vision = not getattr(args, 'no_vision', False) and os.getenv('AGENT_VISION', 'true').lower() not in ('false', '0', 'no')
                write(f'Настройки перечитаны. Модель: {candidate.model}')
            except Exception as exc:
                write(f'Настройки не применены: {exc}', error=True)
            continue
        if task == '/retry':
            if not last_task:
                write('Пока нет задачи для повтора.', error=True)
                continue
            task = 'Продолжи предыдущую задачу: ' + last_task + '\nСначала проверь фактическое состояние. Не повторяй уже выполненные действия.'
        elif task.startswith('/'):
            write('Неизвестная команда. Введите /help.', error=True)
            continue
        else:
            last_task = task
        task_number += 1
        runner.run_dir = Path(session_dir) / f'task-{task_number:03d}'
        runner.run_dir.mkdir(parents=True, exist_ok=True)
        write('Агент выполняет задачу. Следующее сообщение можно ввести после ответа.')
        result = await runner.run(task, continue_session=bool(runner.messages))
        write(f'Отчёт: {runner.run_dir.resolve()}')
        if once:
            return 0 if result.get('status') in ('completed', 'answered') else 2
        if result.get('status') == 'blocked':
            write('Можно уточнить задачу, исправить .env и ввести /reload, затем /retry.', error=True)
