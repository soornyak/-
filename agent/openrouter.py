"""OpenRouter setup uses public model metadata; no inference or key upload."""
import json
import urllib.request
from decimal import Decimal, InvalidOperation
from getpass import getpass
from pathlib import Path
from dotenv import dotenv_values, set_key


def catalog(vision=True, free_only=True):
    with urllib.request.urlopen('https://openrouter.ai/api/v1/models', timeout=25) as response:
        data = json.load(response)
    result = []
    for model in data.get('data', []):
        if 'tools' not in (model.get('supported_parameters') or []):
            continue
        if vision and 'image' not in (model.get('architecture', {}).get('input_modalities') or []):
            continue
        prices = model.get('pricing') or {}
        try:
            free = all(Decimal(str(prices.get(k, '-1'))) == 0 for k in ('prompt', 'completion'))
        except InvalidOperation:
            free = False
        if free_only and not free:
            continue
        result.append(model)
    return sorted(result, key=lambda m: m['id'])


def show_models(vision=True):
    print('Бесплатные модели с tools' + (' и изображениями:' if vision else ':'))
    models = catalog(vision)
    for index, model in enumerate(models, 1):
        print(f"{index}. {model['id']}")
    print('Наличие в каталоге не гарантирует свободную квоту или доступность маршрута.')
    return models


def setup_openrouter(path, read=input, secret=getpass):
    path = Path(path)
    values = dotenv_values(path) if path.exists() else {}
    print('Настройка OpenRouter. Платная модель автоматически не выбирается.')
    key = secret('Новый OpenRouter API-ключ (скрыт; Enter — оставить текущий): ').strip() or values.get('OPENROUTER_API_KEY')
    if not key or key == 'your-key':
        raise ValueError('Введите действующий OpenRouter API-ключ')
    vision = read('Скриншоты [Y/n]: ').strip().lower() not in ('n', 'no', 'нет', 'н')
    models = []
    try:
        models = show_models(vision)
    except Exception:
        print('Каталог недоступен. Можно указать точный ID вручную или выбрать бесплатный маршрутизатор.')
    print('0. openrouter/free — маршрутизатор бесплатных моделей; квоты сохраняются.')
    choice = read('Номер, точный ID или Enter для openrouter/free: ').strip()
    if not choice or choice == '0':
        model_id = 'openrouter/free'
    elif choice.isdigit():
        index = int(choice) - 1
        if not 0 <= index < len(models):
            raise ValueError('Номер отсутствует в списке')
        model_id = models[index]['id']
    else:
        model_id = choice
        print('Выбран ручной ID: стоимость и поддержку tools/изображений проверьте в кабинете.')
    updates = {'AGENT_PROVIDER':'openrouter', 'OPENROUTER_API_KEY':key,
               'AGENT_MODEL':model_id, 'OPENROUTER_BASE_URL':'https://openrouter.ai/api/v1',
               'AGENT_VISION':'true' if vision else 'false'}
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, value in updates.items():
        set_key(str(path), name, value)
    print(f'Сохранено: {path.resolve()}\nМодель: {model_id}\nТеперь: python -m agent.cli --check')
