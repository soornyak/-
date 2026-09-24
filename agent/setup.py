"""Local configuration wizard. Keys are entered without echo."""
from getpass import getpass
from pathlib import Path
from dotenv import dotenv_values, set_key
from .model import PROVIDERS


def setup(path, read=input, secret=getpass):
    path = Path(path)
    values = dotenv_values(path) if path.exists() else {}
    print('Провайдеры: ' + ', '.join(PROVIDERS))
    provider = read('Провайдер: ').strip().lower()
    if provider not in PROVIDERS:
        raise ValueError('Выберите название из списка, не URL')
    model = read('Точный ID модели из кабинета провайдера: ').strip()
    if not model:
        raise ValueError('ID модели обязателен')
    key_name, _ = PROVIDERS[provider]
    key = secret('API-ключ (скрыт; Enter — сохранить существующий): ').strip() or values.get(key_name)
    if not key:
        raise ValueError('API-ключ обязателен')
    updates = {'AGENT_PROVIDER': provider, 'AGENT_MODEL': model, key_name: key}
    if provider == 'compatible':
        base = read('Base URL, например http://localhost:11434/v1: ').strip()
        from .model import CompatibleModel
        CompatibleModel(key, model, base, provider)
        updates['AGENT_BASE_URL'] = base
    vision = read('Отправлять скриншоты? Нужна поддержка изображений [Y/n]: ').strip().lower()
    updates['AGENT_VISION'] = 'false' if vision in ('n', 'no', 'нет', 'н') else 'true'
    path.parent.mkdir(parents=True, exist_ok=True)
    for name, value in updates.items():
        set_key(str(path), name, value)
    print(f'Настройки сохранены: {path.resolve()}\nПроверка: python -m agent.cli --check')
