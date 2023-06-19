# main_handler.py
import uuid
import logging
from typing import Union, List, Dict
from fastapi import HTTPException
import asyncio

async def process_data(data: Union[str, List[Dict[str, str]], Dict[str, str]], queues, futures_dict, logger):
    if data is None or data == '':
        # Если данные пустые, создаем словарь с ключом 'data' и значением "Empty request"
        data = {'data': "Empty request"}
    elif isinstance(data, list):
        # Если входные данные - список словарей, оставляем его как есть
        data = {'data': data}
    elif isinstance(data, str):
        # Если входные данные - просто строка, преобразуем её в словарь с ключом 'data'
        data = {'data': data}
    elif isinstance(data, dict) and 'data' in data:
        # Если входные данные - словарь и содержат ключ 'data', используем его
        pass
    else:
        # Если ни одно из условий не выполнилось, возвращаем сообщение об ошибке
        raise HTTPException(status_code=400, detail="Invalid format. Expected JSON with a 'data' key or a simple text.")

    if not isinstance(data['data'], (str, list)):
        # Если 'data' не является строкой или списком, возвращаем сообщение об ошибке
        raise HTTPException(status_code=400, detail="Invalid 'data' format. Expected a string or a list.")

    correlation_id = str(uuid.uuid4())
    logging.info(f'Received data: {data}. Assigned correlation_id: {correlation_id}.')

    # Назначение задачи для воркера
    message = {
        "correlation_id": correlation_id,
        "data": data,
    }

    # Отправляем сообщение в каждую из очередей
    for queue in queues:
        await queue.put(message)
        logging.info(f'Pushed to {id(queue)} : {message}')

    # Создание Future и его сохранение в словаре
    future = asyncio.Future()
    futures_dict[correlation_id] = future
    logging.info(f'Future created for correlation_id: {correlation_id}.')

    # Ожидание результата и его запись в ответ
    result = await future
    logging.info(f'Received result for correlation_id: {correlation_id}. Result: {result}')
    return result
