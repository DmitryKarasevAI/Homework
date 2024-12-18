from app import analyze_data_parallel, analyze_data, get_weather
import pandas as pd
import time
import asyncio
import requests


async def main():
    data = await asyncio.gather(
        get_weather('London', city_coordinates, API_KEY),
        get_weather('New York', city_coordinates, API_KEY)
    )
    return data


if __name__ == "__main__":
    df = pd.read_csv('temperature_data.csv')
    start_time = time.time()
    analyze_data_parallel(df)
    end_time = time.time()
    parallel_analysis_time = end_time - start_time
    print(f'Время выполнения мультипроцессного анализа: {parallel_analysis_time} сек')

    start_time = time.time()
    analyze_data(df)
    end_time = time.time()
    analysis_time = end_time - start_time
    print(f'Время выполнения однопроцессного анализа: {analysis_time} сек')

    with open('analysis_times.txt', 'w', encoding='utf-8') as output:
        output.write(f'Время выполнения однопроцессного анализа: {analysis_time} сек')
        output.write('\n')
        output.write(f'Время выполнения мультипроцессного анализа: {parallel_analysis_time} сек')

# Время выполнения однопроцессного анализа: 10.345757722854614 сек
# Время выполнения мультипроцессного анализа: 4.303915739059448 сек
# Удалось распараллелить, делая разбиение по городам.

# Синхронный запрос к API:
    
    API_KEY = "99a6db7f3997e2367a33a24724a59e7b"

    city_coordinates = {
        "London": {"lat": 51.5074, "lon": -0.1278},
        "New York": {"lat": 40.7128, "lon": -74.0060}
    }

    start_time = time.time()
    result = requests.get(f'https://api.openweathermap.org/data/2.5/weather?lat={city_coordinates["London"]['lat']}&lon={city_coordinates["London"]['lon']}&appid={API_KEY}').json()
    print(result)
    result = requests.get(f'https://api.openweathermap.org/data/2.5/weather?lat={city_coordinates["New York"]['lat']}&lon={city_coordinates["New York"]['lon']}&appid={API_KEY}').json()
    print(result)
    end_time = time.time()
    sync_api_time = end_time - start_time
    print(f'Время выполнения синхронного запроса: {sync_api_time} сек')

    start_time = time.time()
    print(asyncio.run(main()))
    end_time = time.time()
    async_api_time = end_time - start_time
    print(f'Время выполнения асинхронного запроса: {async_api_time} сек')

    with open('API_times.txt', 'w', encoding='utf-8') as output:
        output.write(f'Время выполнения синхронного запроса: {sync_api_time} сек')
        output.write('\n')
        output.write(f'Время выполнения асинхронного запроса: {async_api_time} сек')
