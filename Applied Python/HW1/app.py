import pandas as pd
import multiprocessing
import datetime
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import requests
import streamlit as st
import aiohttp

# Синхронный анализ
def analyze_data(df):
    # 30-дневный MA по городам:
    df_ma = pd.DataFrame(columns=['city', 'timestamp', 'temperature_ma'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    cities = df['city'].unique()
    seasons = df['season'].unique()

    for city in cities:
        city_data = df[df['city'] == city]
        city_data['temperature_ma'] = city_data['temperature'].rolling(window=30, center=True).mean()
        city_data = city_data[['city', 'timestamp', 'temperature_ma']]
        df_ma = pd.concat([df_ma, city_data], ignore_index=True)

    # Среднее и std по сезонам по городам:
    temp_mean = df[['city', 'season', 'temperature']].groupby(by=['city', 'season']).mean()
    temp_std = df[['city', 'season', 'temperature']].groupby(by=['city', 'season']).std()
    temp_min = df[['city', 'season', 'temperature']].groupby(by=['city', 'season']).min()
    temp_max = df[['city', 'season', 'temperature']].groupby(by=['city', 'season']).max()

    df_city_season_aggs = pd.concat([temp_mean, temp_std, temp_min, temp_max], axis=1)
    df_city_season_aggs.columns = ('temp_mean', 'temp_std', 'temp_min', 'temp_max')

    # Найдём аномалии в данных:
    df_aggs = df_city_season_aggs.reset_index()
    for city in cities:
        city_data = df[df['city'] == city]
        for season in seasons:
            mean = df_aggs[(df_aggs['city'] == city) & (df_aggs['season'] == season)]['temp_mean'].mean()
            std = df_aggs[(df_aggs['city'] == city) & (df_aggs['season'] == season)]['temp_std'].mean()
            city_data['is_anomaly'] = (city_data['temperature'] > mean + 2 * std) | (city_data['temperature'] < mean - 2 * std)

            plt.scatter(city_data[(city_data['is_anomaly'] == False) & (city_data['season'] == season)]['timestamp'],
                        city_data[(city_data['is_anomaly'] == False) & (city_data['season'] == season)]['temperature'],
                        c='b')
            plt.scatter(city_data[(city_data['is_anomaly'] == True) & (city_data['season'] == season)]['timestamp'],
                        city_data[(city_data['is_anomaly'] == True) & (city_data['season'] == season)]['temperature'],
                        c='r')
            plt.title(f'City: {city}, Season: {season}')
            plt.xlabel('Timestamp')
            plt.ylabel('Temperature')
            plt.savefig(str(city) + '_' + str(season) + ".png")
            plt.close()

    # Построим ACF и PACF для каждого города:
    for city in cities:
        city_data = df[df['city'] == city]

        plt.figure(figsize=(12, 6))

        # ACF
        plt.subplot(2, 1, 1)
        plot_acf(city_data['temperature'], lags=30, ax=plt.gca())
        plt.title("Auto-Correlation Function (ACF)")

        # PACF
        plt.subplot(2, 1, 2)
        plot_pacf(city_data['temperature'], lags=30, ax=plt.gca(), method='ywm')
        plt.title("Partial Auto-Correlation Function (PACF)")

        plt.tight_layout()
        plt.savefig(str(city) + ".png")
        plt.close()

    # Графики показывают decay ACF и обрубление PACF примерно на 10 лаге (дне),
    # соответственно, хорошей моделью для данных городов была бы AR(10)
    return df_ma, df_aggs


# Мультипроцессный анализ
def analyze_data_parallel(df):
    cities = df['city'].unique()
    with multiprocessing.Pool(processes=4) as pool:
        pool.starmap(analyze_data, [(df[df['city'] == city], ) for city in cities])

# Асинхронный API запрос
async def get_weather(city_picked, city_coordinates, API_KEY):
    url = f'https://api.openweathermap.org/data/2.5/weather?lat={city_coordinates[city_picked]["lat"]}&lon={city_coordinates[city_picked]["lon"]}&appid={API_KEY}'
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            result = await response.json()
            return result

if __name__=="__main__":

    city_coordinates = {
        "New York": {"lat": 40.71, "lon": -74.01},
        "London": {"lat": 51.51, "lon": -0.13},
        "Paris": {"lat": 48.86, "lon": 2.35},
        "Tokyo": {"lat": 35.68, "lon": 139.76},
        "Moscow": {"lat": 55.75, "lon": 37.62},
        "Sydney": {"lat": -33.87, "lon": 151.21},
        "Berlin": {"lat": 52.52, "lon": 13.41},
        "Beijing": {"lat": 39.90, "lon": 116.40},
        "Rio de Janeiro": {"lat": -22.91, "lon": -43.17},
        "Dubai": {"lat": 25.20, "lon": 55.27},
        "Los Angeles": {"lat": 34.05, "lon": -118.24},
        "Singapore": {"lat": 1.35, "lon": 103.82},
        "Mumbai": {"lat": 19.08, "lon": 72.88},
        "Cairo": {"lat": 30.04, "lon": 31.24},
        "Mexico City": {"lat": 19.43, "lon": -99.13}
        }

    st.title("Детекция аномальной погоды")

    st.header("Получение значения текущей температуры в городе через API")
    API_KEY = st.text_input("Enter Your API Key", "")

    city_picked = st.selectbox("City: ",
                    [
                        "New York",
                        "London",
                        "Paris",
                        "Tokyo",
                        "Moscow",
                        "Sydney",
                        "Berlin",
                        "Beijing",
                        "Rio de Janeiro",
                        "Dubai",
                        "Los Angeles",
                        "Singapore",
                        "Mumbai",
                        "Cairo",
                        "Mexico City"
                    ])

    if (st.button('Check weather')):
        try:
            result = requests.get(f'https://api.openweathermap.org/data/2.5/weather?lat={city_coordinates[city_picked]['lat']}&lon={city_coordinates[city_picked]['lon']}&appid={API_KEY}').json()
            st.success(f'Текущая температура в городе {city_picked}: {result["main"]["temp"] - 273.15:.1f} градусов Цельсия!')
        except KeyError:
            st.error('{"cod":401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}')
    
    st.header("Анализ исторических данных")
    df = st.file_uploader("Upload a CSV File", type=["csv"])

    if df is not None:
        df = pd.read_csv(df)
        df_ma, df_aggs = analyze_data(df)

    st.subheader("Выберите города для анализа")

    check_box_for_ma = st.multiselect("Select cities:", [
        "New York",
        "London",
        "Paris",
        "Tokyo",
        "Moscow",
        "Sydney",
        "Berlin",
        "Beijing",
        "Rio de Janeiro",
        "Dubai",
        "Los Angeles",
        "Singapore",
        "Mumbai",
        "Cairo",
        "Mexico City"
    ])

    if check_box_for_ma:
        seasons = df_aggs['season'].unique()
        
        for city in check_box_for_ma:
            st.write(f'In {city}:')
            for season in seasons:
                st.write(f"""\tat {season} average temp is {df_aggs[(df_aggs['city'] == city) & (df_aggs['season'] == season)]['temp_mean'].iloc[0]:.1f},
                        min temp is {df_aggs[(df_aggs['city'] == city) & (df_aggs['season'] == season)]['temp_min'].iloc[0]:.1f},
                        max temp is {df_aggs[(df_aggs['city'] == city) & (df_aggs['season'] == season)]['temp_max'].iloc[0]:.1f},
                        standard deviation is {df_aggs[(df_aggs['city'] == city) & (df_aggs['season'] == season)]['temp_std'].iloc[0]:.1f}""")
                st.write('\n')
        
        fig, ax = plt.subplots(len(check_box_for_ma), 4, figsize=(20, 10 * len(check_box_for_ma)), squeeze=False)

        for i, city in enumerate(check_box_for_ma):
            city_ma = df_ma[df_ma['city'] == city]
            ax[i, 0].plot(city_ma['timestamp'], city_ma['temperature_ma'], c='black')

            minimum = df_aggs[df_aggs['city'] == city].groupby('city')['temp_min'].min().iloc[0]
            maximum = df_aggs[df_aggs['city'] == city].groupby('city')['temp_max'].max().iloc[0]
            mean_val = df_aggs[df_aggs['city'] == city].groupby('city')['temp_mean'].mean().iloc[0]

            ax[i, 0].plot((pd.Timestamp('2010-01-01'), pd.Timestamp('2019-12-29')), (minimum, minimum), c='b')
            ax[i, 0].plot((pd.Timestamp('2010-01-01'), pd.Timestamp('2019-12-29')), (maximum, maximum), c='r')
            ax[i, 0].plot((pd.Timestamp('2010-01-01'), pd.Timestamp('2019-12-29')), (mean_val, mean_val), c='g')

            ax[i, 0].set_title(f"City: {city}")
            ax[i, 0].set_xlabel("Timestamp")
            ax[i, 0].set_ylabel("Temperature (MA)")

        for i, city in enumerate(check_box_for_ma):
            city_data = df[df['city'] == city].copy()
            for season in seasons:
                mean_s = df_aggs[(df_aggs['city'] == city) & (df_aggs['season'] == season)]['temp_mean'].iloc[0]
                std_s = df_aggs[(df_aggs['city'] == city) & (df_aggs['season'] == season)]['temp_std'].iloc[0]

                city_data['is_anomaly'] = (city_data['temperature'] > mean_s + 2 * std_s) | (city_data['temperature'] < mean_s - 2 * std_s)

                ax[i, 1].scatter(
                    city_data[(city_data['is_anomaly'] == False) & (city_data['season'] == season)]['timestamp'],
                    city_data[(city_data['is_anomaly'] == False) & (city_data['season'] == season)]['temperature'],
                    c='b'
                )
                ax[i, 1].scatter(
                    city_data[(city_data['is_anomaly'] == True) & (city_data['season'] == season)]['timestamp'],
                    city_data[(city_data['is_anomaly'] == True) & (city_data['season'] == season)]['temperature'],
                    c='r'
                )
                ax[i, 1].set_title(f'City: {city}, Season: {season}')
                ax[i, 1].set_xlabel('Timestamp')
                ax[i, 1].set_ylabel('Temperature')

        for i, city in enumerate(check_box_for_ma):
            city_data = df[df['city'] == city]

            # Plot ACF on ax[i, 2]
            plot_acf(city_data['temperature'], lags=30, ax=ax[i, 2])
            ax[i, 2].set_title("Auto-Correlation Function (ACF)")
            ax[i, 2].set_xlabel("Lag")
            ax[i, 2].set_ylabel("Autocorrelation")

            # Plot PACF on ax[i, 3]
            plot_pacf(city_data['temperature'], lags=30, ax=ax[i, 3], method='ywm')
            ax[i, 3].set_title("Partial Auto-Correlation Function (PACF)")
            ax[i, 3].set_xlabel("Lag")
            ax[i, 3].set_ylabel("Partial Autocorrelation")

        st.pyplot(fig)

    if (st.button('Check weather anomality of your city')):
        try:
            result = requests.get(f'https://api.openweathermap.org/data/2.5/weather?lat={city_coordinates[city_picked]['lat']}&lon={city_coordinates[city_picked]['lon']}&appid={API_KEY}').json()
            st.success(f'Текущая температура в городе {city_picked}: {result["main"]["temp"] - 273.15:.1f} градусов Цельсия!')
        
            current_month = datetime.datetime.now().month
            month_to_season = {1: 'winter', 2: 'winter', 3: 'spring', 4: 'spring', 
                            5: 'spring', 6: 'summer', 7: 'summer', 8: 'summer', 
                            9: 'autumn', 10: 'autumn', 11: 'autumn', 12: 'winter', }
            current_season = month_to_season[current_month]
            mean_s = df_aggs[(df_aggs['city'] == city_picked) & (df_aggs['season'] == current_season)]['temp_mean'].iloc[0]
            std_s = df_aggs[(df_aggs['city'] == city_picked) & (df_aggs['season'] == current_season)]['temp_std'].iloc[0]

            is_anomaly = (result["main"]["temp"] - 273.15 > mean_s + 2 * std_s) | (result["main"]["temp"] - 273.15 < mean_s - 2 * std_s)

            if is_anomaly:
                st.success("Это аномалия!")
            else:
                st.success("Это нормально!")
        except KeyError:
            st.error('{"cod":401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}')
