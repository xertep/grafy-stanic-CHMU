import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib.dates as mdates
import numpy as np
import math

st.set_page_config(layout="wide")

# ---------------- CONFIG ----------------
elements = ['T', 'TPM', 'Fmax', 'Fprum', 'H', 'SSV10M', 'D', 'P', 'SRA10M', 'SCEa', 'SCE']
BASE_URL = "https://opendata.chmi.cz/meteorology/climate/now/data/"

element_names = {
    "T": "Teplota (°C)",
    "TPM": "Teplota přízemní (°C)",
    "Fprum": "Vítr průměrný (m/s)",
    "Fmax": "Vítr nárazy (m/s)",
    "SRA10M": "Srážky (mm)",
    "H": "Vlhkost (%)"
}

# ---------------- LOAD STATIONS ----------------
@st.cache_data
def load_stations():
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
    url = f"https://opendata.chmi.cz/meteorology/climate/now/metadata/meta1-{yesterday}.json"
    r = requests.get(url)
    r.raise_for_status()
    meta_json = r.json()

    return {
        f"{row[2]} ({row[1]})": row[0]
        for row in meta_json['data']['data']['values']
    }

stations = load_stations()

# ---------------- DATA FETCH ----------------
def fetch_station_data(wsi):
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y%m%d") for i in [2,1,0]]
    combined_df = pd.DataFrame()

    for date_str in dates:
        url = f"{BASE_URL}10m-{wsi}-{date_str}.json"
        try:
            r = requests.get(url)
            if r.status_code != 200:
                continue
            data = r.json()
        except:
            continue

        header = data['data']['data']['header'].split(',')
        values = data['data']['data']['values']

        df = pd.DataFrame(values, columns=header)
        df['DT'] = pd.to_datetime(df['DT'], utc=True)\
                       .dt.tz_convert('Europe/Prague')\
                       .dt.tz_localize(None)

        df['VAL'] = pd.to_numeric(df['VAL'], errors='coerce')
        df = df[df['ELEMENT'].isin(elements)]

        combined_df = pd.concat([combined_df, df])

    return combined_df


# ---------------- PLOT ----------------
def plot_station(df, station_name):
    if df.empty:
        st.error("No data available")
        return

    df_pivot = df.pivot(index='DT', columns='ELEMENT', values='VAL')

    fig, ax = plt.subplots(figsize=(16,6))

    # --- Temperature ---
    if 'T' in df_pivot:
        ax.plot(df_pivot.index, df_pivot['T'], label='T', color='red')

    if 'TPM' in df_pivot:
        ax.plot(df_pivot.index, df_pivot['TPM'], label='TPM', color='purple')

    # --- Wind ---
    if 'Fprum' in df_pivot:
        ax.plot(df_pivot.index, df_pivot['Fprum'], label='Wind avg', color='green')

    if 'Fmax' in df_pivot:
        ax.plot(df_pivot.index, df_pivot['Fmax'], label='Wind max', color='brown')

    # --- Humidity ---
    if 'H' in df_pivot:
        ax.plot(df_pivot.index, df_pivot['H'], label='Humidity', color='cyan')

    # --- Rain ---
    if 'SRA10M' in df_pivot:
        ax.plot(df_pivot.index, df_pivot['SRA10M'], label='Rain', color='blue')

    # --- X axis ---
    end_time = df_pivot.index.max()
    start_time = end_time - pd.Timedelta(hours=48)
    ax.set_xlim(start_time, end_time)

    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0,4,8,12,16,20]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d.%m'))

    plt.xticks(rotation=0)

    ax.set_title(station_name)
    ax.legend()

    st.pyplot(fig)


# ---------------- UI ----------------
st.title("ČHMÚ meteostanice 🌦️")

# Station select
station_name = st.selectbox("Vyber stanici", list(stations.keys()))

if st.button("Zobraz data"):
    wsi = stations[station_name]

    with st.spinner("Načítám data..."):
        df = fetch_station_data(wsi)

    plot_station(df, station_name)
