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
    try:
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y%m%d")
        url = f"https://opendata.chmi.cz/meteorology/climate/now/metadata/meta1-{yesterday}.json"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        meta_json = r.json()

        return {
            f"{row[2]} ({row[1]})": row[0]
            for row in meta_json['data']['data']['values']
        }
    except Exception as e:
        st.error(f"Error loading stations: {e}")
        return {}

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
def centered_axis(ax, series, pad):
    if series is None or series.empty:
        return
    ax.set_ylim(series.min() - pad, series.max() + pad)


def plot_station(df, station_name):
    if df.empty:
        st.error("No data available")
        return

    df_pivot = df.pivot(index='DT', columns='ELEMENT', values='VAL')

    fig, ax_base = plt.subplots(figsize=(16,6))
    ax_base.set_yticks([])
    ax_base.spines['left'].set_visible(False)

    # ---------------- TEMPERATURE ----------------
    temp_series = None
    if 'T' in df_pivot and 'TPM' in df_pivot:
        temp_series = pd.concat([df_pivot['T'], df_pivot['TPM']])
    elif 'T' in df_pivot:
        temp_series = df_pivot['T']
    elif 'TPM' in df_pivot:
        temp_series = df_pivot['TPM']

    ax_temp = None
    if temp_series is not None and not temp_series.empty:
        ax_temp = ax_base.twinx()
        ax_temp.spines['right'].set_position(('outward', 0))
        ax_temp.tick_params(axis='y', colors='red')

        if 'T' in df_pivot:
            ax_temp.plot(df_pivot.index, df_pivot['T'], color='red')

        if 'TPM' in df_pivot:
            ax_temp.plot(df_pivot.index, df_pivot['TPM'], color='#9636b6', linewidth=1)

        centered_axis(ax_temp, temp_series, 10)

        # horizontal lines every 5°C
        ymin, ymax = ax_temp.get_ylim()
        for y in range(int(ymin//5*5), int(ymax//5*5)+5, 5):
            ax_temp.axhline(y=y, color='lightblue', linestyle='--', linewidth=0.5)

    # ---------------- WIND ----------------
    ax_wind = None
    if 'Fmax' in df_pivot or 'Fprum' in df_pivot:
        ax_wind = ax_base.twinx()
        ax_wind.spines['right'].set_position(('outward', 30))

        if 'Fmax' in df_pivot:
            ax_wind.plot(df_pivot.index, df_pivot['Fmax'], color='#967b60')

        if 'Fprum' in df_pivot:
            ax_wind.plot(df_pivot.index, df_pivot['Fprum'], color='green')

        ax_wind.tick_params(axis='y', colors='green')
        ax_wind.set_ylim(0, max(5, df_pivot[['Fmax','Fprum']].max().max()*1.2))

    # ---------------- HUMIDITY ----------------
    ax_h = None
    if 'H' in df_pivot:
        ax_h = ax_base.twinx()
        ax_h.spines['right'].set_position(('outward', 60))
        ax_h.plot(df_pivot.index, df_pivot['H'], color='#09f8f8')
        ax_h.set_ylim(0, 100)
        ax_h.tick_params(axis='y', colors='#09f8f8')

    # ---------------- RAIN ----------------
    ax_r = None
    if 'SRA10M' in df_pivot:
        ax_r = ax_base.twinx()
        ax_r.spines['right'].set_position(('outward', 90))
        ax_r.plot(df_pivot.index, df_pivot['SRA10M'], color='blue')
        ax_r.tick_params(axis='y', colors='blue')

    # ---------------- TIME AXIS ----------------
    end_time = df_pivot.index.max()
    start_time = end_time - pd.Timedelta(hours=48)

    if ax_temp:
        ax_temp.set_xlim(start_time, end_time)

        ax_temp.xaxis.set_major_locator(mdates.HourLocator(byhour=[0,4,8,12,16,20]))
        ax_temp.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M\n%d.%m'))

    # ---------------- FINAL ----------------
    plt.title(station_name)
    fig.subplots_adjust(left=0.05, right=0.75, bottom=0.15)

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
