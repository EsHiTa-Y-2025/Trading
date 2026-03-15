from datetime import date, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Range Instrument Levels", layout="wide")


# -----------------------------
# Instrument universe
# -----------------------------
DEFAULT_INSTRUMENTS = {
    "Bank Nifty": "^NSEBANK",
    "Nifty 50": "^NSEI",
    "FinNifty": "NIFTY_FIN_SERVICE.NS",
    "Sensex": "^BSESN",
    "Nifty Next 50": "^NSMIDCP",
    "Reliance": "RELIANCE.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "Infosys": "INFY.NS",
    "TCS": "TCS.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Bank": "KOTAKBANK.NS",
    "ITC": "ITC.NS",
    "LT": "LT.NS",
    "Bharti Airtel": "BHARTIARTL.NS",
    "HCL Tech": "HCLTECH.NS",
    "Wipro": "WIPRO.NS",
    "Maruti": "MARUTI.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Sun Pharma": "SUNPHARMA.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Asian Paints": "ASIANPAINT.NS",
    "Titan": "TITAN.NS",
    "UltraTech Cement": "ULTRACEMCO.NS",
    "Power Grid": "POWERGRID.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "Coal India": "COALINDIA.NS",
}


# -----------------------------
# Helpers
# -----------------------------
def digit_sum_reduce(n: int) -> int:
    """Reduce number by repeated digit sum until one digit remains."""
    n = abs(int(n))
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n


def normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle possible MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        level0 = list(df.columns.get_level_values(0))
        level1 = list(df.columns.get_level_values(1))
        price_fields = {"Open", "High", "Low", "Close", "Adj Close", "Volume"}

        if set(level0) & price_fields:
            df.columns = df.columns.get_level_values(0)
        elif set(level1) & price_fields:
            df.columns = df.columns.get_level_values(1)
        else:
            df.columns = [
                "_".join([str(x) for x in col if str(x) != ""]).strip("_")
                for col in df.columns.to_flat_index()
            ]
    return df


@st.cache_data(ttl=300)
def fetch_instrument_daily(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetch daily candles for the selected instrument.
    End gets +1 day because yfinance end is effectively exclusive.
    """
    df = yf.download(
        symbol,
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date) + pd.Timedelta(days=1),
        interval="1d",
        auto_adjust=False,
        progress=False,
        multi_level_index=False,
    )

    if df.empty:
        return df

    df = normalize_yf_columns(df)
    df = df.dropna(how="all")

    required = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in required if c in df.columns]
    if len(available) < 5:
        raise ValueError(f"Expected OHLCV columns not found. Got: {list(df.columns)}")

    df = df[required].copy().dropna()
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def calculate_levels_from_date_range(range_df: pd.DataFrame) -> dict:
    """
    New logic:
    - highest high from all dates in range
    - lowest low from all dates in range
    - close of the end date (practically: last available trading day's close in range)
    """
    highest_high = int(range_df["High"].max())
    lowest_low = int(range_df["Low"].min())
    main_number = highest_high - lowest_low

    reduced_digit = digit_sum_reduce(main_number)
    if reduced_digit == 0:
        raise ValueError("Reduced digit became 0, so division is not possible.")

    derived_number = main_number // reduced_digit

    last_row = range_df.iloc[-1]
    close_used = int(last_row["Close"])
    high_used = int(last_row["High"])
    low_used = int(last_row["Low"])
    close_date_used = pd.to_datetime(range_df.index[-1]).strftime("%d/%m/%Y")

    upper_point = close_used + derived_number
    lower_point = close_used - derived_number
    buying_point = high_used - derived_number
    selling_point = low_used - derived_number

    return {
        "highest_high": highest_high,
        "lowest_low": lowest_low,
        "main_number": main_number,
        "reduced_digit": reduced_digit,
        "derived_number": derived_number,
        "close_used": close_used,
        "high_used": high_used,
        "low_used": low_used,
        "close_date_used": close_date_used,
        "upper_point": upper_point,
        "lower_point": lower_point,
        "buying_point": buying_point,
        "selling_point": selling_point,
    }


# -----------------------------
# UI
# -----------------------------
st.title("Instrument Range Calculator")
st.caption(
    "Choose an instrument and date range. The app takes the highest high and lowest low from all dates in that range, uses the close of the end-date trading candle, and calculates the levels."
)

with st.sidebar:
    st.header("Inputs")
    today = date.today()
    default_end = today
    default_start = today - timedelta(days=60)

    instrument_names = list(DEFAULT_INSTRUMENTS.keys())

    selected_instruments = st.multiselect(
        "Search or select instrument(s)",
        options=instrument_names,
        default=["Bank Nifty"],
        help="Select one or more instruments from the list.",
    )

    custom_symbol = st.text_input(
        "Or enter custom Yahoo symbol",
        value="",
        placeholder="Example: ^NSEBANK or RELIANCE.NS",
        help="Use this if your instrument is not in the list.",
    ).strip()

    start_date = st.date_input(
        "Start date (DD/MM/YYYY)",
        value=default_start,
        format="DD/MM/YYYY",
    )

    end_date = st.date_input(
        "End date (DD/MM/YYYY)",
        value=default_end,
        format="DD/MM/YYYY",
    )

    calculate = st.button("Fetch data and calculate", type="primary", use_container_width=True)

st.write(
    f"**Selected Range:** {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}"
)

instrument_map = {}
for name in selected_instruments:
    instrument_map[name] = DEFAULT_INSTRUMENTS[name]

if custom_symbol:
    instrument_map[f"Custom ({custom_symbol})"] = custom_symbol

if calculate:
    try:
        if start_date > end_date:
            st.error("Start date must be earlier than or equal to end date.")
            st.stop()

        if not instrument_map:
            st.error("Please select at least one instrument or enter a custom Yahoo symbol.")
            st.stop()

        for instrument_name, symbol in instrument_map.items():
            st.divider()
            st.header(f"{instrument_name}  •  {symbol}")

            with st.spinner(f"Fetching live data for {instrument_name}..."):
                daily_df = fetch_instrument_daily(symbol, start_date, end_date)

            if daily_df.empty:
                st.warning(f"No data was returned for {instrument_name} in that date range.")
                continue

            # Keep only exact selected date range
            range_df = daily_df[
                (daily_df.index.date >= start_date) & (daily_df.index.date <= end_date)
            ].copy()

            if range_df.empty:
                st.warning(f"No trading data found for {instrument_name} in the selected date range.")
                continue

            result = calculate_levels_from_date_range(range_df)

            # Five points in descending order
            points = {
                "Upper Point": result["upper_point"],
                "Buying Point": result["buying_point"],
                "Close Used": result["close_used"],
                "Lower Point": result["lower_point"],
                "Selling Point": result["selling_point"],
            }
            sorted_points = sorted(points.items(), key=lambda x: x[1], reverse=True)

            st.subheader(f"Calculated levels for {instrument_name} (Descending Order)")
            cols = st.columns(len(sorted_points))
            for i, (label, value) in enumerate(sorted_points):
                cols[i].metric(label, f"{value:,}")

            st.divider()

            left, right = st.columns([1.15, 1])

            with left:
                st.subheader(f"Daily candles used for calculation ({instrument_name})")
                display_df = range_df.copy()
                display_df.index = display_df.index.strftime("%d/%m/%Y")
                st.dataframe(
                    display_df[["Open", "High", "Low", "Close", "Volume"]],
                    use_container_width=True,
                )

            with right:
                st.subheader("Calculation details")
                st.markdown(
                    f"""
- **Instrument**: {instrument_name}
- **Symbol**: `{symbol}`
- **Range Selected**: {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}
- **Highest High in Range**: {result['highest_high']:,}
- **Lowest Low in Range**: {result['lowest_low']:,}
- **High - Low**: {result['main_number']:,}
- **Digit reduction**: {result['reduced_digit']}
- **Derived number**: {result['derived_number']:,}
- **Close Used**: {result['close_used']:,}
- **Close Date Used**: {result['close_date_used']}
- **High of Close Date Candle**: {result['high_used']:,}
- **Low of Close Date Candle**: {result['low_used']:,}
                    """
                )

        st.divider()
        st.subheader("Formula used")
        st.code(
            """1. Take the highest High from all dates in the selected range
2. Take the lowest Low from all dates in the selected range
3. main_number = highest_high - lowest_low
4. Reduce main_number by adding digits repeatedly until one digit remains
5. derived_number = floor(main_number / reduced_digit)
6. close_used = close of the end-date trading candle (last available trading day in range)
7. upper_point   = close_used + derived_number
8. lower_point   = close_used - derived_number
9. buying_point  = high_of_close_date_candle - derived_number
10. selling_point = low_of_close_date_candle - derived_number""",
            language="text",
        )

    except Exception as e:
        st.exception(e)

else:
    st.info("Choose one or more instruments and a date range in the sidebar, then click **Fetch data and calculate**.")

    st.markdown(
        """
### Features
- Search and select one or more instruments
- Optionally enter a custom Yahoo Finance symbol
- Date display in **DD/MM/YYYY**
- Uses the **exact selected date range**
- Takes **highest high** and **lowest low** from all dates in the range
- Uses the **last available trading close** in the range
- Displays **5 values** in **descending order**:
  - Upper Point
  - Buying Point
  - Close Used
  - Lower Point
  - Selling Point
        """
    )
