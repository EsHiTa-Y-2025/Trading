import math
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Bank Nifty 5-Week Levels", layout="wide")


# -----------------------------
# Helpers
# -----------------------------
def digit_sum_reduce(n: int) -> int:
    """Reduce a number by repeatedly summing digits until one digit remains."""
    n = abs(int(n))
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n


def normalize_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Handle yfinance single-ticker MultiIndex columns safely."""
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
def fetch_banknifty_daily(start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetch Bank Nifty daily candles from Yahoo Finance.
    We add a small forward buffer because yfinance end date is exclusive in practice.
    """
    df = yf.download(
        "^NSEBANK",
        start=pd.Timestamp(start_date),
        end=pd.Timestamp(end_date) + pd.Timedelta(days=1),
        interval="1d",
        auto_adjust=False,
        progress=False,
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
    return df



def build_monday_to_friday_weeks(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily candles into Monday-Friday trading weeks.
    W-FRI means: each weekly bar ends on Friday and contains Mon->Fri data.
    """
    weekly = daily_df.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    ).dropna()
    return weekly



def calculate_levels_from_5_weeks(weekly_5: pd.DataFrame) -> dict:
    five_week_high = int(weekly_5["High"].max())
    five_week_low = int(weekly_5["Low"].min())
    main_number = five_week_high - five_week_low

    reduced_digit = digit_sum_reduce(main_number)
    if reduced_digit == 0:
        raise ValueError("Reduced digit became 0, so division is not possible.")

    derived_number = main_number // reduced_digit

    last_week = weekly_5.iloc[-1]
    last_week_close = int(last_week["Close"])
    last_week_high = int(last_week["High"])
    last_week_low = int(last_week["Low"])

    upper_point = last_week_close + derived_number
    lower_point = last_week_close - derived_number
    buying_point = last_week_high - derived_number
    selling_point = last_week_low - derived_number

    return {
        "five_week_high": five_week_high,
        "five_week_low": five_week_low,
        "main_number": main_number,
        "reduced_digit": reduced_digit,
        "derived_number": derived_number,
        "last_week_close": last_week_close,
        "last_week_high": last_week_high,
        "last_week_low": last_week_low,
        "upper_point": upper_point,
        "lower_point": lower_point,
        "buying_point": buying_point,
        "selling_point": selling_point,
    }


# -----------------------------
# UI
# -----------------------------
st.title("Bank Nifty 5-Week Range Calculator")
st.caption(
    "Pick a date range, fetch live Bank Nifty data, and calculate Upper, Lower, Buying, and Selling points from the last 5 completed weeks inside that range."
)

with st.sidebar:
    st.header("Inputs")
    today = date.today()
    default_end = today
    default_start = today - timedelta(days=140)

    start_date = st.date_input("Start date", value=default_start)
    end_date = st.date_input("End date", value=default_end)

    use_last_5_only = st.checkbox(
        "Use the last 5 completed weeks inside the selected range",
        value=True,
        help="If your selected range contains more than 5 weeks, the app uses the latest 5 completed weeks from that range.",
    )

    calculate = st.button("Fetch data and calculate", type="primary", use_container_width=True)


if calculate:
    try:
        if start_date >= end_date:
            st.error("Start date must be earlier than end date.")
            st.stop()

        with st.spinner("Fetching live Bank Nifty data..."):
            daily_df = fetch_banknifty_daily(start_date, end_date)

        if daily_df.empty:
            st.error("No Bank Nifty data was returned for that date range.")
            st.stop()

        weekly_df = build_monday_to_friday_weeks(daily_df)

        if weekly_df.empty:
            st.error("No completed Monday-Friday weekly candles were found in that range.")
            st.stop()

        if len(weekly_df) < 5:
            st.error(
                f"The selected range only contains {len(weekly_df)} completed week(s). Please choose a larger range with at least 5 completed weeks."
            )
            st.dataframe(weekly_df, use_container_width=True)
            st.stop()

        selected_weeks = weekly_df.tail(5) if use_last_5_only else weekly_df.tail(5)
        result = calculate_levels_from_5_weeks(selected_weeks)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Upper Point", f"{result['upper_point']:,}")
        c2.metric("Lower Point", f"{result['lower_point']:,}")
        c3.metric("Buying Point", f"{result['buying_point']:,}")
        c4.metric("Selling Point", f"{result['selling_point']:,}")

        st.divider()

        left, right = st.columns([1.1, 1])

        with left:
            st.subheader("5-week set used for calculation")
            display_weeks = selected_weeks.copy()
            display_weeks.index = display_weeks.index.date
            st.dataframe(display_weeks[["Open", "High", "Low", "Close", "Volume"]], use_container_width=True)

        with right:
            st.subheader("Calculation details")
            st.markdown(
                f"""
- **5-week High**: {result['five_week_high']:,}
- **5-week Low**: {result['five_week_low']:,}
- **High - Low**: {result['main_number']:,}
- **Digit reduction**: {result['reduced_digit']}
- **Derived number**: {result['derived_number']:,}
- **Last week Close**: {result['last_week_close']:,}
- **Last week High**: {result['last_week_high']:,}
- **Last week Low**: {result['last_week_low']:,}
                """
            )

        st.divider()
        st.subheader("Formula used")
        st.code(
            """1. Take the highest High and lowest Low from the 5-week set
2. main_number = High - Low
3. Reduce main_number by adding digits repeatedly until one digit remains
4. derived_number = floor(main_number / reduced_digit)
5. upper_point   = last_week_close + derived_number
6. lower_point   = last_week_close - derived_number
7. buying_point  = last_week_high  - derived_number
8. selling_point = last_week_low   - derived_number""",
            language="text",
        )

        st.subheader("All completed weekly candles in selected range")
        weekly_display = weekly_df.copy()
        weekly_display.index = weekly_display.index.date
        st.dataframe(weekly_display[["Open", "High", "Low", "Close", "Volume"]], use_container_width=True)

    except Exception as e:
        st.exception(e)
else:
    st.info("Choose a date range in the sidebar, then click **Fetch data and calculate**.")

    st.markdown(
        """
### What this app does
- Fetches **live Bank Nifty** daily data
- Converts it into **Monday-Friday weekly candles**
- Uses the **last 5 completed weeks** inside your selected date range
- Calculates:
  - **Upper Point**
  - **Lower Point**
  - **Buying Point**
  - **Selling Point**
        """
    )

    st.markdown(
        """
### Run locally
```bash
pip install streamlit yfinance pandas
streamlit run banknifty_range_levels_app.py
```
        """
    )
