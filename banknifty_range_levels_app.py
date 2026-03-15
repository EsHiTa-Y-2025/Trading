from datetime import date, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="5-Week Instrument Levels", layout="wide")


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


def format_date_ddmmyyyy(value) -> str:
    return pd.to_datetime(value).strftime("%d/%m/%Y")


@st.cache_data(ttl=300)
def fetch_instrument_daily(symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
    """
    Fetch daily candles for the selected instrument.
    yfinance end behaves effectively like exclusive, so add 1 day.
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
    return df


def build_monday_to_friday_weeks(daily_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily candles into Monday-Friday trading weeks.
    W-FRI means each weekly bar ends on Friday and contains Mon->Fri data.
    """
    weekly = daily_df.resample("W-FRI").agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )

    # Keep reasonably complete weeks only
    counts = daily_df["Close"].resample("W-FRI").count()
    weekly["days_count"] = counts
    weekly = weekly[weekly["days_count"] >= 4].drop(columns=["days_count"])

    return weekly.dropna()


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
st.title("5-Week Instrument Range Calculator")
st.caption(
    "Choose an instrument and date range, fetch live data, and calculate Upper, Buying, Close Used, Lower, and Selling points from the last 5 completed weeks inside that range."
)

with st.sidebar:
    st.header("Inputs")
    today = date.today()
    default_end = today
    default_start = today - timedelta(days=140)

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

    start_date = st.date_input("Start date (DD/MM/YYYY)", value=default_start)
    end_date = st.date_input("End date (DD/MM/YYYY)", value=default_end)

    calculate = st.button("Fetch data and calculate", type="primary", use_container_width=True)

st.write(
    f"**Selected Range:** {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}"
)

# Final instrument set to process
instrument_map = {}

for name in selected_instruments:
    instrument_map[name] = DEFAULT_INSTRUMENTS[name]

if custom_symbol:
    instrument_map[f"Custom ({custom_symbol})"] = custom_symbol

if calculate:
    try:
        if start_date >= end_date:
            st.error("Start date must be earlier than end date.")
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

            weekly_df = build_monday_to_friday_weeks(daily_df)

            if weekly_df.empty:
                st.warning(f"No completed Monday-Friday weekly candles were found for {instrument_name}.")
                continue

            weekly_df = weekly_df[
                (weekly_df.index.date >= start_date) & (weekly_df.index.date <= end_date)
            ]

            if len(weekly_df) < 5:
                st.warning(
                    f"{instrument_name}: only {len(weekly_df)} completed week(s) found. Choose a larger range with at least 5 completed weeks."
                )
                weekly_preview = weekly_df.copy()
                if not weekly_preview.empty:
                    weekly_preview.index = weekly_preview.index.strftime("%d/%m/%Y")
                    st.dataframe(
                        weekly_preview[["Open", "High", "Low", "Close", "Volume"]],
                        use_container_width=True,
                    )
                continue

            selected_weeks = weekly_df.tail(5)
            result = calculate_levels_from_5_weeks(selected_weeks)

            # Five points, sorted descending
            points = {
                "Upper Point": result["upper_point"],
                "Buying Point": result["buying_point"],
                "Close Used": result["last_week_close"],
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
                st.subheader(f"5-week set used for calculation ({instrument_name})")
                display_weeks = selected_weeks.copy()
                display_weeks.index = display_weeks.index.strftime("%d/%m/%Y")
                st.dataframe(
                    display_weeks[["Open", "High", "Low", "Close", "Volume"]],
                    use_container_width=True,
                )

            with right:
                st.subheader("Calculation details")
                st.markdown(
                    f"""
- **Instrument**: {instrument_name}
- **Symbol**: `{symbol}`
- **Range Selected**: {start_date.strftime('%d/%m/%Y')} → {end_date.strftime('%d/%m/%Y')}
- **5-week High**: {result['five_week_high']:,}
- **5-week Low**: {result['five_week_low']:,}
- **High - Low**: {result['main_number']:,}
- **Digit reduction**: {result['reduced_digit']}
- **Derived number**: {result['derived_number']:,}
- **Last week Close used**: {result['last_week_close']:,}
- **Last week High**: {result['last_week_high']:,}
- **Last week Low**: {result['last_week_low']:,}
                    """
                )

            st.subheader(f"All completed weekly candles in selected range ({instrument_name})")
            weekly_display = weekly_df.copy()
            weekly_display.index = weekly_display.index.strftime("%d/%m/%Y")
            st.dataframe(
                weekly_display[["Open", "High", "Low", "Close", "Volume"]],
                use_container_width=True,
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
8. selling_point = last_week_low   - derived_number
9. close_used    = last_week_close""",
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
- Weekly candles built as **Monday-Friday**
- Uses the **last 5 completed weeks**
- Displays **5 values** in **descending order**:
  - Upper Point
  - Buying Point
  - Close Used
  - Lower Point
  - Selling Point
        """
    )
