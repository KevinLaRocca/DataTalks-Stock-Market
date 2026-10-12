import re
import numpy as np
import pandas as pd
import yfinance as yf
import gdown

url_q1 = "https://www.iposcoop.com/ipos-recently-filed/"
tables_q1 = pd.read_html(url_q1, storage_options={"User-Agent": "Mozilla/5.0"})
df_q1 = tables_q1[0]

df_withdrawn = df_q1[df_q1["Expected To Trade"] == "Withdrawn"].copy()


def classify_company(name):
    if not isinstance(name, str):
        return "Other"
    if "Technologies" in name:
        return "Technologies"
    elif any(
        kw in name
        for kw in ["Acquisition Corp", "Acquisition Corporation", "Corp"]
    ):
        return "Acquisition Corp"
    elif "Inc" in name or "Incorporated" in name:
        return "Inc."
    elif "Group" in name:
        return "Group"
    elif "Ltd" in name or "Limited" in name:
        return "Limited"
    elif "Holding" in name or "Holdings" in name:
        return "Holdings"
    else:
        return "Other"


df_withdrawn["Company Type"] = df_withdrawn["Company"].apply(classify_company)


def clean_numeric(val):
    if pd.isna(val) or str(val).strip() in ["-", "", "N/A"]:
        return np.nan
    cleaned = re.sub(r"[^\d.]", "", str(val))
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


price_low = df_withdrawn["Price Low"].apply(clean_numeric)
price_high = df_withdrawn["Price High"].apply(clean_numeric)
df_withdrawn["Avg_price"] = (price_low + price_high) / 2.0

df_withdrawn["Shares_clean"] = df_withdrawn["Shares (millions)"].apply(
    clean_numeric
)
df_withdrawn["Est_Vol_clean"] = df_withdrawn["Est $ Vol (millions)"].apply(
    clean_numeric
)

calculated_val = df_withdrawn["Shares_clean"] * df_withdrawn["Avg_price"]
df_withdrawn["Shares_offered_value"] = np.where(
    calculated_val.notna(), calculated_val, df_withdrawn["Est_Vol_clean"]
)

result_q1 = (
    df_withdrawn.groupby("Company Type")["Shares_offered_value"]
    .sum()
    .reset_index()
)
result_q1 = result_q1.sort_values(by="Shares_offered_value", ascending=False)

top_class = result_q1.iloc[0]["Company Type"]
top_value = result_q1.iloc[0]["Shares_offered_value"]

print(f"Clase con mayor valor cancelado: {top_class}")
print(f"Monto total: ${top_value:,.2f} millones")


url_q2 = "https://www.iposcoop.com/2025-pricings/"
tables_q2 = pd.read_html(url_q2, storage_options={"User-Agent": "Mozilla/5.0"})
df_ipos = tables_q2[0]

df_ipos["Offer Date"] = pd.to_datetime(df_ipos["Offer Date"])

return_col = [c for c in df_ipos.columns if "Return" in c or "1st Day" in c][0]


def parse_return(val):
    if pd.isna(val):
        return np.nan
    val_str = re.sub(r"[^\d.-]", "", str(val))
    try:
        return float(val_str)
    except ValueError:
        return np.nan


df_ipos["Return_num"] = df_ipos[return_col].apply(parse_return)

df_filtered = df_ipos[
    (df_ipos["Offer Date"] < "2025-09-01")
    & (df_ipos["Return_num"].notna())
    & (df_ipos["Return_num"] != 0)
].copy()

symbol_col = [c for c in df_filtered.columns if "Symbol" in c or "Ticker" in c][
    0
]
tickers = (
    df_filtered[symbol_col].dropna().astype(str).str.strip().str.upper().tolist()
)

print(f"Tickers filtrados: {len(tickers)}")

data = yf.download(
    tickers=tickers,
    start="2025-01-01",
    end="2026-09-13",
    group_by="ticker",
    threads=True,
)

processed_dfs = []

for ticker in tickers:
    try:
        if isinstance(data.columns, pd.MultiIndex):
            if ticker in data.columns.levels[0]:
                df_ticker = data[ticker].dropna(how="all").copy()
            else:
                continue
        else:
            df_ticker = data.dropna(how="all").copy()

        if not df_ticker.empty and "Close" in df_ticker:
            close_series = df_ticker["Close"]
            if isinstance(close_series, pd.DataFrame):
                close_series = close_series.iloc[:, 0]

            if close_series.dropna().empty:
                continue

            df_ticker["Ticker"] = ticker
            df_ticker["growth_252d"] = close_series / close_series.shift(252)
            df_ticker["volatility"] = close_series.rolling(30).std() * np.sqrt(
                252
            )
            df_ticker["Sharpe"] = (
                df_ticker["growth_252d"] - 0.05
            ) / df_ticker["volatility"]

            processed_dfs.append(df_ticker)
    except Exception:
        continue

print(f"Tickers con datos válidos: {len(processed_dfs)}")

stocks_df = pd.concat(processed_dfs)
stocks_df.index = pd.to_datetime(stocks_df.index)

df_target = stocks_df[stocks_df.index == "2026-09-11"]

print(df_target["Sharpe"].describe())
print(f"Mediana del Sharpe Ratio: {df_target['Sharpe'].median():.4f}")


#Ejercicio 3

monthly_medians = {}
monthly_means = {}

aligned_records = []

for ticker, group in stocks_df.groupby("Ticker"):
    group = group.sort_index()
    if group.empty:
        continue

    ipo_date = group.index.min()
    ipo_close = group.loc[ipo_date, "Close"]
    if isinstance(ipo_close, pd.Series):
        ipo_close = ipo_close.iloc[0]

    record = {"Ticker": ticker, "IPO_Date": ipo_date, "IPO_Close": ipo_close}

    for m in range(1, 13):
        days = m * 21
        if len(group) > days:
            future_close = group["Close"].iloc[days]
            if isinstance(future_close, pd.Series):
                future_close = future_close.iloc[0]
            record[f"future_growth_{m}_m"] = future_close / ipo_close
        else:
            record[f"future_growth_{m}_m"] = np.nan

    aligned_records.append(record)

growth_df = pd.DataFrame(aligned_records)

growth_cols = [f"future_growth_{m}_m" for m in range(1, 13)]
stats_df = growth_df[growth_cols].describe().T

for m in range(1, 13):
    col = f"future_growth_{m}_m"
    monthly_medians[m] = growth_df[col].median()
    monthly_means[m] = growth_df[col].mean()

optimal_month = max(monthly_medians, key=monthly_medians.get)
max_median_growth = monthly_medians[optimal_month]

print(stats_df[["mean", "50%", "min", "max"]])
print(
    f"\nOptimal Holding Period: {optimal_month} month(s) with median growth: {max_median_growth:.4f}"
)


file_id = "1grCTCzMZKY5sJRtdbLVCXg8JXA8VPyg-"
gdown.download(
    f"https://drive.google.com/uc?id={file_id}", "data.parquet", quiet=False
)

df = pd.read_parquet("data.parquet")

df["Date"] = pd.to_datetime(df["Date"])

selected_df = df[
    (df["Date"] >= "2000-01-01")
    & (df["Date"] <= "2025-06-01")
    & (df["rsi"] < 30)
].copy()

net_income = 1000 * (selected_df["growth_future_30d"] - 1).sum()
net_income_thousands = net_income / 1000

print(f"Total trades: {len(selected_df)}")
print(f"Net income: ${net_income:,.2f}")
print(f"Net income (in $ thousands): ${net_income_thousands:,.2f}k")
print(f"Net income (in $ thousands): ${net_income_thousands:,.2f}k")