import io
import pandas as pd
import requests
import yfinance as yf

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

response = requests.get(url, headers=headers)
tables = pd.read_html(io.StringIO(response.text))

df_sp500 = tables[0].copy()
df_sp500.columns = [str(col).strip() for col in df_sp500.columns]

df_sp500['Date added'] = pd.to_datetime(df_sp500['Date added'], errors='coerce')
df_sp500['Year'] = df_sp500['Date added'].dt.year

additions_since_2020 = df_sp500[df_sp500['Year'] >= 2020]['Year'].value_counts().sort_index()

top_year = int(additions_since_2020.idxmax())
top_count = int(additions_since_2020.max())

cutoff_year = 2026 - 20
more_than_20_yrs = (df_sp500['Year'] <= cutoff_year).sum()

print("Adiciones por año desde 2020:")
print(additions_since_2020)
print(f"\nAño con más adiciones: {top_year} ({top_count} empresas)")
print(f"Empresas con más de 20 años en el índice: {more_than_20_yrs}")

tickers = {
    'US - S&P 500': '^GSPC',
    'China - Shanghai Composite': '000001.SS',
    'Hong Kong - HANG SENG': '^HSI',
    'Australia - S&P/ASX 200': '^AXJO',
    'India - Nifty 50': '^NSEI',
    'Canada - S&P/TSX Composite': '^GSPTSE',
    'Germany - DAX': '^GDAXI',
    'UK - FTSE 100': '^FTSE',
    'Japan - Nikkei 225': '^N225',
    'Mexico - IPC Mexico': '^MXX',
    'Brazil - Ibovespa': '^BVSP'
}

data = yf.download(list(tickers.values()), start='2026-01-01', end='2026-08-22')['Close']

ytd_returns = {}

for country, ticker in tickers.items():
    series = data[ticker].dropna()
    if not series.empty:
        first_price = series.iloc[0]
        last_price = series.iloc[-1]
        ytd_return = ((last_price - first_price) / first_price) * 100
        ytd_returns[country] = ytd_return

df_results = pd.DataFrame(list(ytd_returns.items()), columns=['Index', 'YTD_Return_%'])
df_results = df_results.sort_values(by='YTD_Return_%', ascending=False).reset_index(drop=True)

sp500_return = df_results.loc[df_results['Index'] == 'US - S&P 500', 'YTD_Return_%'].values[0]
better_than_us = df_results[df_results['YTD_Return_%'] > sp500_return]

print("--- Rendimientos YTD (01-Ene a 21-Ago 2026) ---")
print(df_results.to_string(index=False))

print(f"\nRendimiento S&P 500 (EE.UU.): {sp500_return:.2f}%")
print(f"Cantidad de índices que superaron al S&P 500: {len(better_than_us)} de 10")

sp500 = yf.download("^GSPC", start="1950-01-01")
if isinstance(sp500.columns, pd.MultiIndex):
    sp500.columns = sp500.columns.get_level_values(0)

sp500['Historical_High'] = sp500['Close'].cummax()
ath_df = sp500[sp500['Close'] == sp500['Historical_High']]

correcciones = []
for i in range(len(ath_df) - 1):
    start = ath_df.index[i]
    end = ath_df.index[i + 1]

    precio_ath_inicio = ath_df['Close'].iloc[i]

    ventana = sp500.loc[start:end]
    
    precio_minimo = ventana['Close'].min()
    fecha_minimo = ventana['Close'].idxmin()

    drawdown_pct = ((precio_ath_inicio - precio_minimo) / precio_ath_inicio) * 100

    correcciones.append({
        'ath_inicio_fecha': start,
        'ath_inicio_precio': precio_ath_inicio,
        'minimo_fecha': fecha_minimo,
        'minimo_precio': precio_minimo,
        'ath_fin_fecha': end,
        'drawdown_pct': drawdown_pct
    })

df_correcciones = pd.DataFrame(correcciones)
correcciones5 = df_correcciones[df_correcciones['drawdown_pct'] >= 5.0]

correcciones5 = correcciones5.copy()
correcciones5['dias_hasta_recuperar'] = (correcciones5['ath_fin_fecha'] - correcciones5['ath_inicio_fecha']).dt.days
correcciones5['dias_hasta_piso'] = (correcciones5['minimo_fecha'] - correcciones5['ath_inicio_fecha']).dt.days

dd_p25 = correcciones5['drawdown_pct'].quantile(0.25)
dd_p50 = correcciones5['drawdown_pct'].quantile(0.50)
dd_p75 = correcciones5['drawdown_pct'].quantile(0.75)

dur_p25 = correcciones5['dias_hasta_recuperar'].quantile(0.25)
dur_p50 = correcciones5['dias_hasta_recuperar'].quantile(0.50)
dur_p75 = correcciones5['dias_hasta_recuperar'].quantile(0.75)

piso_p25 = correcciones5['dias_hasta_piso'].quantile(0.25)
piso_p50 = correcciones5['dias_hasta_piso'].quantile(0.50)
piso_p75 = correcciones5['dias_hasta_piso'].quantile(0.75)

print("\n=== PERCENTILES DE CORRECCIONES (>= 5% Drawdown) ===")
print(f"Drawdown (%):    25th: {dd_p25:.2f}% | 50th (Mediana): {dd_p50:.2f}% | 75th: {dd_p75:.2f}%")
print(f"Días a Nuevo ATH: 25th: {dur_p25:.0f} días | 50th (Mediana): {dur_p50:.0f} días | 75th: {dur_p75:.0f} días")
print(f"Días al Piso:    25th: {piso_p25:.0f} días | 50th (Mediana): {piso_p50:.0f} días | 75th: {piso_p75:.0f} días")

ticker = 'AMZN'
ticker_obj = yf.Ticker(ticker)

earnings_df = ticker_obj.get_earnings_dates(limit=25)
earnings_df.index = earnings_df.index.tz_localize(None)

positive_surprises = earnings_df[earnings_df['Surprise(%)'] > 0].dropna(subset=['Surprise(%)']).copy()

prices_amzn_df = yf.download(ticker, period="max")
if isinstance(prices_amzn_df.columns, pd.MultiIndex):
    prices_amzn_df.columns = prices_amzn_df.columns.get_level_values(0)
    
prices_amzn = prices_amzn_df['Close']
if isinstance(prices_amzn, pd.DataFrame):
    prices_amzn = prices_amzn.squeeze()

prices_amzn.index = pd.to_datetime(prices_amzn.index).tz_localize(None).normalize()

returns_2d = []
surprises_list = []

for event_date, row in positive_surprises.iterrows():
    event_dt = pd.to_datetime(event_date).normalize()
    
    idx_search = prices_amzn.index.get_indexer([event_dt], method='pad')
    if idx_search[0] == -1:
        idx_search = prices_amzn.index.get_indexer([event_dt], method='bfill')
        
    idx_day2 = idx_search[0]
    idx_day1 = idx_day2 - 1
    idx_day3 = idx_day2 + 1
    
    if idx_day1 >= 0 and idx_day3 < len(prices_amzn):
        p1 = float(prices_amzn.iloc[idx_day1])
        p3 = float(prices_amzn.iloc[idx_day3])
        
        return_2d = (p3 / p1) - 1
        returns_2d.append(return_2d)
        surprises_list.append(row['Surprise(%)'])

df_q4 = pd.DataFrame({
    'Surprise_Magnitude': surprises_list,
    'Return_2D': returns_2d
})

median_2d_return_pct = df_q4['Return_2D'].median() * 100
correlation = df_q4.corr().iloc[0, 1]

print("\n=== Question 4 Results (AMZN) ===")
print(f"Mediana del retorno a 2 días (Positive Surprises): {median_2d_return_pct:.3f}%")
print(f"Correlación entre magnitud de sorpresa y retorno a 2 días: {correlation:.4f}")