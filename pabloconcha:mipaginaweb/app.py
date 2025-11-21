#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 15:51:32 2025

@author: pabloconcha
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ================== CONFIGURACIÓN GENERAL ==================
st.set_page_config(page_title="Concha Quant Lab", page_icon="💸", layout="wide")

st.title("Concha Quant Lab 💼📊")
st.write("Plataforma de análisis financiero by Pablo Concha (yfinance).")

# ================== LISTAS Y DICCIONARIOS ==================
DOW30 = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX",
    "DIS", "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM",
    "KO", "MCD", "MMM", "MRK", "MSFT", "NKE", "PG", "TRV",
    "UNH", "V", "VZ", "WBA", "WMT", "DOW"
]

COMPANIAS_DOW30 = {
    "AAPL": "Apple Inc.",
    "AMGN": "Amgen Inc.",
    "AXP": "American Express Company",
    "BA": "The Boeing Company",
    "CAT": "Caterpillar Inc.",
    "CRM": "Salesforce, Inc.",
    "CSCO": "Cisco Systems, Inc.",
    "CVX": "Chevron Corporation",
    "DIS": "The Walt Disney Company",
    "GS": "The Goldman Sachs Group, Inc.",
    "HD": "The Home Depot, Inc.",
    "HON": "Honeywell International Inc.",
    "IBM": "International Business Machines Corporation",
    "INTC": "Intel Corporation",
    "JNJ": "Johnson & Johnson",
    "JPM": "JPMorgan Chase & Co.",
    "KO": "The Coca-Cola Company",
    "MCD": "McDonald's Corporation",
    "MMM": "3M Company",
    "MRK": "Merck & Co., Inc.",
    "MSFT": "Microsoft Corporation",
    "NKE": "NIKE, Inc.",
    "PG": "The Procter & Gamble Company",
    "TRV": "The Travelers Companies, Inc.",
    "UNH": "UnitedHealth Group Incorporated",
    "V": "Visa Inc.",
    "VZ": "Verizon Communications Inc.",
    "WBA": "Walgreens Boots Alliance, Inc.",
    "WMT": "Walmart Inc.",
    "DOW": "Dow Inc."
}

OPCIONES_DOW30 = {f"{COMPANIAS_DOW30[t]} ({t})": t for t in DOW30}

# ================== FUNCIONES MERCADO ==================

@st.cache_data
def get_prices(tickers, period="1y"):
    data = yf.download(tickers, period=period, progress=False)

    if data is None or data.empty:
        raise ValueError("No se pudieron descargar precios para los tickers seleccionados.")

    # Si viene con MultiIndex (varios campos: Open, High, Low, Close, Adj Close, Volume)
    if isinstance(data.columns, pd.MultiIndex):
        first_level = data.columns.get_level_values(0)

        if "Adj Close" in first_level:
            precios = data["Adj Close"]
        elif "Close" in first_level:
            precios = data["Close"]
        else:
            raise ValueError("No se encontró columna de precios ('Adj Close' ni 'Close').")
    else:
        # DataFrame con columnas simples
        if "Adj Close" in data.columns:
            precios = data["Adj Close"]
        elif "Close" in data.columns:
            precios = data["Close"]
        else:
            raise ValueError("No se encontró columna de precios ('Adj Close' ni 'Close').")

    if isinstance(precios, pd.Series):
        precios = precios.to_frame()

    return precios



@st.cache_data
def get_returns(tickers, period="1y"):
    data = get_prices(tickers, period=period)
    rets = data.pct_change().dropna()
    return rets


def get_daily_returns(ticker, period="1y"):
    data = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if data is None or data.empty or "Close" not in data.columns:
        raise ValueError(f"No se pudieron descargar datos de {ticker}")
    precios = data["Close"]
    if isinstance(precios, pd.DataFrame):
        precios = precios.iloc[:, 0]
    rend = precios.pct_change().dropna()
    if rend.empty:
        raise ValueError(f"No hay rendimientos diarios para {ticker}")
    return rend

# ================== FUNCIONES ESTADOS FINANCIEROS ==================

def get_balance_income(ticker):
    emp = yf.Ticker(ticker)
    try:
        balance = emp.balance_sheet
        income = emp.financials
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
    return balance, income


def find_row_fuzzy(df, keywords):
    if df is None or df.empty:
        return None
    candidates = []
    for idx in df.index:
        name = str(idx).lower()
        if all(k.lower() in name for k in keywords):
            candidates.append(idx)
    if not candidates:
        return None
    return candidates[0]


def calc_roe(ticker):
    balance, income = get_balance_income(ticker)
    if balance.empty or income.empty:
        return None
    col = balance.columns[0]
    equity_row = find_row_fuzzy(balance, ["stockholder", "equity"])
    if equity_row is None:
        equity_row = find_row_fuzzy(balance, ["equity"])
    ni_row = find_row_fuzzy(income, ["net", "income"])
    if equity_row is None or ni_row is None:
        return None
    equity = balance.loc[equity_row, col]
    net_income = income.loc[ni_row, col]
    if equity == 0 or pd.isna(equity) or pd.isna(net_income):
        return None
    return float(net_income / equity)


def calc_roa(ticker):
    balance, income = get_balance_income(ticker)
    if balance.empty or income.empty:
        return None
    col = balance.columns[0]
    assets_row = find_row_fuzzy(balance, ["total", "assets"])
    ni_row = find_row_fuzzy(income, ["net", "income"])
    if assets_row is None or ni_row is None:
        return None
    assets = balance.loc[assets_row, col]
    net_income = income.loc[ni_row, col]
    if assets == 0 or pd.isna(assets) or pd.isna(net_income):
        return None
    return float(net_income / assets)


def calc_roic(ticker):
    balance, income = get_balance_income(ticker)
    if balance.empty or income.empty:
        return None
    col = balance.columns[0]

    ebit_row = find_row_fuzzy(income, ["ebit"])
    if ebit_row is None:
        ebit_row = find_row_fuzzy(income, ["operating", "income"])
    if ebit_row is None:
        return None
    ebit = income.loc[ebit_row, col]

    equity_row = find_row_fuzzy(balance, ["stockholder", "equity"])
    if equity_row is None:
        equity_row = find_row_fuzzy(balance, ["equity"])

    debt_short_row = find_row_fuzzy(balance, ["short", "debt"])
    debt_long_row = find_row_fuzzy(balance, ["long", "debt"])
    cash_row = find_row_fuzzy(balance, ["cash"])

    if equity_row is None:
        return None

    equity = balance.loc[equity_row, col]
    debt_short = balance.loc[debt_short_row, col] if debt_short_row else 0
    debt_long = balance.loc[debt_long_row, col] if debt_long_row else 0
    cash = balance.loc[cash_row, col] if cash_row else 0

    total_debt = (0 if pd.isna(debt_short) else debt_short) + (0 if pd.isna(debt_long) else debt_long)
    invested_capital = equity + total_debt - (0 if pd.isna(cash) else cash)

    if invested_capital == 0 or pd.isna(invested_capital) or pd.isna(ebit):
        return None

    return float(ebit / invested_capital)


def calc_pe(ticker):
    emp = yf.Ticker(ticker)
    info = emp.info if hasattr(emp, "info") else {}
    pe = info.get("trailingPE", None)
    if pe is not None and np.isfinite(pe):
        return float(pe)
    balance, income = get_balance_income(ticker)
    if income.empty:
        return None
    col = income.columns[0]
    ni_row = find_row_fuzzy(income, ["net", "income"])
    if ni_row is None:
        return None
    net_income = income.loc[ni_row, col]
    mcap = info.get("marketCap", None)
    if mcap is None or net_income == 0 or pd.isna(net_income):
        return None
    return float(mcap / net_income)


def calc_pb(ticker):
    emp = yf.Ticker(ticker)
    info = emp.info if hasattr(emp, "info") else {}
    mcap = info.get("marketCap", None)
    balance, _ = get_balance_income(ticker)
    if balance.empty or mcap is None:
        return None
    col = balance.columns[0]
    equity_row = find_row_fuzzy(balance, ["stockholder", "equity"])
    if equity_row is None:
        equity_row = find_row_fuzzy(balance, ["equity"])
    if equity_row is None:
        return None
    equity = balance.loc[equity_row, col]
    if equity == 0 or pd.isna(equity):
        return None
    return float(mcap / equity)


def calc_current_ratio(ticker):
    balance, _ = get_balance_income(ticker)
    if balance.empty:
        return None
    col = balance.columns[0]
    ca_row = find_row_fuzzy(balance, ["total", "current", "assets"])
    if ca_row is None:
        ca_row = find_row_fuzzy(balance, ["current", "assets"])
    cl_row = find_row_fuzzy(balance, ["total", "current", "liabilities"])
    if cl_row is None:
        cl_row = find_row_fuzzy(balance, ["current", "liabilities"])
    if ca_row is None or cl_row is None:
        return None
    ca = balance.loc[ca_row, col]
    cl = balance.loc[cl_row, col]
    if cl == 0 or pd.isna(ca) or pd.isna(cl):
        return None
    return float(ca / cl)


def calc_quick_ratio(ticker):
    balance, _ = get_balance_income(ticker)
    if balance.empty:
        return None
    col = balance.columns[0]
    ca_row = find_row_fuzzy(balance, ["total", "current", "assets"])
    if ca_row is None:
        ca_row = find_row_fuzzy(balance, ["current", "assets"])
    cl_row = find_row_fuzzy(balance, ["total", "current", "liabilities"])
    if cl_row is None:
        cl_row = find_row_fuzzy(balance, ["current", "liabilities"])
    inv_row = find_row_fuzzy(balance, ["inventory"])
    if ca_row is None or cl_row is None:
        return None
    ca = balance.loc[ca_row, col]
    cl = balance.loc[cl_row, col]
    inv = balance.loc[inv_row, col] if inv_row else 0
    if pd.isna(inv):
        inv = 0
    quick_assets = ca - inv
    if cl == 0 or pd.isna(quick_assets):
        return None
    return float(quick_assets / cl)

# ================== OTRAS FUNCIONES ==================

def calc_desviacion_ticker(ticker, period="1y"):
    rend = get_daily_returns(ticker, period=period)
    desv_diaria = float(rend.std())
    desv_anual = desv_diaria * np.sqrt(252)
    return desv_diaria, desv_anual


def top_bottom_metric_dow(metric_func, metric_name, n=5, mayores=True):
    resultados = []
    for t in DOW30:
        try:
            val = metric_func(t)
            if val is not None and np.isfinite(val):
                resultados.append((t, val))
        except Exception:
            continue
    if not resultados:
        return pd.DataFrame(columns=["Ticker", metric_name])
    df = pd.DataFrame(resultados, columns=["Ticker", metric_name])
    df = df.sort_values(metric_name, ascending=not mayores).head(n)
    return df


def top_bottom_desviacion(n=5, mayores=True, period="1y"):
    resultados = []
    for t in DOW30:
        try:
            rend = get_daily_returns(t, period=period)
            desv = float(rend.std())
            resultados.append((t, desv))
        except Exception:
            continue
    if not resultados:
        return pd.DataFrame(columns=["Ticker", "Desv_diaria"])
    df = pd.DataFrame(resultados, columns=["Ticker", "Desv_diaria"])
    df = df.sort_values("Desv_diaria", ascending=not mayores).head(n)
    return df


def top_bottom_volatilidad(n=5, mayores=True, period="1y"):
    resultados = []
    for t in DOW30:
        try:
            rend = get_daily_returns(t, period=period)
            desv_diaria = float(rend.std())
            desv_anual = desv_diaria * np.sqrt(252)
            resultados.append((t, desv_anual))
        except Exception:
            continue
    if not resultados:
        return pd.DataFrame(columns=["Ticker", "Volatilidad_anual"])
    df = pd.DataFrame(resultados, columns=["Ticker", "Volatilidad_anual"])
    df = df.sort_values("Volatilidad_anual", ascending=not mayores).head(n)
    return df


def min_var_weights_cov(cov: np.ndarray):
    n = cov.shape[0]
    ones = np.ones((n, 1))
    inv_cov = np.linalg.inv(cov)
    num = inv_cov @ ones
    den = (ones.T @ inv_cov @ ones)[0, 0]
    w = num / den
    return w.flatten()

# ================== PERFIL Y LOGO ==================

@st.cache_data
def get_profile_yf(ticker: str):
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        profile = {
            "name": info.get("longName", ticker),
            "sector": info.get("sector", "N/D"),
            "industry": info.get("industry", "N/D"),
            "country": info.get("country", "N/D"),
            "website": info.get("website", ""),
            "logo_url": info.get("logo_url", "")
        }
        return profile
    except Exception:
        return {
            "name": ticker,
            "sector": "N/D",
            "industry": "N/D",
            "country": "N/D",
            "website": "",
            "logo_url": ""
        }

# ================== SIDEBAR ==================

st.sidebar.title("Concha Capital IQ – Menú")

modulo = st.sidebar.selectbox(
    "¿Qué quieres hacer?",
    [
        "Métricas fundamentales Dow (ROE, ROA, ROIC, P/E, P/B, Current, Quick)",
        "Desviación estándar (σ diaria)",
        "Volatilidad anual (σ anualizada)",
        "Portafolio mínima varianza (N empresas)",
        "Comparar múltiples empresas (fundamentales yfinance)",
        "Desviación y volatilidad Dow 30",
        "Perfil de empresa (tipo Bloomberg)",
    ],
)

periodo_mercado = st.sidebar.selectbox(
    "Periodo de datos de mercado:",
    ("6mo", "1y", "2y", "5y"),
    index=1
)

# ============ MÓDULO 1: FUNDAMENTALES DOW ============

if modulo == "Métricas fundamentales Dow (ROE, ROA, ROIC, P/E, P/B, Current, Quick)":
    st.subheader("Métricas fundamentales Dow (estados financieros yfinance)")

    metrica = st.selectbox(
        "Métrica:",
        ["ROE", "ROA", "ROIC", "P/E", "P/B", "Current Ratio", "Quick Ratio (Acid Test)"]
    )

    modo = st.radio(
        "Tipo de análisis:",
        ["Empresa específica", "Top X Dow", "Low X Dow"],
        horizontal=True
    )

    metric_funcs = {
        "ROE": calc_roe,
        "ROA": calc_roa,
        "ROIC": calc_roic,
        "P/E": calc_pe,
        "P/B": calc_pb,
        "Current Ratio": calc_current_ratio,
        "Quick Ratio (Acid Test)": calc_quick_ratio,
    }
    func = metric_funcs[metrica]

    if modo == "Empresa específica":
        ticker = st.text_input("Ticker:", "AAPL").upper()
        if st.button("Calcular métrica"):
            with st.spinner("Calculando..."):
                val = func(ticker)
            if val is None:
                st.error(f"No se pudo calcular {metrica} de {ticker}.")
            else:
                # TODO EN %
                if metrica in ["P/E", "P/B"]:
                    st.success(f"{metrica} de {ticker}: {val:.2f}")
                else:
                    st.success(f"{metrica} de {ticker}: {val:.2%}")
    else:
        n = st.number_input("¿Cuántas empresas (X)?", min_value=1, max_value=30, value=5)
        mayores = True if modo == "Top X Dow" else False
        if st.button("Calcular ranking"):
            with st.spinner("Calculando sobre el Dow Jones..."):
                df = top_bottom_metric_dow(func, metrica, n=int(n), mayores=mayores)
            if df.empty:
                st.error("No se pudieron calcular las métricas para ninguna empresa.")
            else:
                df = df.set_index("Ticker")
                # Cambiar None → NaN para que el styling no truene
                df = df.replace({None: np.nan})
                if metrica in ["P/E", "P/B"]:
                    st.dataframe(df.style.format({metrica: "{:.2f}"}))
                else:
                    st.dataframe(df.style.format({metrica: "{:.2%}"}))

# ============ MÓDULO 2: DESVIACIÓN DIARIA ============

elif modulo == "Desviación estándar (σ diaria)":
    st.subheader("Desviación estándar diaria")

    modo = st.radio("Tipo de análisis:", ["Empresa específica", "Top X Dow", "Low X Dow"], horizontal=True)
    periodo = st.text_input("Periodo de yfinance (ej. 1y, 6mo, 2y):", "1y")

    if modo == "Empresa específica":
        ticker = st.text_input("Ticker:", "AAPL").upper()
        if st.button("Calcular desviación"):
            try:
                desv_diaria, desv_anual = calc_desviacion_ticker(ticker, period=periodo)
                st.write(f"**{ticker}**")
                st.write(f"σ diaria: {desv_diaria:.4f} ({desv_diaria:.2%})")
                st.write(f"σ anual (volatilidad): {desv_anual:.4f} ({desv_anual:.2%})")
            except Exception as e:
                st.error(f"Error calculando la desviación de {ticker}: {e}")
    else:
        n = st.number_input("¿Cuántas empresas (X)?", min_value=1, max_value=30, value=5)
        mayores = True if modo == "Top X Dow" else False
        if st.button("Calcular ranking de σ diaria"):
            df = top_bottom_desviacion(int(n), mayores=mayores, period=periodo)
            if df.empty:
                st.error("No se pudo calcular la desviación estándar del Dow.")
            else:
                df = df.set_index("Ticker")
                df = df.replace({None: np.nan})
                st.dataframe(df.style.format({"Desv_diaria": "{:.2%}"}))

# ============ MÓDULO 3: VOLATILIDAD ANUAL ============

elif modulo == "Volatilidad anual (σ anualizada)":
    st.subheader("Volatilidad anualizada")

    modo = st.radio("Tipo de análisis:", ["Empresa específica", "Top X Dow", "Low X Dow"], horizontal=True)
    periodo = st.text_input("Periodo de yfinance (ej. 1y, 6mo, 2y):", "1y")

    if modo == "Empresa específica":
        ticker = st.text_input("Ticker:", "AAPL").upper()
        if st.button("Calcular volatilidad"):
            try:
                desv_diaria, desv_anual = calc_desviacion_ticker(ticker, period=periodo)
                st.write(f"**{ticker}**")
                st.write(f"σ diaria: {desv_diaria:.4f} ({desv_diaria:.2%})")
                st.write(f"σ anual (volatilidad): {desv_anual:.4f} ({desv_anual:.2%})")
            except Exception as e:
                st.error(f"Error calculando volatilidad de {ticker}: {e}")
    else:
        n = st.number_input("¿Cuántas empresas (X)?", min_value=1, max_value=30, value=5)
        mayores = True if modo == "Top X Dow" else False
        if st.button("Calcular ranking de volatilidad"):
            df = top_bottom_volatilidad(int(n), mayores=mayores, period=periodo)
            if df.empty:
                st.error("No se pudo calcular la volatilidad del Dow.")
            else:
                df = df.set_index("Ticker")
                df = df.replace({None: np.nan})
                st.dataframe(df.style.format({"Volatilidad_anual": "{:.2%}"}))

# ============ MÓDULO 4: PORTAFOLIO MIN VAR N ACTIVOS ============

elif modulo == "Portafolio mínima varianza (N empresas)":
    st.subheader("Portafolio de mínima varianza con N activos (Markowitz)")

    seleccion = st.multiselect(
        "Selecciona activos (mínimo 2, máximo 15)",
        DOW30,
        default=["AAPL", "MSFT", "JPM", "KO"]
    )

    if len(seleccion) < 2:
        st.info("Selecciona al menos 2 activos para construir el portafolio.")
    else:
        if st.button("Calcular portafolio óptimo (mínima varianza)"):
            rets = get_returns(seleccion, period=periodo_mercado)
            mean_rets = rets.mean()
            cov = rets.cov().values
            try:
                w = min_var_weights_cov(cov)
                port_mu = np.dot(w, mean_rets.values)
                port_sigma = np.sqrt(w @ cov @ w)

                tabla_w = pd.DataFrame({
                    "Ticker": seleccion,
                    "Empresa": [COMPANIAS_DOW30.get(t, "Desconocida") for t in seleccion],
                    "Peso": w
                }).set_index("Ticker")

                st.write("📌 **Pesos del portafolio (pueden ser negativos si hay short selling):**")
                st.dataframe(tabla_w.style.format({"Peso": "{:.2%}"}))

                st.write("📌 **Rendimiento esperado y riesgo del portafolio:**")
                st.write(f"Rendimiento esperado diario: {port_mu:.2%}")
                st.write(f"Volatilidad diaria: {port_sigma:.2%}")
                st.write(f"Volatilidad anualizada: {(port_sigma * np.sqrt(252)):.2%}")
            except np.linalg.LinAlgError:
                st.error("No se pudo invertir la matriz de covarianzas. Prueba con otros activos.")

# ============ MÓDULO 5: COMPARAR MÚLTIPLES EMPRESAS ============

elif modulo == "Comparar múltiples empresas (fundamentales yfinance)":
    st.subheader("Comparador de múltiples empresas (fundamentales yfinance)")

    seleccion = st.multiselect(
        "Selecciona empresas (Dow 30):",
        list(OPCIONES_DOW30.keys()),
        default=list(OPCIONES_DOW30.keys())[:5]
    )

    if len(seleccion) == 0:
        st.warning("Selecciona al menos una empresa.")
    else:
        tickers = [OPCIONES_DOW30[e] for e in seleccion]
        rows = []
        for t in tickers:
            roe = calc_roe(t)
            roa = calc_roa(t)
            roic = calc_roic(t)
            pe = calc_pe(t)
            pb = calc_pb(t)
            curr = calc_current_ratio(t)
            quick = calc_quick_ratio(t)
            try:
                desv, vol = calc_desviacion_ticker(t, period=periodo_mercado)
            except Exception:
                desv, vol = None, None

            row = {
                "Ticker": t,
                "Empresa": COMPANIAS_DOW30.get(t, t),
                "ROE": roe,
                "ROA": roa,
                "ROIC": roic,
                "P/E": pe,
                "P/B": pb,
                "Current Ratio": curr,
                "Quick Ratio": quick,
                "σ diaria": desv,
                "Vol anual": vol
            }
            # None -> NaN
            for k, v in row.items():
                if v is None:
                    row[k] = np.nan
            rows.append(row)

        df = pd.DataFrame(rows).set_index("Ticker")

        st.dataframe(
            df.style.format({
                "ROE": "{:.2%}",
                "ROA": "{:.2%}",
                "ROIC": "{:.2%}",
                "Current Ratio": "{:.2%}",
                "Quick Ratio": "{:.2%}",
                "σ diaria": "{:.2%}",
                "Vol anual": "{:.2%}",
                "P/E": "{:.2f}",
                "P/B": "{:.2f}",
            })
        )

# ============ MÓDULO 6: DESVIACIÓN Y VOL DOW30 ============

elif modulo == "Desviación y volatilidad Dow 30":
    st.subheader("Desviación estándar y volatilidad anualizada - Dow 30")

    resultados = []
    for t in DOW30:
        try:
            r = get_daily_returns(t, period=periodo_mercado)
            desv = float(r.std())
            vol = desv * np.sqrt(252)
            resultados.append((t, desv, vol))
        except Exception:
            continue

    if not resultados:
        st.error("No se pudieron calcular datos de volatilidad para el Dow 30.")
    else:
        tabla = pd.DataFrame(
            resultados,
            columns=["Ticker", "Desviación diaria", "Volatilidad anualizada"]
        )
        tabla["Empresa"] = tabla["Ticker"].map(lambda x: COMPANIAS_DOW30.get(x, "Desconocida"))
        tabla = tabla.set_index("Ticker").sort_values("Volatilidad anualizada", ascending=False)

        st.dataframe(
            tabla.style.format({
                "Desviación diaria": "{:.2%}",
                "Volatilidad anualizada": "{:.2%}"
            })
        )

        st.bar_chart(tabla["Volatilidad anualizada"])

        st.subheader("Listado Dow 30 (Ticker – Empresa)")
        listado = pd.DataFrame(
            {"Empresa": [COMPANIAS_DOW30[t] for t in DOW30]},
            index=DOW30
        )
        listado.index.name = "Ticker"
        st.table(listado)

# ============ MÓDULO 7: PERFIL TIPO BLOOMBERG ============

elif modulo == "Perfil de empresa (tipo Bloomberg)":
    st.subheader("Perfil de empresa – estilo Bloomberg (yfinance)")

    empresa_sel = st.selectbox("Selecciona empresa", list(OPCIONES_DOW30.keys()), index=0)
    ticker = OPCIONES_DOW30[empresa_sel]

    profile = get_profile_yf(ticker)

    col_logo, col_info = st.columns([1, 3])

    with col_logo:
        if profile["logo_url"]:
            st.image(profile["logo_url"], use_container_width=True)
        st.write(f"**{ticker}**")

    with col_info:
        st.markdown(f"### {profile['name']}")
        st.write(f"**Sector:** {profile['sector']}")
        st.write(f"**Industria:** {profile['industry']}")
        st.write(f"**País:** {profile['country']}")
        if profile["website"]:
            st.write(f"[Página web oficial]({profile['website']})")

    st.write("---")
    st.write("📈 **Precio histórico**")
    precios = get_prices([ticker], period=periodo_mercado)
    st.line_chart(precios)
