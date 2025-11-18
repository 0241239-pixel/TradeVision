# app.py - TradeVision (updated with richer "Mis intereses" + IA chat)

import math
import datetime as dt
from datetime import date, datetime
from typing import Optional, Tuple, Dict, List
from pathlib import Path
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

# Gemini SDK (optional)
try:
    from google import genai
    GENAI_SDK_AVAILABLE = True
except Exception:
    GENAI_SDK_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# API KEY (desde archivo privado o variable de entorno / secrets)
# ─────────────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = ""

# 1) Desarrollo local: config_private.py (no se sube a GitHub)
try:
    from config_private import GEMINI_API_KEY as _LOCAL_GEMINI_API_KEY
    GEMINI_API_KEY = _LOCAL_GEMINI_API_KEY
except Exception:
    import os
    # 2) Variable de entorno estándar (por si la defines así)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# 3) Streamlit Cloud: secrets (GEMINI_API_KEY en Settings → Secrets)
try:
    if not GEMINI_API_KEY:
        GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", "")
except Exception:
    pass

API_KEY = GEMINI_API_KEY
# ─────────────────────────────────────────────────────────────────────────────

# Paths
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
USERS_PATH = DATA_DIR / "users.csv"

# Streamlit config & CSS (light theme)
st.set_page_config(
    page_title="TradeVision",
    page_icon="📊",
    layout="wide"
)

CSS = """
<style>
:root{
  --accent: #0B8043;
  --accent-soft: #0F9D58;
  --bg: #FFFFFF;
  --bg-soft: #F5F7FA;
  --border-soft: #E5E7EB;
  --text-main: #111827;
  --text-muted: #6B7280;
}

/* Background */
[data-testid="stAppViewContainer"]{
  background-color: var(--bg);
}

/* Sidebar */
section[data-testid="stSidebar"]{
  background-color: #F9FAFB;
  border-right: 1px solid var(--border-soft);
}

/* Titles */
h1, h2, h3, h4{
  color: var(--text-main);
}

/* Cards */
.card {
  border: 1px solid var(--border-soft);
  border-radius: 12px;
  padding: 14px 16px;
  background: var(--bg-soft);
}
.kv {
  display:flex;
  justify-content:space-between;
  padding:6px 0;
  border-bottom:1px dashed rgba(148,163,184,0.6);
  font-size:0.92rem;
}
.kv:last-child { border-bottom:none; }

.metric-good { color:#16A34A; font-weight:600; }
.metric-bad  { color:#DC2626; font-weight:600; }
.metric-warn { color:#D97706; font-weight:600; }
.subtle      { color:var(--text-muted); font-size:0.9rem; }

/* Primary buttons */
div.stButton > button:first-child,
button[kind="primary"]{
  background-color: var(--accent-soft);
  color:white;
  border-radius:999px;
  border:1px solid var(--accent);
  font-weight:600;
}
div.stButton > button:first-child:hover,
button[kind="primary"]:hover{
  background-color: var(--accent);
  border-color: var(--accent);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"]{
  gap:0.25rem;
}
.stTabs [data-baseweb="tab"]{
  border-radius:999px;
  padding:0.3rem 0.9rem;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# ---- Date utilities ----
def ytd_start(d: date) -> date:
    return date(d.year, 1, 1)

def months_ago(n: int, ref: date) -> date:
    y, m = ref.year, ref.month - n
    while m <= 0:
        m += 12
        y -= 1
    last_day = [31,29 if (y%4==0 and (y%100!=0 or y%400==0)) else 28,31,30,31,30,31,31,30,31,30,31][m-1]
    return date(y, m, min(ref.day, last_day))

def slice_by_period(idx: pd.DatetimeIndex, period_key: str) -> pd.Index:
    today = date.today()
    starts = {
        "YTD": ytd_start(today),
        "3M": months_ago(3, today),
        "6M": months_ago(6, today),
        "9M": months_ago(9, today),
        "1Y": months_ago(12, today),
        "3Y": months_ago(36, today),
        "5Y": months_ago(60, today),
        "ALL": None
    }
    d0 = starts.get(period_key)
    if d0 is None:
        return idx
    return idx[idx.date >= d0]

# ---- Formatters ----
def fmt_money(x, currency="$"):
    try:
        return f"{currency}{float(x):,.2f}"
    except Exception:
        return "—"

def fmt_pct(x):
    try:
        return f"{float(x)*100:.2f}%"
    except Exception:
        return "—"

def fmt_int(x):
    try:
        return f"{int(x):,}"
    except Exception:
        return "—"

def fmt_float(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return "—"

def fmt_date(x):
    try:
        if isinstance(x, (int, float)):
            return datetime.fromtimestamp(x).strftime("%Y-%m-%d")
        if isinstance(x, pd.Timestamp):
            return x.strftime("%Y-%m-%d")
        if isinstance(x, (datetime, date)):
            return x.strftime("%Y-%m-%d")
        return str(x)[:10]
    except Exception:
        return "—"

# ---- Yahoo Finance helpers ----
@st.cache_data(show_spinner=False, ttl=3600)
def fetch_history(ticker: str, period="5y", interval="1d") -> pd.DataFrame:
    try:
        return yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_info(ticker: str) -> dict:
    tk = yf.Ticker(ticker)
    try:
        info = tk.info
    except Exception:
        info = {}
    try:
        fi = tk.fast_info
        info["_fast"] = {
            "last_price": getattr(fi, "last_price", None),
            "year_low": getattr(fi, "year_low", None),
            "year_high": getattr(fi, "year_high", None),
        }
    except Exception:
        pass
    return info

def get_last_price(ticker: str, info: dict) -> Optional[float]:
    try:
        return float(info.get("_fast", {}).get("last_price"))
    except Exception:
        pass
    try:
        h = yf.Ticker(ticker).history(period="1d")["Close"]
        if not h.empty:
            return float(h.iloc[-1])
    except Exception:
        pass
    return None

@st.cache_data(ttl=900)
def get_last_and_prev_close(ticker: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        h = yf.Ticker(ticker).history(period="5d")["Close"]
        if len(h) >= 2:
            return float(h.iloc[-1]), float(h.iloc[-2])
        elif len(h) == 1:
            return float(h.iloc[-1]), None
    except Exception:
        pass
    return None, None

def cumulative_return(s: pd.Series) -> float:
    s = s.dropna()
    if len(s) < 2:
        return np.nan
    return float(s.iloc[-1] / s.iloc[0] - 1)

def annualized_vol(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(252))

def get_company_name(info: Dict) -> str:
    for k in ("longName","shortName","symbol"):
        if info.get(k):
            return str(info[k])
    return ""

def get_business_summary(info: Dict) -> str:
    for k in ("longBusinessSummary","businessSummary","description"):
        if info.get(k):
            return str(info[k])
    return ""

def get_52w_range(info: dict) -> Tuple[Optional[float], Optional[float]]:
    low = info.get("fiftyTwoWeekLow") or info.get("_fast", {}).get("year_low")
    high = info.get("fiftyTwoWeekHigh") or info.get("_fast", {}).get("year_high")
    return (float(low) if low else None, float(high) if high else None)

# ---- Symbol search ----
@st.cache_data(ttl=300)
def search_symbols(q: str) -> List[dict]:
    if not q or len(q) < 2:
        return []
    out: List[dict] = []
    try:
        res = yf.search(q) or {}
        for x in res.get("quotes", []):
            qt = (x.get("quoteType") or "").upper()
            if qt and "EQUITY" not in qt:
                continue
            out.append({
                "symbol": x.get("symbol"),
                "shortname": x.get("shortname") or x.get("longname",""),
                "exchange": x.get("exchange") or "",
            })
    except Exception:
        pass
    uniq = []
    seen = set()
    for x in out:
        s = x["symbol"]
        if s not in seen:
            uniq.append(x)
            seen.add(s)
    return uniq[:12]

# ---- Gemini helpers ----
def make_genai_client(api_key: str):
    if not GENAI_SDK_AVAILABLE:
        raise RuntimeError("Instala google-genai para usar Gemini.")
    return genai.Client(api_key=api_key)

def translate_summary(desc: str, api_key: str, max_chars: int = 600) -> str:
    if not desc:
        return "Sin descripción disponible."
    try:
        c = make_genai_client(api_key)
        prompt = (
            "Traduce al español financiero formal el siguiente texto. "
            f"Limita la salida a {max_chars} caracteres:\n\n{desc}"
        )
        r = c.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        txt = getattr(r, "text", "") or desc
        return txt[:max_chars]
    except Exception:
        return desc[:max_chars]

def ai_portfolio_comment(
    total_value: float,
    total_pl: float,
    day_pl: float,
    positions_summary: List[dict],
    api_key: str
) -> Optional[str]:
    if not GENAI_SDK_AVAILABLE or not api_key:
        return None
    try:
        c = make_genai_client(api_key)

        pos_txt = ""
        for p in positions_summary[:10]:
            pos_txt += (
                f"- {p['ticker']}: inv={p['inversion']:.2f}, "
                f"valor={p['valor_actual']:.2f}, "
                f"ret_total={p['pct_total']*100:.2f}%, "
                f"ret_dia={p['pct_dia']*100:.2f}%\n"
            )

        prompt = f"""
Eres un analista financiero profesional.
Resume este portafolio en máximo 200 palabras.

Valor total: {total_value:.2f}
Ganancia/pérdida total: {total_pl:.2f}
Ganancia/pérdida del día: {day_pl:.2f}

Detalle de posiciones:
{pos_txt or "Sin posiciones."}

Explica:
1) Cómo va en general.
2) De dónde viene la ganancia/pérdida.
3) En qué debería fijarse el inversionista.

No des órdenes directas. Habla en términos de cosas a observar.
Escribe en español claro.
"""
        r = c.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return getattr(r, "text", "").strip()
    except Exception:
        return None

def ai_position_comment(
    ticker: str,
    company_name: str,
    last_price: float,
    buy_price: float,
    qty: float,
    days_held: int,
    ann_return: Optional[float],
    pct_gain: float,
    rec_summary: str,
    api_key: str
) -> Optional[str]:
    if not GENAI_SDK_AVAILABLE or not api_key:
        return None
    try:
        c = make_genai_client(api_key)
        prompt = f"""
Eres un analista financiero que explica en lenguaje simple.
Analiza esta posición y da un comentario de máximo 200 palabras.

Ticker: {ticker}
Empresa: {company_name}
Precio compra: {buy_price}
Precio actual: {last_price}
Cantidad: {qty}
Días en la posición: {days_held}
Retorno total: {pct_gain*100:.2f}%
"""
    except Exception:
        return None
    if ann_return is not None:
        prompt += f"Retorno anualizado aprox: {ann_return*100:.2f}%\n"
    prompt += f"""
Opinión de analistas: {rec_summary}

Estructura la respuesta en:
1) Situación actual
2) Factores clave
3) Qué podría considerar el inversionista (sin órdenes directas).

Escribe en español, tono cercano pero profesional.
"""
    try:
        r = c.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return getattr(r, "text", "").strip()
    except Exception:
        return None

def ai_watchlist_comment(rows: List[dict], api_key: str) -> Optional[str]:
    if not GENAI_SDK_AVAILABLE or not api_key or not rows:
        return None
    try:
        c = make_genai_client(api_key)
        txt = ""
        for r in rows:
            txt += (
                f"- {r['ticker']}: 3M={r['ret_3m']*100:.1f}%, "
                f"1Y={r['ret_1y']*100:.1f}%, "
                f"dist_max={r['dist_high']*100:.1f}%\n"
            )
        prompt = f"""
Eres un analista de renta variable.
Te paso algunas acciones con buena inercia reciente.
Explica en máximo 200 palabras por qué podrían ser interesantes para vigilar, 
sin recomendar comprar explícitamente.

Datos:
{txt}

Responde en español, ordenando ideas en viñetas o párrafos cortos.
"""
        r = c.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return getattr(r, "text", "").strip()
    except Exception:
        return None

def ai_interest_chat(
    interest: str,
    companies: List[Dict[str, str]],
    user_holdings: List[str],
    question: str,
    api_key: str
) -> Optional[str]:
    if not GENAI_SDK_AVAILABLE or not api_key or not question.strip():
        return None
    try:
        c = make_genai_client(api_key)
        comp_text = ""
        for cinfo in companies[:12]:
            comp_text += f"- {cinfo['ticker']}: {cinfo['empresa']} ({cinfo['segmento']})\n"
        holdings_text = ", ".join(user_holdings) if user_holdings else "ninguna empresa específica de este tema"
        prompt = f"""
Eres un analista financiero educativo especializado por sectores.
El usuario está interesado en el tema: {interest}

Ejemplos de empresas representativas de este tema:
{comp_text or "Sin ejemplos de empresas."}

En su portafolio actualmente tiene: {holdings_text}.

Pregunta del usuario:
{question}

Responde en español claro, sin dar órdenes de comprar o vender, solo explicando:
- Conceptos clave relacionados con el interés.
- Cómo pensar el riesgo y el horizonte de inversión.
- Qué métricas suelen ser importantes revisar.

Máximo 220 palabras.
"""
        r = c.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return getattr(r, "text", "").strip()
    except Exception:
        return None

# ---- Analyst recommendations ----
def summarize_recommendations(ticker: str, info: dict):
    tk = yf.Ticker(ticker)
    rec = {"Buy":0,"Hold":0,"Sell":0}
    src = None
    try:
        rs = tk.recommendations_summary
        if isinstance(rs, pd.DataFrame) and not rs.empty:
            row = rs.iloc[-1]
            rec["Buy"] = int(row.get("strongBuy",0))+int(row.get("buy",0))
            rec["Hold"] = int(row.get("hold",0))
            rec["Sell"] = int(row.get("sell",0))+int(row.get("strongSell",0))
            src = "summary"
    except Exception:
        pass
    if src is None:
        try:
            tr = info.get("recommendationTrend",{})
            if tr and tr.get("trend"):
                r = tr["trend"][0]
                rec["Buy"] = int(r.get("strongBuy",0))+int(r.get("buy",0))
                rec["Hold"] = int(r.get("hold",0))
                rec["Sell"] = int(r.get("sell",0))+int(r.get("strongSell",0))
                src = "trend"
        except Exception:
            pass
    total = rec["Buy"]+rec["Hold"]+rec["Sell"]
    return rec, total, src

def analyst_conclusion(rec: Dict):
    b,h,s = rec["Buy"], rec["Hold"], rec["Sell"]
    if b>=h and b>=s:
        return "Predominan **compras**", "Comprar"
    if s>=b and s>=h:
        return "Predominan **ventas**", "Vender"
    return "Predominan **mantener**", "Mantener"

# ---- Auth helpers ----
def load_users() -> pd.DataFrame:
    if USERS_PATH.exists():
        try:
            df = pd.read_csv(USERS_PATH)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    needed = ["user_id","name","email","password","intereses"]
    for c in needed:
        if c not in df.columns:
            df[c] = ""
    return df[needed]

def save_users(df: pd.DataFrame):
    df.to_csv(USERS_PATH, index=False)

def get_next_user_id(df: pd.DataFrame) -> int:
    if df.empty:
        return 1
    try:
        return int(df["user_id"].astype(int).max()) + 1
    except Exception:
        return 1

def parse_intereses_str(s: str) -> List[str]:
    if not isinstance(s, str) or not s.strip():
        return []
    return [p.strip() for p in s.split("|") if p.strip()]

def intereses_to_str(lst: List[str]) -> str:
    return "|".join(lst) if lst else ""

def ensure_auth_state():
    for k, v in {
        "is_auth": False,
        "user_id": None,
        "user_name": "",
        "user_email": "",
        "user_intereses": [],
        "selected_interest": None,
    }.items():
        st.session_state.setdefault(k, v)

def auth_view():
    ensure_auth_state()
    st.title("TradeVision")
    st.subheader("Gestión inteligente de tu portafolio de inversión")

    col1, col2 = st.columns(2)

    # Login
    with col1:
        st.markdown("#### Inicia sesión")
        email = st.text_input("Correo electrónico", key="login_email")
        pwd = st.text_input("Contraseña", type="password", key="login_pwd")
        if st.button("Entrar", use_container_width=True, key="btn_login", type="primary"):
            users = load_users()
            row = users[users["email"].str.lower() == email.strip().lower()]
            if row.empty:
                st.error("No encontramos una cuenta con ese correo.")
            else:
                r = row.iloc[0]
                if str(r["password"]) != pwd:
                    st.error("Contraseña incorrecta.")
                else:
                    st.session_state["is_auth"] = True
                    st.session_state["user_id"] = int(r["user_id"])
                    st.session_state["user_name"] = str(r["name"])
                    st.session_state["user_email"] = str(r["email"])
                    st.session_state["user_intereses"] = parse_intereses_str(r["intereses"])
                    st.rerun()

    # Register
    with col2:
        st.markdown("#### Crea tu cuenta")
        name2 = st.text_input("Nombre completo", key="reg_name")
        email2 = st.text_input("Correo electrónico", key="reg_email")
        pwd2 = st.text_input("Elige una contraseña", type="password", key="reg_pwd")
        st.markdown("Selecciona tus principales intereses de inversión:")
        intereses_opciones = [
            "Tecnología / Big Tech",
            "Finanzas / Bancos",
            "Crecimiento agresivo",
            "Consumo defensivo",
            "Dividendos / Ingreso pasivo",
        ]
        intereses_sel = st.multiselect(
            "Intereses",
            intereses_opciones,
            key="reg_intereses"
        )
        if st.button("Crear cuenta", use_container_width=True, key="btn_register", type="primary"):
            if not (name2 and email2 and pwd2):
                st.error("Completa nombre, correo y contraseña.")
            else:
                users = load_users()
                if not users[users["email"].str.lower() == email2.strip().lower()].empty:
                    st.error("Ya existe una cuenta registrada con ese correo.")
                else:
                    uid = get_next_user_id(users)
                    new_row = {
                        "user_id": uid,
                        "name": name2.strip(),
                        "email": email2.strip().lower(),
                        "password": pwd2,
                        "intereses": intereses_to_str(intereses_sel),
                    }
                    users = pd.concat([users, pd.DataFrame([new_row])], ignore_index=True)
                    save_users(users)
                    st.success("Cuenta creada. Ahora puedes iniciar sesión con tu correo y contraseña.")

    st.markdown("---")
    st.caption("TradeVision no ofrece asesoría financiera personalizada. La información mostrada es únicamente con fines informativos.")
    return False

# ---- Portfolio helpers ----
def get_current_user_id() -> str:
    return str(st.session_state.get("user_id") or "demo")

def portfolio_path() -> Path:
    return DATA_DIR / f"portfolio_{get_current_user_id()}.csv"

def ensure_portfolio_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "ticker", "precio_compra", "cantidad", "fecha_compra",
        "nota", "cuenta", "activa", "fecha_cierre", "precio_venta"
    ]
    for c in cols:
        if c not in df.columns:
            if c in ("precio_compra", "cantidad", "precio_venta"):
                df[c] = np.nan
            elif c == "activa":
                df[c] = 1
            else:
                df[c] = ""
    for c in ["fecha_compra", "fecha_cierre"]:
        df[c] = df[c].fillna("").astype(str)
    return df[cols]

def load_portfolio() -> pd.DataFrame:
    p = portfolio_path()
    if p.exists():
        try:
            df = pd.read_csv(p)
            return ensure_portfolio_columns(df)
        except Exception:
            pass
    df = pd.DataFrame(columns=[
        "ticker","precio_compra","cantidad","fecha_compra",
        "nota","cuenta","activa","fecha_cierre","precio_venta"
    ])
    return ensure_portfolio_columns(df)

def save_portfolio(df: pd.DataFrame):
    df_to_save = df.copy()
    def _to_date_str(x):
        if isinstance(x, (datetime, date, pd.Timestamp)):
            return x.strftime("%Y-%m-%d")
        if x in [None, "", "NaT", "NaN"] or (isinstance(x, float) and math.isnan(x)):
            return ""
        s = str(x)
        if len(s) >= 10:
            return s[:10]
        return s
    for col in ["fecha_compra", "fecha_cierre"]:
        if col in df_to_save.columns:
            df_to_save[col] = df_to_save[col].apply(_to_date_str)
    df_to_save.to_csv(portfolio_path(), index=False)

# ---- Momentum watchlist ----
UNIVERSE_MOMENTUM = [
    "AAPL","MSFT","NVDA","META","AMZN","GOOGL","TSLA","COST","LLY",
    "JPM","V","MA","ADBE","NFLX","AVGO","PEP","KO","XOM","UNH","RTX"
]

@st.cache_data(ttl=1800)
def momentum_metrics(ticker: str) -> Optional[dict]:
    info = fetch_info(ticker)
    hist = fetch_history(ticker, period="1y", interval="1d")
    if hist.empty:
        return None
    last = get_last_price(ticker, info) or float(hist["Close"].iloc[-1])
    close = hist["Close"]
    today = date.today()
    d3 = months_ago(3, today)
    d12 = months_ago(12, today)
    c3 = close[close.index.date >= d3]
    c12 = close[close.index.date >= d12]
    ret3 = cumulative_return(c3)
    ret12 = cumulative_return(c12)
    vol = annualized_vol(close.pct_change())
    low52, high52 = get_52w_range(info)
    dist_high = (last/high52 - 1) if high52 else np.nan
    return {
        "ticker": ticker,
        "name": get_company_name(info) or ticker,
        "last": last,
        "ret_3m": ret3,
        "ret_1y": ret12,
        "vol": vol,
        "dist_high": dist_high,
    }

# ---- Single stock analysis ----
def render_single_stock_analysis(key_prefix: str, use_gemini: bool, api_key: str):
    st.markdown("#### Selecciona la acción a analizar")
    c1, c2 = st.columns([3,1])
    with c1:
        query = st.text_input(
            "Ticker o nombre (ej. DIS o Disney)",
            st.session_state.get(f"{key_prefix}_ticker", "DIS"),
            key=f"{key_prefix}_ticker_input"
        ).strip()
        suggestions = search_symbols(query) if len(query) >= 2 else []
        if suggestions:
            labels = [
                f"{s['symbol']} — {s['shortname']} ({s['exchange']})"
                for s in suggestions
            ]
            pick = st.selectbox(
                "Coincidencias",
                labels,
                index=0,
                key=f"{key_prefix}_ac_list"
            )
            chosen = suggestions[labels.index(pick)]["symbol"]
            if st.button(
                f"Usar «{chosen}»",
                type="primary",
                key=f"{key_prefix}_btn_use_chosen"
            ):
                st.session_state[f"{key_prefix}_ticker"] = chosen
    with c2:
        analyze = st.button(
            "🔎 Analizar acción",
            use_container_width=True,
            key=f"{key_prefix}_btn_analyze"
        )
    if analyze and query:
        st.session_state[f"{key_prefix}_ticker"] = query.upper()
    ticker = st.session_state.get(f"{key_prefix}_ticker", None)
    if not ticker:
        return
    info = fetch_info(ticker)
    hist = fetch_history(ticker)
    if hist.empty:
        st.error("No hay datos históricos para este ticker.")
        return
    name = get_company_name(info) or ticker
    desc = get_business_summary(info)
    st.markdown(f"### {name} ({ticker})")
    if desc:
        if use_gemini and api_key:
            st.markdown(translate_summary(desc, api_key))
        else:
            st.markdown(desc[:600])
    low52, high52 = get_52w_range(info)
    st.markdown("#### Datos clave")
    k1, k2, k3 = st.columns(3)
    k1.markdown(f"""
    <div class='card'>
      <div class='kv'><span>Previous Close</span><span>{fmt_money(info.get('previousClose'))}</span></div>
      <div class='kv'><span>Open</span><span>{fmt_money(info.get('open'))}</span></div>
      <div class='kv'><span>Day Range</span><span>{fmt_money(info.get('dayLow'))} – {fmt_money(info.get('dayHigh'))}</span></div>
      <div class='kv'><span>52W Range</span><span>{fmt_money(low52)} – {fmt_money(high52)}</span></div>
    </div>
    """, unsafe_allow_html=True)
    k2.markdown(f"""
    <div class='card'>
      <div class='kv'><span>Market Cap</span><span>{fmt_int(info.get('marketCap'))}</span></div>
      <div class='kv'><span>Beta (5Y)</span><span>{fmt_float(info.get('beta'))}</span></div>
      <div class='kv'><span>PE (TTM)</span><span>{fmt_float(info.get('trailingPE'))}</span></div>
      <div class='kv'><span>EPS (TTM)</span><span>{fmt_float(info.get('trailingEps'))}</span></div>
    </div>
    """, unsafe_allow_html=True)
    k3.markdown(f"""
    <div class='card'>
      <div class='kv'><span>Dividend Yield</span><span>{fmt_pct(info.get('dividendYield'))}</span></div>
      <div class='kv'><span>1Y Target</span><span>{fmt_money(info.get('targetMeanPrice'))}</span></div>
      <div class='kv'><span>Earnings Date</span><span>{fmt_date(info.get('earningsDate') or info.get('earningsTimestamp'))}</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("#### Rendimiento y volatilidad histórica")
    today = date.today()
    per_map = {
        "YTD": ytd_start(today),
        "3M": months_ago(3, today),
        "6M": months_ago(6, today),
        "9M": months_ago(9, today),
        "1Y": months_ago(12, today),
        "3Y": months_ago(36, today),
        "5Y": months_ago(60, today),
        "ALL": None
    }
    close = hist["Close"]
    ret = close.pct_change()
    rows = []
    for key, start_d in per_map.items():
        if start_d:
            s_close = close[close.index.date >= start_d]
            s_ret = ret[ret.index.date >= start_d]
        else:
            s_close = close
            s_ret = ret
        rows.append({
            "Periodo": key,
            "Rendimiento": cumulative_return(s_close),
            "Volatilidad anualizada": annualized_vol(s_ret)
        })
    df = pd.DataFrame(rows)
    df["Rendimiento"] = df["Rendimiento"].apply(fmt_pct)
    df["Volatilidad anualizada"] = df["Volatilidad anualizada"].apply(fmt_pct)
    st.dataframe(df, use_container_width=True)
    st.markdown("##### Periodo para las gráficas")
    period_choice = st.radio(
        "Periodo",
        ["YTD","3M","6M","9M","1Y","3Y","5Y","ALL"],
        horizontal=True,
        label_visibility="collapsed",
        key=f"{key_prefix}_chart_period"
    )
    idx_slice = slice_by_period(hist.index, period_choice)
    hist_f = hist.loc[idx_slice]
    st.markdown("#### Gráfico de velas")
    fig = go.Figure([go.Candlestick(
        x=hist_f.index,
        open=hist_f["Open"],
        high=hist_f["High"],
        low=hist_f["Low"],
        close=hist_f["Close"]
    )])
    fig.update_layout(
        template="plotly_white",
        height=450,
        margin=dict(l=20,r=20,t=40,b=20),
        xaxis_title="Fecha",
        yaxis_title="Precio",
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("#### Comparativo base 0 vs S&P 500")
    spx = fetch_history("^GSPC", period="5y", interval="1d")
    if spx.empty:
        st.info("No se pudo cargar el S&P 500. Se muestra solo la acción.")
        series = pd.DataFrame({ticker: hist_f["Close"]})
    else:
        spx_f = spx.loc[slice_by_period(spx.index, period_choice)]
        series = pd.concat(
            [hist_f["Close"].rename(ticker), spx_f["Close"].rename("S&P 500")],
            axis=1
        ).dropna()
    norm = series.apply(lambda s: s / s.iloc[0] - 1.0)
    fig2 = go.Figure()
    for c in norm.columns:
        fig2.add_trace(go.Scatter(x=norm.index, y=norm[c], mode="lines", name=c))
    fig2.update_layout(
        template="plotly_white",
        height=400,
        margin=dict(l=20,r=20,t=40,b=20),
        xaxis_title="Fecha",
        yaxis_title="Rendimiento normalizado",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---- Interest config & insights ----
def get_interest_config(interest: str) -> Dict:
    """Return structured config for each interest: description, companies, metrics, risks."""
    interest = interest or ""
    # Default empty config
    cfg = {
        "title": interest or "Selecciona un interés",
        "descripcion": "",
        "perfil_riesgo": "",
        "empresas": [],
        "metricas_clave": [],
        "oportunidades": [],
        "riesgos": [],
    }
    if "Tecnología" in interest or "Tech" in interest:
        cfg["title"] = "Tecnología / Big Tech"
        cfg["descripcion"] = (
            "Las grandes tecnológicas combinan modelos escalables (software, nube, publicidad digital) "
            "con márgenes altos y fuerte generación de flujo de efectivo. Suelen liderar los ciclos de mercado."
        )
        cfg["perfil_riesgo"] = "Volatilidad media-alta, muy sensibles a tasas de interés, regulación y ciclos de inversión en tecnología."
        cfg["empresas"] = [
            {"ticker": "AAPL", "empresa": "Apple", "segmento": "Dispositivos & ecosistema", "nota": "Ecosistema cerrado y servicios en expansión"},
            {"ticker": "MSFT", "empresa": "Microsoft", "segmento": "Nube & software empresarial", "nota": "Líder en nube y productividad"},
            {"ticker": "GOOGL", "empresa": "Alphabet", "segmento": "Publicidad digital & nube", "nota": "Search, YouTube y Google Cloud"},
            {"ticker": "AMZN", "empresa": "Amazon", "segmento": "E-commerce & nube (AWS)", "nota": "Escala global en retail y cloud"},
            {"ticker": "META", "empresa": "Meta Platforms", "segmento": "Redes sociales & publicidad", "nota": "Facebook, Instagram, WhatsApp"},
            {"ticker": "NVDA", "empresa": "NVIDIA", "segmento": "Chips para IA & gaming", "nota": "Arquitecturas clave para IA generativa"},
            {"ticker": "ADBE", "empresa": "Adobe", "segmento": "Software creativo & suscripciones", "nota": "Suite creativa estándar de la industria"},
            {"ticker": "CRM", "empresa": "Salesforce", "segmento": "Software CRM & nube", "nota": "Plataforma de relación con clientes"},
        ]
        cfg["metricas_clave"] = [
            "Crecimiento de ingresos (YoY).",
            "Margen operativo y margen bruto.",
            "Gasto en I+D como % de ventas.",
            "Participación de mercado en nube / software / publicidad digital.",
            "Valuación relativa (P/E, P/S) vs histórico y sector.",
        ]
        cfg["oportunidades"] = [
            "Beneficio del crecimiento estructural en digitalización y nube.",
            "Modelos de negocio con recurrencia (suscripciones, contratos de largo plazo).",
            "Escalabilidad: cada dólar adicional de ventas suele requerir menos costos marginales.",
        ]
        cfg["riesgos"] = [
            "Mayor sensibilidad a cambios en tasas de interés (afecta valuaciones).",
            "Riesgos regulatorios (antimonopolio, privacidad, uso de datos).",
            "Ritmo de innovación: riesgo de quedar atrás frente a nuevos competidores.",
        ]
    elif "Finanzas" in interest or "Bancos" in interest:
        cfg["title"] = "Finanzas / Bancos"
        cfg["descripcion"] = (
            "El sector financiero incluye bancos tradicionales, gestoras de activos y firmas de inversión. "
            "Su rentabilidad depende de márgenes de interés, calidad de crédito y eficiencia operativa."
        )
        cfg["perfil_riesgo"] = "Riesgo medio, muy ligado al ciclo económico y a la regulación."
        cfg["empresas"] = [
            {"ticker": "JPM", "empresa": "JPMorgan Chase", "segmento": "Banca universal", "nota": "Principal banco de EE.UU. por activos"},
            {"ticker": "BAC", "empresa": "Bank of America", "segmento": "Banca universal", "nota": "Fuerte presencia en banca de consumo"},
            {"ticker": "WFC", "empresa": "Wells Fargo", "segmento": "Banca minorista & comercial", "nota": "Exposición a créditos hipotecarios"},
            {"ticker": "C", "empresa": "Citigroup", "segmento": "Banca global", "nota": "Alta exposición internacional"},
            {"ticker": "GS", "empresa": "Goldman Sachs", "segmento": "Banca de inversión", "nota": "Trading, asesoría y gestión de activos"},
            {"ticker": "MS", "empresa": "Morgan Stanley", "segmento": "Wealth management & banca de inversión", "nota": "Fuerte negocio de gestión de patrimonio"},
            {"ticker": "AXP", "empresa": "American Express", "segmento": "Pagos & servicios financieros", "nota": "Tarjetas premium y red de pagos"},
            {"ticker": "V", "empresa": "Visa", "segmento": "Red de pagos", "nota": "Puente entre bancos, comercios y usuarios"},
            {"ticker": "MA", "empresa": "Mastercard", "segmento": "Red de pagos", "nota": "Modelo muy similar a Visa"},
        ]
        cfg["metricas_clave"] = [
            "ROE (rentabilidad sobre capital) y ROA (sobre activos).",
            "P/BV (precio / valor en libros) respecto a históricos.",
            "Calidad de cartera: NPLs (créditos vencidos) y provisiones.",
            "Margen financiero neto (NIM) y sensibilidad a tasas de interés.",
            "Diversificación de ingresos (intereses, comisiones, trading, gestión de activos).",
        ]
        cfg["oportunidades"] = [
            "Beneficio cuando las tasas suben de forma ordenada (mayor margen financiero).",
            "Participación en crecimiento de crédito al consumo y empresarial.",
            "Transformación digital (banca móvil, pagos digitales, fintech).",
        ]
        cfg["riesgos"] = [
            "Recesiones fuertes elevan la morosidad y las provisiones.",
            "Cambios regulatorios pueden limitar dividendos o recompras.",
            "Exposición a riesgos de liquidez o de confianza en el sistema.",
        ]
    elif "Crecimiento" in interest:
        cfg["title"] = "Crecimiento agresivo"
        cfg["descripcion"] = (
            "Empresas enfocadas en crecer ventas muy por encima del promedio, "
            "a menudo sacrificando ganancias en el corto plazo para ganar mercado."
        )
        cfg["perfil_riesgo"] = "Alto riesgo y alta volatilidad; dependen de ejecutar bien su plan de crecimiento."
        cfg["empresas"] = [
            {"ticker": "SHOP", "empresa": "Shopify", "segmento": "E-commerce enablement", "nota": "Plataforma para tiendas en línea"},
            {"ticker": "SNOW", "empresa": "Snowflake", "segmento": "Data cloud", "nota": "Plataforma de datos en la nube"},
            {"ticker": "CRWD", "empresa": "CrowdStrike", "segmento": "Ciberseguridad", "nota": "Seguridad en endpoints y nube"},
            {"ticker": "NET", "empresa": "Cloudflare", "segmento": "Infraestructura web", "nota": "Red distribuida para aplicaciones"},
            {"ticker": "MELI", "empresa": "MercadoLibre", "segmento": "E-commerce & fintech", "nota": "Plataforma líder en Latam"},
            {"ticker": "UBER", "empresa": "Uber", "segmento": "Movilidad & entregas", "nota": "Plataforma de transporte y delivery"},
        ]
        cfg["metricas_clave"] = [
            "Crecimiento de ingresos (>20% anual suele considerarse crecimiento fuerte).",
            "Evolución del margen bruto y operativo.",
            "Cash burn y caja disponible (runway).",
            "Relación entre crecimiento y dilución (emisión de nuevas acciones).",
            "Camino hacia rentabilidad: guidance de margen futuro.",
        ]
        cfg["oportunidades"] = [
            "Capacidad de capturar mercados poco penetrados con alto potencial.",
            "Escalabilidad: si el modelo funciona, los beneficios pueden crecer rápido.",
            "Innovación en modelos de negocio (suscripciones, plataformas, marketplace).",
        ]
        cfg["riesgos"] = [
            "No alcanzar la escala necesaria para ser rentables.",
            "Necesidad de capital constante (deuda o nuevas emisiones).",
            "Mayor caída en escenarios de aversión al riesgo o subidas de tasas.",
        ]
    elif "Consumo" in interest:
        cfg["title"] = "Consumo defensivo"
        cfg["descripcion"] = (
            "Empresas que venden productos esenciales (alimentos, higiene, salud básica) "
            "suelen tener ingresos más estables a lo largo del ciclo económico."
        )
        cfg["perfil_riesgo"] = "Riesgo bajo-medio, típicamente menor volatilidad que el mercado."
        cfg["empresas"] = [
            {"ticker": "PG", "empresa": "Procter & Gamble", "segmento": "Consumo básico", "nota": "Portafolio amplio de marcas de hogar"},
            {"ticker": "KO", "empresa": "Coca-Cola", "segmento": "Bebidas", "nota": "Marca global de bebidas no alcohólicas"},
            {"ticker": "PEP", "empresa": "PepsiCo", "segmento": "Snacks & bebidas", "nota": "Portafolio diversificado de marcas"},
            {"ticker": "WMT", "empresa": "Walmart", "segmento": "Retail defensivo", "nota": "Cadena de supermercados y tiendas de descuento"},
            {"ticker": "COST", "empresa": "Costco", "segmento": "Membresías & retail", "nota": "Modelo de membresía con lealtad alta"},
        ]
        cfg["metricas_clave"] = [
            "Estabilidad de ingresos y márgenes a lo largo del ciclo.",
            "Capacidad de trasladar inflación a precios (pricing power).",
            "Participación de mercado vs competidores locales.",
            "Política de dividendos y consistencia en pagos.",
        ]
        cfg["oportunidades"] = [
            "Ingresos más predecibles en recesiones.",
            "Atractivo para estrategias de largo plazo y menor volatilidad.",
            "Potencial de crecimiento en mercados emergentes.",
        ]
        cfg["riesgos"] = [
            "Presión en márgenes por costos de materias primas.",
            "Competencia de marcas propias (white label) en supermercados.",
            "Crecimiento estructural más moderado que otros sectores.",
        ]
    elif "Dividendos" in interest or "Ingreso pasivo" in interest:
        cfg["title"] = "Dividendos / Ingreso pasivo"
        cfg["descripcion"] = (
            "Empresas que reparten parte relevante de sus utilidades vía dividendos, "
            "buscando ofrecer un flujo de efectivo recurrente al accionista."
        )
        cfg["perfil_riesgo"] = "Riesgo medio, dependiendo del sector; foco en estabilidad de flujo y balance sólido."
        cfg["empresas"] = [
            {"ticker": "JNJ", "empresa": "Johnson & Johnson", "segmento": "Salud & consumo", "nota": "Historial largo de aumentos de dividendos"},
            {"ticker": "T", "empresa": "AT&T", "segmento": "Telecomunicaciones", "nota": "Históricamente con yield elevado"},
            {"ticker": "VZ", "empresa": "Verizon", "segmento": "Telecomunicaciones", "nota": "Pagos constantes de dividendos"},
            {"ticker": "XOM", "empresa": "Exxon Mobil", "segmento": "Energía", "nota": "Dividendos sostenidos en el tiempo"},
            {"ticker": "KO", "empresa": "Coca-Cola", "segmento": "Consumo defensivo", "nota": "Dividendos crecientes por décadas"},
        ]
        cfg["metricas_clave"] = [
            "Dividend yield (dividendo anual / precio).",
            "Payout ratio (porcentaje de utilidades que se reparten).",
            "Historial de aumentos, recortes o suspensiones de dividendos.",
            "Nivel de endeudamiento y capacidad de generar caja.",
        ]
        cfg["oportunidades"] = [
            "Flujo de efectivo recurrente que puede complementar otros ingresos.",
            "Disciplina de capital: ciertas empresas mantienen política estable de pagos.",
            "Posible reinversión de dividendos para acelerar crecimiento de patrimonio.",
        ]
        cfg["riesgos"] = [
            "Dividend yield muy alto puede esconder problemas de negocio.",
            "Recortes de dividendos suelen ser castigados por el mercado.",
            "Menor reinversión en el negocio si el payout es demasiado alto.",
        ]
    return cfg

def render_interest_insights(selected_interest: Optional[str], use_gemini: bool, api_key: str):
    st.subheader("Insights según tus intereses")
    if not selected_interest:
        st.info("Selecciona un interés en la barra lateral para ver información específica.")
        return
    cfg = get_interest_config(selected_interest)
    st.markdown(f"### {cfg['title']}")
    col1, col2 = st.columns([2,1])
    with col1:
        if cfg["descripcion"]:
            st.markdown(f"**Resumen del tema:** {cfg['descripcion']}")
        if cfg["perfil_riesgo"]:
            st.markdown(f"**Perfil de riesgo típico:** {cfg['perfil_riesgo']}")
    with col2:
        st.markdown("**Tipo de interés seleccionado:**")
        st.markdown(f"- {selected_interest}")
    # Empresas representativas
    if cfg["empresas"]:
        st.markdown("#### Mapa de empresas representativas")
        df_emp = pd.DataFrame(cfg["empresas"])
        df_show = df_emp[["ticker","empresa","segmento","nota"]].rename(columns={
            "ticker": "Ticker",
            "empresa": "Empresa",
            "segmento": "Subsegmento",
            "nota": "Comentario"
        })
        st.dataframe(df_show, use_container_width=True)
        # Conexión con portafolio del usuario
        port = load_portfolio()
        activos = port[port["activa"] == 1]
        tickers_user = set(str(t).upper() for t in activos["ticker"].tolist())
        tickers_interest = set(df_emp["ticker"].tolist())
        interseccion = sorted(tickers_user & tickers_interest)
        if interseccion:
            st.markdown(
                f"🔎 Actualmente tienes exposición a este tema a través de: **{', '.join(interseccion)}** "
                "(esta información es solo para que ubiques tus posiciones, no es una recomendación)."
            )
        else:
            st.caption("Por ahora no tienes posiciones activas en las empresas listadas para este tema.")
    # Métricas clave
    if cfg["metricas_clave"]:
        st.markdown("#### Qué deberías revisar en este tema")
        for m in cfg["metricas_clave"]:
            st.markdown(f"- {m}")
    # Oportunidades y riesgos
    col_o, col_r = st.columns(2)
    with col_o:
        st.markdown("#### ✅ Oportunidades frecuentes")
        if cfg["oportunidades"]:
            for o in cfg["oportunidades"]:
                st.markdown(f"- {o}")
        else:
            st.markdown("- Identificar empresas sólidas dentro del tema.")
    with col_r:
        st.markdown("#### ⚠️ Riesgos habituales")
        if cfg["riesgos"]:
            for r in cfg["riesgos"]:
                st.markdown(f"- {r}")
        else:
            st.markdown("- Volatilidad y cambios estructurales en el sector.")
    st.markdown("---")
    st.markdown("### Chat IA sobre este interés")
    if not use_gemini or not api_key or not GENAI_SDK_AVAILABLE:
        st.info("Para usar el chat de IA de este tema, activa Gemini y configura tu API key en el servidor.")
        return
    question = st.text_area(
        "Escribe una pregunta específica sobre este tema (no se darán recomendaciones de comprar/vender):",
        key="interest_chat_question",
        height=100,
        placeholder="Ejemplos: ¿Qué métricas son clave para analizar bancos de EE.UU.? ¿Cómo afecta la subida de tasas a las Big Tech?"
    )
    c1, c2 = st.columns([1,3])
    with c1:
        ask = st.button("Preguntar a la IA del sector", type="primary", use_container_width=True)
    if ask and question.strip():
        port = load_portfolio()
        activos = port[port["activa"] == 1]
        user_holdings = sorted(set(str(t).upper() for t in activos["ticker"].tolist()))
        answer = ai_interest_chat(
            interest=cfg["title"],
            companies=cfg["empresas"],
            user_holdings=user_holdings,
            question=question,
            api_key=api_key
        )
        if answer:
            st.markdown("#### Respuesta IA")
            st.markdown(answer)
        else:
            st.warning("No se pudo generar la respuesta en este momento. Intenta más tarde.")

# ---- Main app ----
def main_app():
    use_gemini = st.toggle("Activar comentarios con IA (Gemini)", True)
    st.title("TradeVision")
    st.caption("Control de portafolio, análisis de acciones y radar de oportunidades en un solo lugar.")
    tab_inicio, tab_port, tab_ai_tab, tab_stock, tab_watch, tab_interest = st.tabs(
        ["Visión general","Mi portafolio","Análisis IA","Acción individual","Watchlist","Mis intereses"]
    )
    # INICIO
    with tab_inicio:
        st.subheader("Resumen del portafolio")
        portafolio = load_portfolio()
        port_activo = portafolio[portafolio["activa"] == 1].copy()
        if port_activo.empty:
            st.info("Todavía no has registrado posiciones. Empieza en la pestaña **Mi portafolio**.")
        else:
            resumen_pos = []
            for _, row in port_activo.iterrows():
                t = str(row["ticker"])
                buy = float(row["precio_compra"])
                qty = float(row["cantidad"])
                if qty <= 0 or buy <= 0:
                    continue
                info_t = fetch_info(t)
                last, prev = get_last_and_prev_close(t)
                if last is None:
                    last = get_last_price(t, info_t) or buy
                valor_actual = last * qty
                inversion = buy * qty
                pl_total = valor_actual - inversion
                pct_total = (valor_actual/inversion - 1) if inversion>0 else 0.0
                if prev is not None:
                    day_ret = last/prev - 1
                    pl_dia = (last-prev)*qty
                else:
                    day_ret = 0.0
                    pl_dia = 0.0
                resumen_pos.append({
                    "ticker": t,
                    "cuenta": row["cuenta"],
                    "valor_actual": valor_actual,
                    "inversion": inversion,
                    "pl_total": pl_total,
                    "pct_total": pct_total,
                    "pl_dia": pl_dia,
                    "pct_dia": day_ret
                })
            if resumen_pos:
                df_res = pd.DataFrame(resumen_pos)
                total_valor = df_res["valor_actual"].sum()
                total_inv = df_res["inversion"].sum()
                total_pl = df_res["pl_total"].sum()
                total_pl_dia = df_res["pl_dia"].sum()
            else:
                df_res = pd.DataFrame()
                total_valor = total_inv = total_pl = total_pl_dia = 0.0
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Valor total del portafolio", fmt_money(total_valor))
            c2.metric("Capital invertido", fmt_money(total_inv))
            c3.metric(
                "Ganancia / pérdida total",
                fmt_money(total_pl),
                delta=fmt_pct(total_pl/total_inv) if total_inv>0 else None
            )
            c4.metric("Resultado del día", fmt_money(total_pl_dia))
            col_w, col_l = st.columns(2)
            with col_w:
                st.markdown("##### Mejores posiciones")
                if not df_res.empty:
                    winners = df_res.sort_values("pct_total", ascending=False).head(3)
                    for _, r in winners.iterrows():
                        st.markdown(f"- **{r['ticker']}** · {fmt_pct(r['pct_total'])} · {r['cuenta']}")
                else:
                    st.write("Sin datos.")
            with col_l:
                st.markdown("##### Posiciones a vigilar")
                if not df_res.empty:
                    losers = df_res.sort_values("pct_total", ascending=True).head(3)
                    for _, r in losers.iterrows():
                        st.markdown(f"- **{r['ticker']}** · {fmt_pct(r['pct_total'])} · {r['cuenta']}")
                else:
                    st.write("Sin datos.")
            st.markdown("##### Desempeño vs S&P 500")
            if not df_res.empty:
                period_choice_home = st.radio(
                    "Periodo",
                    ["3M", "6M", "1Y", "3Y", "ALL"],
                    horizontal=True,
                    key="home_perf_period"
                )
                tickers_activos = sorted(set(df_res["ticker"]))
                opciones_series = tickers_activos + ["S&P 500"]
                series_sel = st.multiselect(
                    "Series a mostrar",
                    opciones_series,
                    default=opciones_series,
                    key="home_series_sel",
                    help="Selecciona uno o varios tickers para ver su rendimiento vs S&P 500."
                )
                if not series_sel:
                    st.info("Selecciona al menos una serie para mostrar el gráfico.")
                else:
                    series_dict = {}
                    for t in tickers_activos:
                        if t not in series_sel:
                            continue
                        h = fetch_history(t, period="5y", interval="1d")
                        if h.empty:
                            continue
                        idx_slice = slice_by_period(
                            h.index,
                            period_choice_home if period_choice_home != "ALL" else "ALL"
                        )
                        h_f = h.loc[idx_slice]
                        if h_f.empty:
                            continue
                        series_dict[t] = h_f["Close"]
                    if "S&P 500" in series_sel:
                        spx = fetch_history("^GSPC", period="5y", interval="1d")
                        if not spx.empty:
                            idx_spx = slice_by_period(
                                spx.index,
                                period_choice_home if period_choice_home != "ALL" else "ALL"
                            )
                            spx_f = spx.loc[idx_spx]
                            if not spx_f.empty:
                                series_dict["S&P 500"] = spx_f["Close"]
                    if not series_dict:
                        st.info("No se pudieron generar las series de precios para el gráfico.")
                    else:
                        series = pd.DataFrame(series_dict)
                        series = series.sort_index().ffill().dropna()
                        norm = series.apply(lambda s: s / s.iloc[0] - 1.0)
                        figp = go.Figure()
                        for c in norm.columns:
                            figp.add_trace(
                                go.Scatter(x=norm.index, y=norm[c], mode="lines", name=c)
                            )
                        figp.update_layout(
                            template="plotly_white",
                            height=420,
                            margin=dict(l=20, r=20, t=40, b=20),
                            xaxis_title="Fecha",
                            yaxis_title="Rendimiento normalizado (base 0)",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="left",
                                x=0
                            )
                        )
                        st.plotly_chart(figp, use_container_width=True)
            else:
                st.info("Agrega posiciones para ver el gráfico vs S&P 500.")
    # MI PORTAFOLIO
    with tab_port:
        st.subheader("Gestión de portafolio")
        portafolio = load_portfolio()
        cuentas_existentes = sorted([c for c in portafolio["cuenta"].unique() if c])
        cuentas_base = ["Cuenta principal","Broker 1","Broker 2"]
        opciones_cuentas = list(dict.fromkeys(cuentas_base + cuentas_existentes))
        colA, colB = st.columns([1.1,1.9])
        with colA:
            st.markdown("##### Registrar nueva posición")
            cuenta_sel = st.selectbox("Cuenta", opciones_cuentas, key="port_cuenta_sel")
            with st.form("port_add_pos", clear_on_submit=True):
                t = st.text_input("Ticker", value="AAPL", key="port_add_ticker").upper().strip()
                c1, c2 = st.columns(2)
                with c1:
                    buy = st.number_input(
                        "Precio de compra por acción",
                        min_value=0.0,
                        step=0.01,
                        value=150.0,
                        key="port_add_buy"
                    )
                with c2:
                    qty = st.number_input(
                        "Cantidad de acciones",
                        min_value=0.0,
                        step=1.0,
                        value=10.0,
                        key="port_add_qty"
                    )
                fecha = st.date_input(
                    "Fecha de compra",
                    value=max(date.today()-dt.timedelta(days=30), date(2000,1,1)),
                    min_value=date(1990,1,1),
                    max_value=date.today(),
                    key="port_add_fecha"
                )
                nota = st.text_input("Nota (opcional)", value="", key="port_add_nota")
                submitted_add = st.form_submit_button("Guardar posición", use_container_width=True)
            if submitted_add and t and qty>0 and buy>0:
                new_row = {
                    "ticker": t,
                    "precio_compra": buy,
                    "cantidad": qty,
                    "fecha_compra": fecha.strftime("%Y-%m-%d"),
                    "nota": nota if nota else "",
                    "cuenta": cuenta_sel,
                    "activa": 1,
                    "fecha_cierre": "",
                    "precio_venta": np.nan
                }
                portafolio = pd.concat([portafolio, pd.DataFrame([new_row])], ignore_index=True)
                save_portfolio(portafolio)
                st.success(f"Posición en {t} guardada en «{cuenta_sel}».")
            st.markdown("##### Herramientas")
            if st.button("Limpiar cuenta (todas las posiciones)", type="secondary"):
                portafolio = portafolio[portafolio["cuenta"] != cuenta_sel].copy()
                save_portfolio(portafolio)
                st.warning(f"Se eliminaron todas las posiciones de «{cuenta_sel}».")
        with colB:
            st.markdown("##### Portafolio actual")
            df_cuenta = portafolio[portafolio["cuenta"] == cuenta_sel].copy()
            if df_cuenta.empty:
                st.info("Esta cuenta no tiene posiciones.")
            else:
                df_show = df_cuenta.copy()
                df_show["fecha_compra"] = df_show["fecha_compra"].apply(fmt_date)
                df_show["fecha_cierre"] = df_show["fecha_cierre"].apply(
                    lambda x: fmt_date(x) if x else ""
                )
                st.dataframe(df_show, use_container_width=True)
                st.markdown("###### Cerrar posición (registrar venta)")
                activas_cuenta = df_cuenta[df_cuenta["activa"] == 1]
                if activas_cuenta.empty:
                    st.info("No hay posiciones activas en esta cuenta.")
                else:
                    opciones_idx = activas_cuenta.index.tolist()
                    def fmt_pos(i):
                        r = activas_cuenta.loc[i]
                        return f"{r['ticker']} · {int(r['cantidad'])} @ {r['precio_compra']:.2f} ({fmt_date(r['fecha_compra'])})"
                    idx_sel = st.selectbox(
                        "Selecciona la posición",
                        opciones_idx,
                        format_func=fmt_pos,
                        key="port_close_idx"
                    )
                    precio_venta = st.number_input(
                        "Precio de venta por acción",
                        min_value=0.0,
                        step=0.01,
                        value=150.0,
                        key="port_close_price"
                    )
                    if st.button("Cerrar posición", key="port_close_btn"):
                        portafolio.loc[idx_sel, "activa"] = 0
                        portafolio.loc[idx_sel, "fecha_cierre"] = date.today().strftime("%Y-%m-%d")
                        portafolio.loc[idx_sel, "precio_venta"] = precio_venta
                        save_portfolio(portafolio)
                        st.success("Posición cerrada correctamente.")
                        st.rerun()
                st.markdown("###### Editar fecha de compra")
                idx_opciones_fecha = df_cuenta.index.tolist()
                if idx_opciones_fecha:
                    def fmt_pos_fecha(i):
                        r = df_cuenta.loc[i]
                        return f"{r['ticker']} · {int(r['cantidad'])} acciones"
                    idx_sel_fecha = st.selectbox(
                        "Selecciona la posición a editar",
                        idx_opciones_fecha,
                        format_func=fmt_pos_fecha,
                        key="edit_fecha_idx"
                    )
                    fecha_actual = df_cuenta.loc[idx_sel_fecha, "fecha_compra"]
                    if not fecha_actual:
                        fecha_default = date.today()
                    else:
                        try:
                            fecha_default = pd.to_datetime(fecha_actual).date()
                        except Exception:
                            fecha_default = date.today()
                    nueva_fecha = st.date_input(
                        "Nueva fecha de compra",
                        value=fecha_default,
                        key="edit_fecha_input"
                    )
                    if st.button("Guardar nueva fecha de compra", key="edit_fecha_btn"):
                        portafolio.loc[idx_sel_fecha, "fecha_compra"] = nueva_fecha.strftime("%Y-%m-%d")
                        save_portfolio(portafolio)
                        st.success("Fecha de compra actualizada.")
                        st.rerun()
    # ANÁLISIS IA
    with tab_ai_tab:
        st.subheader("Resumen del portafolio con IA")
        portafolio = load_portfolio()
        port_activo = portafolio[portafolio["activa"] == 1].copy()
        if port_activo.empty:
            st.info("No tienes posiciones activas.")
        else:
            resumen_pos = []
            for _, row in port_activo.iterrows():
                t = str(row["ticker"])
                buy = float(row["precio_compra"])
                qty = float(row["cantidad"])
                if qty<=0 or buy<=0:
                    continue
                last, prev = get_last_and_prev_close(t)
                if last is None:
                    info_t = fetch_info(t)
                    last = get_last_price(t, info_t) or buy
                valor_actual = last * qty
                inversion = buy * qty
                pl_total = valor_actual - inversion
                pct_total = (valor_actual/inversion - 1) if inversion>0 else 0.0
                if prev is not None:
                    day_ret = last/prev - 1
                    pl_dia = (last-prev)*qty
                else:
                    day_ret = 0.0
                    pl_dia = 0.0
                resumen_pos.append({
                    "ticker": t,
                    "cuenta": row["cuenta"],
                    "valor_actual": valor_actual,
                    "inversion": inversion,
                    "pl_total": pl_total,
                    "pct_total": pct_total,
                    "pl_dia": pl_dia,
                    "pct_dia": day_ret
                })
            df_res = pd.DataFrame(resumen_pos) if resumen_pos else pd.DataFrame()
            total_valor = df_res["valor_actual"].sum() if not df_res.empty else 0.0
            total_inv = df_res["inversion"].sum() if not df_res.empty else 0.0
            total_pl = df_res["pl_total"].sum() if not df_res.empty else 0.0
            total_pl_dia = df_res["pl_dia"].sum() if not df_res.empty else 0.0
            c1, c2, c3 = st.columns(3)
            c1.metric("Valor total", fmt_money(total_valor))
            c2.metric(
                "Ganancia / pérdida total",
                fmt_money(total_pl),
                delta=fmt_pct(total_pl/total_inv) if total_inv>0 else None
            )
            c3.metric("Resultado del día", fmt_money(total_pl_dia))
            st.markdown("##### Comentario del portafolio")
            if not GENAI_SDK_AVAILABLE or not API_KEY or not use_gemini:
                st.info("Para ver el comentario de IA, instala `google-genai` y configura tu API key de Gemini.")
            else:
                txt = ai_portfolio_comment(
                    total_value=total_valor,
                    total_pl=total_pl,
                    day_pl=total_pl_dia,
                    positions_summary=resumen_pos,
                    api_key=API_KEY
                )
                if txt:
                    st.markdown(txt)
                else:
                    st.info("No se pudo generar el comentario IA en este momento.")
            st.markdown("---")
            st.markdown("##### Análisis IA por posición")
            opciones_idx = port_activo.index.tolist()
            def fmt_pos2(i):
                r = port_activo.loc[i]
                return f"{r['ticker']} · {int(r['cantidad'])} @ {r['precio_compra']:.2f} ({fmt_date(r['fecha_compra'])})"
            idx_sel = st.selectbox(
                "Selecciona una posición",
                opciones_idx,
                format_func=fmt_pos2,
                key="ai_pos_idx"
            )
            pos = port_activo.loc[idx_sel]
            t = str(pos["ticker"])
            buy = float(pos["precio_compra"])
            qty = float(pos["cantidad"])
            fecha_compra = pos["fecha_compra"]
            if isinstance(fecha_compra, str):
                fecha_compra = pd.to_datetime(fecha_compra, errors="coerce")
            fecha_compra = fecha_compra.date() if isinstance(fecha_compra, pd.Timestamp) else fecha_compra
            h = fetch_history(t)
            if h.empty:
                st.error("No hay datos para esta posición.")
            else:
                info2 = fetch_info(t)
                last = get_last_price(t, info2) or float(h["Close"].iloc[-1])
                rec, declared, src = summarize_recommendations(t, info2)
                concl, lab = analyst_conclusion(rec)
                low52, high52 = get_52w_range(info2)
                inv = buy*qty
                val = last*qty
                diff = val-inv
                pct = (diff/inv) if inv>0 else 0.0
                dias = (date.today()-fecha_compra).days if isinstance(fecha_compra, date) else 0
                ann = (last/buy)**(365/dias)-1 if (dias>0 and buy>0) else None
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Precio de compra", fmt_money(buy))
                c2.metric("Precio actual", fmt_money(last))
                c3.metric("Ganancia / pérdida", fmt_money(diff), delta=fmt_pct(pct))
                c4.metric("Días en posición", dias)
                d1, d2, d3 = st.columns(3)
                d1.metric("Rendimiento anualizado", fmt_pct(ann) if ann is not None else "—")
                d2.metric("Mín. 52 semanas", fmt_money(low52))
                d3.metric("Máx. 52 semanas", fmt_money(high52))
                st.markdown("###### Opinión de analistas")
                b1, b2, b3 = st.columns(3)
                b1.metric("Comprar", rec["Buy"])
                b2.metric("Mantener", rec["Hold"])
                b3.metric("Vender", rec["Sell"])
                st.caption(f"Opiniones consideradas: {declared or (rec['Buy']+rec['Hold']+rec['Sell'])}")
                color = "metric-good" if lab=="Comprar" else "metric-bad" if lab=="Vender" else "metric-warn"
                st.markdown(f"**Conclusión:** <span class='{color}'>{concl}</span>", unsafe_allow_html=True)
                st.markdown("###### Comentario IA sobre esta posición")
                if not GENAI_SDK_AVAILABLE or not API_KEY or not use_gemini:
                    st.info("Activa Gemini y coloca tu API key para ver el análisis IA.")
                else:
                    company_name = get_company_name(info2) or t
                    rec_sum = concl.replace("**","")
                    ai_txt = ai_position_comment(
                        ticker=t,
                        company_name=company_name,
                        last_price=last,
                        buy_price=buy,
                        qty=qty,
                        days_held=dias,
                        ann_return=ann,
                        pct_gain=pct,
                        rec_summary=rec_sum,
                        api_key=API_KEY
                    )
                    if ai_txt:
                        st.markdown(ai_txt)
                    else:
                        st.info("No se pudo generar el comentario IA en este momento.")
    # ACCIÓN INDIVIDUAL
    with tab_stock:
        st.subheader("Análisis puntual de una acción")
        render_single_stock_analysis("stock", use_gemini, API_KEY)
    # WATCHLIST
    with tab_watch:
        st.subheader("Ideas de acciones para vigilar")
        rows = []
        for t in UNIVERSE_MOMENTUM:
            m = momentum_metrics(t)
            if m:
                rows.append(m)
        if not rows:
            st.info("No se pudieron calcular métricas de momentum.")
        else:
            df_m = pd.DataFrame(rows)
            df_m["score"] = df_m["ret_3m"].fillna(0) - df_m["dist_high"].fillna(0).clip(lower=-0.3)
            df_m = df_m.sort_values("score", ascending=False)
            top_n = df_m.head(10).copy()
            top_n_display = top_n[[
                "ticker","name","last","ret_3m","ret_1y","vol","dist_high"
            ]].copy()
            top_n_display["Precio actual"] = top_n_display["last"].apply(fmt_money)
            top_n_display["Rend 3M"] = top_n_display["ret_3m"].apply(fmt_pct)
            top_n_display["Rend 1Y"] = top_n_display["ret_1y"].apply(fmt_pct)
            top_n_display["Vol anualizada"] = top_n_display["vol"].apply(fmt_pct)
            top_n_display["Dist máx 52w"] = top_n_display["dist_high"].apply(fmt_pct)
            top_n_display = top_n_display[[
                "ticker","name","Precio actual","Rend 3M","Rend 1Y",
                "Vol anualizada","Dist máx 52w"
            ]]
            st.markdown("##### Acciones con momentum reciente interesante")
            st.dataframe(top_n_display, use_container_width=True)
            st.markdown(
                "Esta lista se basa en rendimiento reciente y cercanía al máximo de 52 semanas. "
                "No constituye una recomendación de compra o venta."
            )
            st.markdown("##### Comentario IA sobre estas ideas")
            if not GENAI_SDK_AVAILABLE or not API_KEY or not use_gemini:
                st.info("Activa Gemini y coloca tu API key para ver el análisis IA.")
            else:
                ai_txt = ai_watchlist_comment(rows=top_n.to_dict(orient="records"), api_key=API_KEY)
                if ai_txt:
                    st.markdown(ai_txt)
                else:
                    st.info("No se pudo generar el comentario IA en este momento.")
    # MIS INTERESES
    with tab_interest:
        render_interest_insights(st.session_state.get("selected_interest"), use_gemini, API_KEY)
    st.markdown("---")
    st.caption(
        "La información mostrada en TradeVision es de carácter informativo y puede contener errores o retrasos. "
        "No constituye una recomendación de inversión ni una oferta de compra o venta de valores."
    )
    st.caption("TradeVision · Desarrollado por Manuel Verdugo Navarro · 2025.")

# ---- Entrypoint ----
def main():
    ensure_auth_state()
    if not st.session_state.get("is_auth", False):
        auth_view()
        return
    st.sidebar.markdown("### Usuario")
    st.sidebar.write(st.session_state.get("user_name", ""))
    st.sidebar.write(st.session_state.get("user_email", ""))
    intereses = st.session_state.get("user_intereses", [])
    if intereses:
        st.sidebar.markdown("**Intereses principales:**")
        selected_interest = st.sidebar.radio(
            "Elige un interés para ver detalles:",
            intereses,
            key="sidebar_interest",
            label_visibility="collapsed"
        )
        st.session_state["selected_interest"] = selected_interest
    else:
        st.sidebar.caption("Puedes configurar tus intereses al crear o editar tu cuenta.")
    if st.sidebar.button("Cerrar sesión", key="btn_logout"):
        for k in ["is_auth","user_id","user_name","user_email","user_intereses","selected_interest"]:
            st.session_state.pop(k, None)
        st.rerun()
    main_app()

if __name__ == "__main__":
    main()
