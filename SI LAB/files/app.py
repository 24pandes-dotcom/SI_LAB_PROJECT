"""Daily Temperature Patterns & Weather Analysis - final-year project (Streamlit)."""
import io
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

st.set_page_config(page_title="Daily Temperature Patterns", page_icon="🌦️", layout="wide")

DATA_PATH = Path(__file__).parent / "daily_temperature_patterns.csv"
TEAL, AMBER, INDIGO, RED, GREEN = "#0E7C86", "#F59E0B", "#6366F1", "#EF4444", "#10B981"
SEASON_ORDER = ["Winter", "Summer", "Monsoon", "Post-Monsoon"]
SEASON_COLORS = {"Winter": "#3B82F6", "Summer": RED, "Monsoon": GREEN, "Post-Monsoon": AMBER}
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# canonical name -> accepted (normalised) column names
ALIASES = {
    "Date": ["date"], "Temperature": ["temperature", "temp", "temperaturec"],
    "Humidity": ["humidity", "humiditypercent"], "Rainfall": ["rainfallmm", "rainfall", "rain"],
    "Wind_Speed": ["windspeedkmph", "windspeed", "wind"], "Pressure": ["pressurehpa", "pressure"],
    "Cloud_Cover": ["cloudcoverpercent", "cloudcover", "cloudcoverage", "cloud"],
    "Sunshine": ["sunshinehours", "sunshine"], "Season": ["season"],
    "Day_Type": ["daytype", "datetype"], "Feels_Like": ["feelsliketemperature", "feelslike"],
    "Temp_Change": ["temperaturechange", "tempchange"],
}
LABELS = {
    "Temperature": "Temperature (°C)", "Humidity": "Humidity (%)", "Rainfall": "Rainfall (mm)",
    "Wind_Speed": "Wind Speed (km/h)", "Pressure": "Pressure (hPa)", "Cloud_Cover": "Cloud Cover (%)",
    "Sunshine": "Sunshine (hours)", "Feels_Like": "Feels-Like Temp (°C)", "Temp_Change": "Temp Change (°C)",
}
WEATHER_VARS = ["Humidity", "Rainfall", "Wind_Speed", "Pressure", "Cloud_Cover", "Sunshine"]
SECTIONS = ["🏠 Home Dashboard", "📊 Exploratory Analysis", "📐 Statistical Analysis", "🎯 Sampling Techniques",
            "🔔 Sampling Distribution & CLT", "🔁 Bootstrap Analysis", "📏 Confidence Intervals",
            "🔗 Weather Relationships", "🤖 Predictive Analysis", "ℹ️ About the Project"]

st.markdown(f"""<style>
.hero{{background:linear-gradient(120deg,{TEAL},{INDIGO});padding:1.6rem 2rem;border-radius:16px;color:#fff;margin-bottom:1.2rem}}
.hero h1{{color:#fff;margin:0;font-size:2.1rem}} .hero p{{margin:.4rem 0 0;opacity:.92;font-size:1.02rem}}
div[data-testid="stMetric"]{{background:#fff;border:1px solid #D9E6E8;border-left:5px solid {TEAL};padding:.8rem 1rem;border-radius:12px;box-shadow:0 1px 4px rgba(0,0,0,.06)}}
h2,h3{{color:#0B5D66}} .block-container{{padding-top:1.5rem}}
</style>""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------- data
def norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@st.cache_data(show_spinner="Loading dataset...")
def load_data(raw: bytes):
    df = pd.read_csv(io.BytesIO(raw))
    lookup = {norm(c): c for c in df.columns}
    rename, missing = {}, []
    for canon, names in ALIASES.items():
        hit = next((lookup[n] for n in names if n in lookup), None)
        if hit is not None and hit not in rename:
            rename[hit] = canon
        else:
            missing.append(canon)
    df = df.rename(columns=rename)
    notes = []
    for c in LABELS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ("Season", "Day_Type"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan})
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        bad = int(df["Date"].isna().sum())
        if bad:
            notes.append(f"{bad} rows with unreadable dates were removed.")
            df = df.dropna(subset=["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
        df["Month"] = df["Date"].dt.month
        df["Month_Name"] = df["Date"].dt.strftime("%b")
        df["Year_Month"] = df["Date"].dt.to_period("M").astype(str)
    return df, missing, notes


def style(fig, h=380):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=50, b=10),
                      title_font_size=16, legend_title_text="")
    return fig


def need(d, cols, what):
    miss = [LABELS.get(c, c) for c in cols if c not in d.columns]
    if miss:
        st.warning(f"**{what}** needs these columns, which were not found in the dataset: {', '.join(miss)}.")
        return False
    return True


def season_list(d):
    return [s for s in SEASON_ORDER if s in set(d["Season"].dropna())] + \
           sorted(set(d["Season"].dropna()) - set(SEASON_ORDER))


# ----------------------------------------------------------------------------- load
raw = None
if DATA_PATH.exists():
    raw = DATA_PATH.read_bytes()
else:
    st.sidebar.error("`daily_temperature_patterns.csv` not found next to app.py.")
    up = st.sidebar.file_uploader("Upload the CSV instead", type="csv")
    if up:
        raw = up.getvalue()
if raw is None:
    st.error("Dataset missing. Save `daily_temperature_patterns.csv` in the same folder as `app.py` and refresh.")
    st.stop()
try:
    full, missing_cols, load_notes = load_data(raw)
except Exception as e:  # noqa
    st.error(f"Could not read the CSV: {e}")
    st.stop()
if "Temperature" not in full.columns or full["Temperature"].notna().sum() < 5:
    st.error("A usable Temperature column is required (at least 5 numeric values).")
    st.stop()

# ----------------------------------------------------------------------------- sidebar
st.sidebar.markdown("## 🌦️ Weather Analytics")
section = st.sidebar.radio("Navigate", SECTIONS, label_visibility="collapsed")
st.sidebar.markdown("---\n### Filters")
d = full.copy()
if "Date" in d.columns:
    lo, hi = d["Date"].min().date(), d["Date"].max().date()
    rng_sel = st.sidebar.date_input("Date range", (lo, hi), min_value=lo, max_value=hi)
    if isinstance(rng_sel, (tuple, list)) and len(rng_sel) == 2:
        d = d[(d["Date"].dt.date >= rng_sel[0]) & (d["Date"].dt.date <= rng_sel[1])]
if "Season" in d.columns:
    chosen = st.sidebar.multiselect("Season", season_list(full), default=season_list(full))
    d = d[d["Season"].isin(chosen)]
SEED = int(st.sidebar.number_input("Random seed (reproducibility)", 0, 99999, 42))
st.sidebar.caption(f"Filtered records: **{len(d)}** of {len(full)}")
d = d.reset_index(drop=True)

st.markdown("""<div class="hero"><h1>🌦️ Daily Temperature Patterns & Weather Analysis</h1>
<p>Statistics, sampling, Central Limit Theorem, bootstrap, confidence intervals and machine-learning prediction on daily weather data.</p></div>""",
            unsafe_allow_html=True)
for n in load_notes:
    st.info(n)
if len(d) < 10:
    st.warning("Fewer than 10 records match the current filters. Widen the filters in the sidebar.")
    st.stop()
T = d["Temperature"].dropna()
N = len(T)
num_cols = [c for c in LABELS if c in d.columns]


def scatter_fit(df, x, y, color=None, title=None):
    sub = df[[x, y] + ([color] if color else [])].dropna()
    fig = px.scatter(sub, x=x, y=y, color=color, opacity=.7, color_discrete_map=SEASON_COLORS,
                     labels={x: LABELS.get(x, x), y: LABELS.get(y, y)}, title=title)
    if len(sub) > 2 and sub[x].nunique() > 1:
        m, b = np.polyfit(sub[x], sub[y], 1)
        xs = np.array([sub[x].min(), sub[x].max()])
        fig.add_trace(go.Scatter(x=xs, y=m * xs + b, mode="lines", name="Trend line", line=dict(color="#111", dash="dash")))
    return style(fig)


# ============================================================================= A. HOME
if section == SECTIONS[0]:
    st.header("Home Dashboard")
    c = st.columns(5)
    c[0].metric("Total records", f"{len(d):,}")
    c[1].metric("Avg temperature", f"{T.mean():.2f} °C")
    c[2].metric("Avg humidity", f"{d['Humidity'].mean():.1f} %" if "Humidity" in d else "n/a")
    c[3].metric("Total rainfall", f"{d['Rainfall'].sum():,.1f} mm" if "Rainfall" in d else "n/a")
    c[4].metric("Avg wind speed", f"{d['Wind_Speed'].mean():.1f} km/h" if "Wind_Speed" in d else "n/a")
    if missing_cols:
        st.caption("Columns not found in the dataset (related features are disabled): " + ", ".join(missing_cols))
    st.subheader("Dataset preview")
    st.dataframe(d.drop(columns=["Month", "Month_Name", "Year_Month"], errors="ignore").head(50), width="stretch")
    if "Date" in d.columns:
        fig = px.line(d, x="Date", y="Temperature", title="Daily temperature (filtered range)", labels={"Temperature": LABELS["Temperature"]})
        fig.update_traces(line_color=TEAL)
        st.plotly_chart(style(fig, 320), width="stretch")
    st.info("Use the sidebar to change the date range and seasons. Every section recalculates using the filtered data.")

# ============================================================================= B. EDA
elif section == SECTIONS[1]:
    st.header("Exploratory Data Analysis")
    a, b = st.columns(2)
    fig = px.histogram(d, x="Temperature", nbins=25, marginal="rug", title="Temperature distribution", labels={"Temperature": LABELS["Temperature"]})
    fig.update_traces(marker_color=TEAL, selector=dict(type="histogram"))
    a.plotly_chart(style(fig), width="stretch")
    fig = px.box(d, y="Temperature", points="outliers", title="Temperature boxplot", labels={"Temperature": LABELS["Temperature"]})
    fig.update_traces(marker_color=INDIGO)
    b.plotly_chart(style(fig), width="stretch")
    if "Date" in d.columns:
        t = d[["Date", "Temperature"]].copy()
        t["7-day rolling mean"] = t["Temperature"].rolling(7, min_periods=1).mean()
        fig = px.line(t, x="Date", y=["Temperature", "7-day rolling mean"], title="Daily temperature trend",
                      color_discrete_sequence=["#9CC9CE", RED], labels={"value": LABELS["Temperature"], "variable": ""})
        st.plotly_chart(style(fig), width="stretch")
        a, b = st.columns(2)
        m = d.groupby(["Month", "Month_Name"])["Temperature"].mean().reset_index()
        fig = px.bar(m, x="Month_Name", y="Temperature", title="Monthly average temperature", text_auto=".1f",
                     labels={"Month_Name": "Month", "Temperature": LABELS["Temperature"]}, color_discrete_sequence=[TEAL])
        a.plotly_chart(style(fig), width="stretch")
    else:
        a, b = st.columns(2)
        st.info("Date column not found: trend and monthly charts are unavailable.")
    if "Season" in d.columns:
        s = d.groupby("Season")["Temperature"].mean().reindex(season_list(d)).reset_index()
        fig = px.bar(s, x="Season", y="Temperature", color="Season", color_discrete_map=SEASON_COLORS, text_auto=".1f",
                     title="Seasonal average temperature", labels={"Temperature": LABELS["Temperature"]})
        b.plotly_chart(style(fig), width="stretch")

    st.subheader("Other weather variables")
    wv = [v for v in WEATHER_VARS if v in d.columns]
    if wv:
        v = st.selectbox("Choose a variable", wv, format_func=lambda x: LABELS[x])
        a, b = st.columns(2)
        a.plotly_chart(style(px.histogram(d, x=v, nbins=25, title=f"Distribution of {LABELS[v]}", color_discrete_sequence=[AMBER],
                                          labels={v: LABELS[v]})), width="stretch")
        if "Date" in d.columns:
            b.plotly_chart(style(px.line(d, x="Date", y=v, title=f"{LABELS[v]} over time", color_discrete_sequence=[INDIGO],
                                         labels={v: LABELS[v]})), width="stretch")
        if "Season" in d.columns:
            st.plotly_chart(style(px.box(d, x="Season", y=v, color="Season", color_discrete_map=SEASON_COLORS,
                                         category_orders={"Season": season_list(d)}, title=f"{LABELS[v]} by season",
                                         labels={v: LABELS[v]}), 330), width="stretch")
    a, b = st.columns(2)
    if "Humidity" in d.columns:
        a.plotly_chart(scatter_fit(d, "Humidity", "Temperature", "Season" if "Season" in d else None, "Temperature vs Humidity"), width="stretch")
    if "Rainfall" in d.columns:
        b.plotly_chart(scatter_fit(d, "Rainfall", "Temperature", "Season" if "Season" in d else None, "Temperature vs Rainfall"), width="stretch")

    st.subheader("Correlation heatmap")
    cm = d[num_cols].loc[:, d[num_cols].nunique() > 1]
    if cm.shape[1] >= 2:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.heatmap(cm.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=.5, ax=ax)
        ax.set_title("Pearson correlation between numerical variables")
        st.pyplot(fig)
        plt.close(fig)
        st.caption("Feels-Like Temperature and Temperature Change are calculated from temperature, so their high correlation with Temperature is expected.")

    st.subheader("Main observations (calculated from the filtered data)")
    obs = [f"Temperature ranges from **{T.min():.1f} °C to {T.max():.1f} °C** with mean **{T.mean():.2f} °C** (skewness {stats.skew(T):.2f}, "
           f"{'roughly symmetric' if abs(stats.skew(T)) < .5 else 'noticeably skewed'})."]
    if "Month" in d.columns and d["Month"].nunique() > 1:
        mm = d.groupby("Month_Name")["Temperature"].mean()
        obs.append(f"Warmest month: **{mm.idxmax()}** ({mm.max():.1f} °C); coolest month: **{mm.idxmin()}** ({mm.min():.1f} °C).")
    if "Season" in d.columns and d["Season"].nunique() > 1:
        ss = d.groupby("Season")["Temperature"].mean()
        obs.append(f"Warmest season: **{ss.idxmax()}** ({ss.max():.1f} °C); coolest season: **{ss.idxmin()}** ({ss.min():.1f} °C).")
    cor = d[[v for v in wv] + ["Temperature"]].corr()["Temperature"].drop("Temperature").dropna() if wv else pd.Series(dtype=float)
    if len(cor):
        k = cor.abs().idxmax()
        obs.append(f"The weather variable most strongly correlated with temperature is **{LABELS[k]}** (r = {cor[k]:.2f}). Correlation does not prove causation.")
    st.markdown("\n".join(f"- {o}" for o in obs))

# ============================================================================= C. STATS
elif section == SECTIONS[2]:
    st.header("Statistical Analysis")
    sk, ku = stats.skew(T), stats.kurtosis(T)
    rows = {"Mean": T.mean(), "Median": T.median(), "Variance": T.var(ddof=1), "Standard deviation": T.std(ddof=1),
            "Skewness": sk, "Kurtosis (excess)": ku, "Minimum": T.min(), "Maximum": T.max(), "Range": T.max() - T.min(),
            "IQR": T.quantile(.75) - T.quantile(.25)}
    c = st.columns(5)
    for col, (k, v) in zip(c, list(rows.items())[:5]):
        col.metric(k, f"{v:.3f}")
    st.dataframe(pd.DataFrame({"Statistic": rows.keys(), "Temperature (°C)": [round(v, 4) for v in rows.values()]}), hide_index=True, width="stretch")
    st.subheader("What do these mean?")
    st.markdown(f"""
- **Mean** – the average temperature ({T.mean():.2f} °C). **Median** – the middle value ({T.median():.2f} °C); if it is close to the mean, the data is fairly balanced.
- **Variance / Standard deviation** – how spread out daily temperatures are around the mean. Typical day differs from the mean by about **{T.std():.2f} °C**.
- **Skewness** ({sk:.2f}) – 0 means symmetric; positive means a longer tail of hot days; negative means a longer tail of cold days.
- **Kurtosis** ({ku:.2f}) – 0 is similar to a normal curve; negative means flatter, with fewer extreme values.
- **IQR** – the range of the middle 50% of days.""")
    if 3 <= N <= 5000:
        w, p = stats.shapiro(T)
        st.info(f"Shapiro–Wilk normality test: W = {w:.4f}, p = {p:.4g}. " + ("p < 0.05 → the temperatures are **not** perfectly normal." if p < .05 else "p ≥ 0.05 → no strong evidence against normality."))
    st.subheader("Descriptive statistics for all numerical variables")
    ds = d[num_cols].describe().T
    ds["variance"] = d[num_cols].var()
    ds["skewness"] = d[num_cols].skew()
    ds.index = [LABELS[i] for i in ds.index]
    st.dataframe(ds.round(3), width="stretch")

# ============================================================================= D. SAMPLING
elif section == SECTIONS[3]:
    st.header("Sampling Techniques")
    st.markdown("A **sample** is a small part of the data used to estimate things about the whole (the *population*). Here the filtered dataset is the population.")
    pop = d.dropna(subset=["Temperature"]).reset_index(drop=True)
    n = st.slider("Sample size", 5, min(200, len(pop)), min(30, len(pop)))
    st.caption(f"Population: {len(pop)} days, mean temperature **{pop['Temperature'].mean():.3f} °C**")
    k = max(len(pop) // n, 1)
    samples = {"Simple random": pop.sample(n=n, random_state=SEED)}
    samples["Systematic"] = pop.iloc[np.random.default_rng(SEED).integers(0, k)::k].head(n)
    if "Season" in pop.columns and pop["Season"].notna().any():
        g = pop.dropna(subset=["Season"]).groupby("Season")
        alloc = (g.size() / g.size().sum() * n).round().astype(int).clip(lower=1)
        samples["Stratified (by Season)"] = pd.concat([x.sample(min(len(x), alloc[s]), random_state=SEED) for s, x in g])
    if "Year_Month" in pop.columns:
        months = pop["Year_Month"].unique()
        m = st.slider("Cluster sampling: number of months (clusters) to select", 1, len(months), min(3, len(months)))
        pick = np.random.default_rng(SEED).choice(months, size=m, replace=False)
        samples["Cluster (by month)"] = pop[pop["Year_Month"].isin(pick)]
    else:
        st.info("Cluster sampling needs a valid Date column.")
    if "Season" not in pop.columns:
        st.info("Stratified sampling needs the Season column.")
    mu = pop["Temperature"].mean()
    cmp = pd.DataFrame([{"Method": k_, "Sample size": len(s), "Mean temp (°C)": s["Temperature"].mean(), "Std dev": s["Temperature"].std(),
                         "Min": s["Temperature"].min(), "Max": s["Temperature"].max(),
                         "Error vs population mean": s["Temperature"].mean() - mu} for k_, s in samples.items()])
    st.subheader("Comparison")
    st.dataframe(cmp.round(3), hide_index=True, width="stretch")
    fig = px.bar(cmp, x="Method", y="Mean temp (°C)", color="Method", text_auto=".2f", title="Sample means vs population mean (dashed line)")
    fig.add_hline(y=mu, line_dash="dash", line_color="#111")
    lo_, hi_ = cmp["Mean temp (°C)"].min(), cmp["Mean temp (°C)"].max()
    fig.update_yaxes(range=[min(lo_, mu) - 1.5, max(hi_, mu) + 1.5])
    st.plotly_chart(style(fig, 340), width="stretch")
    st.subheader("Sample previews")
    cols = [c for c in ["Date", "Temperature", "Humidity", "Rainfall", "Season"] if c in pop.columns]
    tabs = st.tabs(list(samples))
    expl = {"Simple random": "Every day has an equal chance of selection.",
            "Systematic": f"Every {k}-th day after a random start.",
            "Stratified (by Season)": "Random picks from each season in proportion to the season's size.",
            "Cluster (by month)": "Whole months are picked at random and every day in them is used."}
    for tab, (name, s) in zip(tabs, samples.items()):
        with tab:
            st.caption(expl[name])
            st.dataframe(s[cols].head(15), width="stretch")
    st.info("Because sampling is random, a single sample's mean differs a little from the population mean. Change the seed in the sidebar to see this.")

# ============================================================================= E. CLT
elif section == SECTIONS[4]:
    st.header("Sampling Distributions & the Central Limit Theorem")
    st.markdown("**CLT:** if we take many random samples and compute each sample's mean, the means form a roughly normal (bell-shaped) distribution centred on the population mean, "
                "with spread (standard error) **σ/√n** — and it narrows as the sample size *n* grows.")
    vals = T.to_numpy()
    reps = st.slider("Number of repeated samples", 200, 5000, 1000, 100)
    sizes = [s for s in (10, 30, 50) if s <= N] or [max(2, N // 2)]
    if N < 50:
        st.warning(f"Only {N} records available, so sample sizes were adjusted to {sizes}.")

    @st.cache_data(show_spinner=False)
    def sample_means(v, n, reps, seed):
        return np.random.default_rng(seed + n).choice(v, size=(reps, n), replace=True).mean(axis=1)

    mu, sd = vals.mean(), vals.std(ddof=1)
    fig = make_subplots(rows=1, cols=len(sizes), shared_yaxes=True, subplot_titles=[f"n = {s}" for s in sizes])
    out = []
    for i, s in enumerate(sizes, 1):
        sm = sample_means(vals, s, reps, SEED)
        fig.add_trace(go.Histogram(x=sm, histnorm="probability density", nbinsx=30, marker_color=TEAL, opacity=.75, name="Sample means", showlegend=i == 1), row=1, col=i)
        xs = np.linspace(sm.min(), sm.max(), 200)
        fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, mu, sd / np.sqrt(s)), line=dict(color=RED), name="Normal curve (theory)", showlegend=i == 1), row=1, col=i)
        out.append({"Sample size n": s, "Mean of sample means": sm.mean(), "Empirical SE (std of means)": sm.std(ddof=1),
                    "Theoretical SE (σ/√n)": sd / np.sqrt(s), "Skewness of means": stats.skew(sm)})
    fig.update_xaxes(title_text="Sample mean (°C)")
    fig.update_yaxes(title_text="Density", col=1)
    fig.update_layout(title=f"Sampling distribution of the mean ({reps} samples each)")
    st.plotly_chart(style(fig, 400), width="stretch")
    a, b = st.columns(2)
    a.plotly_chart(style(px.histogram(x=vals, nbins=25, title="Original (population) temperature distribution", labels={"x": LABELS["Temperature"]},
                                      color_discrete_sequence=[AMBER]), 320), width="stretch")
    res = pd.DataFrame(out)
    with b:
        st.markdown("**Standard errors**")
        st.dataframe(res.round(4), hide_index=True, width="stretch")
    st.info(f"Population mean = {mu:.3f} °C, σ = {sd:.3f} °C (skewness {stats.skew(vals):.2f}). The means of the sample means stay close to the population mean, and the standard error falls "
            f"from {res.iloc[0, 2]:.3f} (n={sizes[0]}) to {res.iloc[-1, 2]:.3f} (n={sizes[-1]}). "
            "If the original data is already close to symmetric, the bell shape appears quickly; the narrowing with larger n is the key CLT result to observe.")

# ============================================================================= F. BOOTSTRAP
elif section == SECTIONS[5]:
    st.header("Bootstrap Analysis")
    st.markdown("**Bootstrap in simple words:** take one sample, then repeatedly *resample it with replacement* (same size) and compute the mean each time. "
                "The spread of these bootstrap means shows how uncertain the sample mean is — without assuming a formula.")
    vals = T.to_numpy()
    B = st.slider("Bootstrap resamples (B)", 500, 10000, 2000, 500)
    sizes = [s for s in (30, 50, 100) if s <= N]
    if not sizes:
        st.warning("Not enough records for bootstrap sample sizes of 30 or more.")
        st.stop()
    if len(sizes) < 3:
        st.warning(f"Only {N} records available: using sample sizes {sizes}.")

    @st.cache_data(show_spinner=False)
    def boot(v, n, B, seed):
        r = np.random.default_rng(seed + n)
        samp = r.choice(v, size=n, replace=False)
        means = r.choice(samp, size=(B, n), replace=True).mean(axis=1)
        return samp.mean(), means

    cols, rows = st.columns(len(sizes)), []
    for c, s in zip(cols, sizes):
        m0, bm = boot(vals, s, B, SEED)
        lo, hi = np.percentile(bm, [2.5, 97.5])
        fig = px.histogram(x=bm, nbins=30, title=f"n = {s}", labels={"x": "Bootstrap mean (°C)"}, color_discrete_sequence=[INDIGO])
        fig.add_vline(x=lo, line_dash="dash", line_color=RED)
        fig.add_vline(x=hi, line_dash="dash", line_color=RED)
        fig.add_vline(x=m0, line_color="#111")
        c.plotly_chart(style(fig, 330), width="stretch")
        rows.append({"Sample size": s, "Sample mean": m0, "Bootstrap mean": bm.mean(), "Bootstrap SE": bm.std(ddof=1), "95% CI lower": lo, "95% CI upper": hi, "CI width": hi - lo})
    st.caption("Black line = original sample mean; red dashed lines = 95% percentile confidence interval.")
    st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    st.info("Larger samples give a narrower bootstrap interval, i.e. a more precise estimate of the mean temperature. Results are reproducible for a fixed seed.")

# ============================================================================= G. CI
elif section == SECTIONS[6]:
    st.header("Confidence Intervals for the Mean Temperature")
    vals = T.to_numpy()
    a, b, c = st.columns(3)
    level = a.selectbox("Confidence level", [90, 95, 99], index=1) / 100
    mode = b.radio("Data used", ["Entire filtered data", "Random sample"], horizontal=True)
    n = c.slider("Sample size", 5, min(200, N), min(30, N)) if mode == "Random sample" else N
    x = np.random.default_rng(SEED).choice(vals, size=n, replace=False) if mode == "Random sample" else vals
    m, s = x.mean(), x.std(ddof=1)
    se = s / np.sqrt(len(x))
    tc = stats.t.ppf(1 - (1 - level) / 2, df=len(x) - 1)
    lo, hi = m - tc * se, m + tc * se
    cc = st.columns(5)
    for col, (k, v) in zip(cc, [("Sample size", f"{len(x)}"), ("Sample mean", f"{m:.3f}"), ("Standard error", f"{se:.4f}"),
                                ("Lower bound", f"{lo:.3f}"), ("Upper bound", f"{hi:.3f}")]):
        col.metric(k, v)
    st.success(f"{int(level*100)}% CI for mean temperature: **({lo:.3f}, {hi:.3f}) °C**  (t critical value = {tc:.3f}, margin of error = {tc*se:.3f} °C)")
    st.markdown(f"""**Meaning:** if we repeated this sampling many times and built an interval each time, about **{int(level*100)}%** of the intervals would contain the true mean temperature.
It does *not* mean there is a {int(level*100)}% chance that this one interval contains it, and it is not the range of individual daily temperatures.""")
    st.subheader("Simulation: how often does the interval capture the population mean?")
    k = st.slider("Sample size for simulation", 5, min(100, N), min(30, N), key="simn")

    @st.cache_data(show_spinner=False)
    def coverage(v, n, level, seed, reps=200):
        r = np.random.default_rng(seed)
        out = []
        for _ in range(reps):
            xs = r.choice(v, size=n, replace=False)
            e = stats.t.ppf(1 - (1 - level) / 2, n - 1) * xs.std(ddof=1) / np.sqrt(n)
            out.append((xs.mean() - e, xs.mean() + e))
        return np.array(out)

    iv = coverage(vals, k, level, SEED)
    pm = vals.mean()
    hit = (iv[:, 0] <= pm) & (pm <= iv[:, 1])
    fig = go.Figure()
    for i in range(min(60, len(iv))):
        fig.add_trace(go.Scatter(x=iv[i], y=[i, i], mode="lines", line=dict(color=TEAL if hit[i] else RED, width=2), showlegend=False))
    fig.add_vline(x=pm, line_dash="dash", line_color="#111")
    fig.update_layout(title="First 60 of 200 simulated intervals (red = missed the population mean)", xaxis_title="Temperature (°C)", yaxis_title="Simulation #")
    st.plotly_chart(style(fig, 420), width="stretch")
    st.info(f"{hit.sum()} of {len(iv)} intervals ({hit.mean()*100:.1f}%) contained the population mean {pm:.3f} °C (target {int(level*100)}%). Sampling is without replacement from a finite dataset, so coverage can differ slightly from the target.")

# ============================================================================= H. RELATIONSHIPS
elif section == SECTIONS[7]:
    st.header("Weather Relationships")
    wv = [v for v in WEATHER_VARS if v in d.columns]
    if not wv:
        st.warning("No other weather variables found.")
        st.stop()
    rows = []
    for v in wv:
        sub = d[["Temperature", v]].dropna()
        if len(sub) > 3 and sub[v].nunique() > 1:
            r, p = stats.pearsonr(sub[v], sub["Temperature"])
            rho, _ = stats.spearmanr(sub[v], sub["Temperature"])
            strength = "strong" if abs(r) >= .6 else "moderate" if abs(r) >= .3 else "weak" if abs(r) >= .1 else "negligible"
            rows.append({"Variable": LABELS[v], "n": len(sub), "Pearson r": r, "Spearman ρ": rho, "R²": r * r, "p-value": p,
                         "Direction": "positive" if r > 0 else "negative", "Strength": strength})
    if rows:
        st.dataframe(pd.DataFrame(rows).round(4), hide_index=True, width="stretch")
    for i in range(0, len(wv), 2):
        cols = st.columns(2)
        for c, v in zip(cols, wv[i:i + 2]):
            c.plotly_chart(scatter_fit(d, v, "Temperature", "Season" if "Season" in d else None, f"Temperature vs {LABELS[v]}"), width="stretch")
    st.info("**Pearson r** measures straight-line association (−1 to +1); **Spearman ρ** measures rank (monotonic) association; **R²** is the share of temperature variation "
            "linearly explained by that one variable. A small p-value (< 0.05) means the correlation is unlikely to be exactly zero by chance. **Correlation does not prove causation** — "
            "e.g. seasons can drive several variables at once.")

# ============================================================================= I. PREDICTION
elif section == SECTIONS[8]:
    st.header("Predictive Analysis – Temperature Model")
    feats_num = [v for v in WEATHER_VARS if v in full.columns] + (["Month"] if "Month" in full.columns else [])
    feats_cat = [c for c in ("Season", "Day_Type") if c in full.columns]
    feats = feats_num + feats_cat
    if len(feats) < 2 or full["Temperature"].notna().sum() < 50:
        st.warning("Not enough usable features or records (need ≥ 50 rows with temperature) to build a reliable model.")
        st.stop()
    st.markdown("**Target:** Temperature (°C). **Features:** " + ", ".join(LABELS.get(f, f) for f in feats) + ".  \n"
                "**Excluded to avoid data leakage:** *Feels-Like Temperature* and *Temperature Change* (both are computed from temperature itself). "
                "The model is trained on the **entire** dataset (sidebar filters do not apply here).")
    c1, c2 = st.columns(2)
    split = c1.radio("Train/test split", ["Random 80/20", "Chronological (last 20% = test)"] if "Date" in full.columns else ["Random 80/20"], horizontal=True)
    c2.caption("Chronological split is stricter: it tests on the last 20% of days. With only one year of data those days fall in seasons the model never saw in training, so poor scores here are expected and show the limits of a one-year dataset.")

    @st.cache_data(show_spinner="Training models...")
    def train(df, feats_num, feats_cat, chrono, seed):
        df = df.dropna(subset=["Temperature"])
        if chrono:
            df = df.sort_values("Date")
            cut = int(len(df) * .8)
            tr, te = df.iloc[:cut], df.iloc[cut:]
        else:
            tr, te = train_test_split(df, test_size=.2, random_state=seed)
        Xtr, Xte, ytr, yte = tr[feats_num + feats_cat], te[feats_num + feats_cat], tr["Temperature"], te["Temperature"]

        def pipe(model):
            pre = ColumnTransformer([
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), feats_num),
                ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore"))]), feats_cat)])
            return Pipeline([("pre", pre), ("m", model)])

        res = {}
        base = np.full(len(yte), ytr.mean())
        res["Baseline (predict train mean)"] = dict(model=None, pred=base, mae=mean_absolute_error(yte, base), rmse=mean_squared_error(yte, base) ** .5, r2=r2_score(yte, base))
        for name, mdl in {"Linear Regression": LinearRegression(),
                          "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=-1)}.items():
            p = pipe(mdl).fit(Xtr, ytr)
            pr = p.predict(Xte)
            cv = cross_val_score(pipe(mdl), Xtr, ytr, cv=KFold(5, shuffle=True, random_state=seed), scoring="r2")
            imp = permutation_importance(p, Xte, yte, n_repeats=10, random_state=seed)
            res[name] = dict(model=p, pred=pr, mae=mean_absolute_error(yte, pr), rmse=mean_squared_error(yte, pr) ** .5, r2=r2_score(yte, pr),
                             train_r2=r2_score(ytr, p.predict(Xtr)), cv_r2=cv.mean(), cv_sd=cv.std(),
                             imp=pd.Series(imp.importances_mean, index=feats_num + feats_cat).sort_values())
        return res, yte.to_numpy(), te["Date"].to_numpy() if "Date" in te else None, len(tr), len(te)

    res, ytest, dates, ntr, nte = train(full, feats_num, feats_cat, split.startswith("Chrono"), SEED)
    st.caption(f"Training rows: {ntr} · Test rows: {nte} (test data never used for fitting or tuning)")
    mt = pd.DataFrame([{"Model": k, "MAE (°C)": v["mae"], "RMSE (°C)": v["rmse"], "R² (test)": v["r2"], "R² (train)": v.get("train_r2", np.nan),
                        "5-fold CV R² (train)": v.get("cv_r2", np.nan)} for k, v in res.items()])
    st.subheader("Test-set evaluation")
    st.dataframe(mt.round(3), hide_index=True, width="stretch")
    models = {k: v for k, v in res.items() if v["model"] is not None}
    best = max(models, key=lambda k: models[k]["r2"])
    br = models[best]
    if br["r2"] <= 0.1 or br["rmse"] >= res["Baseline (predict train mean)"]["rmse"] * 0.95:
        st.warning(f"**No reliable model could be built from these features.** The best model ({best}) has test R² = {br['r2']:.3f}, barely better than (or not better than) simply predicting the average temperature. "
                   "Predictions below should not be trusted.")
    else:
        st.success(f"Best model on the test set: **{best}** — R² = {br['r2']:.3f}, MAE = {br['mae']:.2f} °C, RMSE = {br['rmse']:.2f} °C "
                   f"(baseline RMSE = {res['Baseline (predict train mean)']['rmse']:.2f} °C).")
    a, b = st.columns(2)
    ap = pd.DataFrame({"Actual": ytest, "Predicted": br["pred"]})
    fig = px.scatter(ap, x="Actual", y="Predicted", title=f"Actual vs predicted temperature – {best} (test set)", color_discrete_sequence=[TEAL])
    lim = [min(ap.min()), max(ap.max())]
    fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", name="Perfect prediction", line=dict(color=RED, dash="dash")))
    a.plotly_chart(style(fig), width="stretch")
    b.plotly_chart(style(px.histogram(x=ytest - br["pred"], nbins=20, title="Residuals (actual − predicted)", labels={"x": "Error (°C)"}, color_discrete_sequence=[INDIGO])), width="stretch")
    imp = br["imp"]
    fig = px.bar(x=imp.values, y=[LABELS.get(i, i) for i in imp.index], orientation="h", title=f"Permutation importance on test set – {best}",
                 labels={"x": "Drop in R² when feature is shuffled", "y": ""}, color_discrete_sequence=[AMBER])
    st.plotly_chart(style(fig, 340), width="stretch")

    st.subheader("Interactive prediction")
    st.caption("Enter weather conditions to get a **model estimate**. This is *not* an observed temperature.")
    with st.form("predict"):
        choice = st.selectbox("Model", list(models), index=list(models).index(best))
        cols = st.columns(3)
        inp = {}
        for i, f in enumerate(feats_num):
            col = cols[i % 3]
            if f == "Month":
                inp[f] = MONTHS.index(col.selectbox("Month", MONTHS)) + 1
            else:
                s = full[f].dropna()
                inp[f] = col.number_input(LABELS[f], float(s.min()), float(s.max()), float(s.median()), step=float((s.max() - s.min()) / 100))
        for i, f in enumerate(feats_cat):
            inp[f] = cols[(len(feats_num) + i) % 3].selectbox(f.replace("_", " "), sorted(full[f].dropna().unique(), key=lambda x: (SEASON_ORDER + [x]).index(x) if f == "Season" else 0))
        go_ = st.form_submit_button("Predict temperature", type="primary")
    if go_:
        pred = models[choice]["model"].predict(pd.DataFrame([inp])[feats_num + feats_cat])[0]
        x1, x2 = st.columns(2)
        x1.metric("Predicted temperature (model estimate)", f"{pred:.1f} °C")
        x2.metric("Typical error on test data (RMSE)", f"± {models[choice]['rmse']:.1f} °C")
        if "Season" in inp and "Month" in inp and "Month" in full.columns:
            usual = full[full["Month"] == inp["Month"]]["Season"].mode()
            if len(usual) and usual.iloc[0] != inp["Season"]:
                st.warning(f"In this dataset, {MONTHS[inp['Month']-1]} is normally labelled **{usual.iloc[0]}**, not {inp['Season']}; the combination is unusual.")

# ============================================================================= J. ABOUT
else:
    st.header("About the Project")
    st.subheader("Objectives")
    st.markdown("- Understand daily temperature patterns and how they relate to other weather variables.\n- Apply descriptive statistics, sampling techniques, the Central Limit Theorem, bootstrap and confidence intervals.\n"
                "- Build and honestly evaluate a machine-learning model for temperature prediction.\n- Present everything in an interactive web dashboard.")
    st.subheader("Dataset description")
    dd = full
    st.markdown(f"- **File:** `daily_temperature_patterns.csv` · **Rows:** {len(dd)} · **Columns:** {dd.shape[1] - 3 if 'Month' in dd else dd.shape[1]}")
    if "Date" in dd.columns:
        st.markdown(f"- **Period:** {dd['Date'].min().date()} to {dd['Date'].max().date()} (daily records)")
    miss = dd[num_cols + [c for c in ('Season', 'Day_Type') if c in dd]].isna().sum()
    st.markdown("- **Missing values:** " + (", ".join(f"{LABELS.get(k, k)}: {v}" for k, v in miss[miss > 0].items()) or "none") +
                (" (Temperature Change is empty for the first day because it needs a previous day.)" if "Temp_Change" in miss[miss > 0] else ""))
    st.markdown("- **Units:** the file does not state units; temperature is assumed to be °C and other units are taken from the column names (mm, km/h, hPa, %, hours).")
    st.subheader("Tools and libraries")
    st.markdown("Python · Streamlit · Pandas · NumPy · SciPy · Matplotlib · Seaborn · Plotly · scikit-learn")
    st.subheader("Methodology")
    st.markdown("1. Load and clean data (column-name matching, type conversion, date parsing).  2. Exploratory analysis & descriptive statistics.  3. Sampling methods (simple random, systematic, stratified, cluster).  "
                "4. Sampling distributions and CLT by repeated sampling.  5. Bootstrap percentile intervals.  6. t-based confidence intervals with a coverage simulation.  "
                "7. Correlation analysis.  8. Regression models with a held-out test set, cross-validation, and a baseline comparison.")
    st.subheader("Key findings (calculated from the full dataset)")
    Tf = full["Temperature"].dropna()
    kf = [f"{len(full)} daily records; mean temperature **{Tf.mean():.2f} °C**, standard deviation **{Tf.std():.2f} °C**, range {Tf.min():.1f}–{Tf.max():.1f} °C."]
    if "Date" in full.columns:
        kf.append(f"Hottest day: {full.loc[full['Temperature'].idxmax(), 'Date'].date()} ({Tf.max():.1f} °C); coldest day: {full.loc[full['Temperature'].idxmin(), 'Date'].date()} ({Tf.min():.1f} °C).")
    if "Season" in full.columns:
        ss = full.groupby("Season")["Temperature"].mean()
        kf.append(f"Warmest season: {ss.idxmax()} ({ss.max():.1f} °C); coolest: {ss.idxmin()} ({ss.min():.1f} °C).")
    wv = [v for v in WEATHER_VARS if v in full.columns]
    if wv:
        cr = full[wv + ["Temperature"]].corr()["Temperature"].drop("Temperature").dropna()
        kf.append(f"Strongest correlation with temperature: {LABELS[cr.abs().idxmax()]} (r = {cr[cr.abs().idxmax()]:.2f}).")
    se = Tf.std() / np.sqrt(len(Tf))
    tc = stats.t.ppf(.975, len(Tf) - 1)
    kf.append(f"95% CI for mean temperature: ({Tf.mean() - tc*se:.2f}, {Tf.mean() + tc*se:.2f}) °C.")
    st.markdown("\n".join(f"- {k}" for k in kf))
    st.caption("Model results are shown on the Predictive Analysis page, where the train/test split can be chosen.")
    st.subheader("Limitations and future improvements")
    st.markdown("- Only one year of data from an unnamed location: findings may not generalise to other years or places, and seasonal effects cannot be compared across years.\n"
                "- Small dataset (a few hundred rows): model scores can vary with the split.\n- Units are assumed from column names.\n"
                "- Same-day weather is used as model input, so the model *explains* temperature rather than forecasting future days.\n"
                "- Future work: multi-year data, lag features for true forecasting, time-series models (ARIMA/Prophet), hyper-parameter tuning, more locations.")
