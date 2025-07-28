import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Used Car Dashboard", layout="wide")
st.title("🚗 Used Car Sales Dashboard")

# Veri yükleme ve temizleme
@st.cache_data
def load_data():
    df = pd.read_csv("car_prices.csv")

    df["make_model"] = df["make"].astype(str) + " " + df["model"].astype(str)
    df['saledate_clean'] = df['saledate'].str.extract(r'(\w{3} \w{3} \d{2} \d{4} \d{2}:\d{2}:\d{2})')[0]
    df['saledate_clean'] = pd.to_datetime(df['saledate_clean'], format='%a %b %d %Y %H:%M:%S', errors='coerce')
    df['period'] = df['saledate_clean'].dt.to_period("M").astype(str)
    df['state'] = df['state'].str.upper().str.strip()
    return df

df = load_data()

# --- SIDEBAR FILTERS ---
st.sidebar.header("📌 Filters")


# 1. Period filtresi
selected_periods = st.sidebar.multiselect("Select Period(s):", sorted(df['period'].dropna().unique()))

# 2. Make_Model filtresi
selected_models = st.sidebar.multiselect("Select Make & Model:", sorted(df['make_model'].dropna().unique()))

# 3. Year filtresi
selected_years = st.sidebar.multiselect("Select Years:", sorted(df['year'].dropna().unique()))

# Odometer filtresi (min-max slider)
min_odo = int(df["odometer"].min())
max_odo = int(df["odometer"].max())

selected_odo = st.sidebar.slider(
    "Select Odometer Range:",
    min_value=min_odo,
    max_value=max_odo,
    value=(min_odo, max_odo)
)

# --- VERİYİ FİLTRELE ---
filtered_df = df.copy()

if selected_periods:
    filtered_df = filtered_df[filtered_df['period'].isin(selected_periods)]
if selected_models:
    filtered_df = filtered_df[filtered_df['make_model'].isin(selected_models)]
if selected_years:
    filtered_df = filtered_df[filtered_df['year'].isin(selected_years)]
filtered_df = filtered_df[
    (filtered_df["odometer"] >= selected_odo[0]) &
    (filtered_df["odometer"] <= selected_odo[1])
]


# --- KPI METRICS ---
st.markdown("### 📊 Key Performance Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Cars Sold", len(filtered_df))

with col2:
    avg_price = round(filtered_df["sellingprice"].mean(), 0)
    st.metric("Avg Selling Price", f"${avg_price:,.0f}")

with col3:
    avg_mmr = round(filtered_df["mmr"].mean(), 0)
    st.metric("Avg MMR", f"${avg_mmr:,.0f}")

col4, col5, col6 = st.columns(3)

with col4:
    avg_odo = round(filtered_df["odometer"].mean(), 0)
    st.metric("Avg Odometer", f"{avg_odo:,.0f} km")

with col5:
    # En çok satılan model
    if not filtered_df.empty:
        top_model = (
            filtered_df["make_model"]
            .value_counts()
            .idxmax()
        )
        st.metric("Top Selling Model", top_model)
    else:
        st.metric("Top Selling Model", "N/A")

with col6:
    # En yoğun satış dönemi
    if not filtered_df.empty and filtered_df["period"].notna().any():
        top_period = (
            filtered_df["period"]
            .value_counts()
            .idxmax()
        )
        st.metric("Most Active Month", top_period)
    else:
        st.metric("Most Active Month", "N/A")

# Ekstra: Satışların yüzde kaçı MMR'ın altında?
if not filtered_df.empty:
    below_mmr_ratio = (filtered_df["sellingprice"] < filtered_df["mmr"]).mean()
    st.markdown(f"**📉 % Sold Below MMR:** {below_mmr_ratio * 100:.1f}%")


# --- GRAFİKLER ---

# 1) Harita: Eyaletlere göre ortalama fiyat
st.subheader("📍 Average Selling Price by State")
state_price = (
    filtered_df.groupby("state")["sellingprice"]
    .mean()
    .reset_index()
)
fig_map = px.choropleth(
    state_price,
    locations="state",
    locationmode="USA-states",
    color="sellingprice",
    scope="usa",
    color_continuous_scale="Blues",
    labels={"sellingprice": "Avg Price"},
)
st.plotly_chart(fig_map, use_container_width=True)

# 2) MMR vs Selling Price - y=x doğrulu scatterplot
st.subheader("💸 Market Value vs Selling Price ")
fig_scatter = px.scatter(
    filtered_df,
    x="mmr",
    y="sellingprice",
    hover_data=["make_model", "state", "year", "odometer"],
    opacity=0.4,
    title="Market Value vs Selling Price"
)

# Referans çizgisi: y = x
min_val = min(filtered_df["mmr"].min(), filtered_df["sellingprice"].min())
max_val = max(filtered_df["mmr"].max(), filtered_df["sellingprice"].max())
fig_scatter.add_shape(
    type='line',
    x0=min_val, y0=min_val,
    x1=max_val, y1=max_val,
    line=dict(color="red", dash="dash")
)

fig_scatter.update_layout(showlegend=False)
st.plotly_chart(fig_scatter, use_container_width=True)

# 3) Tablo - filtrelenmiş veri
st.subheader("📋 Raw Data (filtered)")
st.dataframe(filtered_df.head(100))

# 4) Line Chart: Period vs Sales Count
st.subheader("📈 Sales Trend Over Time (by Period)")
period_sales = (
    filtered_df["period"]
    .value_counts()
    .sort_index()
    .reset_index()
)
period_sales.columns = ["period", "sales_count"]

fig_line = px.line(
    period_sales,
    x="period",
    y="sales_count",
    markers=True,
    title="Monthly Sales Count",
    labels={"sales_count": "Number of Sales", "period": "Period"},
)

fig_line.update_layout(xaxis_tickangle=45)
st.plotly_chart(fig_line, use_container_width=True)


