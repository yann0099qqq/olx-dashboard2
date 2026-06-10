import streamlit as st
import pandas as pd

st.set_page_config(page_title="Análise OLX Paraíba", layout="wide")

@st.cache_data
def carregar_dados():
    return pd.read_csv("dataset_olx_raw.csv")

df = carregar_dados()

st.title("🏠 Análise OLX Paraíba — Preço por m² e Anomalias")

df = df.dropna(subset=["preco", "area_m2", "tipo", "cidade"])
df = df[(df["preco"] > 10000) & (df["area_m2"] > 10)]

df["preco_m2"] = df["preco"] / df["area_m2"]

q1 = df["preco_m2"].quantile(0.25)
q3 = df["preco_m2"].quantile(0.75)
iqr = q3 - q1

limite_inf = q1 - 1.5 * iqr
limite_sup = q3 + 1.5 * iqr

df["anomalia"] = (
    (df["preco_m2"] < limite_inf) |
    (df["preco_m2"] > limite_sup)
)

df_limpo = df[df["anomalia"] == False]

c1, c2, c3 = st.columns(3)
c1.metric("Anúncios Originais", len(df))
c2.metric("Registros Válidos", len(df_limpo))
c3.metric("Anomalias Detectadas", int(df["anomalia"].sum()))

st.divider()

st.subheader("📈 Recomendação de Preço por m²")

col1, col2, col3, col4 = st.columns(4)

tipo = col1.selectbox("Tipo do Imóvel", sorted(df_limpo["tipo"].unique()))
cidade = col2.selectbox("Cidade", sorted(df_limpo["cidade"].unique()))

base = df_limpo[(df_limpo["tipo"] == tipo) & (df_limpo["cidade"] == cidade)]

bairro = col3.selectbox("Bairro", ["Todos"] + sorted(base["bairro"].dropna().unique()))
area = col4.number_input("Área do imóvel (m²)", value=80.0)

if bairro != "Todos":
    base = base[base["bairro"] == bairro]

if len(base) > 0:
    mediana_m2 = base["preco_m2"].median()
    preco_recomendado = mediana_m2 * area

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Preço recomendado", f"R$ {preco_recomendado:,.2f}")
    r2.metric("Mediana R$/m²", f"R$ {mediana_m2:,.2f}")
    r3.metric("Faixa inferior", f"R$ {preco_recomendado * 0.85:,.2f}")
    r4.metric("Faixa superior", f"R$ {preco_recomendado * 1.15:,.2f}")

st.divider()

st.subheader("🚨 Detecção de Anomalias")

st.dataframe(
    df[df["anomalia"] == True][["tipo", "cidade", "bairro", "preco", "area_m2", "preco_m2"]]
)
