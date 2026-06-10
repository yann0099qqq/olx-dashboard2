import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="OLX Paraíba — Custo-benefício e Preço por m²",
    layout="wide"
)

@st.cache_data
def carregar_dados():
    return pd.read_csv("dataset_olx_raw.csv")

df_original = carregar_dados()

st.title("🏠 Análise OLX Paraíba — Custo-benefício por Bairro e Preço por m²")
st.caption("Dashboard para recomendação de preço de anúncio e ranking de bairros com melhor custo-benefício.")

# =========================
# LIMPEZA DOS DADOS
# =========================
df = df_original.copy()

df = df.dropna(subset=["preco", "area_m2", "tipo", "cidade", "bairro"])
df = df[(df["preco"] > 10000) & (df["area_m2"] > 10)]

df["preco_m2"] = df["preco"] / df["area_m2"]

# =========================
# DETECÇÃO DE ANOMALIAS
# =========================
q1 = df["preco_m2"].quantile(0.25)
q3 = df["preco_m2"].quantile(0.75)
iqr = q3 - q1

limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

df["anomalia"] = (
    (df["preco_m2"] < limite_inferior) |
    (df["preco_m2"] > limite_superior)
)

df_limpo = df[df["anomalia"] == False].copy()

# =========================
# MÉTRICAS GERAIS
# =========================
col1, col2, col3 = st.columns(3)
col1.metric("Anúncios originais", len(df_original))
col2.metric("Registros válidos", len(df_limpo))
col3.metric("Registros removidos", len(df_original) - len(df_limpo))

st.divider()

# =========================
# RECOMENDAÇÃO DE PREÇO POR M²
# =========================
st.subheader("📈 Recomendação de preço por m²")

f1, f2, f3, f4 = st.columns(4)

tipo = f1.selectbox("Tipo do imóvel", sorted(df_limpo["tipo"].dropna().unique()))
cidade = f2.selectbox("Cidade", sorted(df_limpo["cidade"].dropna().unique()))

base_recomendacao = df_limpo[
    (df_limpo["tipo"] == tipo) &
    (df_limpo["cidade"] == cidade)
]

bairro = f3.selectbox(
    "Bairro",
    ["Todos"] + sorted(base_recomendacao["bairro"].dropna().unique())
)

area = f4.number_input(
    "Área do imóvel (m²)",
    min_value=1.0,
    value=80.0,
    step=1.0
)

if bairro != "Todos":
    base_recomendacao = base_recomendacao[base_recomendacao["bairro"] == bairro]

if len(base_recomendacao) > 0:
    mediana_m2 = base_recomendacao["preco_m2"].median()
    preco_recomendado = mediana_m2 * area
    faixa_inferior = preco_recomendado * 0.85
    faixa_superior = preco_recomendado * 1.15

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Preço recomendado", f"R$ {preco_recomendado:,.2f}")
    r2.metric("Mediana R$/m²", f"R$ {mediana_m2:,.2f}")
    r3.metric("Faixa inferior", f"R$ {faixa_inferior:,.2f}")
    r4.metric("Faixa superior", f"R$ {faixa_superior:,.2f}")

    st.caption(f"Base usada: tipo + cidade + bairro selecionado | Amostras: {len(base_recomendacao)}")
else:
    st.warning("Não há dados suficientes para esse filtro.")

st.divider()

# =========================
# RANKING DE CUSTO-BENEFÍCIO POR BAIRRO
# =========================
st.subheader("🏆 Ranking de custo-benefício por bairro")

st.write(
    "O ranking considera bairros com menor mediana de preço por m². "
    "Quanto menor o valor por m², melhor o custo-benefício para compra ou investimento."
)

c1, c2, c3 = st.columns(3)

cidade_ranking = c1.selectbox(
    "Cidade para o ranking",
    sorted(df_limpo["cidade"].dropna().unique()),
    key="cidade_ranking"
)

tipo_ranking = c2.selectbox(
    "Tipo de imóvel para o ranking",
    sorted(df_limpo["tipo"].dropna().unique()),
    key="tipo_ranking"
)

min_amostras = c3.number_input(
    "Mínimo de anúncios por bairro",
    min_value=1,
    value=5,
    step=1
)

base_ranking = df_limpo[
    (df_limpo["cidade"] == cidade_ranking) &
    (df_limpo["tipo"] == tipo_ranking)
]

ranking = (
    base_ranking
    .groupby("bairro")
    .agg(
        anuncios=("preco", "count"),
        preco_mediano=("preco", "median"),
        area_mediana=("area_m2", "median"),
        preco_m2_mediano=("preco_m2", "median")
    )
    .reset_index()
)

ranking = ranking[ranking["anuncios"] >= min_amostras]
ranking = ranking.sort_values("preco_m2_mediano", ascending=True)

ranking["posicao"] = range(1, len(ranking) + 1)

ranking_exibir = ranking[
    ["posicao", "bairro", "anuncios", "preco_mediano", "area_mediana", "preco_m2_mediano"]
].copy()

ranking_exibir = ranking_exibir.rename(columns={
    "posicao": "Posição",
    "bairro": "Bairro",
    "anuncios": "Anúncios",
    "preco_mediano": "Preço mediano",
    "area_mediana": "Área mediana",
    "preco_m2_mediano": "Mediana R$/m²"
})

if len(ranking_exibir) > 0:
    st.dataframe(
        ranking_exibir,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Top 10 bairros com melhor custo-benefício")
    top10 = ranking.head(10).set_index("bairro")["preco_m2_mediano"]
    st.bar_chart(top10)

else:
    st.warning("Não há bairros suficientes com esse filtro. Tente diminuir o mínimo de anúncios.")

st.divider()

# =========================
# ANOMALIAS
# =========================
with st.expander("🚨 Ver imóveis detectados como anomalias"):
    st.write(
        "Anomalias são imóveis com preço por m² muito abaixo ou muito acima do comportamento geral da base."
    )

    colunas_anomalias = ["tipo", "cidade", "bairro", "preco", "area_m2", "preco_m2"]

    st.dataframe(
        df[df["anomalia"] == True][colunas_anomalias],
        use_container_width=True
    )
