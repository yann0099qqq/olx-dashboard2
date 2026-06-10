import streamlit as st
import pandas as pd
import numpy as np
import re

st.set_page_config(
    page_title="OLX Paraíba — Análise Imobiliária",
    layout="wide"
)

@st.cache_data
def carregar_dados():
    return pd.read_csv("dataset_olx_raw.csv")

def formatar_moeda(valor):
    if pd.isna(valor):
        return "R$ 0,00"
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def limpar_numero(coluna):
    return pd.to_numeric(coluna, errors="coerce")

def classificar_imovel(titulo, descricao, tipo_original):
    texto = f"{tipo_original} {titulo} {descricao}".lower()

    regras = [
        ("Cobertura", ["cobertura"]),
        ("Kitnet", ["kitnet", "kitinete", "quitinete", "studio", "stúdio"]),
        ("Loft", ["loft"]),
        ("Galpão", ["galpão", "galpao", "galpões", "galpoes"]),
        ("Chácara", ["chácara", "chacara", "sítio", "sitio", "granja"]),
        ("Loja", ["loja", "ponto comercial", "sala comercial", "comercial"]),
        ("Apartamento", ["apartamento", "apto", "flat"]),
        ("Casa", ["casa", "sobrado", "duplex", "triplex"])
    ]

    for categoria, palavras in regras:
        for palavra in palavras:
            if palavra in texto:
                return categoria

    return "Outros"

# =========================
# CARREGAMENTO
# =========================
df_original = carregar_dados()
df = df_original.copy()

# =========================
# PADRONIZAÇÃO INICIAL
# =========================
colunas_texto = ["titulo", "descricao", "tipo", "bairro", "cidade", "estado"]
for coluna in colunas_texto:
    if coluna in df.columns:
        df[coluna] = df[coluna].fillna("").astype(str).str.strip()

colunas_numericas = ["preco", "area_m2", "quartos", "banheiros", "garagens"]
for coluna in colunas_numericas:
    if coluna in df.columns:
        df[coluna] = limpar_numero(df[coluna])

# =========================
# CLASSIFICAÇÃO PADRONIZADA DO IMÓVEL
# =========================
df["tipo_padrao"] = df.apply(
    lambda linha: classificar_imovel(
        linha.get("titulo", ""),
        linha.get("descricao", ""),
        linha.get("tipo", "")
    ),
    axis=1
)

# =========================
# LIMPEZA E FILTROS DE ANOMALIAS BRUTAS
# =========================
antes_limpeza = len(df)

df = df.dropna(subset=["preco", "area_m2", "cidade", "bairro"])
df = df[df["tipo_padrao"] != "Outros"]

df = df[
    (df["preco"] >= 10000) &
    (df["preco"] <= 20000000)
]

df = df[
    (df["area_m2"] >= 15) &
    (df["area_m2"] <= 5000)
]

for coluna in ["quartos", "banheiros", "garagens"]:
    if coluna in df.columns:
        df[coluna] = df[coluna].fillna(0)
        df = df[
            (df[coluna] >= 0) &
            (df[coluna] <= 15)
        ]

df["cidade"] = df["cidade"].str.title()
df["bairro"] = df["bairro"].str.title()

# Remove possíveis valores estranhos em cidade/bairro
df = df[
    ~df["cidade"].str.match(r"^\d{1,2}:\d{2}$", na=False)
]
df = df[
    ~df["bairro"].str.match(r"^\d{1,2}:\d{2}$", na=False)
]

df["preco_m2"] = df["preco"] / df["area_m2"]

# Filtro adicional de preço por m² absurdo
df = df[
    (df["preco_m2"] >= 300) &
    (df["preco_m2"] <= 50000)
]

apos_limpeza_bruta = len(df)

# =========================
# DETECÇÃO ESTATÍSTICA DE ANOMALIAS POR IQR
# =========================
q1 = df["preco_m2"].quantile(0.25)
q3 = df["preco_m2"].quantile(0.75)
iqr = q3 - q1

limite_inferior = q1 - 1.5 * iqr
limite_superior = q3 + 1.5 * iqr

df["anomalia_iqr"] = (
    (df["preco_m2"] < limite_inferior) |
    (df["preco_m2"] > limite_superior)
)

df_limpo = df[df["anomalia_iqr"] == False].copy()
df_anomalias = df[df["anomalia_iqr"] == True].copy()

# =========================
# INTERFACE
# =========================
st.title("🏠 Análise OLX Paraíba — Preço por m² e Custo-benefício")
st.caption(
    "Dashboard com limpeza de anomalias, padronização de tipos de imóvel, "
    "recomendação de preço de anúncio e ranking de custo-benefício por bairro."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Anúncios originais", len(df_original))
m2.metric("Após limpeza inicial", apos_limpeza_bruta)
m3.metric("Registros válidos", len(df_limpo))
m4.metric("Anomalias removidas", len(df_original) - len(df_limpo))

st.divider()

# =========================
# ABA 1: RECOMENDAÇÃO
# =========================
st.subheader("📈 Recomendação de preço por m²")

f1, f2, f3, f4 = st.columns(4)

tipos_disponiveis = sorted(df_limpo["tipo_padrao"].dropna().unique())
cidades_disponiveis = sorted(df_limpo["cidade"].dropna().unique())

tipo_escolhido = f1.selectbox("Tipo do imóvel", tipos_disponiveis)
cidade_escolhida = f2.selectbox("Cidade", cidades_disponiveis)

base_recomendacao = df_limpo[
    (df_limpo["tipo_padrao"] == tipo_escolhido) &
    (df_limpo["cidade"] == cidade_escolhida)
]

bairros_disponiveis = ["Todos"] + sorted(base_recomendacao["bairro"].dropna().unique())

bairro_escolhido = f3.selectbox("Bairro", bairros_disponiveis)

area_informada = f4.number_input(
    "Área do imóvel (m²)",
    min_value=15.0,
    value=80.0,
    step=1.0
)

if bairro_escolhido != "Todos":
    base_recomendacao = base_recomendacao[
        base_recomendacao["bairro"] == bairro_escolhido
    ]

if len(base_recomendacao) > 0:
    mediana_m2 = base_recomendacao["preco_m2"].median()
    preco_recomendado = mediana_m2 * area_informada
    faixa_inferior = preco_recomendado * 0.85
    faixa_superior = preco_recomendado * 1.15

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Preço recomendado", formatar_moeda(preco_recomendado))
    r2.metric("Mediana R$/m²", formatar_moeda(mediana_m2))
    r3.metric("Faixa inferior", formatar_moeda(faixa_inferior))
    r4.metric("Faixa superior", formatar_moeda(faixa_superior))

    st.caption(
        f"Base usada: {tipo_escolhido} em {cidade_escolhida}"
        + (f", bairro {bairro_escolhido}" if bairro_escolhido != "Todos" else "")
        + f" | Amostras: {len(base_recomendacao)}"
    )
else:
    st.warning("Não há dados suficientes para esse filtro.")

st.divider()

# =========================
# ABA 2: RANKING
# =========================
st.subheader("🏆 Ranking de custo-benefício por bairro")

st.write(
    "O ranking usa a mediana do preço por m². "
    "Quanto menor a mediana de R$/m², melhor o custo-benefício do bairro."
)

c1, c2, c3 = st.columns(3)

cidade_ranking = c1.selectbox(
    "Cidade para o ranking",
    cidades_disponiveis,
    key="cidade_ranking"
)

tipo_ranking = c2.selectbox(
    "Tipo de imóvel para o ranking",
    tipos_disponiveis,
    key="tipo_ranking"
)

minimo_anuncios = c3.number_input(
    "Mínimo de anúncios por bairro",
    min_value=1,
    value=3,
    step=1
)

base_ranking = df_limpo[
    (df_limpo["cidade"] == cidade_ranking) &
    (df_limpo["tipo_padrao"] == tipo_ranking)
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

ranking = ranking[ranking["anuncios"] >= minimo_anuncios]
ranking = ranking.sort_values("preco_m2_mediano", ascending=True)
ranking["posicao"] = range(1, len(ranking) + 1)

if len(ranking) > 0:
    ranking_exibir = ranking.copy()
    ranking_exibir["Preço mediano"] = ranking_exibir["preco_mediano"].apply(formatar_moeda)
    ranking_exibir["Área mediana"] = ranking_exibir["area_mediana"].round(2)
    ranking_exibir["Mediana R$/m²"] = ranking_exibir["preco_m2_mediano"].apply(formatar_moeda)

    ranking_exibir = ranking_exibir.rename(columns={
        "posicao": "Posição",
        "bairro": "Bairro",
        "anuncios": "Anúncios"
    })

    st.dataframe(
        ranking_exibir[
            ["Posição", "Bairro", "Anúncios", "Preço mediano", "Área mediana", "Mediana R$/m²"]
        ],
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
# ANOMALIAS E BASE
# =========================
with st.expander("🚨 Ver imóveis detectados como anomalias estatísticas"):
    st.write(
        "Esses imóveis passaram pela limpeza inicial, mas foram marcados pelo método IQR "
        "por terem preço por m² muito fora do padrão geral."
    )

    colunas = ["tipo_padrao", "cidade", "bairro", "preco", "area_m2", "quartos", "banheiros", "garagens", "preco_m2"]
    colunas = [c for c in colunas if c in df_anomalias.columns]

    st.dataframe(
        df_anomalias[colunas],
        use_container_width=True,
        hide_index=True
    )

with st.expander("📋 Ver amostra da base limpa"):
    colunas = ["titulo", "tipo_padrao", "cidade", "bairro", "preco", "area_m2", "quartos", "banheiros", "garagens", "preco_m2"]
    colunas = [c for c in colunas if c in df_limpo.columns]

    st.dataframe(
        df_limpo[colunas].head(100),
        use_container_width=True,
        hide_index=True
    )
