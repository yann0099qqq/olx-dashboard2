import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="OLX Paraíba — Machine Learning",
    layout="wide"
)

@st.cache_data
def carregar_dados():
    return pd.read_csv("dataset_olx_raw.csv")

def formatar_moeda(valor):
    if pd.isna(valor):
        return "R$ 0,00"
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

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

@st.cache_data
def preparar_base(df_original):
    df = df_original.copy()

    for coluna in ["titulo", "descricao", "tipo", "bairro", "cidade", "estado"]:
        if coluna in df.columns:
            df[coluna] = df[coluna].fillna("").astype(str).str.strip()

    for coluna in ["preco", "area_m2", "quartos", "banheiros", "garagens"]:
        if coluna in df.columns:
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df["tipo_padrao"] = df.apply(
        lambda linha: classificar_imovel(
            linha.get("titulo", ""),
            linha.get("descricao", ""),
            linha.get("tipo", "")
        ),
        axis=1
    )

    df = df.dropna(subset=["preco", "area_m2", "cidade", "bairro"])
    df = df[df["tipo_padrao"] != "Outros"]

    df = df[(df["preco"] >= 10000) & (df["preco"] <= 20000000)]
    df = df[(df["area_m2"] >= 15) & (df["area_m2"] <= 5000)]

    for coluna in ["quartos", "banheiros", "garagens"]:
        if coluna in df.columns:
            df[coluna] = df[coluna].fillna(0)
            df = df[(df[coluna] >= 0) & (df[coluna] <= 15)]

    df["cidade"] = df["cidade"].str.title()
    df["bairro"] = df["bairro"].str.title()

    df = df[~df["cidade"].str.match(r"^\d{1,2}:\d{2}$", na=False)]
    df = df[~df["bairro"].str.match(r"^\d{1,2}:\d{2}$", na=False)]

    df["preco_m2"] = df["preco"] / df["area_m2"]
    df = df[(df["preco_m2"] >= 300) & (df["preco_m2"] <= 50000)]

    return df

@st.cache_data
def treinar_modelo_ml(df):
    dados = df.copy()

    features = ["area_m2", "quartos", "banheiros", "garagens", "tipo_padrao", "cidade"]
    X = dados[features]
    y = dados["preco"]

    colunas_numericas = ["area_m2", "quartos", "banheiros", "garagens"]
    colunas_categoricas = ["tipo_padrao", "cidade"]

    preprocessador = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), colunas_categoricas),
            ("num", "passthrough", colunas_numericas)
        ]
    )

    modelo = Pipeline(
        steps=[
            ("preprocessador", preprocessador),
            ("regressor", GradientBoostingRegressor(random_state=42))
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    modelo.fit(X_train, y_train)

    pred_test = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, pred_test)
    r2 = r2_score(y_test, pred_test)

    dados["preco_previsto_ml"] = modelo.predict(X)
    dados["desvio_ml"] = dados["preco"] - dados["preco_previsto_ml"]
    dados["desvio_percentual_ml"] = (dados["desvio_ml"] / dados["preco_previsto_ml"]) * 100

    def classificar_ml(desvio):
        if desvio <= -30:
            return "Oportunidade ML"
        if desvio >= 50:
            return "Suspeito ML"
        return "Dentro do mercado"

    dados["classificacao_ml"] = dados["desvio_percentual_ml"].apply(classificar_ml)

    return modelo, dados, mae, r2

@st.cache_data
def aplicar_isolation_forest(df):
    dados = df.copy()

    features = ["preco", "preco_m2", "area_m2", "quartos", "banheiros", "garagens"]
    X = dados[features].fillna(0)

    iso = IsolationForest(
        n_estimators=150,
        contamination=0.08,
        random_state=42
    )

    dados["resultado_isolation"] = iso.fit_predict(X)
    dados["anomalia_isolation"] = dados["resultado_isolation"].map({
        1: "Normal",
        -1: "Anômalo"
    })

    return dados

# =====================
# BASE + MODELOS
# =====================
df_original = carregar_dados()
df_limpo = preparar_base(df_original)
modelo_ml, df_ml, mae, r2 = treinar_modelo_ml(df_limpo)
df_ml = aplicar_isolation_forest(df_ml)

oportunidades_ml = df_ml[df_ml["classificacao_ml"] == "Oportunidade ML"]
suspeitos_ml = df_ml[df_ml["classificacao_ml"] == "Suspeito ML"]
anomalias_iso = df_ml[df_ml["anomalia_isolation"] == "Anômalo"]

# =====================
# DASHBOARD
# =====================
st.title("🏠 OLX Paraíba — Dashboard com Machine Learning")
st.caption(
    "Análise preditiva com Gradient Boosting Regressor e detecção de anomalias com Isolation Forest."
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Anúncios originais", len(df_original))
m2.metric("Base limpa", len(df_limpo))
m3.metric("MAE do modelo", formatar_moeda(mae))
m4.metric("R² do modelo", f"{r2:.2f}")

st.divider()

aba1, aba2, aba3, aba4 = st.tabs([
    "🤖 Previsão de preço ML",
    "🚨 Anomalias ML",
    "📊 Gráficos ML",
    "📋 Base com previsões"
])

tipos = sorted(df_ml["tipo_padrao"].dropna().unique())
cidades = sorted(df_ml["cidade"].dropna().unique())

# =====================
# ABA 1
# =====================
with aba1:
    st.subheader("🤖 Recomendação de preço usando Machine Learning")
    st.write(
        "O modelo usa área, quartos, banheiros, garagens, tipo do imóvel e cidade "
        "para prever o preço estimado do imóvel."
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)

    tipo = c1.selectbox("Tipo", tipos)
    cidade = c2.selectbox("Cidade", cidades)
    area = c3.number_input("Área (m²)", min_value=15.0, value=80.0, step=1.0)
    quartos = c4.number_input("Quartos", min_value=0, max_value=15, value=2)
    banheiros = c5.number_input("Banheiros", min_value=0, max_value=15, value=1)
    garagens = c6.number_input("Garagens", min_value=0, max_value=15, value=1)

    entrada = pd.DataFrame([{
        "area_m2": area,
        "quartos": quartos,
        "banheiros": banheiros,
        "garagens": garagens,
        "tipo_padrao": tipo,
        "cidade": cidade
    }])

    preco_previsto = modelo_ml.predict(entrada)[0]

    base_comparacao = df_ml[
        (df_ml["tipo_padrao"] == tipo) &
        (df_ml["cidade"] == cidade)
    ]

    mediana_m2 = base_comparacao["preco_m2"].median()
    preco_mediana = mediana_m2 * area

    preco_final = (preco_previsto * 0.6) + (preco_mediana * 0.4)

    faixa_inferior = preco_final * 0.90
    faixa_superior = preco_final * 1.10

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Preço previsto ML", formatar_moeda(preco_previsto))
    r2.metric("Referência por mediana", formatar_moeda(preco_mediana))
    r3.metric("Preço recomendado final", formatar_moeda(preco_final))
    r4.metric("Faixa ±10%", f"{formatar_moeda(faixa_inferior)} até {formatar_moeda(faixa_superior)}")

    st.info(
        "Preço final = 60% previsão do modelo Gradient Boosting + 40% referência da mediana local."
    )

# =====================
# ABA 2
# =====================
with aba2:
    st.subheader("🚨 Identificação de anomalias com Machine Learning")

    st.write(
        "Aqui o sistema compara o preço real com o preço previsto pelo modelo. "
        "Valores muito abaixo são oportunidades; valores muito acima são suspeitos."
    )

    c1, c2, c3 = st.columns(3)
    cidade_f = c1.selectbox("Cidade", ["Todas"] + cidades, key="cidade_f")
    tipo_f = c2.selectbox("Tipo", ["Todos"] + tipos, key="tipo_f")
    classe_f = c3.selectbox(
        "Classificação ML",
        ["Todas", "Oportunidade ML", "Dentro do mercado", "Suspeito ML"]
    )

    tabela = df_ml.copy()

    if cidade_f != "Todas":
        tabela = tabela[tabela["cidade"] == cidade_f]

    if tipo_f != "Todos":
        tabela = tabela[tabela["tipo_padrao"] == tipo_f]

    if classe_f != "Todas":
        tabela = tabela[tabela["classificacao_ml"] == classe_f]

    colunas = [
        "titulo", "tipo_padrao", "cidade", "bairro", "preco",
        "preco_previsto_ml", "desvio_percentual_ml",
        "classificacao_ml", "anomalia_isolation"
    ]

    tabela_view = tabela[colunas].copy()
    tabela_view["preco"] = tabela_view["preco"].apply(formatar_moeda)
    tabela_view["preco_previsto_ml"] = tabela_view["preco_previsto_ml"].apply(formatar_moeda)
    tabela_view["desvio_percentual_ml"] = tabela_view["desvio_percentual_ml"].round(2)

    st.dataframe(tabela_view, use_container_width=True, hide_index=True)

# =====================
# ABA 3
# =====================
with aba3:
    st.subheader("📊 Gráficos de Machine Learning")

    g1, g2 = st.columns(2)

    contagem = df_ml["classificacao_ml"].value_counts().reset_index()
    contagem.columns = ["classificacao", "quantidade"]

    fig1 = px.pie(
        contagem,
        names="classificacao",
        values="quantidade",
        hole=0.45,
        title="Classificação por preço previsto pelo ML"
    )
    fig1.update_layout(template="plotly_dark", height=430)
    g1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(
        df_ml[
            (df_ml["desvio_percentual_ml"] >= -100) &
            (df_ml["desvio_percentual_ml"] <= 200)
        ],
        x="desvio_percentual_ml",
        color="classificacao_ml",
        nbins=50,
        title="Desvio entre preço real e preço previsto pelo ML",
        labels={"desvio_percentual_ml": "Desvio percentual (%)"}
    )
    fig2.add_vline(x=-30, line_dash="dash", line_color="green")
    fig2.add_vline(x=50, line_dash="dash", line_color="red")
    fig2.update_layout(template="plotly_dark", height=430)
    g2.plotly_chart(fig2, use_container_width=True)

    st.subheader("Preço real x Preço previsto")

    amostra = df_ml.copy()
    if len(amostra) > 1500:
        amostra = amostra.sample(1500, random_state=42)

    fig3 = px.scatter(
        amostra,
        x="preco_previsto_ml",
        y="preco",
        color="classificacao_ml",
        hover_data=["titulo", "cidade", "bairro", "tipo_padrao"],
        title="Comparação entre preço previsto pelo ML e preço real",
        labels={
            "preco_previsto_ml": "Preço previsto ML",
            "preco": "Preço real",
            "classificacao_ml": "Classificação"
        }
    )
    fig3.update_layout(template="plotly_dark", height=550)
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Anomalias pelo Isolation Forest")

    fig4 = px.scatter(
        amostra,
        x="area_m2",
        y="preco_m2",
        color="anomalia_isolation",
        hover_data=["titulo", "cidade", "bairro", "tipo_padrao", "preco"],
        title="Detecção de anomalias por preço/m² e área",
        labels={
            "area_m2": "Área (m²)",
            "preco_m2": "Preço por m²",
            "anomalia_isolation": "Resultado"
        }
    )
    fig4.update_layout(template="plotly_dark", height=550)
    st.plotly_chart(fig4, use_container_width=True)

# =====================
# ABA 4
# =====================
with aba4:
    st.subheader("📋 Base com previsões de Machine Learning")

    colunas = [
        "titulo", "tipo_padrao", "cidade", "bairro", "preco",
        "preco_previsto_ml", "desvio_percentual_ml",
        "classificacao_ml", "anomalia_isolation"
    ]

    tabela = df_ml[colunas].copy()
    tabela["preco"] = tabela["preco"].apply(formatar_moeda)
    tabela["preco_previsto_ml"] = tabela["preco_previsto_ml"].apply(formatar_moeda)
    tabela["desvio_percentual_ml"] = tabela["desvio_percentual_ml"].round(2)

    st.dataframe(tabela, use_container_width=True, hide_index=True)
