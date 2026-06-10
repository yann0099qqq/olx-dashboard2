import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="OLX Paraíba — Dashboard Imobiliário",
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

    df = df.dropna(subset=["preco", "cidade", "bairro"])
    df = df[df["tipo_padrao"] != "Outros"]

    df = df[(df["preco"] >= 10000) & (df["preco"] <= 20000000)]

    if "area_m2" in df.columns:
        df = df.dropna(subset=["area_m2"])
        df = df[(df["area_m2"] >= 15) & (df["area_m2"] <= 5000)]
        df["preco_m2"] = df["preco"] / df["area_m2"]
        df = df[(df["preco_m2"] >= 300) & (df["preco_m2"] <= 50000)]
    else:
        df["preco_m2"] = np.nan

    for coluna in ["quartos", "banheiros", "garagens"]:
        if coluna in df.columns:
            df[coluna] = df[coluna].fillna(0)
            df = df[(df[coluna] >= 0) & (df[coluna] <= 15)]

    df["cidade"] = df["cidade"].str.title()
    df["bairro"] = df["bairro"].str.title()

    df = df[~df["cidade"].str.match(r"^\d{1,2}:\d{2}$", na=False)]
    df = df[~df["bairro"].str.match(r"^\d{1,2}:\d{2}$", na=False)]

    return df

def detectar_por_mediana(df):
    dados = df.copy()

    grupo = dados.groupby(["tipo_padrao", "cidade"])["preco"]
    dados["mediana_grupo"] = grupo.transform("median")
    dados["qtd_grupo"] = grupo.transform("count")

    dados["desvio_percentual"] = (
        (dados["preco"] - dados["mediana_grupo"]) /
        dados["mediana_grupo"]
    ) * 100

    def classificar(linha):
        if linha["qtd_grupo"] < 5:
            return "Dados insuficientes"

        desvio = linha["desvio_percentual"]

        if desvio <= -40:
            return "Excelente oportunidade"
        elif desvio <= -20:
            return "Oportunidade"
        elif desvio >= 80:
            return "Suspeito grave"
        elif desvio >= 40:
            return "Suspeito"
        else:
            return "Dentro do mercado"

    def alerta(classe):
        if classe == "Excelente oportunidade":
            return "Preço muito abaixo da mediana local. Pode ser oportunidade, mas verifique documentação, localização e condições do imóvel."
        if classe == "Oportunidade":
            return "Preço abaixo da mediana do grupo. Vale comparar com imóveis semelhantes."
        if classe == "Suspeito":
            return "Preço muito acima da mediana local. Pode indicar especulação ou erro de cadastro."
        if classe == "Suspeito grave":
            return "Preço extremamente acima da mediana local. Recomenda-se cautela antes de negociar."
        if classe == "Dados insuficientes":
            return "Há poucos anúncios semelhantes para uma comparação segura."
        return "Preço compatível com o comportamento do mercado local."

    dados["classificacao"] = dados.apply(classificar, axis=1)
    dados["alerta_comprador"] = dados["classificacao"].apply(alerta)

    return dados

def grafico_barra(df, x, y, title, labels=None):
    fig = px.bar(
        df,
        x=x,
        y=y,
        title=title,
        text=y,
        labels=labels,
        color=y,
        color_continuous_scale="Viridis"
    )
    fig.update_traces(texttemplate="%{text:.2s}", textposition="outside")
    fig.update_layout(
        height=430,
        template="plotly_dark",
        showlegend=False,
        margin=dict(l=20, r=20, t=70, b=20),
        title=dict(font=dict(size=22)),
        coloraxis_showscale=False
    )
    return fig

# =========================
# BASE
# =========================
df_original = carregar_dados()
df_tratada = preparar_base(df_original)
df = detectar_por_mediana(df_tratada)

# =========================
# LAYOUT
# =========================
st.title("🏠 OLX Paraíba — Análise de Preço, Oportunidades e Custo-benefício")
st.caption(
    "Dashboard com limpeza da base, recomendação de preço por m², ranking por bairro "
    "e identificação de oportunidades/suspeitos com base na mediana por tipo e cidade."
)

oportunidades = df[df["classificacao"].isin(["Excelente oportunidade", "Oportunidade"])]
suspeitos = df[df["classificacao"].isin(["Suspeito", "Suspeito grave"])]

m1, m2, m3, m4 = st.columns(4)
m1.metric("Anúncios originais", len(df_original))
m2.metric("Após limpeza", len(df))
m3.metric("Oportunidades", len(oportunidades))
m4.metric("Suspeitos", len(suspeitos))

st.divider()

aba1, aba2, aba3, aba4 = st.tabs([
    "📈 Recomendação",
    "🏆 Ranking por bairro",
    "🚨 Oportunidades e suspeitos",
    "📊 Gráficos"
])

tipos = sorted(df["tipo_padrao"].dropna().unique())
cidades = sorted(df["cidade"].dropna().unique())

# =========================
# ABA 1
# =========================
with aba1:
    st.subheader("📈 Recomendação de preço por m²")

    c1, c2, c3, c4 = st.columns(4)

    tipo = c1.selectbox("Tipo do imóvel", tipos)
    cidade = c2.selectbox("Cidade", cidades)

    base = df[
        (df["tipo_padrao"] == tipo) &
        (df["cidade"] == cidade)
    ]

    bairros = ["Todos"] + sorted(base["bairro"].dropna().unique())
    bairro = c3.selectbox("Bairro", bairros)
    area = c4.number_input("Área do imóvel (m²)", min_value=15.0, value=80.0, step=1.0)

    if bairro != "Todos":
        base = base[base["bairro"] == bairro]

    if len(base) > 0:
        mediana_m2 = base["preco_m2"].median()
        preco_recomendado = mediana_m2 * area
        faixa_inferior = preco_recomendado * 0.85
        faixa_superior = preco_recomendado * 1.15

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Preço recomendado", formatar_moeda(preco_recomendado))
        r2.metric("Mediana R$/m²", formatar_moeda(mediana_m2))
        r3.metric("Faixa inferior", formatar_moeda(faixa_inferior))
        r4.metric("Faixa superior", formatar_moeda(faixa_superior))

        st.info(f"Base usada: {tipo} em {cidade} | Amostras: {len(base)}")
    else:
        st.warning("Não há dados suficientes para esse filtro.")

# =========================
# ABA 2
# =========================
with aba2:
    st.subheader("🏆 Ranking de custo-benefício por bairro")
    st.write("Quanto menor a mediana de preço por m², melhor o custo-benefício.")

    c1, c2, c3 = st.columns(3)
    cidade_r = c1.selectbox("Cidade", cidades, key="cidade_r")
    tipo_r = c2.selectbox("Tipo", tipos, key="tipo_r")
    min_anuncios = c3.number_input("Mínimo de anúncios por bairro", min_value=1, value=3, step=1)

    base_r = df[
        (df["cidade"] == cidade_r) &
        (df["tipo_padrao"] == tipo_r)
    ]

    ranking = (
        base_r.groupby("bairro")
        .agg(
            anuncios=("preco", "count"),
            preco_mediano=("preco", "median"),
            area_mediana=("area_m2", "median"),
            preco_m2_mediano=("preco_m2", "median")
        )
        .reset_index()
    )

    ranking = ranking[ranking["anuncios"] >= min_anuncios]
    ranking = ranking.sort_values("preco_m2_mediano", ascending=True)
    ranking["posicao"] = range(1, len(ranking) + 1)

    if len(ranking) > 0:
        ranking_view = ranking.copy()
        ranking_view["Preço mediano"] = ranking_view["preco_mediano"].apply(formatar_moeda)
        ranking_view["Mediana R$/m²"] = ranking_view["preco_m2_mediano"].apply(formatar_moeda)
        ranking_view["Área mediana"] = ranking_view["area_mediana"].round(2)

        st.dataframe(
            ranking_view.rename(columns={
                "posicao": "Posição",
                "bairro": "Bairro",
                "anuncios": "Anúncios"
            })[["Posição", "Bairro", "Anúncios", "Preço mediano", "Área mediana", "Mediana R$/m²"]],
            use_container_width=True,
            hide_index=True
        )

        top10 = ranking.head(10).copy()
        fig = px.bar(
            top10,
            x="preco_m2_mediano",
            y="bairro",
            orientation="h",
            title="Top 10 bairros com melhor custo-benefício",
            labels={"preco_m2_mediano": "Mediana R$/m²", "bairro": "Bairro"},
            color="preco_m2_mediano",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(
            template="plotly_dark",
            height=500,
            yaxis=dict(autorange="reversed"),
            margin=dict(l=20, r=20, t=70, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Não há bairros suficientes com esse filtro.")

# =========================
# ABA 3
# =========================
with aba3:
    st.subheader("🚨 Identificação de anomalias de preço: oportunidades e suspeitos")
    st.write(
        "Esta análise usa apenas **preço, tipo e cidade**. "
        "O sistema compara cada imóvel com a mediana do seu grupo. "
        "Abaixo da mediana: oportunidade. Acima da mediana: possível suspeita."
    )

    c1, c2, c3 = st.columns(3)
    cidade_a = c1.selectbox("Cidade", ["Todas"] + cidades, key="cidade_a")
    tipo_a = c2.selectbox("Tipo", ["Todos"] + tipos, key="tipo_a")
    classe_a = c3.selectbox(
        "Classificação",
        ["Todas", "Excelente oportunidade", "Oportunidade", "Dentro do mercado", "Suspeito", "Suspeito grave"]
    )

    alertas = df.copy()

    if cidade_a != "Todas":
        alertas = alertas[alertas["cidade"] == cidade_a]

    if tipo_a != "Todos":
        alertas = alertas[alertas["tipo_padrao"] == tipo_a]

    if classe_a != "Todas":
        alertas = alertas[alertas["classificacao"] == classe_a]

    colunas_alerta = [
        "titulo", "tipo_padrao", "cidade", "bairro", "preco",
        "mediana_grupo", "desvio_percentual", "classificacao", "alerta_comprador"
    ]
    colunas_alerta = [c for c in colunas_alerta if c in alertas.columns]

    alertas_view = alertas[colunas_alerta].copy()
    if len(alertas_view) > 0:
        alertas_view["preco"] = alertas_view["preco"].apply(formatar_moeda)
        alertas_view["mediana_grupo"] = alertas_view["mediana_grupo"].apply(formatar_moeda)
        alertas_view["desvio_percentual"] = alertas_view["desvio_percentual"].round(2)

    st.dataframe(alertas_view, use_container_width=True, hide_index=True)

# =========================
# ABA 4
# =========================
with aba4:
    st.subheader("📊 Gráficos organizados")

    g1, g2 = st.columns(2)

    contagem = (
        df["classificacao"]
        .value_counts()
        .reset_index()
    )
    contagem.columns = ["classificacao", "quantidade"]

    fig1 = px.pie(
        contagem,
        names="classificacao",
        values="quantidade",
        hole=0.45,
        title="Distribuição das classificações"
    )
    fig1.update_layout(template="plotly_dark", height=430, margin=dict(l=20, r=20, t=70, b=20))
    g1.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(
        df,
        x="desvio_percentual",
        color="classificacao",
        nbins=50,
        title="Desvio percentual em relação à mediana do grupo",
        labels={"desvio_percentual": "Desvio percentual (%)"}
    )
    fig2.add_vline(x=-40, line_dash="dash", line_color="green")
    fig2.add_vline(x=-20, line_dash="dash", line_color="lime")
    fig2.add_vline(x=40, line_dash="dash", line_color="orange")
    fig2.add_vline(x=80, line_dash="dash", line_color="red")
    fig2.update_layout(template="plotly_dark", height=430, margin=dict(l=20, r=20, t=70, b=20))
    g2.plotly_chart(fig2, use_container_width=True)

    st.subheader("Preço por m² x Área")

    filtro_grafico = df.copy()
    if len(filtro_grafico) > 1500:
        filtro_grafico = filtro_grafico.sample(1500, random_state=42)

    fig3 = px.scatter(
        filtro_grafico,
        x="area_m2",
        y="preco_m2",
        color="classificacao",
        hover_data=["titulo", "cidade", "bairro", "tipo_padrao", "preco"],
        title="Relação entre área e preço por m²",
        labels={
            "area_m2": "Área (m²)",
            "preco_m2": "Preço por m²",
            "classificacao": "Classificação"
        }
    )
    fig3.update_layout(
        template="plotly_dark",
        height=550,
        margin=dict(l=20, r=20, t=70, b=20)
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Top grupos com maior mediana de preço")

    top_grupos = (
        df.groupby(["cidade", "tipo_padrao"])
        .agg(mediana_preco=("preco", "median"), anuncios=("preco", "count"))
        .reset_index()
    )
    top_grupos = top_grupos[top_grupos["anuncios"] >= 5]
    top_grupos["grupo"] = top_grupos["tipo_padrao"] + " — " + top_grupos["cidade"]
    top_grupos = top_grupos.sort_values("mediana_preco", ascending=False).head(10)

    fig4 = px.bar(
        top_grupos,
        x="mediana_preco",
        y="grupo",
        orientation="h",
        title="Top 10 grupos com maior preço mediano",
        labels={"mediana_preco": "Preço mediano", "grupo": "Grupo"},
        color="mediana_preco",
        color_continuous_scale="Plasma"
    )
    fig4.update_layout(
        template="plotly_dark",
        height=500,
        yaxis=dict(autorange="reversed"),
        margin=dict(l=20, r=20, t=70, b=20),
        coloraxis_showscale=False
    )
    st.plotly_chart(fig4, use_container_width=True)

with st.expander("📋 Ver base tratada"):
    colunas = [
        "titulo", "tipo_padrao", "cidade", "bairro", "preco", "area_m2",
        "preco_m2", "mediana_grupo", "desvio_percentual", "classificacao", "alerta_comprador"
    ]
    colunas = [c for c in colunas if c in df.columns]

    base_view = df[colunas].head(300).copy()
    st.dataframe(base_view, use_container_width=True, hide_index=True)
