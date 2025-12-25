import streamlit as st
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import ast
import plotly.express as px
import time

# --- KONFIGURACJA ---
GROQ_API_KEY = "groq_klucz"

st.set_page_config(page_title="🎥 CineMate AI Pro", page_icon="🧠", layout="wide")

# --- CSS (Stylizacja) ---
st.markdown("""
<style>
    /* Karta filmu */
    .movie-card { 
        background-color: #1E1E1E; 
        padding: 15px; 
        border-radius: 8px; 
        margin-bottom: 10px; 
        border-left: 5px solid #FF4B4B;
    }
    .match-score { color: #00ff00; font-weight: bold; font-size: 14px; }

    /* Pasek czatu na samym dole */
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        padding-bottom: 20px;
        padding-top: 10px;
        background-color: #0E1117;
        z-index: 100;
    }
    /* Margines dolny dla treści */
    .block-container {
        padding-bottom: 150px; 
    }
</style>
""", unsafe_allow_html=True)

# --- PAMIĘĆ SESJI ---
if "watchlist" not in st.session_state:
    st.session_state.watchlist = []
if "messages" not in st.session_state:
    st.session_state.messages = []
# KLUCZOWA POPRAWKA: Pamięć ostatnich wyników, żeby nie znikały po kliknięciu przycisku
if "last_results" not in st.session_state:
    st.session_state.last_results = []


# --- FUNKCJE POMOCNICZE ---
def parse_json_column(x):
    try:
        if pd.isna(x): return ""
        data = ast.literal_eval(str(x))
        names = [item['name'] for item in data]
        return ", ".join(names)
    except:
        return ""


def get_year(date_str):
    try:
        return int(str(date_str).split('-')[0])
    except:
        return 0


def add_to_watchlist(movie):
    # Sprawdzamy duplikaty po tytule
    if not any(m['title'] == movie['title'] for m in st.session_state.watchlist):
        st.session_state.watchlist.append({
            'title': movie['title'],
            'year': movie['year'],
            'overview': movie['overview'],
            'genres': movie['genres_clean']
        })
        st.toast(f"✅ Dodano: {movie['title']}", icon="💾")
    else:
        st.toast("Już masz ten film na liście!", icon="ℹ️")


# --- ŁADOWANIE DANYCH ---
@st.cache_resource
def load_data():
    try:
        df = pd.read_csv("tmdb_5000_movies.csv")

        # Czyszczenie
        df['genres_clean'] = df['genres'].apply(parse_json_column)
        df['keywords_clean'] = df['keywords'].apply(parse_json_column)
        df['overview'] = df['overview'].fillna('')
        df['title'] = df['title'].fillna('Unknown')
        df['year'] = df['release_date'].apply(get_year)
        df['vote_average'] = df['vote_average'].fillna(0)

        # Tekst do AI
        df['combined_info'] = (
                "Title: " + df['title'] +
                " | Genres: " + df['genres_clean'] +
                " | Plot: " + df['overview']
        )

        # Embeddingi
        embed_model = SentenceTransformer('all-MiniLM-L6-v2')
        with st.spinner("🚀 Inicjalizacja systemu AI..."):
            vectors = embed_model.encode(df['combined_info'].tolist(), show_progress_bar=True)

        vectors = np.array(vectors).astype('float32')
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        all_genres = set()
        for g in df['genres_clean']:
            if g: all_genres.update(g.split(', '))

        return index, df, embed_model, sorted(list(all_genres))

    except Exception as e:
        st.error(f"Błąd danych: {e}")
        return None, None, None, []


index, df, embed_model, all_genres = load_data()
client = Groq(api_key=GROQ_API_KEY)

# --- UKŁAD STRONY ---
tab1, tab2, tab3 = st.tabs(["🎬 Czat AI", "📊 Analityka", "❤️ Moja Lista"])

# === ZAKŁADKA 1: CZAT ===
with tab1:
    col_filters, col_chat = st.columns([1, 3])

    with col_filters:
        st.subheader("Filtry")
        # USUNIĘTE ZDJĘCIA - TYLKO TEKST
        selected_genre = st.selectbox("Gatunek", ["Wszystkie"] + all_genres)
        min_rating = st.slider("Min. Ocena", 0.0, 10.0, 6.0)
        creativity = st.slider("Kreatywność", 0.0, 1.0, 0.6)

        st.divider()
        st.info("💡 Wyniki są zapamiętywane, więc możesz klikać 'Dodaj' bez znikania listy.")

    with col_chat:
        st.title("🍿 CineMate AI")

        # 1. KONTENER NA TREŚĆ (Tutaj ląduje historia i wyniki)
        chat_container = st.container()

        # 2. PASEK INPUTU (NA SAMYM DOLE)
        user_input = st.chat_input("Napisz, jaki film chcesz obejrzeć...")

        # --- LOGIKA 1: PRZETWARZANIE NOWEGO ZAPYTANIA ---
        # Uruchamia się TYLKO gdy user wciśnie Enter
        if user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})

            if index is not None:
                # Wektoryzacja
                q_vec = embed_model.encode([user_input]).astype('float32')
                D, I = index.search(q_vec, 100)

                # Filtrowanie i ZAPIS DO PAMIĘCI
                new_results = []
                for dist, idx in zip(D[0], I[0]):
                    if idx == -1: continue
                    m = df.iloc[idx]
                    if selected_genre != "Wszystkie" and selected_genre not in m['genres_clean']: continue
                    if m['vote_average'] < min_rating: continue
                    score = max(0, 100 - dist * 5)
                    new_results.append((m, score))
                    if len(new_results) >= 4: break  # Top 4

                # Zapisujemy wyniki w sesji! To klucz do naprawy przycisku.
                st.session_state.last_results = new_results

                # Generowanie odpowiedzi AI
                context_text = "\n".join([f"{m['title']}: {m['overview']}" for m, s in new_results])
                try:
                    final_resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system",
                             "content": f"Jesteś ekspertem. Krótko (2 zdania) poleć jeden najlepszy film z listy: {context_text}. Odpowiedz po polsku."},
                            {"role": "user", "content": user_input}
                        ],
                        temperature=creativity
                    )
                    ans = final_resp.choices[0].message.content
                    st.session_state.messages.append({"role": "assistant", "content": ans})
                except Exception as e:
                    st.error(f"API Error: {e}")

        # --- LOGIKA 2: RENDEROWANIE EKRANU (Zawsze) ---
        with chat_container:
            # A. Historia
            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

            # B. Wyniki wyszukiwania (Zawsze renderujemy to co jest w pamięci)
            if st.session_state.last_results:
                st.divider()
                st.caption("🎬 Znalezione filmy (Kliknij +, aby dodać):")

                cols = st.columns(4)
                for i, (movie, score) in enumerate(st.session_state.last_results):
                    with cols[i]:
                        with st.container():
                            # Karta filmu (Bez obrazka)
                            st.markdown(f"**{movie['title']}**")
                            st.caption(f"📅 {movie['year']} | ⭐ {movie['vote_average']}")
                            st.markdown(f"<span class='match-score'>{int(score)}% Match</span>", unsafe_allow_html=True)

                            # PRZYCISK - TERAZ DZIAŁA BO DANE SĄ W SESJI
                            # Klucz musi być unikalny dla filmu
                            btn_key = f"btn_{movie['title'].replace(' ', '_')}_{i}"
                            if st.button("➕ Dodaj", key=btn_key):
                                add_to_watchlist(movie)
                            st.divider()

# === ZAKŁADKA 2: ANALITYKA (BOGATSZA WERSJA) ===
with tab2:
    st.title("📊 Laboratorium Danych")
    st.markdown("Analiza zbioru danych wykorzystywanego przez model.")

    # 1. Duże Liczby (KPI)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Liczba Filmów", len(df))
    m2.metric("Średnia Ocena", round(df['vote_average'].mean(), 2))
    m3.metric("Liczba Gatunków", len(all_genres))
    m4.metric("Najstarszy Film", df['year'].min())

    st.divider()

    # 2. Wykresy w dwóch kolumnach
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        st.subheader("📈 Jak oceniane są filmy?")
        # Histogram
        fig_hist = px.histogram(df, x="vote_average", nbins=20, title="Rozkład Ocen (IMDB)",
                                color_discrete_sequence=['#ffbd45'])
        fig_hist.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_chart2:
        st.subheader("🎭 Najpopularniejsze Gatunki")
        # Przetwarzanie gatunków do wykresu (rozdzielamy po przecinku)
        all_genres_list = [g for sublist in df['genres_clean'].str.split(', ') for g in sublist if g]
        genre_counts = pd.Series(all_genres_list).value_counts().head(10)

        # Wykres słupkowy poziomy
        fig_bar = px.bar(genre_counts, x=genre_counts.values, y=genre_counts.index, orientation='h',
                         title="Top 10 Gatunków",
                         color=genre_counts.values,
                         color_continuous_scale='Bluered')
        fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white",
                              yaxis=dict(autorange="reversed"))  # Odwracamy, żeby najpopularniejszy był na górze
        st.plotly_chart(fig_bar, use_container_width=True)

    # 3. Wykres liniowy (Area Chart) - Trendy
    st.subheader("🗓️ Historia Kina: Liczba filmów w czasie")
    year_counts = df['year'].value_counts().sort_index()
    year_counts = year_counts[year_counts.index > 1920]  # Omijamy bardzo stare błędy danych

    fig_area = px.area(x=year_counts.index, y=year_counts.values,
                       title="Wzrost produkcji filmowej",
                       color_discrete_sequence=['#00cc96'])
    fig_area.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="white")
    st.plotly_chart(fig_area, use_container_width=True)

    st.info(
        "💡 **Wniosek:** Baza danych wykazuje eksponencjalny wzrost liczby filmów po roku 1990, co może wpływać na to, że model częściej poleca nowsze produkcje.")

# === ZAKŁADKA 3: WATCHLISTA ===
with tab3:
    st.header(f"❤️ Moja Lista ({len(st.session_state.watchlist)})")

    if not st.session_state.watchlist:
        st.info("Lista jest pusta. Wróć do czatu i dodaj coś!")
    else:
        wl_df = pd.DataFrame(st.session_state.watchlist)
        st.dataframe(wl_df[['title', 'year', 'genres']], use_container_width=True)

        # Usuwanie
        st.subheader("Zarządzanie")
        titles = [m['title'] for m in st.session_state.watchlist]
        movie_to_delete = st.selectbox("Wybierz film do usunięcia", titles)

        if st.button("🗑️ Usuń wybrany"):
            st.session_state.watchlist = [m for m in st.session_state.watchlist if m['title'] != movie_to_delete]
            st.rerun()

        # Eksport (IDP)
        st.divider()
        csv = wl_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Pobierz listę (CSV)",
            data=csv,
            file_name='moja_lista.csv',
            mime='text/csv'
        )