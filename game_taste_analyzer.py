"""Metacritic 리뷰로 취향을 분석하는 Streamlit 앱."""

import html
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train-00000-of-00001.parquet"
DATA_URL = "https://huggingface.co/datasets/Wada1/Metacritic_Games_Reviews_Dataset/resolve/main/data/train-00000-of-00001.parquet"
LOCAL_REVIEWS = [BASE_DIR / f"metacritic_reviews_{i}.csv" for i in range(1, 4)]
star_rating = components.declare_component("half_star_rating", path=str(BASE_DIR / "star_rating"))


@st.cache_data(show_spinner=False)
def load_data():
    """노트북과 같은 원본 Parquet을 우선 사용하고, 연결 실패 시 CSV로 대체한다."""
    try:
        if not DATA_PATH.exists():
            response = requests.get(DATA_URL, timeout=(10, 60))
            response.raise_for_status()
            DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            DATA_PATH.write_bytes(response.content)
        df = pd.read_parquet(DATA_PATH)
        data_source = "Hugging Face 원본 Parquet"
    except Exception as parquet_error:
        local_files = [path for path in LOCAL_REVIEWS if path.exists()]
        if not local_files:
            raise RuntimeError(f"원본 리뷰 데이터를 불러오지 못했습니다: {parquet_error}") from parquet_error
        df = pd.concat([pd.read_csv(path) for path in local_files], ignore_index=True)
        data_source = "로컬 CSV 대체 데이터 (원본 Parquet 연결 실패)"
        if {"game", "name", "rating"}.issubset(df.columns):
            df = df.rename(columns={"name": "user"})
    columns = {str(column).lower(): column for column in df.columns}

    def find(*names):
        return next((columns[name] for name in names if name in columns), None)

    game_col = find("game", "game_title", "title") or (find("name") if "game" not in columns else None)
    user_col = find("user", "user_id", "username", "critic_name", "author", "name")
    rating_col = find("rating", "score", "review_score", "metascore")
    review_col = find("review", "review_text", "text", "content")
    if not all((game_col, user_col, rating_col)):
        raise ValueError(f"게임, 사용자, 평점 컬럼을 찾지 못했습니다: {list(df.columns)}")

    df = df.rename(columns={game_col: "game", user_col: "user", rating_col: "rating"})
    if review_col:
        df = df.rename(columns={review_col: "review"})
    else:
        df["review"] = ""
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["game", "user", "rating"])
    # 데이터가 0~100점 척도이면 0~10점으로 먼저 바꾼다.
    if df["rating"].max() > 10:
        df["rating"] /= 10
    df = df[df["rating"].between(0, 10)].copy()
    df["rating_5"] = df["rating"] / 2
    df["game"] = df["game"].astype(str).str.strip()
    df["user"] = df["user"].astype(str)
    df.attrs["source"] = data_source
    return df


@st.cache_resource(show_spinner="리뷰 데이터로 취향 모델을 처음 학습하는 중입니다…")
def train_model():
    """리뷰를 사용자별 게임 평점으로 집계한다."""
    df = load_data()
    user_games = df.groupby(["user", "game"], as_index=False)["rating_5"].mean()
    return df, user_games


@st.cache_data(ttl=600, show_spinner=False)
def poster_url(title):
    """Steam 이미지를 먼저 사용하고, 없을 때만 다른 제공처를 검색한다."""
    def normalize(value):
        value = re.sub(r"\s*\((?:\d{4}\s+)?video game\)\s*$", "", value, flags=re.IGNORECASE)
        value = re.sub(r"[^a-z0-9]", "", value.lower())
        return re.sub(r"(?:directorscut|digitaldeluxeedition|deluxeedition|completeedition|gotyedition|gameoftheyearedition)$", "", value)

    wanted = normalize(title)
    cover_terms = ("cover", "boxart", "box", "poster", "packart", "keyart", "frontcover")
    lookup_failed = False

    try:
        response = requests.get(
            "https://store.steampowered.com/api/storesearch/",
            params={"term": title, "l": "english", "cc": "us"},
            timeout=3,
        )
        response.raise_for_status()
        games = response.json().get("items", [])
        match = next((item for item in games if normalize(item.get("name", "")) == wanted), None)
        if match:
            app = {}
            try:
                details = requests.get(
                    "https://store.steampowered.com/api/appdetails/",
                    params={"appids": match["id"], "cc": "us", "l": "en"},
                    timeout=3,
                )
                details.raise_for_status()
                app = details.json().get(str(match["id"]), {}).get("data", {})
                if app and normalize(app.get("name", "")) != wanted:
                    app = {}
            except requests.RequestException:
                pass
            image = app.get("header_image") or app.get("capsule_imagev5") or match.get("tiny_image")
            if image:
                return image, False
    except Exception:
        lookup_failed = True

    commons_candidates = []
    try:
        response = requests.get(
            "https://commons.wikimedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": f'{title} "video game" (cover OR box art OR poster)',
                "gsrnamespace": 6,
                "gsrlimit": 12,
                "prop": "imageinfo",
                "iiprop": "url",
                "iiurlwidth": 900,
                "format": "json",
            },
            headers={"User-Agent": "GameTasteAnalyzer/1.0"},
            timeout=3,
        )
        response.raise_for_status()
        for page in response.json().get("query", {}).get("pages", {}).values():
            filename = normalize(page.get("title", "").replace("File:", ""))
            if not any(term in filename for term in cover_terms):
                continue
            if not wanted or wanted not in filename:
                continue
            info = (page.get("imageinfo") or [{}])[0]
            thumbnail = info.get("thumburl") or info.get("url", "")
            width, height = info.get("thumbwidth", info.get("width", 0)), info.get("thumbheight", info.get("height", 0))
            if thumbnail and height and width / height < 0.9:
                commons_candidates.append((len(filename), thumbnail))
    except Exception:
        lookup_failed = True
    if commons_candidates:
        return min(commons_candidates, key=lambda item: item[0])[1], True

    # Wikipedia page images are allowed only when their source filename identifies cover art.
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "generator": "search",
                "gsrsearch": f"{title} video game cover",
                "gsrnamespace": 0,
                "gsrlimit": 8,
                "prop": "pageimages",
                "piprop": "thumbnail|name",
                "pilicense": "any",
                "pithumbsize": 900,
                "format": "json",
            },
            headers={"User-Agent": "GameTasteAnalyzer/1.0"},
            timeout=3,
        )
        response.raise_for_status()
        candidates = []
        for page in response.json().get("query", {}).get("pages", {}).values():
            page_title = normalize(page.get("title", ""))
            # Wikipedia sometimes uses the main title without its subtitle (e.g. Sekiro).
            if page_title not in {wanted, normalize(title.split(":", 1)[0])}:
                continue
            page_image = normalize(page.get("pageimage", ""))
            thumb = page.get("thumbnail", {})
            width, height = thumb.get("width", 0), thumb.get("height", 0)
            if thumb.get("source") and any(term in page_image for term in cover_terms) and height and width / height < 0.9:
                candidates.append((page_title == wanted, thumb["source"]))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1], True
    except Exception:
        lookup_failed = True
    if lookup_failed:
        raise ConnectionError("Image provider lookup failed")
    return "", False


def lookup_poster(title):
    try:
        return poster_url(title)
    except ConnectionError:
        return None


def image_with_fallback(title, image, api_key):
    if image is None:
        st.warning(f"{title}: 이미지 제공처 연결에 실패했습니다. GPT 생성 없이 다음 조회 때 다시 시도합니다.")
        return "", False
    if image[0] or not api_key:
        return image
    generated = st.session_state.setdefault("generated_posters", {})
    if title in generated:
        return generated[title]
    try:
        with st.spinner("gpt로 이미지 생성중..."):
            response = OpenAI(api_key=api_key, timeout=120, max_retries=0).images.generate(
                model="gpt-image-1-mini",
                prompt=(f"Create original cinematic landscape poster artwork inspired by the video game {title!r}. "
                        "Show an illustrated scene conveying the game's mood, not just lettering. No text or logos."),
                size="1536x1024", quality="low", output_format="jpeg",
            )
            encoded = response.data[0].b64_json
            if not encoded:
                raise ValueError("Empty image response")
            generated[title] = (f"data:image/jpeg;base64,{encoded}", False)
            return generated[title]
    except Exception:
        st.warning("대체 이미지 생성에 실패했습니다. 입력한 API 키의 모델 접근 권한과 잔액을 확인해 주세요.")
        return image


def poster_markup(image, safe_title):
    if isinstance(image, str):
        image = (image, False)
    url, portrait = image
    if not url:
        return f'<div class="poster poster-fallback">🎮<br>{safe_title}</div>'
    orientation = " portrait" if portrait else ""
    label = '<p class="muted">AI 생성 대체 이미지</p>' if url.startswith("data:image/") else ""
    return f'<div class="poster{orientation}"><img src="{html.escape(url, quote=True)}" alt="{safe_title}"></div>{label}'


@st.cache_data(show_spinner=False)
def popular_games():
    """실제로 학습하는 리뷰 데이터의 행 수로 게임을 정렬한다."""
    counts = load_data().groupby("game").size()
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def neighbor_recommendations(user_games, known):
    """공통 평가의 오차와 겹친 게임 수로 이웃을 고르고 실제 평균만 집계한다."""
    common_rows = user_games[user_games["game"].isin(known)].copy()
    common_rows["squared_error"] = (common_rows["rating_5"] - common_rows["game"].map(known)) ** 2
    distances = common_rows.groupby("user").agg(overlap=("game", "nunique"), mse=("squared_error", "mean"))
    target_overlap = min(5, max(2, (len(known) + 4) // 5))
    min_overlap = target_overlap
    while min_overlap > 2 and (distances["overlap"] >= min_overlap).sum() < 10:
        min_overlap -= 1
    distances = distances[distances["overlap"] >= min_overlap].copy()
    distances["distance"] = np.sqrt(distances["mse"]) + 1 / distances["overlap"]
    shortlist = distances.sort_values(["distance", "overlap"], ascending=[True, False]).head(50)
    if len(shortlist) < 10:
        raise ValueError(f"공통으로 {min_overlap}개 이상 평가한 사용자가 {len(shortlist)}명입니다. 유사 사용자 10명을 찾으려면 다른 게임도 평가해 주세요.")
    # 리뷰 수 가점과 허용 거리 차이를 0.25점으로 제한해 유사도를 우선한다.
    shortlist = shortlist[shortlist["distance"] <= shortlist.iloc[9]["distance"] + 0.25].copy()
    unseen_counts = user_games[~user_games["game"].isin(known)].groupby("user")["game"].nunique()
    shortlist["unseen_count"] = unseen_counts.reindex(shortlist.index, fill_value=0)
    shortlist["selection_distance"] = shortlist["distance"] - 0.25 * shortlist["unseen_count"] / (shortlist["unseen_count"] + 5)
    neighbors = shortlist.sort_values(["selection_distance", "distance", "overlap"], ascending=[True, True, False]).head(10)
    candidates = user_games[user_games["user"].isin(neighbors.index) & ~user_games["game"].isin(known)]
    stats = candidates.groupby("game").agg(score=("rating_5", "mean"), raters=("user", "nunique"))
    qualified = stats[stats["raters"] >= 3].sort_values(["score", "raters"], ascending=[False, False])
    selected = qualified.head(3).assign(fallback=False)
    if len(selected) < 3:
        fallback = stats.drop(index=selected.index).sort_values(["raters", "score"], ascending=[False, False])
        selected = pd.concat([selected, fallback.head(3 - len(selected)).assign(fallback=True)])
    diagnostics = {
        "candidate_count": len(stats), "qualified_count": len(qualified), "rated_count": len(known),
        "target_overlap": target_overlap, "min_overlap": min_overlap,
        "shortlist_count": len(shortlist), "neighbors": neighbors.index.tolist(),
    }
    return selected, diagnostics


def recommend(ratings, api_key):
    df, user_games = train_model()
    submitted = {str(game).strip(): float(score) for game, score in (ratings or {}).items() if score is not None and 0.5 <= float(score) <= 5}
    lookup = {str(game).strip().casefold(): game for game in user_games["game"].unique()}
    known = {lookup[game.casefold()]: score for game, score in submitted.items() if game.casefold() in lookup}
    if len(known) < 5:
        raise ValueError(f"저장된 평점 {len(submitted)}개 중 리뷰 데이터와 일치하는 게임은 {len(known)}개입니다. 평가 목록의 게임 5개 이상을 평가해 주세요.")

    candidate_stats, diagnostics = neighbor_recommendations(user_games, known)
    st.session_state.recommendation_diagnostics = diagnostics
    if candidate_stats.empty:
        raise ValueError("유사 사용자 10명이 평가한 게임을 이미 모두 평가해서 추천할 게임이 없습니다.")
    predictions = candidate_stats["score"].head(3)
    with st.spinner("이미지 가져오는중..."):
        with ThreadPoolExecutor(max_workers=3) as pool:
            image_urls = list(pool.map(lookup_poster, map(str, predictions.index)))
    image_lookup = dict(zip(predictions.index, image_urls))
    for game in predictions.index:
        image_lookup[game] = image_with_fallback(str(game), image_lookup[game], api_key)

    review_context = {}
    for game in predictions.index:
        game_reviews = df.loc[(df["game"] == game) & df["review"].notna(), ["rating_5", "review"]].copy()
        game_reviews["review"] = game_reviews["review"].astype(str).str.strip()
        game_reviews = game_reviews[game_reviews["review"] != ""]
        positives = game_reviews[game_reviews["rating_5"] >= 4.0]["review"].head(3).tolist()
        negatives = game_reviews[game_reviews["rating_5"] <= 2.5]["review"].head(3).tolist()
        if not positives:
            positives = game_reviews.nlargest(2, "rating_5")["review"].tolist()
        if not negatives:
            negatives = game_reviews.nsmallest(2, "rating_5")["review"].tolist()
        review_context[game] = {
            "high_score_reviews": [review[:700] for review in positives],
            "low_score_reviews": [review[:700] for review in negatives],
        }

    recommendation_reasons = {}
    if api_key:
        try:
            response = OpenAI(api_key=api_key, timeout=30, max_retries=0).chat.completions.create(
                model="gpt-4o-mini",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "각 추천 게임마다 왜 이 사용자의 취향에 맞을지 한국어 한 문장으로 직접 설명하세요. 사용자가 높게 평가한 게임과 추천 게임의 리뷰 내용을 연결해 개인화하고, 각 게임의 구체적인 리뷰 근거를 활용하세요. 리뷰와 평점에 없는 게임 특징은 추측하지 말고, 모든 추천에 같은 문구를 반복하지 마세요. JSON 객체만 출력하고 게임 제목을 키, 추천 이유 한 문장을 값으로 작성하세요."},
                    {"role": "user", "content": json.dumps({"user_ratings_5_point_scale": known, "recommendations": {game: {"neighbor_average_rating": float(score), "neighbor_raters": int(candidate_stats.loc[game, "raters"]), "reviews": review_context[game]} for game, score in predictions.items()}}, ensure_ascii=False)},
                ],
            )
            recommendation_reasons = json.loads(response.choices[0].message.content or "{}")
        except Exception:
            recommendation_reasons = {}

    return [
        {
            "game": game,
            "score": float(score),
            "raters": int(candidate_stats.loc[game, "raters"]),
            "neighbor_count": 10,
            "fallback": bool(candidate_stats.loc[game, "fallback"]),
            "reason": str(recommendation_reasons.get(game, "API 키를 입력하면 GPT 추천 이유를 생성할 수 있습니다." if not api_key else "GPT 추천 이유 생성에 실패했습니다. 키 권한과 API 연결을 확인해 주세요.")),
            "image": image_lookup[game],
        }
        for game, score in predictions.items()
    ]


def save_current_rating(game, score_key):
    if score_key != f"stars_{st.session_state.game_index}":
        return
    score = st.session_state.get(score_key)
    if score is None or not 0.5 <= float(score) <= 5 or float(score) * 2 != int(float(score) * 2):
        return
    st.session_state.ratings[game] = float(score)
    st.session_state.recommendations = None
    st.session_state.game_index += 1


def skip_current_game():
    st.session_state.game_index += 1


def reset_app():
    st.session_state.game_index = 0
    st.session_state.ratings = {}
    st.session_state.recommendations = None
    for key in list(st.session_state):
        if key.startswith(("score_", "stars_")):
            del st.session_state[key]


def main():
    st.set_page_config(page_title="게임 취향 분석기", page_icon="🎮", layout="wide")
    st.markdown("""
    <style>
    .stApp, .stApp *, .stApp *::before, .stApp *::after {animation:none!important;transition:none!important;scroll-behavior:auto!important}
    .stApp {background:linear-gradient(145deg,#0b1020 0%,#111827 55%,#17132b 100%);color:#f5f7ff}
    [data-testid="stHeader"] {background:transparent}
    .hero {padding:2rem 2.2rem;border:1px solid #ffffff20;border-radius:24px;background:linear-gradient(120deg,#20294cdd,#30214bdd);margin-bottom:1.2rem}
    .hero h1 {font-size:2.6rem;margin:0 0 .4rem 0}
    .hero p {color:#c7cbe0;margin:0}
    .poster {width:min(100%,520px);aspect-ratio:16/9;overflow:hidden;border-radius:18px;background:#1d2438;margin:0 auto 1rem;box-shadow:0 18px 50px #0008}
    .poster img {width:100%;height:100%;object-fit:cover;background:#111827}
    .poster.portrait img {object-fit:contain}
    .game-meta,.muted {color:#aeb6d2}
    .result-card {padding:1.4rem;border:1px solid #ffffff22;border-radius:20px;background:#ffffff0b;margin:1rem 0 1.6rem;box-shadow:0 14px 40px #0003}
    .result-card h2 {margin:.2rem 0 .7rem}
    div.stButton > button {border-radius:12px;min-height:2.8rem;font-weight:650}
    div.stButton > button[kind="secondary"] {color:#111827!important;background:#f3f4f6!important;border:1px solid #d1d5db!important}
    .poster-fallback {display:grid;place-items:center;text-align:center;padding:1rem;color:#f3f4ff;font-size:1.2rem;font-weight:700;background:radial-gradient(circle at 70% 20%,#7458b8,#273451 55%,#151a29)}
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="hero"><h1>🎮 게임 취향 분석기</h1><p>리뷰 평점으로 취향 모델을 학습하고, 비슷한 게이머들이 좋아한 게임을 찾아요.</p></div>', unsafe_allow_html=True)
    with st.form("api_key_form"):
        entered_key = st.text_input("OpenAI API 키", type="password", key="user_api_key",
                                    help="입력 후 적용 버튼을 눌러 주세요. 키는 현재 사용자 세션에서만 보관합니다.").strip()
        apply_key = st.form_submit_button("API 키 적용", type="primary")
    if apply_key:
        st.session_state.active_api_key = entered_key
        st.session_state.recommendations = None
    api_key = st.session_state.get("active_api_key", "")
    if api_key:
        st.success("API 키 적용됨 · 이후 GPT 요청에 이 키를 사용합니다.")
        st.caption("키가 적용된 상태이며, 유효성과 모델 접근 권한은 실제 API 요청 시 확인됩니다. 키를 지우고 적용하면 사용을 해제합니다.")
    st.caption("이미지가 없으면 gpt-image-1-mini로 생성합니다. 최초 생성은 시간이 걸리며 입력한 키에 API 요금이 발생합니다.")
    if not api_key:
        st.info("API 키 없이도 5개 이상 평가하면 평점 기반 추천을 받을 수 있습니다. GPT 추천 이유와 이미지 생성에는 API 키가 필요합니다.")
    try:
        with st.spinner("원본 리뷰 데이터를 불러오는 중입니다. 처음 실행은 잠시 걸릴 수 있어요."):
            ranked_games = popular_games()
            data_source = load_data().attrs.get("source", "")
    except Exception as exc:
        st.error(f"게임 목록을 불러오지 못했습니다: {exc}")
        st.stop()
    if data_source.startswith("로컬 CSV"):
        st.warning(f"{data_source}. 게임 순위가 노트북 원본과 다를 수 있습니다.")
    if not ranked_games:
        st.error("평가할 게임이 없습니다.")
        st.stop()

    if "game_index" not in st.session_state:
        st.session_state.game_index = 0
    if "ratings" not in st.session_state:
        st.session_state.ratings = {}
    if "recommendations" not in st.session_state:
        st.session_state.recommendations = None
    rated_count = len(st.session_state.ratings)
    st.progress(min(rated_count / 5, 1.0), text=f"추천 데이터에 반영되는 평가 {rated_count}개 · 5개 이상부터 추천 가능")

    index = st.session_state.game_index
    if index < len(ranked_games):
        game, review_count = ranked_games[index]
        left, center, right = st.columns([1, 1.2, 1])
        with center:
            with st.spinner("이미지 가져오는중..."):
                image = lookup_poster(str(game))
            image = image_with_fallback(str(game), image, api_key)
            safe_game = html.escape(str(game))
            st.markdown(poster_markup(image, safe_game), unsafe_allow_html=True)
            st.markdown(f"### {safe_game}")
            st.markdown(f'<p class="game-meta">인기 순위 {index + 1}위 · 리뷰 {review_count:,}개</p>', unsafe_allow_html=True)
            star_rating(key=f"stars_{index}", default=None,
                        on_change=lambda: save_current_rating(game, f"stars_{index}"))
            st.markdown('<p class="muted" style="text-align:center">별의 왼쪽은 반 점, 오른쪽은 한 점입니다.<br>클릭하면 바로 저장하고 다음 게임으로 넘어갑니다.</p>', unsafe_allow_html=True)
            _, skip_center, _ = st.columns([1, 2, 1])
            with skip_center:
                st.button("아직 안해봤어요", key="skip_game", use_container_width=True, on_click=skip_current_game)
    else:
        st.info("모든 게임을 확인했습니다.")

    st.divider()
    recommend_col, reset_col = st.columns([3, 1])
    with recommend_col:
        get_recommendations = st.button("✨ 내 취향으로 3개 추천받기", type="primary", disabled=rated_count < 5, use_container_width=True)
    with reset_col:
        st.button("처음부터", key="reset_app", use_container_width=True, on_click=reset_app)

    if get_recommendations:
        st.session_state.recommendations = None
        try:
            with st.spinner("리뷰 데이터에서 취향이 비슷한 게이머를 찾고 있어요…"):
                st.session_state.recommendations = recommend(st.session_state.ratings, api_key)
        except Exception as exc:
            st.error(str(exc) if isinstance(exc, ValueError) else "추천 처리에 실패했습니다. API 키와 연결 상태를 확인해 주세요.")

    if st.session_state.recommendations:
        st.markdown("## 당신을 위한 추천")
        if len(st.session_state.recommendations) < 3:
            details = st.session_state.get("recommendation_diagnostics", {})
            st.info(f"유사 사용자 10명이 평가한 미평가 게임 자체가 {details.get('candidate_count', '?')}개라 가능한 게임만 표시합니다.")
        for fallback, heading, description in [
            (False, "취향 맞춤 추천", "비슷한 취향의 사용자 10명 중 3명 이상이 평가한 게임을 평균 별점순으로 추천해요."),
            (True, "비슷한 취향의 인기작", "맞춤 추천이 부족한 만큼, 같은 사용자들이 많이 평가한 게임을 골랐어요. 평가 인원이 같으면 평균 별점이 높은 게임을 먼저 보여드려요."),
        ]:
            items = [item for item in st.session_state.recommendations if bool(item.get("fallback")) == fallback]
            if not items:
                continue
            st.markdown(f"### {heading}")
            st.caption(description)
            for item in items:
                reason = html.escape(item["reason"]).replace("\n", " ")
                safe_title = html.escape(item["game"])
                poster = poster_markup(item["image"], safe_title)
                st.markdown(
                    f'<div class="result-card"><h2>{safe_title}</h2>'
                    f'{poster}'
                    f'<h3>예상 별점&nbsp; {item["score"]:.1f} / 5 ⭐</h3>'
                    f'<p class="muted">유사 사용자 {item["raters"]}/{item["neighbor_count"]}명이 평가</p>'
                    f'<p><b>추천 이유</b><br>{reason}</p></div>',
                    unsafe_allow_html=True,
                )


if __name__ == "__main__":
    main()
