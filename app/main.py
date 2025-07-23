from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re
import asyncio
import random
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import os
import shutil
import logging
import httpx
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.DEBUG,  # ✅ Make sure this is DEBUG
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)


app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Spotify credentials
SPOTIFY_CLIENT_ID = "5cea519d0b1446de97f8c54afed8497d"
SPOTIFY_CLIENT_SECRET = "305c6e938a824fab9683ea78c7b2ae44"

CONCURRENT_TASKS = 5
RETRIES = 3
max_listeners = 1.2e8

ARTIST_URLS_CACHE_FILE = "artist_urls_cache.json"
LISTENERS_CACHE_FILE = "listeners_cache.json"


def safe_load_cache(cache_file):
    if not os.path.exists(cache_file):
        return {}
    try:
        with open(cache_file, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        backup_file = cache_file + ".backup"
        shutil.copy(cache_file, backup_file)
        return {}


def save_cache(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f)


def indie_score(listeners):
    def log_decay(x, x_min, x_max, y_min, y_max):
        x = np.clip(x, x_min, x_max)
        scale = (np.log10(x) - np.log10(x_min)) / (np.log10(x_max) - np.log10(x_min))
        y = y_min + (y_max - y_min) * scale
        return y

    scores = []
    for x in listeners:
        if x <= 1e4:
            score = 100 - log_decay(x, 1, 1e4, 0, 1)
        elif x <= 1e5:
            score = 99 - log_decay(x, 1e4, 1e5, 0, 4)
        elif x <= 1e6:
            score = 95 - log_decay(x, 1e5, 1e6, 0, 10)
        elif x <= 1e7:
            score = 85 - log_decay(x, 1e6, 1e7, 0, 35)
        elif x <= 5e7:
            score = 50 - log_decay(x, 1e7, 5e7, 0, 40)
        else:
            score = 10 - log_decay(x, 5e7, max_listeners, 0, 10)
        scores.append(round(score, 2))
    return scores


def get_artists_from_playlist(playlist_url, sp):
    logging.info("🎵 Extracting artists from playlist")
    pattern = r"playlist/([a-zA-Z0-9]+)"
    match = re.search(pattern, playlist_url)
    if not match:
        raise ValueError("Invalid playlist URL")
    playlist_id = match.group(1)

    logging.info(f"Playlist ID: {playlist_id}")

    artist_counts = defaultdict(int)
    results = sp.playlist_items(playlist_id, additional_types=['track'])

    while results:
        for item in results['items']:
            track = item['track']
            if track:
                for artist in track['artists']:
                    name = artist['name'].strip()
                    if name and name != ',':
                        artist_counts[name] += 1
        if results['next']:
            results = sp.next(results)
        else:
            results = None

    logging.info(f"✅ Found {len(artist_counts)} unique artists")
    return artist_counts


def get_artist_urls(artist_names, sp):
    cache = safe_load_cache(ARTIST_URLS_CACHE_FILE)
    urls = {}

    for name in artist_names:
        if name in cache:
            urls[name] = cache[name]
        else:
            try:
                result = sp.search(q=name, type='artist', limit=1)
                if result['artists']['items']:
                    artist_id = result['artists']['items'][0]['id']
                    url = f"https://open.spotify.com/artist/{artist_id}"
                    urls[name] = url
                    cache[name] = url
                else:
                    urls[name] = None
                    cache[name] = None
            except:
                urls[name] = None
                cache[name] = None

    save_cache(cache, ARTIST_URLS_CACHE_FILE)
    return urls


async def fetch_with_retry(artist, url, retries=RETRIES):
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    for attempt in range(1, retries + 1):
        try:
            async with httpx.AsyncClient(timeout=15.0, headers=headers) as client:
                response = await client.get(url)
                response.raise_for_status()
                html = response.text

            soup = BeautifulSoup(html, 'html.parser')
            match = soup.find(string=re.compile(r'([\d,]+)\s+monthly listeners', re.IGNORECASE))
            if match:
                result = re.search(r'([\d,]+)\s+monthly listeners', match)
                if result:
                    listeners = result.group(1)
                    return (artist, listeners)

            raise ValueError(f"Monthly listeners not found for artist '{artist}'")

        except Exception as e:
            if attempt == retries:
                return (artist, None)
            await asyncio.sleep(1.5 * attempt)

async def fetch_monthly_listeners(artist_urls):
    logging.info("📡 Starting monthly listeners fetch (httpx)")
    results = []
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
    cache = safe_load_cache(LISTENERS_CACHE_FILE)

    async def bound_fetch(artist, url):
        if not url:
            return (artist, None)
        if artist in cache:
            return (artist, cache[artist])
        async with semaphore:
            result = await fetch_with_retry(artist, url)
            if result[1] is not None:
                cache[artist] = result[1]
            return result

    tasks = [bound_fetch(artist, url) for artist, url in artist_urls.items()]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)

    save_cache(cache, LISTENERS_CACHE_FILE)
    logging.info("💾 Cache saved")
    return results


def parse_listeners(listeners_str):
    if listeners_str is None:
        return None
    return int(listeners_str.replace(',', ''))


async def score_artists_from_playlist_async(playlist_url):
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    ))

    # Spotify calls (sync)
    artist_counts = get_artists_from_playlist(playlist_url, sp)
    artist_names = list(artist_counts.keys())

    artist_urls = get_artist_urls(artist_names, sp)

    # Async scraping - await instead of asyncio.run
    data = await fetch_monthly_listeners(artist_urls)

    df = pd.DataFrame(data, columns=["Artist", "Listeners"])
    df['ListenersInt'] = df['Listeners'].apply(parse_listeners)
    df = df.dropna(subset=['ListenersInt'])
    df['IndieScore'] = indie_score(df['ListenersInt'].tolist())

    dict_df = pd.DataFrame(list(artist_counts.items()), columns=['Artist', 'n'])
    df = df[['Artist', 'ListenersInt', 'IndieScore']].sort_values(by='IndieScore', ascending=False)
    df = df.merge(dict_df, on='Artist', how='left')

    return df


def get_artist_image_from_url(artist_url, sp):
    match = re.search(r'artist/([a-zA-Z0-9]+)', artist_url)
    if not match:
        return None
    artist_id = match.group(1)
    artist = sp.artist(artist_id)
    images = artist.get('images', [])
    if images:
        return images[0]['url']
    return None


def extract_playlist_id(playlist_url):
    return re.search(r"playlist/([a-zA-Z0-9]+)", playlist_url).group(1)


def get_playlist_info(playlist_url: str, sp):
    playlist_id = extract_playlist_id(playlist_url)
    logging.info(f"📥 Extracted playlist ID: {playlist_id}")
    data = sp.playlist(playlist_id)
    name = data['name']
    image_url = data['images'][0]['url'] if data['images'] else None
    logging.info("" \
    "" \
    "" \
    "" \
    "" \
    "EXCUSE ME", name)
    owner = data['owner']
    owner_name = owner['display_name']
    owner_url = owner['external_urls']['spotify']
    owner_id = owner['id']

    # Try to fetch owner image from their profile (if public)
    try:
        user = sp.user(owner_id)
        owner_image_url = user['images'][0]['url'] if user.get('images') else None
    except:
        owner_image_url = None

    return name, image_url, owner_name, owner_url, owner_image_url


def get_listener_distribution(df):
    bins = [0, 10_000, 100_000, 1_000_000, 10_000_000, 50_000_000, 1_200_000_000]
    labels = ['0-10k', '10k-100k', '100k-1M', '1M-10M', '10M-50M', '>50M']
    df['ListenerRange'] = pd.cut(df['ListenersInt'], bins=bins, labels=labels, right=False)

    distribution = df.groupby('ListenerRange').agg(
        count=('Artist', 'count'),
        avg_score=('IndieScore', 'mean')
    ).reset_index()
    logging.info("here", distribution)

    total = distribution['count'].sum()
    distribution['percent'] = distribution['count'] / total * 100
    distribution['score_band'] = distribution['avg_score'].round(2)

    distribution = distribution.rename(columns={'ListenerRange': 'label'})
    distribution = distribution.fillna({'score_band': 0, 'percent': 0})
    return distribution.to_dict(orient='records')


LEADERBOARD_FILE = "leaderboard.json"


def load_leaderboard():
    if os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "r") as f:
            return json.load(f)
    return []


def save_leaderboard(data):
    with open(LEADERBOARD_FILE, "w") as f:
        json.dump(data, f)


def update_leaderboard(playlist_id, score, playlist_url=None, playlist_name=None,
                       image_url=None, owner_name=None, owner_url=None, owner_image_url=None):
    leaderboard = load_leaderboard()
    if any(entry['playlist_id'] == playlist_id for entry in leaderboard):
        return leaderboard
    leaderboard.append({
        "playlist_id": playlist_id,
        "score": score,
        "playlist_url": playlist_url,
        "playlist_name": playlist_name,
        "image_url": image_url,
        "owner_name": owner_name,
        "owner_url": owner_url,
        "owner_image_url": owner_image_url
    })
    leaderboard.sort(key=lambda x: x['score'], reverse=True)
    save_leaderboard(leaderboard)
    return leaderboard


def get_playlist_rank(playlist_id, leaderboard):
    for idx, entry in enumerate(leaderboard, 1):
        if entry['playlist_id'] == playlist_id:
            return idx
    return None


def get_similar_playlists(current_id, current_score, leaderboard, top_n=5):
    # Exclude the current playlist
    others = [entry for entry in leaderboard if entry['playlist_id'] != current_id]

    # Sort by score proximity
    others.sort(key=lambda x: abs(x['score'] - current_score))

    return others[:top_n]


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request, playlist: str = None):
    if not playlist:
        return templates.TemplateResponse("indie_report.html", {"request": request, "playlist_name": None})

    try:
        logging.info("🔗 Received playlist URL")

        sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
            client_id=SPOTIFY_CLIENT_ID,
            client_secret=SPOTIFY_CLIENT_SECRET
        ))

        df = await score_artists_from_playlist_async(playlist)
        logging.info("✅ Artists scored")

        weighted_indie_score = round(np.average(df['IndieScore'], weights=df['n']), 2)
        logging.info(f"📊 Weighted Indie Score: {weighted_indie_score}")
        listener_distribution = get_listener_distribution(df)
        playlist_id = extract_playlist_id(playlist)

        # Load leaderboard first (do NOT update here with partial info)
        leaderboard = load_leaderboard()

        top_10 = df.head(10).to_dict(orient='records')

        most_indie_artist = df.iloc[df['IndieScore'].idxmax()]
        artist_urls = get_artist_urls([most_indie_artist['Artist']], sp)
        most_indie_artist_url = artist_urls.get(most_indie_artist['Artist'], None)
        most_indie_artist_image = get_artist_image_from_url(most_indie_artist_url, sp)

        playlist_name, playlist_image_url, owner_name, owner_url, owner_image_url = get_playlist_info(playlist, sp)
        similar_playlists = get_similar_playlists(playlist_id, weighted_indie_score, leaderboard)

        # Now update leaderboard fully
        leaderboard = update_leaderboard(
            playlist_id,
            weighted_indie_score,
            playlist,
            playlist_name,
            playlist_image_url,
            owner_name,
            owner_url,
            owner_image_url
        )

        rank = get_playlist_rank(playlist_id, leaderboard)
        total = len(leaderboard)
        percentile = round((rank / total)*100,0)
        if percentile <= 50:
            percentile_txt = f"Your playlist ranks in the top {percentile}% of all users ({rank} / {total})!"
        else:
            percentile_txt = f"Your playlist ranks in the bottom {100 - percentile}% of all users ({rank} / {total})..."

        return templates.TemplateResponse("indie_report.html", {
            "request": request,
            "playlist_name": playlist_name,
            "playlist_url": playlist,
            "playlist_image_url": playlist_image_url,
            "weighted_indie_score": weighted_indie_score,
            "listener_distribution": listener_distribution,
            "top_10_artists": top_10,
            "most_indie_artist_name": most_indie_artist['Artist'],
            "most_indie_artist_score": most_indie_artist['IndieScore'],
            "most_indie_artist_count": most_indie_artist['n'],
            "most_indie_artist_image": most_indie_artist_image,
            "rank": rank,
            "total": total,
            "percentile": percentile_txt,
            "similar_playlists": similar_playlists
        })
    except Exception as e:
        return templates.TemplateResponse("indie_report.html", {
            "request": request,
            "playlist_name": None,
            "error": str(e)
        })
    