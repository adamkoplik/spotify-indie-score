import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import re
import asyncio
from playwright.async_api import async_playwright
import random
import pandas as pd
import numpy as np
from collections import defaultdict
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import messagebox


# Your Spotify API credentials
SPOTIFY_CLIENT_ID = '5cea519d0b1446de97f8c54afed8497d'
SPOTIFY_CLIENT_SECRET = '305c6e938a824fab9683ea78c7b2ae44'

# --- Config for Playwright scraping ---
CONCURRENT_TASKS = 5
RETRIES = 3

max_listeners = 1.2e8  # 120 million

ARTIST_URLS_CACHE_FILE = "artist_urls_cache.json"
LISTENERS_CACHE_FILE = "listeners_cache.json"

import json
import shutil
import os

def safe_load_cache(cache_file):
    if not os.path.exists(cache_file):
        print(f"Cache file '{cache_file}' does not exist. Creating a new empty cache.")
        return {}

    with open(cache_file, 'r') as f:
        data = f.read()

    try:
        obj = json.loads(data)
        print("Cache loaded successfully.")
        return obj
    except json.JSONDecodeError as e:
        print(f"JSONDecodeError loading cache at line {e.lineno}, column {e.colno} (char {e.pos}): {e.msg}")
        # Backup corrupted cache file
        backup_file = cache_file + ".backup"
        shutil.copy(cache_file, backup_file)
        print(f"Corrupted cache backed up to '{backup_file}'")

        # Optionally delete original corrupted file or keep it for manual fix
        # os.remove(cache_file)  # uncomment if you want to delete it immediately

        # Return empty cache to proceed safely
        return {}

def save_cache(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

# Indie score function ported from R
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

# Extract artists from playlist (with counts)
def get_artists_from_playlist(playlist_url, sp):
    pattern = r"playlist/([a-zA-Z0-9]+)"
    match = re.search(pattern, playlist_url)
    if not match:
        raise ValueError("Invalid playlist URL")
    playlist_id = match.group(1)

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

    return artist_counts

# Get Spotify artist URLs by name with cache
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

# 
def get_top_artist_url(name, sp):
    cache = safe_load_cache(ARTIST_URLS_CACHE_FILE)
    url = cache[name]
    return url

# Async fetch monthly listeners from artist Spotify page with cache
async def fetch_with_retry(page, artist, url, retries=RETRIES):
    for attempt in range(1, retries + 1):
        try:
            await page.goto(url, timeout=15000)
            await page.wait_for_timeout(random.randint(2000, 4000))
            html = await page.content()

            match = re.search(r'([\d,]+)\s+monthly listeners', html, re.IGNORECASE)
            if not match:
                raise ValueError(f"Monthly listeners not found for artist '{artist}'")

            listeners = match.group(1)
            return (artist, listeners)

        except Exception as e:
            if attempt == retries:
                return (artist, None)
            await asyncio.sleep(1.5 * attempt)

async def fetch_monthly_listeners(artist_urls):
    results = []
    semaphore = asyncio.Semaphore(CONCURRENT_TASKS)
    cache = safe_load_cache(LISTENERS_CACHE_FILE)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        async def bound_fetch(artist, url):
            if not url:
                return (artist, None)
            if artist in cache:
                return (artist, cache[artist])
            print("Accessing", artist)
            async with semaphore:
                page = await browser.new_page()
                result = await fetch_with_retry(page, artist, url)
                await page.close()
                if result[1] is not None:
                    cache[artist] = result[1]
                return result

        tasks = [bound_fetch(artist, url) for artist, url in artist_urls.items()]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)

        await browser.close()

    save_cache(cache, LISTENERS_CACHE_FILE)
    return results

# Parse listener count string "1,234,567" → int 1234567
def parse_listeners(listeners_str):
    if listeners_str is None:
        return None
    return int(listeners_str.replace(',', ''))

# The full pipeline function:
def score_artists_from_playlist(playlist_url):
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(
        client_id=SPOTIFY_CLIENT_ID,
        client_secret=SPOTIFY_CLIENT_SECRET
    ))

    # Get artists from playlist
    artist_counts = get_artists_from_playlist(playlist_url, sp)
    artist_names = list(artist_counts.keys())

    # Get artist URLs for scraping (with caching)
    artist_urls = get_artist_urls(artist_names, sp)

    # Fetch monthly listeners asynchronously (with caching)
    data = asyncio.run(fetch_monthly_listeners(artist_urls))

    # Build DataFrame
    df = pd.DataFrame(data, columns=["Artist", "Listeners"])
    df['ListenersInt'] = df['Listeners'].apply(parse_listeners)
    df = df.dropna(subset=['ListenersInt'])

    # Calculate indie scores
    df['IndieScore'] = indie_score(df['ListenersInt'].tolist())
    dict_df = pd.DataFrame(list(artist_counts.items()), columns=['Artist', 'n'])
    df = df[['Artist', 'ListenersInt', 'IndieScore']].sort_values(by='IndieScore', ascending=False)
    df = df.merge(dict_df, on='Artist', how='left')

    return df

def get_artist_image_from_url(artist_url, sp):
    # Extract artist ID from URL
    match = re.search(r'artist/([a-zA-Z0-9]+)', artist_url)
    if not match:
        raise ValueError("Invalid Spotify artist URL")
    artist_id = match.group(1)

    # Get artist info
    artist = sp.artist(artist_id)
    images = artist.get('images', [])
    if images:
        # Usually images[0] is largest, images[-1] smallest
        return images[0]['url']
    return None

def extract_playlist_id(playlist_url):
    # Handles both with and without query parameters
    return playlist_url.split("/")[-1].split("?")[0]