"""
API-Football (api-sports.io) client.

Handles the two things the prediction pipeline needs from a real
fixtures provider:

  1. Historical results, to train/refit the model
     -> get_results(league_id, season)
  2. Upcoming fixtures, to generate predictions for
     -> get_upcoming_fixtures(league_id, season, next_n)

Both return data already shaped for dixon_coles_model.py, so you can
swap this in for the demo's free JSON loader with no other changes.

## Getting an API key

1. Sign up at https://www.api-football.com/ or https://dashboard.api-football.com/
   (free tier: 100 requests/day, no card required, all endpoints included).
2. Copy your key from the dashboard.
3. Set it as an environment variable rather than hardcoding it:

       export API_FOOTBALL_KEY="your-key-here"

## Common league IDs (season = the year the season starts, e.g. 2025 for 2025/26)

    39   English Premier League
    140  Spanish La Liga
    135  Italian Serie A
    78   German Bundesliga
    61   French Ligue 1
    88   Dutch Eredivisie
    203  Turkish Super Lig
    (Full list: GET https://v3.football.api-sports.io/leagues)

## A note on rate limits

The free tier is 100 requests/day. Fetching a full season of results is
ONE request (paginated in blocks of ~100 fixtures, so a full season is
usually 1-4 requests). Don't call this on every prediction — cache
results in your database (see schema.sql) and only re-fetch new/changed
fixtures.

## A note on testing

This module makes live calls to v3.football.api-sports.io. It could not
be executed inside the sandbox this project was built in (that
environment only allows outbound calls to a fixed list of package-registry
domains, not third-party APIs) — so test it in your own environment
with a real key before wiring it into the pipeline. The request/response
shapes follow API-Football's published v3 documentation.
"""

from __future__ import annotations

import os
import time
import urllib.request
import urllib.parse
import json
import pandas as pd

BASE_URL = "https://v3.football.api-sports.io"


def _get_api_key() -> str:
    key = os.environ.get("API_FOOTBALL_KEY")
    if not key:
        raise RuntimeError(
            "Set the API_FOOTBALL_KEY environment variable with your API-Football key."
        )
    return key


def _request(endpoint: str, params: dict) -> dict:
    """Low-level GET against the API-Football v3 REST API."""
    url = f"{BASE_URL}/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"x-apisports-key": _get_api_key()})
    with urllib.request.urlopen(req) as resp:
        payload = json.loads(resp.read().decode())

    if payload.get("errors"):
        raise RuntimeError(f"API-Football error: {payload['errors']}")
    return payload


def get_results(league_id: int, season: int, max_pages: int = 5) -> pd.DataFrame:
    """
    Fetch completed matches for a league/season, shaped for
    dixon_coles_model.fit_dixon_coles().

    Returns columns: date, home_team, away_team, home_goals, away_goals,
    plus fixture_id (useful as the external key back into your `matches` table).
    """
    rows = []
    page = 1
    while page <= max_pages:
        data = _request(
            "fixtures",
            {
                "league": league_id,
                "season": season,
                "status": "FT",  # full-time only, i.e. completed matches
                "page": page,
            },
        )
        for item in data["response"]:
            rows.append(
                {
                    "fixture_id": item["fixture"]["id"],
                    "date": item["fixture"]["date"][:10],
                    "home_team": item["teams"]["home"]["name"],
                    "away_team": item["teams"]["away"]["name"],
                    "home_goals": item["goals"]["home"],
                    "away_goals": item["goals"]["away"],
                }
            )

        paging = data.get("paging", {"current": 1, "total": 1})
        if paging["current"] >= paging["total"]:
            break
        page += 1
        time.sleep(0.5)  # be polite to the free-tier rate limit

    return pd.DataFrame(rows)


def get_upcoming_fixtures(league_id: int, season: int, next_n: int = 10) -> pd.DataFrame:
    """
    Fetch the next N scheduled (not-yet-played) fixtures for a league.

    Returns columns: fixture_id, date, home_team, away_team — feed
    (home_team, away_team) straight into model.predict_match().
    """
    data = _request(
        "fixtures",
        {"league": league_id, "season": season, "next": next_n},
    )
    rows = []
    for item in data["response"]:
        rows.append(
            {
                "fixture_id": item["fixture"]["id"],
                "date": item["fixture"]["date"][:10],
                "home_team": item["teams"]["home"]["name"],
                "away_team": item["teams"]["away"]["name"],
            }
        )
    return pd.DataFrame(rows)
