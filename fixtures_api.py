"""
football-data.org client.

Handles the two things the prediction pipeline needs from a real
fixtures provider:

  1. Historical results, to train/refit the model
     -> get_results(league_id, season)
  2. Upcoming fixtures, to generate predictions for
     -> get_upcoming_fixtures(league_id, season, next_n)

Both return data shaped exactly the same as the previous API-Football
version, so pipeline.py and db_utils.py need NO changes:

  get_results()           -> date, home_team, away_team, home_goals,
                              away_goals, fixture_id
  get_upcoming_fixtures() -> fixture_id, date, home_team, away_team

## Why football-data.org instead of API-Football

API-Football's free tier only allows historical seasons 2022-2024 —
it has no access to the current season, which makes it useless for a
"predict today's/upcoming matches" site. football-data.org's free
tier DOES include current-season data for the competitions below, at
the cost of a stricter rate limit (10 requests/minute) and fewer
leagues covered.

## Getting an API key

1. Sign up at https://www.football-data.org/client/register
   (free, no card required — you get a key immediately by email).
2. Set it as an environment variable rather than hardcoding it:

       export FOOTBALL_DATA_API_KEY="your-key-here"

## League ID mapping

pipeline.py's LEAGUES dict uses API-Football-style numeric IDs. This
module maps those same IDs to football-data.org's competition codes,
so LEAGUES in pipeline.py does not need to change. Only leagues
covered by football-data.org's free tier are included here — if you
add a league ID to pipeline.py that isn't in this map, you'll get a
clear error rather than a silent failure.

    39   -> PL   English Premier League
    140  -> PD   Spanish La Liga
    135  -> SA   Italian Serie A
    78   -> BL1  German Bundesliga
    61   -> FL1  French Ligue 1
    88   -> DED  Dutch Eredivisie
    94   -> PPL  Portuguese Primeira Liga
    2    -> CL   UEFA Champions League

(Full competition list: GET https://api.football-data.org/v4/competitions)

## A note on rate limits

Free tier: 10 requests/minute. Each call here is a single request
(the API returns a whole season's matches in one response, no
pagination needed for these competitions), so a 3-league pipeline run
is well within limits. A small delay + retry-on-429 is included for
safety regardless.

## A note on testing

This module makes live calls to api.football-data.org. It could not
be executed inside the sandbox this project was built in (that
environment only allows outbound calls to a fixed list of
package-registry domains, not third-party APIs) — so test it in your
own environment with a real key before relying on it. The
request/response shapes follow football-data.org's published v4
documentation.
"""

from __future__ import annotations

import os
import time
import urllib.request
import urllib.error
import urllib.parse
import json
import pandas as pd

BASE_URL = "https://api.football-data.org/v4"

# Maps the same numeric league IDs pipeline.py already uses to
# football-data.org's competition codes. Add more here if you add
# leagues to pipeline.py's LEAGUES dict -- check they're covered by
# your plan at https://www.football-data.org/coverage first.
LEAGUE_ID_TO_CODE = {
    39: "PL",
    140: "PD",
    135: "SA",
    78: "BL1",
    61: "FL1",
    88: "DED",
    94: "PPL",
    2: "CL",
}


def _get_api_key() -> str:
    key = os.environ.get("FOOTBALL_DATA_API_KEY")
    if not key:
        raise RuntimeError(
            "Set the FOOTBALL_DATA_API_KEY environment variable with your "
            "football-data.org key (free at https://www.football-data.org/client/register)."
        )
    return key


def _competition_code(league_id: int) -> str:
    code = LEAGUE_ID_TO_CODE.get(league_id)
    if not code:
        raise RuntimeError(
            f"League ID {league_id} isn't mapped to a football-data.org competition "
            f"code. Known IDs: {sorted(LEAGUE_ID_TO_CODE)}. Add it to LEAGUE_ID_TO_CODE "
            f"in fixtures_api.py (check coverage at https://www.football-data.org/coverage first)."
        )
    return code


def _request(path: str, params: dict, max_retries: int = 3) -> dict:
    """Low-level GET against the football-data.org v4 REST API, with
    basic retry on the free tier's rate limit (HTTP 429)."""
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"X-Auth-Token": _get_api_key()})

    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < max_retries:
                time.sleep(6)  # brief backoff, then retry within the 10/min window
                continue
            body = e.read().decode(errors="ignore")
            raise RuntimeError(f"football-data.org error ({e.code}): {body}") from e

    raise RuntimeError("football-data.org error: exceeded retries after repeated 429s")


def get_results(league_id: int, season: int) -> pd.DataFrame:
    """
    Fetch completed matches for a league/season, shaped for
    dixon_coles_model.fit_dixon_coles().

    Returns columns: date, home_team, away_team, home_goals, away_goals,
    plus fixture_id (useful as the external key back into your `matches` table).
    """
    code = _competition_code(league_id)
    data = _request(
        f"/competitions/{code}/matches",
        {"season": season, "status": "FINISHED"},
    )

    rows = []
    for m in data.get("matches", []):
        full_time = m.get("score", {}).get("fullTime", {})
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")
        if home_goals is None or away_goals is None:
            continue  # skip anything without a final score (e.g. abandoned matches)
        rows.append(
            {
                "fixture_id": m["id"],
                "date": m["utcDate"][:10],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )

    time.sleep(1)  # stay comfortably within 10 requests/minute across leagues
    return pd.DataFrame(rows)


def get_upcoming_fixtures(league_id: int, season: int, next_n: int = 10) -> pd.DataFrame:
    """
    Fetch the next N scheduled (not-yet-played) fixtures for a league.

    Returns columns: fixture_id, date, home_team, away_team — feed
    (home_team, away_team) straight into model.predict_match().
    """
    code = _competition_code(league_id)
    data = _request(
        f"/competitions/{code}/matches",
        {"season": season, "status": "SCHEDULED"},
    )

    rows = []
    for m in data.get("matches", []):
        rows.append(
            {
                "fixture_id": m["id"],
                "date": m["utcDate"][:10],
                "home_team": m["homeTeam"]["name"],
                "away_team": m["awayTeam"]["name"],
            }
        )

    # football-data.org doesn't take a "next N" param directly like
    # API-Football did -- it returns all scheduled matches for the
    # season, so sort by date and take the first N ourselves.
    rows.sort(key=lambda r: r["date"])
    time.sleep(1)  # stay comfortably within 10 requests/minute across leagues
    return pd.DataFrame(rows[:next_n])
