"""
Lightweight, frequent live-score refresh.

Deliberately separate from pipeline.py: this does NOT touch training
data, does NOT refit the model, and does NOT need the database. It
just asks "what's the status/score of today's matches, right now?"
and republishes that as live_today.json. This is what lets it run
every ~15 minutes (see live_scores.yml) without burning through the
API rate limit or taking long enough to overlap with itself.

The front end (today.html) merges this against predictions_today.json
by fixture_id to show the model's pre-match prediction alongside the
actual live status/score.
"""

import json

from fixtures_api import get_today_status
from pipeline import LEAGUES


def main():
    all_rows = []
    for league_id, league_name in LEAGUES.items():
        try:
            df = get_today_status(league_id)
        except RuntimeError as e:
            print(f"Skipping {league_name}: {e}")
            continue
        for _, row in df.iterrows():
            r = row.to_dict()
            r["league"] = league_name
            all_rows.append(r)

    with open("live_today.json", "w") as f:
        json.dump(all_rows, f, indent=2)

    print(f"Live update: {len(all_rows)} matches today across {len(LEAGUES)} leagues.")


if __name__ == "__main__":
    main()
