"""
Fully automated daily pipeline: fetch -> store -> predict -> publish.

This is the "AI handles everything" entry point — run once a day (via
cron or the included GitHub Actions workflow) and it does the whole
job with no manual intervention:

  1. Fetch newly completed results from football-data.org, store in SQLite.
     Pulls both the previous completed season AND the current season's
     results-so-far, so there's always enough history to fit a model
     even in the first weeks of a new season.
  2. Fetch upcoming fixtures for the current season, store in SQLite.
  3. Refit the model on all stored results.
  4. Predict every upcoming fixture.
  5. Publish several JSON files for the front end, each already
     filtered to what that page needs:
       - predictions_today.json      fixtures dated today
       - predictions_tomorrow.json   fixtures dated tomorrow
       - predictions_upcoming.json   fixtures beyond tomorrow
       - resolved_predictions.json   past predictions vs actual results,
                                      with correctness flags per market
       - predictions_output.json/csv the full raw list (Power BI / Excel)
     A separate, much more frequent job (live_update.py) refreshes
     live scores for today's matches only — see live_scores.yml.

Configure the leagues to cover in LEAGUES below. Everything else is
automatic.
"""

import json
from datetime import datetime, timedelta, timezone

import pandas as pd

import db_utils
import accuracy_tracker
from fixtures_api import get_results, get_upcoming_fixtures
from dixon_coles_model import fit_dixon_coles

MODEL_VERSION = "dixon-coles-v1"

# Add/remove leagues here — this is the only manual configuration needed.
# IDs must be present in fixtures_api.LEAGUE_ID_TO_CODE.
LEAGUES = {
    39: "English Premier League",
    140: "Spanish La Liga",
    135: "Italian Serie A",
}

SEASON = 2026  # year the CURRENT season started — used for upcoming fixtures
TRAINING_SEASONS = [SEASON - 1, SEASON]  # previous + current, for model fitting


def run_league(conn, league_id: int, league_name: str):
    print(f"\n=== {league_name} (league_id={league_id}) ===")

    print("Fetching completed results...")
    results_frames = []
    for season in TRAINING_SEASONS:
        try:
            season_results = get_results(league_id, season)
        except RuntimeError as e:
            print(f"  Skipping season {season}: {e}")
            continue
        print(f"  Season {season}: {len(season_results)} completed matches.")
        if not season_results.empty:
            results_frames.append(season_results)

    if not results_frames:
        print(f"No completed results available for {league_name} — skipping.")
        return []

    results = pd.concat(results_frames, ignore_index=True)
    db_utils.upsert_matches(conn, results, league_id, status="played")
    print(f"Stored {len(results)} completed matches total.")

    print("Fetching upcoming fixtures...")
    try:
        upcoming = get_upcoming_fixtures(league_id, SEASON, next_n=15)
    except RuntimeError as e:
        print(f"Skipping upcoming fixtures for {league_name}: {e}")
        return []
    db_utils.upsert_matches(conn, upcoming, league_id, status="scheduled")
    print(f"Stored {len(upcoming)} upcoming fixtures.")

    if upcoming.empty:
        print("No upcoming fixtures yet (season may not have a published schedule) — skipping.")
        return []

    training_data = db_utils.load_results_for_training(conn, league_id)
    if len(training_data) < 20:
        print("Not enough completed matches yet to fit a reliable model — skipping.")
        return []

    print(f"Fitting model on {len(training_data)} matches...")
    model = fit_dixon_coles(training_data, half_life_days=180)

    predictions = []
    for _, row in upcoming.iterrows():
        try:
            pred = model.predict_match(row["home_team"], row["away_team"])
        except ValueError:
            continue  # promoted/relegated team with no history yet in this window
        pred["fixture_id"] = int(row["fixture_id"])
        pred["date"] = row["date"]
        predictions.append(pred)

    db_utils.upsert_predictions(conn, predictions, MODEL_VERSION)
    print(f"Generated and stored {len(predictions)} predictions.")
    return predictions


def _predicted_result(row) -> str:
    probs = {
        "home_win": row["prob_home_win"],
        "draw": row["prob_draw"],
        "away_win": row["prob_away_win"],
    }
    return max(probs, key=probs.get)


def build_resolved_predictions(conn) -> list[dict]:
    """
    Every past prediction joined against what actually happened,
    with correctness computed straight from the raw probabilities
    (not from top_pick's text, since its exact wording/market isn't
    guaranteed) — this is what powers Yesterday, All Dates history,
    and the rolling accuracy stats page.
    """
    resolved = db_utils.load_resolved(conn)
    if resolved.empty:
        return []

    resolved["league"] = resolved["league_id"].map(LEAGUES)

    resolved["actual_result"] = [
        "home_win" if h > a else ("away_win" if h < a else "draw")
        for h, a in zip(resolved["actual_home_goals"], resolved["actual_away_goals"])
    ]
    resolved["predicted_result"] = resolved.apply(_predicted_result, axis=1)
    resolved["result_correct"] = resolved["predicted_result"] == resolved["actual_result"]

    resolved["actual_btts"] = (resolved["actual_home_goals"] > 0) & (resolved["actual_away_goals"] > 0)
    resolved["predicted_btts"] = resolved["prob_btts_yes"] >= 0.5
    resolved["btts_correct"] = resolved["actual_btts"] == resolved["predicted_btts"]

    total_goals = resolved["actual_home_goals"] + resolved["actual_away_goals"]
    resolved["actual_over_2_5"] = total_goals > 2.5
    resolved["predicted_over_2_5"] = resolved["prob_over_2_5"] >= 0.5
    resolved["over_2_5_correct"] = resolved["actual_over_2_5"] == resolved["predicted_over_2_5"]

    # Keep the published file to a sane size -- last 90 days is plenty
    # for "Yesterday", "All Dates" history, and 30-day rolling stats.
    cutoff = (datetime.now(timezone.utc).date() - timedelta(days=90)).isoformat()
    resolved = resolved[resolved["date"] >= cutoff]

    return json.loads(resolved.to_json(orient="records"))


def publish(conn, all_predictions: list[dict]):
    today = datetime.now(timezone.utc).date()
    tomorrow_s = (today + timedelta(days=1)).isoformat()
    today_s = today.isoformat()

    todays = [p for p in all_predictions if p.get("date") == today_s]
    tomorrows = [p for p in all_predictions if p.get("date") == tomorrow_s]
    later = [p for p in all_predictions if p.get("date", "") > tomorrow_s]

    with open("predictions_today.json", "w") as f:
        json.dump(todays, f, indent=2)
    with open("predictions_tomorrow.json", "w") as f:
        json.dump(tomorrows, f, indent=2)
    with open("predictions_upcoming.json", "w") as f:
        json.dump(later, f, indent=2)
    print(f"Bucketed: {len(todays)} today, {len(tomorrows)} tomorrow, {len(later)} later.")

    resolved = build_resolved_predictions(conn)
    with open("resolved_predictions.json", "w") as f:
        json.dump(resolved, f, indent=2)
    print(f"Published {len(resolved)} resolved predictions (last 90 days).")

    # Full raw list, unchanged -- kept for Power BI / Excel / any other consumer.
    with open("predictions_output.json", "w") as f:
        json.dump(all_predictions, f, indent=2)
    pd.DataFrame(all_predictions).to_csv("predictions_output.csv", index=False)


def main():
    db_utils.init_db()
    all_predictions = []

    with db_utils.get_connection() as conn:
        for league_id, league_name in LEAGUES.items():
            preds = run_league(conn, league_id, league_name)
            for p in preds:
                p["league"] = league_name
            all_predictions.extend(preds)

        publish(conn, all_predictions)

    print(f"\nDone. {len(all_predictions)} predictions published.")

    # Score every prediction whose match has now been played, and publish
    # an up-to-date accuracy report — this is what turns "trust me" into
    # a number you can show him.
    print("\nUpdating accuracy report...")
    accuracy_tracker.run()


if __name__ == "__main__":
    main()
