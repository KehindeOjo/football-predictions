"""
Fully automated daily pipeline: fetch -> store -> predict -> publish.

This is the "AI handles everything" entry point — run once a day (via
cron or the included GitHub Actions workflow) and it does the whole
job with no manual intervention:

  1. Fetch newly completed results from API-Football, store in SQLite.
  2. Fetch upcoming fixtures, store in SQLite.
  3. Refit the model on all stored results.
  4. Predict every upcoming fixture.
  5. Store predictions in SQLite AND export them as JSON/CSV for
     publishing (Power BI, a website, a WhatsApp digest — whatever the
     front end ends up being, it can just read these files or query
     the DB directly).

Configure the leagues to cover in LEAGUES below. Everything else is
automatic.
"""

import json
import pandas as pd

import db_utils
import accuracy_tracker
from fixtures_api import get_results, get_upcoming_fixtures
from dixon_coles_model import fit_dixon_coles

MODEL_VERSION = "dixon-coles-v1"

# Add/remove leagues here — this is the only manual configuration needed.
LEAGUES = {
    39: "English Premier League",
    140: "Spanish La Liga",
    135: "Italian Serie A",
}
SEASON = 2025  # year the season started


def run_league(conn, league_id: int, league_name: str):
    print(f"\n=== {league_name} (league_id={league_id}) ===")

    print("Fetching completed results...")
    results = get_results(league_id, SEASON)
    results["home_goals"] = results["home_goals"]
    results["away_goals"] = results["away_goals"]
    db_utils.upsert_matches(conn, results, league_id, status="played")
    print(f"Stored {len(results)} completed matches.")

    print("Fetching upcoming fixtures...")
    upcoming = get_upcoming_fixtures(league_id, SEASON, next_n=15)
    db_utils.upsert_matches(conn, upcoming, league_id, status="scheduled")
    print(f"Stored {len(upcoming)} upcoming fixtures.")

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


def main():
    db_utils.init_db()
    all_predictions = []

    with db_utils.get_connection() as conn:
        for league_id, league_name in LEAGUES.items():
            preds = run_league(conn, league_id, league_name)
            for p in preds:
                p["league"] = league_name
            all_predictions.extend(preds)

    # Publish: JSON for any downstream consumer, CSV for Power BI / Excel.
    with open("predictions_output.json", "w") as f:
        json.dump(all_predictions, f, indent=2)

    pd.DataFrame(all_predictions).to_csv("predictions_output.csv", index=False)

    print(f"\nDone. {len(all_predictions)} predictions published to "
          f"predictions_output.json / predictions_output.csv and predictions.db")

    # Score every prediction whose match has now been played, and publish
    # an up-to-date accuracy report — this is what turns "trust me" into
    # a number you can show him.
    print("\nUpdating accuracy report...")
    accuracy_tracker.run()


if __name__ == "__main__":
    main()
