"""
Accuracy tracking.

Every prediction the pipeline ever made is already sitting in the
`predictions` table (see db_utils.py), and `matches` gets updated with
real scores every time pipeline.py re-fetches results. This module
joins the two: for every prediction whose match has now been played,
it checks whether the top pick was right, and rolls that up into a
report you can actually show him — hit rate on match outcome, BTTS,
and Over/Under 2.5, overall and by league.

Run standalone any time:
    python accuracy_tracker.py

pipeline.py also calls this automatically at the end of every daily run,
so accuracy_report.json/csv are always current with zero manual work.
"""

import json
import pandas as pd

import db_utils


def resolve_predictions(conn) -> pd.DataFrame:
    """
    Join predictions to their now-known results. Only returns rows for
    matches that have actually been played (home_goals IS NOT NULL).
    """
    query = """
        SELECT
            p.prediction_id,
            p.model_version,
            p.external_fixture_id,
            p.home_team,
            p.away_team,
            p.match_date,
            p.prob_home_win,
            p.prob_draw,
            p.prob_away_win,
            p.prob_btts_yes,
            p.prob_over_2_5,
            p.top_pick,
            m.home_goals,
            m.away_goals,
            m.league_id
        FROM predictions p
        JOIN matches m ON m.external_fixture_id = p.external_fixture_id
        WHERE m.home_goals IS NOT NULL AND m.away_goals IS NOT NULL
    """
    df = pd.read_sql_query(query, conn)
    if df.empty:
        return df

    def actual_outcome(row):
        if row["home_goals"] > row["away_goals"]:
            return "Home Win"
        elif row["home_goals"] < row["away_goals"]:
            return "Away Win"
        return "Draw"

    df["actual_outcome"] = df.apply(actual_outcome, axis=1)
    df["outcome_correct"] = df["top_pick"] == df["actual_outcome"]

    df["actual_btts"] = (df["home_goals"] > 0) & (df["away_goals"] > 0)
    df["btts_predicted_yes"] = df["prob_btts_yes"] >= 0.5
    df["btts_correct"] = df["btts_predicted_yes"] == df["actual_btts"]

    df["actual_over_2_5"] = (df["home_goals"] + df["away_goals"]) > 2.5
    df["over_predicted"] = df["prob_over_2_5"] >= 0.5
    df["over_under_correct"] = df["over_predicted"] == df["actual_over_2_5"]

    return df


def build_report(resolved: pd.DataFrame) -> dict:
    """Summarise accuracy overall and by league/model version."""
    if resolved.empty:
        return {"resolved_predictions": 0, "message": "No completed matches to score yet."}

    overall = {
        "resolved_predictions": int(len(resolved)),
        "outcome_accuracy": round(resolved["outcome_correct"].mean(), 4),
        "btts_accuracy": round(resolved["btts_correct"].mean(), 4),
        "over_under_2_5_accuracy": round(resolved["over_under_correct"].mean(), 4),
    }

    by_league = (
        resolved.groupby("league_id")
        .agg(
            resolved_predictions=("outcome_correct", "count"),
            outcome_accuracy=("outcome_correct", "mean"),
            btts_accuracy=("btts_correct", "mean"),
            over_under_2_5_accuracy=("over_under_correct", "mean"),
        )
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )

    by_model_version = (
        resolved.groupby("model_version")
        .agg(
            resolved_predictions=("outcome_correct", "count"),
            outcome_accuracy=("outcome_correct", "mean"),
        )
        .round(4)
        .reset_index()
        .to_dict(orient="records")
    )

    return {"overall": overall, "by_league": by_league, "by_model_version": by_model_version}


def run(db_path: str = db_utils.DB_PATH):
    with db_utils.get_connection(db_path) as conn:
        resolved = resolve_predictions(conn)
        report = build_report(resolved)

    with open("accuracy_report.json", "w") as f:
        json.dump(report, f, indent=2)

    if not resolved.empty:
        resolved.to_csv("accuracy_detail.csv", index=False)

    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run()
