"""
SQLite storage layer for the automated pipeline.

Why SQLite instead of the schema.sql (Postgres/MySQL) version: full
automation with "no manual work" means no server to provision or
maintain. SQLite is a single file, needs no setup, and GitHub Actions
(or any cron box) can read/write it directly. The table structure
mirrors schema.sql closely — if the project later needs a real DB
server (e.g. for the Power BI dashboard to query directly), migrating
is a straight port.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager

DB_PATH = "predictions.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS teams (
    team_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT UNIQUE NOT NULL,
    league_id   INTEGER
);

CREATE TABLE IF NOT EXISTS matches (
    match_id             INTEGER PRIMARY KEY AUTOINCREMENT,
    external_fixture_id  INTEGER UNIQUE,
    league_id            INTEGER NOT NULL,
    match_date            TEXT NOT NULL,
    home_team             TEXT NOT NULL,
    away_team             TEXT NOT NULL,
    home_goals            INTEGER,
    away_goals            INTEGER,
    status                TEXT DEFAULT 'scheduled'
);

CREATE TABLE IF NOT EXISTS predictions (
    prediction_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    external_fixture_id   INTEGER NOT NULL,
    model_version         TEXT NOT NULL,
    predicted_at           TEXT DEFAULT CURRENT_TIMESTAMP,
    home_team              TEXT NOT NULL,
    away_team              TEXT NOT NULL,
    match_date             TEXT NOT NULL,
    expected_goals_home    REAL,
    expected_goals_away    REAL,
    prob_home_win          REAL,
    prob_draw              REAL,
    prob_away_win          REAL,
    prob_btts_yes           REAL,
    prob_over_2_5           REAL,
    top_pick                TEXT,
    top_pick_confidence     REAL,
    UNIQUE(external_fixture_id, model_version)
);
"""


@contextmanager
def get_connection(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(db_path: str = DB_PATH):
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)


def upsert_matches(conn, matches_df, league_id: int, status: str):
    """Insert or update matches (results or upcoming fixtures)."""
    for _, row in matches_df.iterrows():
        conn.execute(
            """
            INSERT INTO matches
                (external_fixture_id, league_id, match_date, home_team, away_team,
                 home_goals, away_goals, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(external_fixture_id) DO UPDATE SET
                home_goals = excluded.home_goals,
                away_goals = excluded.away_goals,
                status = excluded.status
            """,
            (
                int(row["fixture_id"]),
                league_id,
                row["date"],
                row["home_team"],
                row["away_team"],
                row.get("home_goals"),
                row.get("away_goals"),
                status,
            ),
        )


def upsert_predictions(conn, predictions: list[dict], model_version: str):
    for p in predictions:
        conn.execute(
            """
            INSERT INTO predictions
                (external_fixture_id, model_version, home_team, away_team, match_date,
                 expected_goals_home, expected_goals_away, prob_home_win, prob_draw,
                 prob_away_win, prob_btts_yes, prob_over_2_5, top_pick, top_pick_confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(external_fixture_id, model_version) DO UPDATE SET
                expected_goals_home = excluded.expected_goals_home,
                expected_goals_away = excluded.expected_goals_away,
                prob_home_win = excluded.prob_home_win,
                prob_draw = excluded.prob_draw,
                prob_away_win = excluded.prob_away_win,
                prob_btts_yes = excluded.prob_btts_yes,
                prob_over_2_5 = excluded.prob_over_2_5,
                top_pick = excluded.top_pick,
                top_pick_confidence = excluded.top_pick_confidence,
                predicted_at = CURRENT_TIMESTAMP
            """,
            (
                p["fixture_id"],
                model_version,
                p["home_team"],
                p["away_team"],
                p["date"],
                p["expected_goals_home"],
                p["expected_goals_away"],
                p["prob_home_win"],
                p["prob_draw"],
                p["prob_away_win"],
                p["prob_btts_yes"],
                p["prob_over_2_5"],
                p["top_pick"],
                p["top_pick_confidence"],
            ),
        )


def load_results_for_training(conn, league_id: int):
    """Pull all completed matches from the DB, shaped for fit_dixon_coles()."""
    import pandas as pd

    return pd.read_sql_query(
        """
        SELECT match_date AS date, home_team, away_team, home_goals, away_goals
        FROM matches
        WHERE league_id = ? AND status = 'played' AND home_goals IS NOT NULL
        """,
        conn,
        params=(league_id,),
    )
