-- Football prediction project — core database schema
-- Works as-is on PostgreSQL/MySQL; minor tweaks for SQLite (no ENUM, use TEXT + CHECK).

CREATE TABLE leagues (
    league_id     SERIAL PRIMARY KEY,
    name          VARCHAR(100) NOT NULL,       -- e.g. 'English Premier League'
    country       VARCHAR(100),
    external_code VARCHAR(20)                  -- code used by your data provider, e.g. 'PL'
);

CREATE TABLE teams (
    team_id       SERIAL PRIMARY KEY,
    name          VARCHAR(150) NOT NULL,
    league_id     INT REFERENCES leagues(league_id)
);

-- One row per played match. This is what the model trains on.
CREATE TABLE matches (
    match_id      SERIAL PRIMARY KEY,
    external_fixture_id INT UNIQUE,             -- the provider's fixture id (e.g. API-Football), for dedup on re-fetch
    league_id     INT REFERENCES leagues(league_id),
    match_date    DATE NOT NULL,
    home_team_id  INT REFERENCES teams(team_id),
    away_team_id  INT REFERENCES teams(team_id),
    home_goals    SMALLINT,                    -- NULL if not yet played
    away_goals    SMALLINT,
    status        VARCHAR(20) DEFAULT 'scheduled'  -- scheduled | played | postponed
);

-- One row per fixture per model run. Kept even after the match is played,
-- so you can compare predicted vs actual and track accuracy over time.
CREATE TABLE predictions (
    prediction_id     SERIAL PRIMARY KEY,
    match_id          INT REFERENCES matches(match_id),
    model_version     VARCHAR(50) NOT NULL,     -- e.g. 'dixon-coles-v1', 'xgb-v2'
    predicted_at      TIMESTAMP NOT NULL DEFAULT NOW(),
    expected_goals_home  NUMERIC(4,2),
    expected_goals_away  NUMERIC(4,2),
    prob_home_win     NUMERIC(5,4),
    prob_draw         NUMERIC(5,4),
    prob_away_win     NUMERIC(5,4),
    prob_btts_yes     NUMERIC(5,4),
    prob_over_2_5     NUMERIC(5,4),
    top_pick          VARCHAR(20),
    top_pick_confidence NUMERIC(5,4)
);

-- Optional: bookmaker odds, only needed if you build "value bets"
-- (comparing your model's probability against the market's implied probability).
CREATE TABLE odds (
    odds_id       SERIAL PRIMARY KEY,
    match_id      INT REFERENCES matches(match_id),
    bookmaker     VARCHAR(50),
    home_odds     NUMERIC(6,2),
    draw_odds     NUMERIC(6,2),
    away_odds     NUMERIC(6,2),
    fetched_at    TIMESTAMP DEFAULT NOW()
);

-- Track model accuracy over time — this is what gives the product credibility.
CREATE VIEW prediction_results AS
SELECT
    p.prediction_id,
    p.model_version,
    m.match_date,
    ht.name AS home_team,
    at.name AS away_team,
    m.home_goals,
    m.away_goals,
    p.top_pick,
    p.top_pick_confidence,
    CASE
        WHEN m.home_goals IS NULL THEN NULL
        WHEN m.home_goals > m.away_goals AND p.top_pick = 'Home Win' THEN TRUE
        WHEN m.home_goals = m.away_goals AND p.top_pick = 'Draw' THEN TRUE
        WHEN m.home_goals < m.away_goals AND p.top_pick = 'Away Win' THEN TRUE
        ELSE FALSE
    END AS correct
FROM predictions p
JOIN matches m ON m.match_id = p.match_id
JOIN teams ht ON ht.team_id = m.home_team_id
JOIN teams at ON at.team_id = m.away_team_id;
