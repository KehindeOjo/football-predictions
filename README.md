# Football Prediction Model — Starter Project

A working starting point for a football (soccer) match prediction system,
similar in spirit to footballpredictionai.com: match outcome probabilities,
a "top pick," BTTS, and Over/Under 2.5 for any fixture.

## What's in here

| File | Purpose |
|---|---|
| `dixon_coles_model.py` | Core model: fits attack/defense strength per team from historical results, predicts full scoreline probabilities for any fixture. |
| `run_demo_predictions.py` | End-to-end demo using a free public-domain dataset: pulls a real EPL season, fits the model, prints predictions for sample fixtures. Runs as-is, no API key needed. |
| `fixtures_api.py` | Real fixtures client (API-Football / api-sports.io): fetches historical results to train on and upcoming fixtures to predict. |
| `run_live_predictions.py` | Same pipeline as the demo, but pulls real historical + upcoming fixtures via `fixtures_api.py`. One-off run, needs an API key. |
| `db_utils.py` | SQLite storage layer — stores matches and predictions, no server required. |
| `pipeline.py` | **The full automation.** Fetch → store → refit → predict → publish → score accuracy, for every configured league, in one run. This is what the daily job executes. |
| `accuracy_tracker.py` | Resolves past predictions against actual results once matches are played, and publishes an accuracy report (overall, by league, by model version). Runs automatically at the end of every `pipeline.py` run. |
| `.github/workflows/daily_predictions.yml` | Runs `pipeline.py` automatically every day via GitHub Actions — no manual steps once set up. |
| `schema.sql` | Reference Postgres/MySQL schema, in case the project later moves off SQLite onto a real DB server (e.g. for Power BI to query directly). |

Run the free demo (no API key needed):

```bash
pip install pandas numpy scipy
python run_demo_predictions.py
```

Run the live pipeline (needs an API key — see "Real fixtures API" below):

```bash
export API_FOOTBALL_KEY="your-key-here"
python run_live_predictions.py
```

## Real fixtures API

`fixtures_api.py` uses **API-Football** (api-sports.io):

- Free tier: 100 requests/day, all endpoints, no card required — enough
  to prototype and even run a small daily job.
- Paid tiers start around $19–50/month for higher request volume if you
  scale to more leagues or more frequent polling.
- Sign up at api-football.com or dashboard.api-football.com, grab your
  key, and set it as the `API_FOOTBALL_KEY` environment variable (never
  hardcode it in the script).
- League IDs (39 = EPL, 140 = La Liga, etc.) are listed in
  `fixtures_api.py`'s docstring.

Fetching a full season of results is cheap (1–4 requests, paginated).
Don't re-fetch on every prediction — pull once, store in the `matches`
table via `schema.sql` (which now has an `external_fixture_id` column
for exactly this), and only fetch new/changed fixtures after that.

**Note on testing:** `fixtures_api.py` was written directly against
API-Football's published v3 documentation and its request/response
parsing was verified against a mocked payload, but it hasn't been
run against the live API — the sandbox this project was built in can't
reach third-party APIs. Test it with a real key in your own environment
before relying on it.

## How the model works

This is the **Dixon-Coles model** (Dixon & Coles, 1997), the standard
statistical baseline for football prediction:

1. Every team gets an **attack strength** and **defense strength**,
   estimated from historical goals scored/conceded.
2. A league-wide **home advantage** factor is estimated alongside them.
3. For any fixture, expected goals for each side come from combining
   attack × opponent's defense × home advantage.
4. Goals are modelled as Poisson-distributed, with a small correlation
   correction (`rho`) for low-scoring results (0-0, 1-0, 0-1, 1-1),
   which plain independent-Poisson models get slightly wrong.
5. More recent matches are weighted more heavily (exponential decay,
   configurable half-life) — form matters more than results from a
   year ago.

From the resulting scoreline probability grid, you get 1X2, BTTS, and
Over/Under markets for free — they're all just sums over the grid.

## Recommended path to a full product

**Phase 1 — prove the model (done)**
`dixon_coles_model.py` + `run_demo_predictions.py`, verified against a
real EPL season.

**Phase 2 — real data pipeline (done)**
`fixtures_api.py` connects to a live provider (API-Football).

**Phase 3 — full automation, no manual work (done)**
This is what "AI handles everything" means in practice: nobody manually
pulls data, runs the model, or publishes picks. `pipeline.py` +
`db_utils.py` + the GitHub Actions workflow do the whole cycle daily:

1. **Fetch** — new completed results + upcoming fixtures for every
   league listed in `pipeline.py`'s `LEAGUES` dict.
2. **Store** — into `predictions.db` (SQLite — no server to manage).
3. **Refit** — the model on all stored results (always up to date,
   automatically incorporates the latest form).
4. **Predict** — every upcoming fixture.
5. **Publish** — writes `predictions_output.json` and
   `predictions_output.csv` (Power BI / Excel can point straight at the
   CSV), and commits them back to the repo so the latest picks are
   always sitting in a known place.

**Setting up the automation:**

1. Push this project to a GitHub repo.
2. In the repo's Settings → Secrets → Actions, add a secret named
   `API_FOOTBALL_KEY` with your API-Football key.
3. That's it — `.github/workflows/daily_predictions.yml` runs it every
   day at 06:00 UTC (adjust the cron schedule as needed), with no
   further manual steps. You can also trigger it on demand from the
   Actions tab.

To run it locally instead of via GitHub Actions (e.g. on your own
server with cron):

```bash
export API_FOOTBALL_KEY="your-key-here"
python pipeline.py
```

**Phase 4 — improve accuracy**
Add a gradient-boosting model (XGBoost/LightGBM) trained on engineered
features (recent form, head-to-head, rest days, injuries if available)
and blend it with the Dixon-Coles output. `accuracy_tracker.py` (see
below) already gives you real numbers to compare model versions against
— don't guess which one is better, check `accuracy_report.json`.

## Accuracy tracking (done)

`accuracy_tracker.py` runs automatically at the end of every
`pipeline.py` run. It joins every past prediction against the actual
final score (once the match has been played) and reports:

- Outcome accuracy (was the top pick right?)
- BTTS accuracy
- Over/Under 2.5 accuracy
- All three broken down by league and by model version

Output: `accuracy_report.json` (summary) and `accuracy_detail.csv`
(every resolved prediction, for digging into specifics). This is the
number to show him as proof the model is working — not a one-off
backtest, but a live, always-current track record.

**Phase 5 — front end**
`predictions_output.csv` / the SQLite DB are ready to be consumed by:
- A Power BI dashboard (fastest to ship, plays to your existing strengths)
- A simple web front end reading the JSON
- A daily WhatsApp/Telegram digest built from the same file

**Optional — value bets**
Only needed if the client specifically wants mispriced-bet flagging.
Requires live bookmaker odds (the `odds` table is there for this) and
comparing your model's probability to the market's implied probability
from the odds.

## Notes on scope

Don't try to cover 145 leagues on day one — that's what the reference
site does after years of iteration. Pick the 1–2 leagues the client
actually cares about, get accuracy and the pipeline solid, then expand.
