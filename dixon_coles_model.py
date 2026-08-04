"""
Dixon-Coles football match prediction model.

Implements the classic Dixon & Coles (1997) extension of the independent
Poisson model for football scores. It estimates, for every team, an
"attack strength" and "defense strength" from historical results, plus a
league-wide home advantage factor and a low-score correlation adjustment
(rho). From those parameters you can simulate the full scoreline
probability grid for any upcoming fixture and derive:

  - 1X2 (home win / draw / away win) probabilities
  - BTTS (both teams to score) probability
  - Over/Under 2.5 goals probability
  - A "confidence" top pick

This is meant as a solid, well-documented starting point — not a
finished product. See README.md for how this fits into the wider
pipeline (data ingestion -> storage -> model -> publishing).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DixonColesModel:
    """Fitted Dixon-Coles model parameters and prediction methods."""

    teams: list
    attack: dict          # team -> attack strength
    defense: dict         # team -> defense strength
    home_advantage: float
    rho: float            # low-score correlation term
    max_goals: int = 8    # scoreline grid size for prediction

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def expected_goals(self, home_team: str, away_team: str) -> tuple[float, float]:
        """Return (expected home goals, expected away goals) for a fixture."""
        lambda_home = np.exp(
            self.attack[home_team] + self.defense[away_team] + self.home_advantage
        )
        lambda_away = np.exp(self.attack[away_team] + self.defense[home_team])
        return lambda_home, lambda_away

    def _tau(self, x: int, y: int, lambda_home: float, lambda_away: float) -> float:
        """Dixon-Coles low-score correction (adjusts 0-0, 1-0, 0-1, 1-1)."""
        rho = self.rho
        if x == 0 and y == 0:
            return 1 - lambda_home * lambda_away * rho
        elif x == 0 and y == 1:
            return 1 + lambda_home * rho
        elif x == 1 and y == 0:
            return 1 + lambda_away * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def scoreline_matrix(self, home_team: str, away_team: str) -> np.ndarray:
        """Full probability grid P(home scores i, away scores j)."""
        lh, la = self.expected_goals(home_team, away_team)
        n = self.max_goals + 1
        home_probs = poisson.pmf(np.arange(n), lh)
        away_probs = poisson.pmf(np.arange(n), la)
        matrix = np.outer(home_probs, away_probs)

        for i in range(2):
            for j in range(2):
                matrix[i, j] *= self._tau(i, j, lh, la)

        matrix /= matrix.sum()  # renormalise after the tau correction
        return matrix

    def predict_match(self, home_team: str, away_team: str) -> dict:
        """Return a full prediction dict for one fixture."""
        for t in (home_team, away_team):
            if t not in self.attack:
                raise ValueError(
                    f"'{t}' not in the fitted teams. "
                    f"Known teams: {sorted(self.teams)}"
                )

        matrix = self.scoreline_matrix(home_team, away_team)
        lh, la = self.expected_goals(home_team, away_team)

        home_win = np.tril(matrix, -1).sum()
        draw = np.trace(matrix)
        away_win = np.triu(matrix, 1).sum()

        # BTTS: both teams score >= 1
        btts_yes = matrix[1:, 1:].sum()

        # Over/Under 2.5 total goals
        total_goals_grid = np.add.outer(
            np.arange(self.max_goals + 1), np.arange(self.max_goals + 1)
        )
        over_2_5 = matrix[total_goals_grid > 2.5].sum()

        outcomes = {"Home Win": home_win, "Draw": draw, "Away Win": away_win}
        top_pick = max(outcomes, key=outcomes.get)

        return {
            "home_team": home_team,
            "away_team": away_team,
            "expected_goals_home": round(lh, 2),
            "expected_goals_away": round(la, 2),
            "prob_home_win": round(home_win, 4),
            "prob_draw": round(draw, 4),
            "prob_away_win": round(away_win, 4),
            "prob_btts_yes": round(btts_yes, 4),
            "prob_over_2_5": round(over_2_5, 4),
            "prob_under_2_5": round(1 - over_2_5, 4),
            "top_pick": top_pick,
            "top_pick_confidence": round(outcomes[top_pick], 4),
        }


def fit_dixon_coles(
    matches: pd.DataFrame,
    half_life_days: float | None = 180,
    reference_date: datetime | None = None,
) -> DixonColesModel:
    """
    Fit a Dixon-Coles model to historical results.

    Parameters
    ----------
    matches : DataFrame with columns
        ['date', 'home_team', 'away_team', 'home_goals', 'away_goals']
    half_life_days : if set, older matches are down-weighted with an
        exponential decay of this half-life (recency matters more in
        football than in most domains — form changes fast). Set to
        None to weight every match equally.
    reference_date : the "as of" date for the decay weighting. Defaults
        to the most recent match date in the data.

    Returns
    -------
    DixonColesModel
    """
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    n_teams = len(teams)
    team_idx = {t: i for i, t in enumerate(teams)}

    if reference_date is None:
        reference_date = pd.to_datetime(matches["date"]).max()

    if half_life_days:
        days_old = (pd.to_datetime(reference_date) - pd.to_datetime(matches["date"])).dt.days
        decay = 0.5 ** (days_old / half_life_days)
        weights = decay.clip(lower=0.01).to_numpy()
    else:
        weights = np.ones(len(matches))

    home_idx = matches["home_team"].map(team_idx).to_numpy()
    away_idx = matches["away_team"].map(team_idx).to_numpy()
    home_goals = matches["home_goals"].to_numpy()
    away_goals = matches["away_goals"].to_numpy()

    # Parameter vector: [attack_1..attack_n, defense_1..defense_n, home_adv, rho]
    # One attack parameter is fixed to 0 for identifiability.
    def unpack(params):
        attack = np.concatenate(([0.0], params[: n_teams - 1]))
        defense = params[n_teams - 1 : 2 * n_teams - 1]
        home_adv = params[2 * n_teams - 1]
        rho = params[2 * n_teams]
        return attack, defense, home_adv, rho

    def neg_log_likelihood(params):
        attack, defense, home_adv, rho = unpack(params)
        lh = np.exp(attack[home_idx] + defense[away_idx] + home_adv)
        la = np.exp(attack[away_idx] + defense[home_idx])

        ll = (
            poisson.logpmf(home_goals, lh) + poisson.logpmf(away_goals, la)
        )

        # low-score correction, applied only where relevant
        tau_adj = np.ones(len(matches))
        for k in range(len(matches)):
            x, y = home_goals[k], away_goals[k]
            if x <= 1 and y <= 1:
                if x == 0 and y == 0:
                    tau_adj[k] = 1 - lh[k] * la[k] * rho
                elif x == 0 and y == 1:
                    tau_adj[k] = 1 + lh[k] * rho
                elif x == 1 and y == 0:
                    tau_adj[k] = 1 + la[k] * rho
                elif x == 1 and y == 1:
                    tau_adj[k] = 1 - rho
        tau_adj = np.clip(tau_adj, 1e-6, None)  # keep log() finite

        ll = ll + np.log(tau_adj)
        return -np.sum(weights * ll)

    x0 = np.zeros(2 * n_teams + 1)  # attack(n-1) + defense(n) + home_adv + rho
    result = minimize(neg_log_likelihood, x0, method="L-BFGS-B")

    attack, defense, home_adv, rho = unpack(result.x)
    attack_dict = {t: attack[i] for t, i in team_idx.items()}
    defense_dict = {t: defense[i] for t, i in team_idx.items()}

    return DixonColesModel(
        teams=teams,
        attack=attack_dict,
        defense=defense_dict,
        home_advantage=home_adv,
        rho=rho,
    )
