import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List


@dataclass
class EloConfig:
    k_factor: float = 24.0
    initial_rating: float = 1500.0


@dataclass
class EloPlayer:
    name: str
    rating: float


@dataclass
class EloGame:
    white: str
    black: str
    result: str  # "1-0", "0-1", or "1/2-1/2"


@dataclass
class EloState:
    ratings: Dict[str, float] = field(default_factory=dict)

    def ensure_player(self, name: str, initial_rating: float) -> None:
        if name not in self.ratings:
            self.ratings[name] = initial_rating

    def to_json(self) -> str:
        return json.dumps(self.ratings, indent=2, sort_keys=True)


class EloSystem:
    def __init__(self, config: EloConfig) -> None:
        self.config = config
        self.state = EloState()

    def _expected_score(self, rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))

    def _score_from_result(self, result: str) -> float:
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return 0.0
        if result == "1/2-1/2":
            return 0.5
        raise ValueError(f"Unknown result: {result}")

    def update_from_games(self, games: Iterable[EloGame]) -> None:
        for game in games:
            self.state.ensure_player(game.white, self.config.initial_rating)
            self.state.ensure_player(game.black, self.config.initial_rating)

            rating_white = self.state.ratings[game.white]
            rating_black = self.state.ratings[game.black]

            expected_white = self._expected_score(rating_white, rating_black)
            expected_black = self._expected_score(rating_black, rating_white)
            score_white = self._score_from_result(game.result)
            score_black = 1.0 - score_white

            rating_white += self.config.k_factor * (score_white - expected_white)
            rating_black += self.config.k_factor * (score_black - expected_black)

            self.state.ratings[game.white] = rating_white
            self.state.ratings[game.black] = rating_black

    def rating_table(self) -> List[EloPlayer]:
        return [
            EloPlayer(name=name, rating=rating)
            for name, rating in sorted(self.state.ratings.items(), key=lambda item: item[1], reverse=True)
        ]
