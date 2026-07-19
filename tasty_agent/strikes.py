from __future__ import annotations

from typing import Any

from tastytrade.dxfeed import Greeks
from tastytrade.instruments import FutureOption, Option

from tasty_agent.core import compact_value


def validate_target_deltas(target_deltas: list[float]) -> None:
    """Validate sign-convention target deltas: positive=call side, negative=put side."""
    if not target_deltas:
        raise ValueError("target_deltas is required")
    for target in target_deltas:
        if target == 0:
            raise ValueError(
                "target_delta = 0 is not valid. Positive targets search calls, negative targets search puts."
            )


def find_nearest_strikes_by_delta(
    options: list[Option | FutureOption],
    greeks_by_symbol: dict[str, Greeks],
    target_deltas: list[float],
) -> list[tuple[float, Option | FutureOption, Greeks]]:
    """For each target delta, find the option with the closest delta on the matching side.

    Positive target_delta searches calls (call delta is 0 to 1); negative searches
    puts (put delta is -1 to 0). Matching minimizes abs(actual_delta - target_delta).
    """
    calls = [option for option in options if option.option_type.value == "C"]
    puts = [option for option in options if option.option_type.value == "P"]

    matches: list[tuple[float, Option | FutureOption, Greeks]] = []
    for target in target_deltas:
        side_options = calls if target > 0 else puts
        side_label = "call" if target > 0 else "put"

        candidates: list[tuple[float, Option | FutureOption, Greeks]] = []
        for option in side_options:
            greek = greeks_by_symbol.get(option.streamer_symbol)
            if greek is None or greek.delta is None:
                continue
            candidates.append((abs(float(greek.delta) - target), option, greek))

        if not candidates:
            raise ValueError(f"No {side_label} strikes with delta data found for target {target}")

        _, option, greek = min(candidates, key=lambda item: item[0])
        matches.append((target, option, greek))

    return matches


def compact_strike_match(target: float, option: Option | FutureOption, greek: Greeks) -> dict[str, Any]:
    """Format one delta-matched strike as a compact table row."""
    data = greek.model_dump()
    return {
        "target": target,
        "sym": compact_value(data.get("event_symbol")),
        "strike": compact_value(option.strike_price),
        "type": option.option_type.value,
        "delta": compact_value(data.get("delta")),
        "price": compact_value(data.get("price")),
        "iv": compact_value(data.get("volatility")),
        "gamma": compact_value(data.get("gamma")),
        "theta": compact_value(data.get("theta")),
        "vega": compact_value(data.get("vega")),
    }
