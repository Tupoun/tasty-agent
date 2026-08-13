from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

from tastytrade.dxfeed import Greeks
from tastytrade.instruments import FutureOption, Option

GREEKS_NUMERIC_FIELDS = ("price", "volatility", "delta", "gamma", "theta", "vega", "rho")


def is_monthly_expiration(expiration: date) -> bool:
    """Return whether expiration is the standard monthly (3rd Friday of its month)."""
    first = expiration.replace(day=1)
    friday_offset = (4 - first.weekday()) % 7
    third_friday = first + timedelta(days=friday_offset + 14)
    return expiration == third_friday


def select_expiration_from_dte_range(
    available_expirations: list[date],
    min_dte: int,
    max_dte: int,
    today: date,
) -> date:
    """Select the best expiration within [min_dte, max_dte] DTE.

    Prefers standard monthly expirations (3rd Friday) closest to the center of
    the range; falls back to the closest candidate if none are monthly.
    """
    min_date = today + timedelta(days=min_dte)
    max_date = today + timedelta(days=max_dte)
    candidates = [exp for exp in available_expirations if min_date <= exp <= max_date]

    if not candidates:
        nearby = sorted(
            available_expirations,
            key=lambda exp: min(abs((exp - min_date).days), abs((exp - max_date).days)),
        )[:5]
        raise ValueError(
            f"No expirations found between {min_dte}-{max_dte} DTE. Nearest available: {nearby}"
        )

    target_date = today + timedelta(days=(min_dte + max_dte) // 2)
    monthlies = [exp for exp in candidates if is_monthly_expiration(exp)]
    pool = monthlies or candidates
    return min(pool, key=lambda exp: abs((exp - target_date).days))


def validate_target_deltas(target_deltas: list[float]) -> None:
    """Validate sign-convention target deltas: positive=call side, negative=put side."""
    if not target_deltas:
        raise ValueError("target_deltas is required")
    for target in target_deltas:
        if target == 0:
            raise ValueError(
                "target_delta = 0 is not valid. Positive targets search calls, negative targets search puts."
            )


def has_usable_greeks(greek: Greeks) -> bool:
    """Return whether every Greek on this strike is a finite number.

    DXLink publishes NaN Greeks for strikes the pricing model has no output for
    (deep wings, no open interest). Such a strike cannot be scored against a
    target delta -- NaN loses every comparison, so min() keeps whichever
    candidate it saw first -- and compact_value() refuses to render it. Treat it
    as not a candidate, the same way a strike without delta data is skipped.
    """
    for field in GREEKS_NUMERIC_FIELDS:
        value = getattr(greek, field, None)
        if isinstance(value, Decimal) and not value.is_finite():
            return False
    return True


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
            if greek is None or greek.delta is None or not has_usable_greeks(greek):
                continue
            candidates.append((abs(float(greek.delta) - target), option, greek))

        if not candidates:
            raise ValueError(f"No {side_label} strikes with delta data found for target {target}")

        _, option, greek = min(candidates, key=lambda item: item[0])
        matches.append((target, option, greek))

    return matches
