import math


def get_kl_weight(
    current,
    total,
    min_beta: float = 0.0,
    max_beta: float = 1.0,
    scaling_factor: float = 5.0,
    method: str = "exp_annealing",
    warmup_fraction: float = 0.2,
):
    """Return a bounded KL weight using completed epochs or optimizer steps.

    ``linear`` ramps over ``warmup_fraction`` of training (zero skips warmup).
    ``exp_annealing`` is a normalized concave exponential. The existing
    ``exp-convex_annealing`` curve interpolates geometrically using
    ``progress ** scaling_factor``; a zero minimum uses a relative 1e-12
    floor inside the interval, while both endpoints remain exact.

    Unknown horizons (None, non-positive, or infinite) use ``max_beta``.
    Bounds must be finite with 0 <= min_beta <= max_beta. Exponential
    scaling factors must be positive; linear warmup fractions lie in [0, 1].
    """
    methods = {"none", "linear", "exp_annealing", "exp-convex_annealing"}
    if method not in methods:
        raise ValueError(
            f"Unknown KL weight method {method!r}; choose from {sorted(methods)}."
        )

    min_beta, max_beta = float(min_beta), float(max_beta)
    if not (math.isfinite(min_beta) and math.isfinite(max_beta)) or not (
        0 <= min_beta <= max_beta
    ):
        raise ValueError("KL weights must be finite with 0 <= min_beta <= max_beta.")

    if method == "none":
        return max_beta

    if method == "linear":
        warmup_fraction = float(warmup_fraction)
        if not math.isfinite(warmup_fraction) or not 0 <= warmup_fraction <= 1:
            raise ValueError("warmup_fraction must be finite and between 0 and 1.")
    else:
        scaling_factor = float(scaling_factor)
        if not math.isfinite(scaling_factor) or scaling_factor <= 0:
            raise ValueError("scaling_factor must be finite and positive.")

    if total is None:
        return max_beta
    total = float(total)
    if math.isnan(total):
        raise ValueError("total must not be NaN.")
    if not math.isfinite(total) or total <= 0:
        return max_beta

    current = float(current)
    if not math.isfinite(current):
        raise ValueError("current must be finite.")
    progress = max(0.0, min(current / total, 1.0))

    if method == "linear":
        if warmup_fraction == 0:
            return max_beta
        ramp = min(progress / warmup_fraction, 1.0)
        weight = min_beta + (max_beta - min_beta) * ramp
    else:
        if progress == 0 or min_beta == max_beta:
            return min_beta
        if progress == 1:
            return max_beta
        if method == "exp_annealing":
            ramp = math.expm1(-scaling_factor * progress) / math.expm1(-scaling_factor)
            weight = min_beta + (max_beta - min_beta) * ramp
        else:
            log_hi = math.log(max_beta)
            log_lo = math.log(min_beta) if min_beta > 0 else log_hi + math.log(1e-12)
            ramp = progress**scaling_factor
            weight = math.exp((1 - ramp) * log_lo + ramp * log_hi)

    return float(max(min_beta, min(weight, max_beta)))
