"""Method registry and variant parsing for distillation evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from compression.cli.common import resolve_methods


_ALLOWED_VARIANT_KEYS = {
    "calib_n",
    "calib_batch_size",
    "calib_seed",
    "include_affine_gamma",
}


@dataclass(frozen=True)
class MethodSpec:
    """Resolved method configuration for one pipeline run.

    Variant syntax:
      <method>
      <method>:key=value;key=value
    """

    requested: str
    compress_method: str
    method_params: dict[str, Any] = field(default_factory=dict)
    calibration_overrides: dict[str, int] = field(default_factory=dict)
    include_affine_gamma: bool | None = None

    @property
    def run_method_name(self) -> str:
        return self.requested

    def to_config(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "compress_method": self.compress_method,
            "method_params": dict(self.method_params),
            "calibration_overrides": dict(self.calibration_overrides),
            "include_affine_gamma": self.include_affine_gamma,
        }


def _parse_bool(value: str) -> bool:
    lower = value.strip().lower()
    if lower in {"1", "true", "yes", "y", "on"}:
        return True
    if lower in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def _parse_variant_params(raw: str) -> dict[str, str]:
    params: dict[str, str] = {}
    for entry in raw.split(";"):
        token = entry.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                f"Malformed variant token {token!r}. Expected key=value entries."
            )
        key, value = token.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in _ALLOWED_VARIANT_KEYS:
            raise ValueError(
                f"Unsupported variant key {key!r}. "
                f"Allowed keys: {sorted(_ALLOWED_VARIANT_KEYS)}"
            )
        if not value:
            raise ValueError(f"Variant key {key!r} is missing a value.")
        params[key] = value
    return params


def parse_method_spec(token: str) -> MethodSpec:
    token = token.strip()
    if not token:
        raise ValueError("Method token cannot be empty")

    if ":" not in token:
        resolve_methods(token)  # validate baseline name
        return MethodSpec(requested=token, compress_method=token)

    method_name, raw_params = token.split(":", 1)
    method_name = method_name.strip()
    resolve_methods(method_name)  # validate baseline name
    parsed = _parse_variant_params(raw_params)

    calibration_overrides: dict[str, int] = {}
    include_affine_gamma: bool | None = None
    method_params: dict[str, Any] = {}
    for key, value in parsed.items():
        if key in {"calib_n", "calib_batch_size", "calib_seed"}:
            parsed_int = int(value)
            if parsed_int <= 0 and key != "calib_seed":
                raise ValueError(f"{key} must be positive, got {parsed_int}")
            if key == "calib_seed" and parsed_int < 0:
                raise ValueError(f"{key} must be non-negative, got {parsed_int}")
            calibration_overrides[key] = parsed_int
            method_params[key] = parsed_int
        elif key == "include_affine_gamma":
            include_affine_gamma = _parse_bool(value)
            method_params[key] = include_affine_gamma

    return MethodSpec(
        requested=token,
        compress_method=method_name,
        method_params=method_params,
        calibration_overrides=calibration_overrides,
        include_affine_gamma=include_affine_gamma,
    )


def resolve_method_specs(methods_arg: str) -> list[MethodSpec]:
    """Resolve method specs while preserving baseline compatibility.

    - "all" returns baseline methods in the canonical order from resolve_methods.
    - comma-separated methods support optional variants via `method:key=value;...`.
    """
    if methods_arg == "all":
        return [MethodSpec(requested=m, compress_method=m) for m in resolve_methods("all")]

    tokens = [x.strip() for x in methods_arg.split(",") if x.strip()]
    if not tokens:
        raise ValueError("No methods specified")
    return [parse_method_spec(token) for token in tokens]
