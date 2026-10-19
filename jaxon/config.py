"""Configuration: strict YAML loading.

Unknown keys and missing mandatory instrument metadata are errors. Every
numeric strategy default is a PROPOSED ASSUMPTION (see docs/assumptions.md),
not something the source video specified.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

import yaml

ASSET_CLASSES = {"equity", "future", "forex", "crypto"}
VOLUME_TYPES = {"exchange", "consolidated", "tick", "none"}


class ConfigError(ValueError):
    pass


# ---------------------------------------------------------------- sections
@dataclass
class SessionConfig:
    timezone: str                       # exchange timezone, IANA name
    open: str                           # "HH:MM" local; if open >= close the session starts the previous day
    close: str                          # "HH:MM" local; "24:00" allowed
    trading_weekdays: list              # 0=Mon .. 6=Sun, weekday of the session's CLOSE date
    holidays: list = field(default_factory=list)          # ["YYYY-MM-DD", ...] close dates with no session
    early_closes: dict = field(default_factory=dict)      # {"YYYY-MM-DD": "HH:MM"}
    flatten_minutes_before_close: int = 5
    last_entry_minutes_before_close: int = 30
    vwap_reset: str = "session_open"    # only "session_open" supported in v1


@dataclass
class BarsConfig:
    base_interval_minutes: int          # interval of the CSV rows; must divide 5
    timestamp_convention: str = "open"  # "open" or "close" — what the CSV timestamp marks
    naive_timestamp_timezone: Optional[str] = None  # required if CSV timestamps lack an offset
    four_hour_alignment: str = "session_open"       # "session_open" | "utc_epoch"
    drop_outside_session: bool = True
    stale_gap_bars: int = 1             # this many (or more) missing base bars inside a session marks the feed stale
    stale_recovery_bars: int = 6        # fresh base bars required before new entries are allowed again


@dataclass
class CostsConfig:
    spread_ticks: float                 # full bid/ask spread, in ticks; half is paid on each side
    slippage_ticks: float               # extra adverse ticks on market and stop fills
    commission_per_unit: float = 0.0
    commission_pct_notional: float = 0.0
    min_commission_per_order: float = 0.0


@dataclass
class DataSource:
    path: str
    synthetic: bool = False             # declare synthetic fixtures; the loader also detects a SYNTHETIC header
    description: str = ""


@dataclass
class InstrumentConfig:
    symbol: str
    asset_class: str
    currency: str
    tick_size: float
    lot_size: float                     # quantity increment
    min_quantity: float
    value_per_point: float              # cash value of a 1.0 price move per 1 unit of quantity
    volume_type: str                    # exchange | consolidated | tick | none
    session: SessionConfig
    bars: BarsConfig
    costs: CostsConfig
    data: DataSource
    min_notional: float = 0.0
    initial_margin_per_unit: Optional[float] = None
    max_quantity: Optional[float] = None   # e.g. prop-firm contract cap


@dataclass
class RiskConfig:
    risk_per_trade_pct: float = 0.0025
    max_daily_drawdown_pct: float = 0.01
    max_entries_per_session: int = 3
    max_open_positions: int = 1
    max_open_risk_pct: float = 0.005
    max_leverage: float = 1.0           # max notional / equity
    max_margin_usage: float = 1.0       # used only when initial_margin_per_unit is set
    size_from: str = "current_equity"   # or "starting_equity" (fixed-dollar risk, e.g. prop challenges)


@dataclass
class PortfolioConfig:
    starting_equity: float
    currency: str
    risk_day_timezone: str = "UTC"      # daily-loss and entry-count limits reset here
    risk_day_reset: str = "00:00"


@dataclass
class StrategyConfig:
    htf_pivot_left: int = 2
    htf_pivot_right: int = 2
    mtf_pivot_left: int = 2
    mtf_pivot_right: int = 2
    ltf_pivot_left: int = 2
    ltf_pivot_right: int = 2
    atr_period: int = 14
    displacement_atr_mult: float = 1.5
    base_lookback_bars: int = 5
    zone_expiry_bars: int = 20
    pullback_declining_closes: int = 2
    confirm_window_bars: int = 6
    min_reward_risk: float = 1.5
    stop_offset_ticks: int = 1
    vwap_allowed_volume_types: list = field(default_factory=lambda: ["exchange", "consolidated"])
    vwap_require_complete_session: bool = True
    cancel_setup_at_session_end: bool = True
    # --- signal mode (PRO Breakout = port of the user's TradingView indicator) ---
    mode: str = "jaxon"                 # jaxon | jaxon_pro | pro_breakout
    target_mode: str = "vwap"           # vwap | pro_atr
    pro_trend_adj: int = 10
    pro_target_adj: float = 1.0
    pro_stop_adj: float = 1.0
    pro_target_n: int = 2
    pro_require_bias: bool = True       # pro_breakout mode: longs only with 4H bullish bias
    pro_exit_on_sell: bool = True       # pro_breakout mode: a SELL signal closes the long


@dataclass
class ExecutionConfig:
    limit_requires_trade_through: bool = True   # target only fills if price trades a tick beyond it
    ambiguous_bar_policy: str = "stop_first"    # only "stop_first" in v1


@dataclass
class EvaluationConfig:
    development_end: Optional[str] = None       # ISO date; dev = [start, development_end)
    validation_end: Optional[str] = None        # validation = [development_end, validation_end); test after
    min_trades_for_inference: int = 100
    sensitivity: dict = field(default_factory=dict)  # {param: [values]} varied one at a time


@dataclass
class ExplainConfig:
    enabled: bool = False
    model: str = "claude-opus-5-5"


@dataclass
class Config:
    config_version: str
    example: bool
    portfolio: PortfolioConfig
    instruments: list
    risk: RiskConfig = field(default_factory=RiskConfig)
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    explain: ExplainConfig = field(default_factory=ExplainConfig)
    source_path: Optional[str] = None

    def instrument(self, symbol: str) -> InstrumentConfig:
        for i in self.instruments:
            if i.symbol == symbol:
                return i
        raise KeyError(symbol)

    def canonical(self) -> dict:
        d = asdict(self)
        d.pop("source_path", None)
        return d

    @property
    def config_hash(self) -> str:
        blob = json.dumps(self.canonical(), sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:16]

    @property
    def version_tag(self) -> str:
        return f"{self.config_version}+{self.config_hash}"


# ---------------------------------------------------------------- building
_NESTED = {
    (InstrumentConfig, "session"): SessionConfig,
    (InstrumentConfig, "bars"): BarsConfig,
    (InstrumentConfig, "costs"): CostsConfig,
    (InstrumentConfig, "data"): DataSource,
    (Config, "portfolio"): PortfolioConfig,
    (Config, "risk"): RiskConfig,
    (Config, "strategy"): StrategyConfig,
    (Config, "execution"): ExecutionConfig,
    (Config, "evaluation"): EvaluationConfig,
    (Config, "explain"): ExplainConfig,
}


def _build(cls, raw: Any, where: str):
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(raw).__name__}")
    known = {f.name: f for f in fields(cls)}
    unknown = set(raw) - set(known)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    kwargs = {}
    for name, f in known.items():
        if name not in raw:
            required = f.default is MISSING and f.default_factory is MISSING
            if required:
                raise ConfigError(f"{where}: missing mandatory field '{name}'")
            continue
        val = raw[name]
        sub = _NESTED.get((cls, name))
        if sub is not None:
            val = _build(sub, val, f"{where}.{name}")
        elif cls is Config and name == "instruments":
            if not isinstance(val, list) or not val:
                raise ConfigError(f"{where}.instruments: need at least one instrument")
            val = [_build(InstrumentConfig, v, f"{where}.instruments[{i}]") for i, v in enumerate(val)]
        kwargs[name] = val
    return cls(**kwargs)


def _hhmm(s: str, where: str, allow_24=False):
    try:
        h, m = (int(x) for x in str(s).split(":"))
    except Exception:
        raise ConfigError(f"{where}: '{s}' is not HH:MM")
    if not (0 <= m < 60 and (0 <= h < 24 or (allow_24 and h == 24 and m == 0))):
        raise ConfigError(f"{where}: '{s}' out of range")


def validate(cfg: Config) -> list[str]:
    """Raise ConfigError on hard problems; return warnings for soft ones."""
    from zoneinfo import ZoneInfo

    warnings = []
    if cfg.portfolio.starting_equity <= 0:
        raise ConfigError("portfolio.starting_equity must be > 0")
    try:
        ZoneInfo(cfg.portfolio.risk_day_timezone)
    except Exception:
        raise ConfigError(f"portfolio.risk_day_timezone: unknown timezone {cfg.portfolio.risk_day_timezone}")
    _hhmm(cfg.portfolio.risk_day_reset, "portfolio.risk_day_reset")
    symbols = [i.symbol for i in cfg.instruments]
    if len(set(symbols)) != len(symbols):
        raise ConfigError("instrument symbols must be unique")

    for i in cfg.instruments:
        w = f"instruments[{i.symbol}]"
        if i.asset_class not in ASSET_CLASSES:
            raise ConfigError(f"{w}.asset_class must be one of {sorted(ASSET_CLASSES)}")
        if i.volume_type not in VOLUME_TYPES:
            raise ConfigError(f"{w}.volume_type must be one of {sorted(VOLUME_TYPES)}")
        for name in ("tick_size", "lot_size", "min_quantity", "value_per_point"):
            v = getattr(i, name)
            if not isinstance(v, (int, float)) or v <= 0:
                raise ConfigError(f"{w}.{name} must be a positive number (got {v!r})")
        if i.currency != cfg.portfolio.currency:
            raise ConfigError(f"{w}.currency {i.currency} != portfolio currency {cfg.portfolio.currency}; "
                              "FX conversion is not implemented in v1")
        try:
            ZoneInfo(i.session.timezone)
        except Exception:
            raise ConfigError(f"{w}.session.timezone: unknown timezone {i.session.timezone}")
        _hhmm(i.session.open, f"{w}.session.open")
        _hhmm(i.session.close, f"{w}.session.close", allow_24=True)
        for d, t in i.session.early_closes.items():
            _hhmm(t, f"{w}.session.early_closes[{d}]")
        if not i.session.trading_weekdays or any(d not in range(7) for d in i.session.trading_weekdays):
            raise ConfigError(f"{w}.session.trading_weekdays must list weekday numbers 0-6")
        if i.session.vwap_reset != "session_open":
            raise ConfigError(f"{w}.session.vwap_reset: only 'session_open' is supported in v1")
        b = i.bars
        if b.base_interval_minutes <= 0 or 5 % b.base_interval_minutes:
            raise ConfigError(f"{w}.bars.base_interval_minutes must divide 5 (1 or 5)")
        if b.timestamp_convention not in ("open", "close"):
            raise ConfigError(f"{w}.bars.timestamp_convention must be 'open' or 'close'")
        if b.four_hour_alignment not in ("session_open", "utc_epoch"):
            raise ConfigError(f"{w}.bars.four_hour_alignment must be 'session_open' or 'utc_epoch'")
        if i.costs.spread_ticks < 0 or i.costs.slippage_ticks < 0:
            raise ConfigError(f"{w}.costs: spread and slippage must be >= 0")
        if i.volume_type not in cfg.strategy.vwap_allowed_volume_types:
            warnings.append(f"{w}: volume_type '{i.volume_type}' is not allowed for VWAP — this instrument "
                            "will never produce a VWAP-targeted trade")
        if i.asset_class == "forex" and i.volume_type == "tick":
            warnings.append(f"{w}: forex tick volume is not traded volume; VWAP would be unreliable")
    r = cfg.risk
    if not (0 < r.risk_per_trade_pct <= r.max_open_risk_pct):
        raise ConfigError("risk.risk_per_trade_pct must be > 0 and <= risk.max_open_risk_pct")
    if r.max_open_positions < 1 or r.max_entries_per_session < 1:
        raise ConfigError("risk position/entry limits must be >= 1")
    if cfg.execution.ambiguous_bar_policy != "stop_first":
        raise ConfigError("execution.ambiguous_bar_policy: only 'stop_first' is supported")
    s = cfg.strategy
    for name in ("htf_pivot_left", "htf_pivot_right", "mtf_pivot_left", "mtf_pivot_right",
                 "ltf_pivot_left", "ltf_pivot_right", "atr_period", "base_lookback_bars",
                 "zone_expiry_bars", "confirm_window_bars"):
        if getattr(s, name) < 1:
            raise ConfigError(f"strategy.{name} must be >= 1")
    if s.mode not in ("jaxon", "jaxon_pro", "pro_breakout"):
        raise ConfigError("strategy.mode must be jaxon, jaxon_pro or pro_breakout")
    if s.target_mode not in ("vwap", "pro_atr"):
        raise ConfigError("strategy.target_mode must be vwap or pro_atr")
    if r.size_from not in ("current_equity", "starting_equity"):
        raise ConfigError("risk.size_from must be current_equity or starting_equity")
    if s.pullback_declining_closes < 2:
        raise ConfigError("strategy.pullback_declining_closes must be >= 2")
    for p in cfg.evaluation.sensitivity:
        if not hasattr(s, p):
            raise ConfigError(f"evaluation.sensitivity: '{p}' is not a strategy parameter")
    return warnings


def load_config(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    cfg = _build(Config, raw, path.name)
    cfg.source_path = str(path)
    # resolve data paths relative to the config file
    for i in cfg.instruments:
        p = Path(i.data.path)
        if not p.is_absolute():
            i.data.path = str((path.parent / p).resolve())
    validate(cfg)
    return cfg


def from_dict(raw: dict) -> Config:
    cfg = _build(Config, raw, "config")
    validate(cfg)
    return cfg


def with_strategy_overrides(cfg: Config, **overrides) -> Config:
    import copy

    new = copy.deepcopy(cfg)
    for k, v in overrides.items():
        if not hasattr(new.strategy, k):
            raise ConfigError(f"unknown strategy parameter {k}")
        setattr(new.strategy, k, v)
    validate(new)
    return new
