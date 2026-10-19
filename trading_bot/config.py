"""All strategy knobs in one place."""
from dataclasses import dataclass, asdict


@dataclass
class StrategyConfig:
    # --- 4H trend filter (must be going UP) ---
    htf_ema_fast: int = 20
    htf_ema_slow: int = 50
    htf_pivot: int = 2               # bars each side to confirm a 4H swing
    htf_require_higher_low: bool = True

    # --- 30m pullback (must be going DOWN) ---
    mtf_ema_fast: int = 9
    mtf_ema_slow: int = 21
    mtf_pivot: int = 2
    pullback_expiry_bars: int = 36   # 5m bars the setup stays armed after the 30m pullback ends (3h)

    # --- 5m structure + VWAP entry ---
    ltf_pivot: int = 3               # bars each side to confirm a 5m swing
    entry_window_bars: int = 12      # 5m bars after the break of structure to get the VWAP entry (1h)
    vwap_touch_tol: float = 0.001    # low may be within 0.1% above VWAP and still count as a touch
    vwap_session_tz: str = "UTC"     # VWAP resets each day in this timezone (use America/New_York for stocks)

    # --- risk / exits ---
    stop_buffer_atr: float = 0.10    # stop = swing low - 0.1 * ATR(14, 5m)
    max_stop_pct: float = 0.03       # skip if stop is wider than 3%
    min_stop_pct: float = 0.002      # skip if stop is tighter than 0.2% (fees would eat it)
    reward_r: float = 2.0            # take profit at 2R
    breakeven_at_r: float = 1.0      # move stop to entry at +1R (0 disables)
    max_hold_bars: int = 288         # close after 24h of 5m bars (0 disables)

    # --- account ---
    risk_per_trade: float = 0.01     # risk 1% of equity per trade
    fee_rate: float = 0.0005         # 0.05% per side
    slippage: float = 0.0002         # 0.02% per fill

    def to_dict(self):
        return asdict(self)
