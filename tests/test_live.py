from trading_bot.live import LiveRunner


def test_paper_position_hits_breakeven_then_target(tmp_path):
    bot = LiveRunner("BTC/USDT", "binance", paper=True, state_dir=str(tmp_path))
    bot.state["position"] = {"entry": 100.0, "qty": 10.0, "stop": 99.0, "risk": 1.0, "be_moved": False,
                             "target": 102.0, "opened": "x", "bars": 0}
    prices = iter([101.2, 102.5])
    bot._price = lambda: next(prices)
    bot.manage(new_bar=True)
    assert bot.state["position"]["stop"] == 100.0          # moved to breakeven at +1R
    bot.manage(new_bar=True)
    assert bot.state["position"] is None                   # closed at target
    assert bot.state["paper_equity"] > 10_000
    assert (tmp_path / "BTC_USDT_trades.csv").exists()
