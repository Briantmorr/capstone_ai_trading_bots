from __future__ import annotations

from trading_system.core.bot import BotContext
from trading_system.strategies.intraday_mean_reversion import IntradayMeanReversionBot
from trading_system.strategies.momentum_volatility import MomentumVolatilityBot
from trading_system.strategies.pead_drift import PEADEventBot


OFFICIAL_BOTS = {
    "momentum_volatility": {
        "class": MomentumVolatilityBot,
        "cadence": "daily",
        "universe": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "META", "TSLA", "NFLX", "GOOGL"],
    },
    "pead_drift": {
        "class": PEADEventBot,
        "cadence": "event",
        "universe": ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL"],
    },
    "intraday_mean_reversion": {
        "class": IntradayMeanReversionBot,
        "cadence": "hourly",
        "universe": ["SPY", "QQQ", "AAPL", "MSFT", "NVDA"],
    },
}


def create_bot(name: str, params=None):
    if name not in OFFICIAL_BOTS:
        raise KeyError(f"Unknown bot: {name}")
    config = OFFICIAL_BOTS[name]
    context = BotContext(name=name, cadence=config["cadence"], universe=config["universe"], params=params or {})
    return config["class"](context)


def list_bots():
    return sorted(OFFICIAL_BOTS.keys())
