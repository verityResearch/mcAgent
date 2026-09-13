from src.data_report.schema import adapt_trading


ARMORER_TRADES = {
    "tiers": [
        {
            "trades": [
                {
                    "wants": [{"item": "minecraft:coal", "quantity": {"min": 16, "max": 24}}],
                    "gives": [{"item": "minecraft:emerald"}],
                },
                {
                    "wants": [{"item": "minecraft:emerald", "quantity": {"min": 4, "max": 6}}],
                    "gives": [{"item": "minecraft:iron_helmet"}],
                },
            ]
        },
        {
            "trades": [
                {
                    "wants": [{"item": "minecraft:iron_ingot", "quantity": {"min": 7, "max": 9}}],
                    "gives": [{"item": "minecraft:emerald"}],
                }
            ]
        },
    ]
}


def test_adapt_trading_emits_one_record_per_trade():
    records = adapt_trading("minecraft:armorer_trades", ARMORER_TRADES)
    assert len(records) == 3
    assert all(r["category"] == "trading" for r in records)
    subject_ids = [r["subject_id"] for r in records]
    assert subject_ids == [
        "minecraft:armorer/tier0/trade0",
        "minecraft:armorer/tier0/trade1",
        "minecraft:armorer/tier1/trade0",
    ]
    assert records[0]["fields"] == {"wants": ["minecraft:coal"], "gives": ["minecraft:emerald"]}
    assert records[1]["fields"] == {"wants": ["minecraft:emerald"], "gives": ["minecraft:iron_helmet"]}


def test_adapt_trading_skips_malformed_trade_keeps_others():
    mixed = {"tiers": [{"trades": [
        {"wants": [], "gives": [{"item": "minecraft:emerald"}]},
        {"wants": [{"item": "minecraft:coal"}], "gives": [{"item": "minecraft:emerald"}]},
    ]}]}
    records = adapt_trading("minecraft:armorer_trades", mixed)
    assert len(records) == 1
    assert records[0]["subject_id"] == "minecraft:armorer/tier0/trade1"
