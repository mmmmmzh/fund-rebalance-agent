from __future__ import annotations

from fund_agent.universe import load_fund_universe_csv


def test_load_fund_universe_csv(tmp_path) -> None:
    path = tmp_path / "funds.csv"
    path.write_text(
        "code,name,category,is_equity_like,instrument_type\n"
        "017074,嘉实清洁能源股票发起式C,equity,true,open_fund\n",
        encoding="utf-8",
    )

    universe = load_fund_universe_csv(path)

    assert len(universe) == 1
    assert universe[0].code == "017074"
    assert universe[0].instrument_type == "open_fund"
    assert universe[0].is_equity_like is True
