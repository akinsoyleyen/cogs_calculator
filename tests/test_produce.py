import pandas as pd

from cogs.produce import GROUP_COLS, _coerce_bool, default_produce, reconcile_groups


def test_default_produce_matches_fruit_first_or_last():
    assert default_produce("Cherry 10 x 500g") == "Cherry"
    assert default_produce("14kg Open Top Grapefruit") == "Grapefruit"  # fruit last
    assert default_produce("15kg Telescopic Lemon") == "Lemon"


def test_default_produce_variety_beats_parent():
    assert default_produce("UFO Peach 10 x 500g") == "UFO Peach"
    assert default_produce("UFO Nectarine 10 x 500g") == "UFO Nectarine"
    assert default_produce("Green Plum 5kg") == "Green Plum"
    assert default_produce("Black Fig 700g") == "Fig"


def test_default_produce_fallback_first_nonsize_token():
    assert default_produce("17kg Japan Telescopic") == "Japan"  # no known fruit
    assert default_produce("Mango 6 x 1kg") == "Mango"
    assert default_produce("") == ""


def test_reconcile_appends_unknown_and_drops_stale():
    groups = pd.DataFrame(
        {
            "ProductID": ["Cherry 10 x 500g", "Old Item"],
            "Produce": ["Cherry", "Old"],
            "AirEligible": [True, False],
        }
    )
    out = reconcile_groups(groups, ["Cherry 10 x 500g", "Plum 10 x 500g"])

    assert list(out.columns) == GROUP_COLS
    ids = out["ProductID"].tolist()
    assert "Cherry 10 x 500g" in ids
    assert "Plum 10 x 500g" in ids  # appended via default_produce
    assert "Old Item" not in ids    # dropped — no longer a live product

    plum = out[out["ProductID"] == "Plum 10 x 500g"].iloc[0]
    assert plum["Produce"] == "Plum"
    assert bool(plum["AirEligible"]) is True


def test_reconcile_coerces_string_bool():
    groups = pd.DataFrame(
        {"ProductID": ["X (Container/Truck)"], "Produce": ["X"], "AirEligible": ["False"]}
    )
    out = reconcile_groups(groups, ["X (Container/Truck)"])
    assert bool(out.iloc[0]["AirEligible"]) is False


def test_reconcile_empty_file_seeds_all():
    out = reconcile_groups(pd.DataFrame(), ["Cherry 10 x 500g", "Apricot 16 x 350g"])
    assert len(out) == 2
    assert out[out["ProductID"] == "Cherry 10 x 500g"].iloc[0]["Produce"] == "Cherry"
    assert bool(out["AirEligible"].all()) is True


def test_coerce_bool_variants():
    assert _coerce_bool(True) is True
    assert _coerce_bool(False) is False
    assert _coerce_bool("False") is False
    assert _coerce_bool("no") is False
    assert _coerce_bool(0) is False
    assert _coerce_bool(1) is True
    assert _coerce_bool(None) is True   # blank cell → eligible
    assert _coerce_bool("") is True
