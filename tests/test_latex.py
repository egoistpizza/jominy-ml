import pytest

from jominy.evaluate import Interval
from jominy.latex import Outputs, num


def test_num_drops_the_sign_of_negative_zero() -> None:
    assert num(-0.04) == "0.0"
    assert num(-0.06) == "-0.1"
    assert num(3.14159, 2) == "3.14"


def test_macros_are_unique_and_letters_only() -> None:
    out = Outputs()
    out.macro("maeHyb", "3.6")
    with pytest.raises(ValueError):
        out.macro("maeHyb", "3.7")
    with pytest.raises(ValueError):
        out.macro("mae42", "1.0")


def test_interval_writes_three_macros_and_tables_render() -> None:
    out = Outputs()
    out.interval("diff", Interval(-0.5, -1.25, 0.25), digits=2)
    out.table("tab", "lr", ["a", "b"], [["x", "1"]])
    out.write_macros()
    assert out.macros == {"diff": "-0.50", "diffLow": "-1.25", "diffHigh": "0.25"}
    assert "\\toprule" in out.files["tab.tex"]
    assert "\\newcommand{\\diffLow}{-1.25}" in out.files["results.tex"]
