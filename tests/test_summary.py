import pandas as pd

from agnews.summary import inject_tables, markdown, pct


def test_inject_tables_fills_empty_and_replaces_existing_blocks():
    text = "intro\n<!-- BEGIN table:main -->\n<!-- END table:main -->\noutro\n"
    once = inject_tables(text, {"main": "| a |"})
    assert once == "intro\n<!-- BEGIN table:main -->\n| a |\n<!-- END table:main -->\noutro\n"
    assert inject_tables(once, {"main": "| b |"}) == once.replace("| a |", "| b |")


def test_markdown_aligns_text_left_and_numbers_right():
    table = markdown(pd.DataFrame({"Model": ["FNN"], "Acc": ["91.5 ± 0.1"], "Note": ["n/a"]}))
    assert table.splitlines()[1] == "| :-- | --: | --: |"


def test_pct_formats_mean_and_optional_std():
    assert pct(0.91537, 0.00061, 2) == "91.54 ± 0.06"
    assert pct(0.9, float("nan")) == "90.0"
