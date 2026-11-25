"""The "Python code" under each chart runs on the downloaded CSV; big data is sampled."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from test_visualization_plots import viz_app


def data() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 120
    days = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "date": days.strftime("%d/%m/%Y"),
            "day": rng.choice(["Thu", "Fri", "Sat"], n),
            "sex": rng.choice(["M", "F"], n),
            "tip": rng.gamma(2, 1.5, n),
            "bill": rng.normal(20, 5, n),
        }
    )


def run_code(app, frame, tmp_path, monkeypatch):
    code = next(c.value for c in app.code if "px." in c.value)
    frame.to_csv(tmp_path / "processed_dataset.csv", index=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(go.Figure, "show", lambda self, *a, **k: None)
    scope = {}
    exec(compile(code, "chart_code", "exec"), scope)
    return scope["fig"]


@pytest.mark.parametrize(
    "plot, mapping, expected",
    [
        ("bars", dict(x="day", y="None", color="sex"), "bar"),
        ("bars", dict(x="date", y="tip"), "bar"),
        ("boxes", dict(x="day", y="tip", color="sex"), "box"),
        ("lineplot", dict(x="date", y="tip", color="sex"), "scatter"),
        ("lineplot", dict(x="day", y="tip"), "scatter"),
        ("histogram", dict(x="tip", y="None", color="sex"), "histogram"),
        ("scatter", dict(x="tip", y="bill", color="day"), "scatter"),
        ("catplot", dict(x="day", y="tip"), "box"),
        ("correlation", dict(x="tip", y="bill"), "heatmap"),
    ],
)
def test_chart_code_runs_on_the_downloaded_csv(
    plot, mapping, expected, tmp_path, monkeypatch
):
    frame = data()
    app = viz_app(frame, plot, **mapping)
    app.run()
    assert not app.exception
    fig = run_code(app, frame, tmp_path, monkeypatch)
    assert fig.data and fig.data[0].type == expected


def test_big_data_is_sampled_unless_asked():
    from edalens.views import visualization as viz

    rows = viz.SAMPLE_ROWS + 1000
    rng = np.random.default_rng(0)
    frame = pd.DataFrame({"a": rng.normal(size=rows), "b": rng.normal(size=rows)})
    app = viz_app(frame, "scatter", x="a", y="b")
    app.run()
    assert not app.exception
    assert any("random sample" in i.value for i in app.info)
    app.sidebar.checkbox(key="viz_use_all_rows").check()
    app.run()
    assert not any("random sample" in i.value for i in app.info)
