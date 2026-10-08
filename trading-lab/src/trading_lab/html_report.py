"""Self-contained HTML report for one run (``trading-lab report RUN_ID --html FILE``).

One file, no external scripts, styles or fonts, so it can be e-mailed or
opened offline. It is built from the read-only dashboard data layer, so
writing a report never changes the database. Every piece of text from the
database (including model rationales) is HTML-escaped.

Charts follow the project's data-viz rules: one axis per chart, thin 2px
lines, recessive grid, a legend plus direct end labels for the two equity
series, a crosshair tooltip on hover, light and dark themes, and a table
view of the same numbers.
"""

from __future__ import annotations

import html
import json
import math
from datetime import datetime, timezone
from typing import Any, Sequence

import pandas as pd

from trading_lab.dashboard.data import DashboardData

MAX_POINTS = 600
W, H, PAD_L, PAD_R, PAD_T, PAD_B = 800, 260, 64, 96, 16, 28

_CSS = """
.viz-root{color-scheme:light;--surface-0:#f4f3ef;--surface-1:#fcfcfb;--text-primary:#0b0b0b;
--text-secondary:#52514e;--text-muted:#76756f;--grid:#e4e3df;--border:#dcdbd5;
--series-1:#2a78d6;--series-2:#eb6834;--good:#0b7a0b;--critical:#c42f2f}
@media (prefers-color-scheme:dark){:root:where(:not([data-theme="light"])) .viz-root{color-scheme:dark;
--surface-0:#121211;--surface-1:#1a1a19;--text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#9b9a91;
--grid:#2f2f2c;--border:#3a3a36;--series-1:#3987e5;--series-2:#d95926;--good:#3fbf3f;--critical:#e66767}}
:root[data-theme="dark"] .viz-root{color-scheme:dark;--surface-0:#121211;--surface-1:#1a1a19;
--text-primary:#ffffff;--text-secondary:#c3c2b7;--text-muted:#9b9a91;--grid:#2f2f2c;--border:#3a3a36;
--series-1:#3987e5;--series-2:#d95926;--good:#3fbf3f;--critical:#e66767}
*{box-sizing:border-box}
body{margin:0;background:var(--surface-0);color:var(--text-primary);
font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:1040px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:28px 0 10px}
.muted{color:var(--text-muted)}.secondary{color:var(--text-secondary)}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:10px;padding:14px 16px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-top:16px}
.tile .label{color:var(--text-secondary);font-size:12px}.tile .value{font-size:20px;font-weight:600;
font-variant-numeric:tabular-nums}
.chart{position:relative}.chart svg{width:100%;height:auto;display:block}
.legend{display:flex;gap:16px;font-size:12px;color:var(--text-secondary);margin-bottom:6px}
.legend i{display:inline-block;width:14px;height:2px;vertical-align:middle;margin-right:6px;border-radius:2px}
.tip{position:absolute;pointer-events:none;background:var(--surface-1);border:1px solid var(--border);
border-radius:8px;padding:6px 8px;font-size:12px;box-shadow:0 2px 8px rgba(0,0,0,.12);display:none;
white-space:nowrap;font-variant-numeric:tabular-nums}
.table-wrap{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:right;padding:6px 8px;border-bottom:1px solid var(--border);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--text-secondary);font-weight:600}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:10px}
.rationale{white-space:normal;color:var(--text-secondary);margin-top:6px}
.pos{color:var(--good)}.neg{color:var(--critical)}
details summary{cursor:pointer;color:var(--text-secondary);margin-top:8px}
.note{margin-top:28px;font-size:12px;color:var(--text-muted)}
"""

_JS = """
document.querySelectorAll('.chart[data-points]').forEach(function(box){
  var pts=JSON.parse(box.getAttribute('data-points'));var svg=box.querySelector('svg');
  var tip=box.querySelector('.tip');var line=svg.querySelector('.cross');if(!pts.length)return;
  function show(ev){var r=svg.getBoundingClientRect();var x=(ev.clientX-r.left)/r.width*%(W)d;
    var i=Math.round((x-%(L)d)/(%(W)d-%(L)d-%(R)d)*(pts.length-1));i=Math.max(0,Math.min(pts.length-1,i));
    var p=pts[i];line.setAttribute('x1',p.x);line.setAttribute('x2',p.x);line.style.display='';
    tip.innerHTML=p.label;tip.style.display='block';var left=p.x/%(W)d*r.width+12;
    if(left+tip.offsetWidth>r.width)left=p.x/%(W)d*r.width-tip.offsetWidth-12;
    tip.style.left=left+'px';tip.style.top='8px';}
  svg.addEventListener('mousemove',show);svg.addEventListener('mouseleave',function(){
    tip.style.display='none';line.style.display='none';});
});
""" % {"W": W, "L": PAD_L, "R": PAD_R}


def _e(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _pct(value: Any, signed: bool = True) -> str:
    if isinstance(value, str):  # e.g. a profit factor of "inf" when nothing was lost
        return _e(value)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:+.2%}" if signed else f"{value:.1%}"


def _num(value: Any, digits: int = 2) -> str:
    if isinstance(value, str):  # e.g. a profit factor of "inf" when nothing was lost
        return _e(value)
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    return f"{value:,.{digits}f}"


def _when(value: Any) -> str:
    """'2024-02-29T18:00:00+00:00' -> '2024-02-29 18:00' (UTC throughout)."""
    return "" if value is None else str(value)[:16].replace("T", " ")


def _cls(value: Any) -> str:
    if not isinstance(value, (int, float)) or value == 0:
        return ""
    return "pos" if value > 0 else "neg"


def _downsample(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) <= MAX_POINTS:
        return frame
    step = math.ceil(len(frame) / MAX_POINTS)
    return pd.concat([frame.iloc[::step], frame.iloc[[-1]]]).loc[lambda f: ~f.index.duplicated(keep="last")]


def _ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if hi <= lo:
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = min((m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw), default=raw)
    start = math.ceil(lo / step) * step
    return [start + k * step for k in range(int((hi - start) / step) + 1)]


def line_chart(frame: pd.DataFrame, series: Sequence[tuple[str, str, str]], *, fmt: str = "num",
               area: bool = False) -> str:
    """SVG chart for ``frame`` columns. ``series`` = (column, label, css colour variable)."""
    frame = _downsample(frame[[c for c, _, _ in series]].dropna(how="all"))
    if frame.empty:
        return '<p class="muted">No data yet.</p>'
    values = frame.to_numpy(dtype="float64")
    lo, hi = float(pd.Series(values.ravel()).min()), float(pd.Series(values.ravel()).max())
    if area:
        hi = max(hi, 0.0)
    if hi == lo:
        lo, hi = lo - 1, hi + 1
    pad = (hi - lo) * 0.06
    lo, hi = lo - (0 if area else pad), hi + pad
    n = len(frame)

    def x(i: int) -> float:
        return PAD_L + (W - PAD_L - PAD_R) * (i / max(n - 1, 1))

    def y(v: float) -> float:
        return PAD_T + (H - PAD_T - PAD_B) * (1 - (v - lo) / (hi - lo))

    def label(v: float) -> str:
        return f"{v:.1%}" if fmt == "pct" else f"{v:,.0f}"

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{_e(", ".join(s[1] for s in series))}">']
    for t in _ticks(lo, hi):
        parts.append(f'<line x1="{PAD_L}" x2="{W - PAD_R}" y1="{y(t):.1f}" y2="{y(t):.1f}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{PAD_L - 8}" y="{y(t) + 4:.1f}" text-anchor="end" font-size="11" '
                     f'fill="var(--text-muted)">{_e(label(t))}</text>')
    first, last = frame.index[0], frame.index[-1]
    for i, ts in ((0, first), (n - 1, last)):
        anchor = "start" if i == 0 else "end"
        parts.append(f'<text x="{x(i):.1f}" y="{H - 8}" text-anchor="{anchor}" font-size="11" '
                     f'fill="var(--text-muted)">{_e(f"{ts:%Y-%m-%d %H:%M}")}</text>')
    for k, (col, name, var) in enumerate(series):
        pts = [(x(i), y(v)) for i, v in enumerate(values[:, k]) if math.isfinite(v)]
        if not pts:
            continue
        path = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        if area:
            base = y(0.0)
            parts.append(f'<polygon points="{pts[0][0]:.1f},{base:.1f} {path} {pts[-1][0]:.1f},{base:.1f}" '
                         f'fill="var({var})" fill-opacity="0.18" stroke="none"/>')
        parts.append(f'<polyline points="{path}" fill="none" stroke="var({var})" stroke-width="2" '
                     'stroke-linejoin="round" stroke-linecap="round"/>')
        if len(series) > 1:  # direct label at the line end
            parts.append(f'<text x="{pts[-1][0] + 6:.1f}" y="{pts[-1][1] + 4:.1f}" font-size="12" '
                         f'fill="var(--text-secondary)">{_e(name)}</text>')
    parts.append(f'<line class="cross" x1="0" x2="0" y1="{PAD_T}" y2="{H - PAD_B}" stroke="var(--text-muted)" '
                 'stroke-width="1" stroke-dasharray="3 3" style="display:none"/>')
    parts.append("</svg>")
    points = []
    for i, (ts, row) in enumerate(zip(frame.index, values)):
        lines = [f"<b>{_e(f'{ts:%Y-%m-%d %H:%M}')}</b>"] + [
            f"{_e(name)}: {_e(label(v))}" for (_, name, _), v in zip(series, row) if math.isfinite(v)
        ]
        points.append({"x": round(x(i), 1), "label": "<br>".join(lines)})
    legend = ""
    if len(series) > 1:
        legend = '<div class="legend">' + "".join(
            f'<span><i style="background:var({var})"></i>{_e(name)}</span>' for _, name, var in series) + "</div>"
    return (f'{legend}<div class="chart" data-points="{_e(json.dumps(points))}">{"".join(parts)}'
            '<div class="tip"></div></div>')


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    head = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(cell for cell in row) + "</tr>" for row in rows)
    return f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _td(text: Any, cls: str = "") -> str:
    return f'<td class="{cls}">{_e(text)}</td>' if cls else f"<td>{_e(text)}</td>"


def build_html_report(db_path: str, run_id: str | None = None, *, horizon: int = 4,
                      now: datetime | None = None) -> str:
    with DashboardData(db_path) as data:
        run_id = run_id or data.default_run_id()
        if run_id is None:
            raise ValueError("no runs stored yet")
        o = data.overview(run_id)
        research = data.research(run_id)
        curve = data.equity_curve(run_id)
        agents = data.agent_performance(run_id, horizon)
        usage = data.usage(run_id)
        trades = data.recent_trades(run_id, limit=50)
        rationales = data.latest_rationales(run_id)
        breakers = data.breaker_status(run_id)
        try:
            from trading_lab.research.robustness import robustness_for_run

            robust = robustness_for_run(data.store, run_id, samples=2000)
        except Exception:  # a report must render even without enough data
            robust = None
        regimes = data.regimes(run_id)
        monthly = data.monthly_returns(run_id)
        breakdown = data.trade_breakdown(run_id)
        outlook = data.outlook(run_id)
        fingerprint = data.store.get_run(run_id)["config_fingerprint"]
        decisions = data.store.load_decisions(run_id, include_holds=False)
    m, b = research["metrics"] or {}, research["benchmark"] or {}
    generated = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%d %H:%M UTC")

    out = [
        "<!doctype html>", '<html lang="en"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>trading-lab run {_e(run_id)}</title><style>{_CSS}</style></head>",
        '<body class="viz-root"><main>',
        f"<h1>trading-lab run {_e(run_id)}</h1>",
        f'<div class="secondary">{_e(o["kind"])} · {_e(o["status"])} · {_e(o["timeframe"])} candles · '
        f'{_e(", ".join(o["symbols"]))} · data {_e(o["data_source"])} · agents {_e(o["agents_mode"])}</div>',
        f'<div class="muted">Period {_e(_when(o["period_start"]))} → {_e(_when(o["period_end"] or o.get("last_bar")) or "now")} UTC · '
        f'config {_e(fingerprint[:16])} · generated {_e(generated)}</div>',
    ]
    tiles = [
        ("Total return", _pct(m.get("total_return")), _cls(m.get("total_return"))),
        ("Buy & hold", _pct(b.get("total_return")), _cls(b.get("total_return"))),
        ("Max drawdown", _pct(-m["max_drawdown"]) if m.get("max_drawdown") is not None else "n/a", ""),
        ("Sharpe", _num(m.get("sharpe_ratio")), ""),
        ("Prob. Sharpe > 0", _pct(m.get("probabilistic_sharpe"), signed=False), ""),
        ("Profit factor", _num(m.get("profit_factor")), ""),
        ("Trades", str(m.get("num_trades", 0)), ""),
        ("Win rate", _pct(m.get("win_rate"), signed=False), ""),
        ("Exposure", _pct(m.get("exposure"), signed=False), ""),
    ]
    rel = research.get("relative") or {}
    if rel:
        tiles += [
            ("Excess vs buy & hold", _pct(rel.get("excess_return")), _cls(rel.get("excess_return"))),
            ("Alpha (per year)", _pct(rel.get("alpha_annualized")), _cls(rel.get("alpha_annualized"))),
            ("Beta", _num(rel.get("beta")), ""),
            ("Information ratio", _num(rel.get("information_ratio")), ""),
        ]
    out.append('<div class="tiles">' + "".join(
        f'<div class="card tile"><div class="label">{_e(label)}</div><div class="value {cls}">{_e(value)}</div></div>'
        for label, value, cls in tiles) + "</div>")

    out.append("<h2>Equity</h2>")
    if curve.empty:
        out.append('<div class="card"><p class="muted">No equity recorded yet.</p></div>')
    else:
        series = [("equity", "Strategy", "--series-1")]
        if "buy_and_hold" in curve:
            series.append(("buy_and_hold", "Buy & hold", "--series-2"))
        out.append(f'<div class="card">{line_chart(curve, series)}</div>')
        out.append("<h2>Drawdown</h2>")
        out.append(f'<div class="card">{line_chart(curve, [("drawdown", "Drawdown", "--series-1")], fmt="pct", area=True)}</div>')
        daily = curve.resample("1D").last().dropna(how="all")
        rows = [[_td(f"{ts:%Y-%m-%d}"), _td(_num(r.equity)), _td(_num(r.get("buy_and_hold"))),
                 _td(_pct(r.drawdown))] for ts, r in daily.iterrows()]
        out.append("<details><summary>Table view (daily close)</summary>"
                   + _table(["Day", "Equity", "Buy & hold", "Drawdown"], rows) + "</details>")

    if robust is not None and robust.bars:
        def rng(r: Any, pct: bool = True) -> str:
            if r is None:
                return "n/a"
            f = _pct if pct else _num
            return f"{f(r.low)} … {f(r.median)} … {f(r.high)}"

        rows = [
            [_td(f"Trade bootstrap ({robust.trades} trades)"), _td(rng(robust.trade_total_return)),
             _td("n/a" if robust.prob_loss is None else f"{robust.prob_loss:.0%}"), _td("")],
            [_td(f"Block bootstrap ({robust.bars} bars)"), _td(rng(robust.bar_total_return)), _td(""),
             _td(rng(robust.sharpe_range, pct=False))],
        ]
        out.append("<h2>Robustness</h2><div class=\"card\">" + _table(
            ["Resampling", "Total return (5% … median … 95%)", "P(loss)", "Sharpe (5% … 95%)"], rows)
            + "".join(f'<p class="muted">⚠ {_e(w)}</p>' for w in robust.warnings) + "</div>")

    if monthly:
        months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        rows = [[_td(row["year"])] + [_td("" if row[str(m)] is None else _pct(row[str(m)]), _cls(row[str(m)]))
                                      for m in range(1, 13)] + [_td(_pct(row["total"]), _cls(row["total"]))]
                for row in monthly]
        out.append("<h2>Monthly returns</h2><div class=\"card\">" + _table(["Year", *months, "Year total"], rows)
                   + '<p class="muted">Each month compares its last equity with the previous month\'s; '
                   'empty = no bars in that month.</p></div>')

    if regimes:
        rows = [[_td(r["name"]), _td(r["bars"]), _td(f'{r["time_share"]:.0%}'),
                 _td(_pct(r["strategy_return"]), _cls(r["strategy_return"])),
                 _td(_pct(r["market_return"]), _cls(r["market_return"])), _td(f'{r["in_market"]:.0%}'),
                 _td(r["trades"]), _td(_num(r["pnl"]), _cls(r["pnl"]))]
                for r in regimes["by_trend"] + regimes["combined"] if r["bars"]]
        out.append("<h2>Market regimes</h2><div class=\"card\">" + _table(
            ["Regime", "Bars", "Time", "Strategy", "Market", "In market", "Trades", "PnL"], rows)
            + f'<p class="muted">Trend from the {regimes["trend_bars"]}-bar average of an equal-weight index of the '
            f'symbols; volatility against its median. Returns compound only that regime\'s bars; '
            f'{regimes["warmup_bars"]} warm-up bars are in no regime.</p></div>')

    if outlook:
        rows = [[_td("Max drawdown"), _td(f'{outlook["drawdown_median"]:.1%}'), _td(f'{outlook["drawdown_bad"]:.1%}'),
                 _td(f'run: {outlook["actual_drawdown"]:.1%}')],
                [_td("Return"), _td(_pct(outlook["return_median"]), _cls(outlook["return_median"])),
                 _td(_pct(outlook["return_low"]), _cls(outlook["return_low"])),
                 _td(f'chance of a loss {outlook["prob_loss"]:.0%}')],
                [_td("Losing streak"), _td(f'{outlook["streak_median"]} trades'), _td(f'{outlook["streak_bad"]} trades'),
                 _td("")]]
        chances = ", ".join(f'{p["level"]:.0%}: {p["probability"]:.0%}' for p in outlook["prob_drawdown"])
        out.append(f'<h2>What to be ready for</h2><div class="card">'
                   f'<p>The next {outlook["horizon"]} trades, resampled {outlook["samples"]:,} times from the run\'s '
                   f'{outlook["trades"]}.</p>'
                   + _table(["", "Median", "Bad case (1 in 20)", ""], rows)
                   + f"<p>Chance of a drawdown of at least {_e(chances)}.</p>"
                   + "".join(f'<p class="muted">⚠ {_e(w)}</p>' for w in outlook["warnings"])
                   + '<p class="muted">Trades are drawn independently and drawdowns measured trade by trade, so '
                   "real streaks and dips can be worse: treat the bad case as a floor to prepare for.</p></div>")

    if breakdown:
        titles = {"exit": "Exit", "side": "Side", "holding": "Holding time", "symbol": "Symbol"}
        rows = [[_td(titles[grouping]), _td(g["name"]), _td(g["trades"]), _td(_pct(g["win_rate"], signed=False)),
                 _td(_num(g["pnl"]), _cls(g["pnl"])), _td(_pct(g["avg_return"]), _cls(g["avg_return"])),
                 _td(f'{g["avg_bars"]:.1f}'), _td(_num(g["best"])), _td(_num(g["worst"])),
                 _td(_pct(g.get("avg_mae"))), _td(_pct(g.get("avg_mfe")))]
                for grouping in ("exit", "side", "holding", "symbol") for g in breakdown["groups"][grouping]]
        from trading_lab.research.trades import excursion_sentences

        out.append("<h2>Where the money comes from</h2><div class=\"card\">" + _table(
            ["By", "Group", "Trades", "Won", "PnL", "Avg return", "Avg bars", "Best", "Worst", "MAE", "MFE"], rows)
            + "".join(f'<p>{_e(s)}</p>' for s in excursion_sentences(breakdown.get("excursions")))
            + "".join(f'<p class="muted">• {_e(n)}</p>' for n in breakdown["observations"])
            + '<p class="muted">Closed trades only. MAE/MFE: the worst and best move against and for the entry '
            "while the trade was open. Small groups are noise; test a change with ab or walkforward "
            "(see <code>trading-lab trades</code>).</p></div>")

    if breakers.get("trips"):
        out.append("<h2>Circuit breaker trips</h2><div class=\"card\"><ul>" + "".join(
            f"<li>{_e(_when(t['timestamp']))}: {_e(t['reason'])}</li>" for t in breakers["trips"]) + "</ul></div>")

    shown = [a for a in agents if a["votes"]]
    if shown:
        out.append(f"<h2>Voters ({horizon}-bar outcome horizon)</h2>")
        rows = [[_td(a["strategy"] + (" (AI)" if a["is_agent"] else "")), _td(a["votes"]),
                 _td(f'{a["buy"]}/{a["sell"]}/{a["hold"]}'), _td(a["errors"]),
                 _td(_num(a["avg_confidence"])), _td(_pct(a["directional_correctness"], signed=False)),
                 _td(a["trades_influenced"]), _td(a["trades_pivotal"]),
                 _td(_pct(a["pnl_agreed_pct"]) if a["trades_agreed"] else "–", _cls(a["pnl_agreed_pct"])),
                 _td(_pct(a["pnl_disagreed_pct"]) if a["trades_disagreed"] else "–",
                     _cls(a["pnl_disagreed_pct"]))] for a in shown]
        out.append('<div class="card">' + _table(
            ["Voter", "Votes", "Buy/Sell/Hold", "Errors", "Avg conf.", "Correct", "Trades influenced",
             "Pivotal", "PnL agreed", "PnL disagreed"], rows) + "</div>")

    if rationales:
        out.append('<h2>Latest AI rationales</h2><div class="cards">')
        for r in rationales:
            label = r.get("regime") or r.get("momentum_state") or r.get("risk_state") or ""
            text = r.get("error") or r.get("rationale") or ""
            out.append(f'<div class="card"><b>{_e(r["strategy"])}</b> · {_e(r["symbol"])} · '
                       f'{_e(str(r["direction"]).upper())} {_e(_num(r["confidence"]))} {_e(label)}'
                       f'<div class="muted">{_e(_when(r["timestamp"]))} UTC</div><div class="rationale">{_e(text)}</div></div>')
        out.append("</div>")

    if usage["per_agent"]:
        t = usage["total"]
        est = " (estimated)" if t["tokens_estimated"] else ""
        out.append("<h2>Model usage</h2>")
        rows = [[_td(name), _td(s["calls"]), _td(s["cache_hits"]), _td(s["failures"]), _td(s["retries"]),
                 _td("n/a" if s["avg_latency_seconds"] is None else f'{s["avg_latency_seconds"]:.2f}s'),
                 _td(f'{s["input_tokens"]:,}'), _td(f'{s["output_tokens"]:,}')]
                for name, s in usage["per_agent"].items()]
        out.append('<div class="card">' + _table(["Agent", "Calls", "Cache hits", "Failures", "Retries",
                                                  "Avg latency", f"Input tokens{est}", f"Output tokens{est}"],
                                                 rows) + "</div>")

    out.append(f"<h2>Closed trades ({len(trades)} most recent)</h2>")
    if trades:
        rows = [[_td(t["symbol"]), _td(t.get("side", "long")), _td(_when(t["opened_at"])), _td(_when(t["closed_at"])),
                 _td(f'{t["quantity"]:.6g}'), _td(f'{t["entry_price"]:.6g}'), _td(f'{t["exit_price"]:.6g}'),
                 _td(_num(t["pnl"]), _cls(t["pnl"])), _td(_pct(t["return_pct"]), _cls(t["return_pct"]))]
                for t in trades]
        out.append('<div class="card">' + _table(["Symbol", "Side", "Opened", "Closed", "Qty", "Entry", "Exit",
                                                  "PnL", "Return"], rows) + "</div>")
    else:
        out.append('<div class="card"><p class="muted">No closed trades.</p></div>')

    if not decisions.empty:
        counts = decisions["action"].value_counts().sort_index()
        out.append("<h2>Decisions</h2><div class=\"card\">" + _table(
            ["Action", "Count"], [[_td(k), _td(int(v))] for k, v in counts.items()]) + "</div>")

    out.append('<p class="note">Paper trading only: simulated fills on public market data. Past, simulated '
               "results do not show that a strategy is profitable; see docs/EXPERIMENT_PROTOCOL.md. "
               "This report is a read-only snapshot of the run database.</p>")
    out.append(f"</main><script>{_JS}</script></body></html>")
    return "\n".join(out)
