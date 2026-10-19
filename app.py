import html as _html
import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from zone_engine import (
    ATR_RANGE, BOUNDS, CHECK_WHY, DEFAULTS, DIRS, FILTERS, MARKETS, SCREEN_RVOL, SORTS, SYD, US_MARKETS,
    alert_lines, apply_view, atr_series, build_events, build_plays, cached, cancel_trade, close_trade,
    code_to_settings, completed, context_lines, decimals, duplicate_open, evaluate, fetch, fetch_earnings,
    fetch_news, fetch_spread, fmp_events_raw, fmt, from_q, index_state_raw, infer, intraday_rvol,
    journal_stats, limit_state, merge_trades, money, new_trade, send_ntfy, settings_to_code, to_q,
    trades_from_csv, trades_to_csv, valid_topic, zone_points,
)

st.set_page_config(page_title="Zone Ticket", page_icon="📈", layout="centered")

JOURNAL_PATH = Path(os.environ.get("ZT_JOURNAL") or Path(__file__).with_name("journal.json"))
SCAN_CACHE_SECS = 180   # price data reused for this long, so a quick re-scan does not hit Yahoo again
SCAN_WORKERS = 4        # fewer parallel requests, so Yahoo is less likely to rate-limit a big scan


def polite(fn, *a, **kw):
    """Small random pause before a Yahoo request so the parallel workers do not fire in a burst."""
    time.sleep(random.uniform(0.05, 0.35))
    return fn(*a, **kw)


@st.cache_resource
def data_store():
    """One cache shared by every scan, so repeated scans and the single-ticker lookup reuse data."""
    return dict(lock=threading.Lock(), d={})


# ---------------- saved settings (kept in the page address) ----------------
if "_init" not in st.session_state:
    q = st.query_params
    for k, d in DEFAULTS.items():
        val = d
        if k in q:
            try:
                val = from_q(k, q[k])
            except Exception:
                val = d
        st.session_state[k] = val
    st.session_state["_saved"] = {k: st.session_state[k] for k in DEFAULTS}
    st.session_state["_init"] = True


def flash(msg):
    """Show a message at the top of the next run, because toasts do not survive a rerun."""
    st.session_state.setdefault("_flash", []).append(msg)


def current_settings():
    return {k: st.session_state[k] for k in DEFAULTS}


def reset_settings():
    for k, d in DEFAULTS.items():
        st.session_state[k] = d


def save_settings():
    """Write the current settings into the page address and remember them as saved."""
    cur = current_settings()
    try:
        st.query_params.from_dict({k: to_q(v) for k, v in cur.items()})
        st.session_state["_saved"] = cur
        flash("Settings saved. Bookmark this page to keep them.")
    except Exception:
        flash("Could not save settings.")


def apply_code():
    try:
        new = code_to_settings(st.session_state.get("import_code", ""))
    except ValueError as e:
        flash(str(e))
        return
    for k, v in new.items():
        st.session_state[k] = v
    st.session_state["import_code"] = ""
    flash(f"Applied {len(new)} settings. Tap Save settings to keep them.")


def ntfy_topic():
    t = str(st.session_state.get("ntfy_topic", "")).strip()
    if t:
        return t
    try:
        return str(st.secrets.get("NTFY_TOPIC", "")).strip()
    except Exception:
        return ""


def test_alert():
    try:
        send_ntfy(ntfy_topic(), "Zone Ticket test", "Alerts are working.")
        flash("Test alert sent. Check your phone.")
    except Exception as e:
        flash(f"Test alert failed: {e}")


# ---------------- journal storage ----------------
def jload():
    try:
        if JOURNAL_PATH.exists():
            data = json.loads(JOURNAL_PATH.read_text())
            if isinstance(data, list):
                st.session_state["journal_mem"] = data
                return data
    except Exception:
        pass
    return st.session_state.get("journal_mem", [])


def jsave(trades):
    st.session_state["journal_mem"] = trades
    try:
        tmp = JOURNAL_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(trades))
        tmp.replace(JOURNAL_PATH)
        return True
    except Exception:
        flash("Could not write the journal file, so it is only kept for this visit. Download a backup.")
        return False


def log_trade(p, s):
    trades = jload()
    if duplicate_open(trades, p):
        flash(f"{p.get('label', p['tkr'])} is already in your open trades at that price.")
        return
    trades.append(new_trade(p, s, st.session_state["acct"]))
    jsave(trades)
    flash(f"Logged {p.get('label', p['tkr'])}. It is in the Journal tab.")


def close_cb(tid):
    trades = jload()
    for t in trades:
        if t["id"] == tid and t["status"] == "open":
            close_trade(t, float(st.session_state[f"x_{tid}"]))
            flash(f"Closed {t['label']}: {t['r']:+.2f}R, {money(t['acct'] == 'AUD' and 'A$' or '$', t['pnl'])}")
    jsave(trades)


def cancel_cb(tid):
    trades = jload()
    for t in trades:
        if t["id"] == tid and t["status"] == "open":
            cancel_trade(t)
            flash(f"Removed {t['label']} (never filled).")
    jsave(trades)


# ---------------- data helpers (cached) ----------------
@st.cache_data(ttl=600, show_spinner=False)
def usd_rate(ccy):
    """USD value of one unit of ccy. Raises when it cannot be read, so a failure is not cached."""
    if ccy == "USD":
        return 1.0
    import yfinance as yf
    for sym, inv in ((f"{ccy}USD=X", False), (f"USD{ccy}=X", True)):
        try:
            c = yf.Ticker(sym).history(period="5d")["Close"].dropna()
            if len(c):
                return 1 / float(c.iloc[-1]) if inv else float(c.iloc[-1])
        except Exception:
            pass
    raise RuntimeError(f"no rate for {ccy}")


def acct_rate(ccy, acct):
    """Value of one unit of ccy in the account currency, or None."""
    try:
        return usd_rate(ccy) / usd_rate(acct)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fmp_events(key):
    return fmp_events_raw(key)


@st.cache_data(ttl=300, show_spinner=False)
def index_state():
    return index_state_raw()


@st.cache_data(ttl=300, show_spinner=False)
def load(tk, us):
    return fetch(tk, with_info=us)


def fmp_key():
    k = str(st.session_state.get("fmp_key_input", "")).strip()
    if k:
        return k
    try:
        return str(st.secrets.get("FMP_KEY", "")).strip()
    except Exception:
        return ""


def get_ctx(S):
    events, status = build_events(S["manual_ev"], fmp_key(), fmp_events)
    try:
        idx = index_state()
    except Exception:
        idx = None
    return dict(events=events, status=status, index=idx)


def enrich(p, us, stock, news, earn, news_ok=True):
    p["us"] = us
    p["news"] = news if us else []
    p["news_ok"] = news_ok
    p["earn"] = earn if stock else []


def md(s):
    """Streamlit treats a pair of dollar signs as maths, so escape them."""
    return str(s).replace("$", "\\$")


# ---------------- rendering ----------------
STYLE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, .stApp, button, input, textarea, select { font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif; }
.block-container { max-width: 780px; padding-top: 1.6rem; padding-bottom: 3rem; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* header */
.zt-hero { display: flex; align-items: center; gap: 14px; margin: 4px 0 18px; }
.zt-logo { width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center;
  font-weight: 700; letter-spacing: .02em; color: #fff; background: linear-gradient(135deg, #3b82f6, #6366f1);
  box-shadow: 0 6px 18px rgba(59, 130, 246, .35); }
.zt-title { font-size: 1.55rem; font-weight: 700; letter-spacing: -.02em; line-height: 1.15; }
.zt-sub { font-size: .85rem; opacity: .6; margin-top: 2px; }

/* controls */
.stButton > button, .stDownloadButton > button { border-radius: 10px; font-weight: 600; transition: transform .12s ease, box-shadow .12s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 16px rgba(0, 0, 0, .18); }
div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="textarea"] { border-radius: 10px; }
button[data-baseweb="tab"] { font-weight: 600; }
div[data-baseweb="tab-list"] { gap: 6px; }

/* cards */
[data-testid="stExpander"] details { border: 1px solid rgba(128, 128, 128, .24); border-radius: 14px; background: rgba(128, 128, 128, .045); overflow: hidden; }
[data-testid="stExpander"] details summary { font-weight: 600; padding: .7rem 1rem; }
[data-testid="stExpander"] details summary:hover { background: rgba(128, 128, 128, .08); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 14px; }
[data-testid="stMetric"] { background: rgba(128, 128, 128, .07); border: 1px solid rgba(128, 128, 128, .18); border-radius: 12px; padding: 10px 14px; }
[data-testid="stMetricLabel"] { opacity: .65; font-size: .72rem; text-transform: uppercase; letter-spacing: .05em; }
[data-testid="stMetricValue"] { font-weight: 650; font-size: 1.25rem; }

/* section headings and summary strip */
.zt-sec { display: flex; align-items: center; gap: 10px; margin: 26px 0 10px; font-size: 1.05rem; font-weight: 700; letter-spacing: -.01em; }
.zt-count { font-size: .75rem; font-weight: 700; padding: 2px 9px; border-radius: 999px; background: rgba(128, 128, 128, .18); }
.zt-kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 12px 0 4px; }
.zt-kpi { border: 1px solid rgba(128, 128, 128, .2); border-radius: 14px; padding: 12px 14px; background: rgba(128, 128, 128, .05); }
.zt-kpi-v { font-size: 1.45rem; font-weight: 700; line-height: 1.1; }
.zt-kpi-l { font-size: .7rem; opacity: .6; text-transform: uppercase; letter-spacing: .06em; margin-top: 4px; }
.zt-t-take .zt-kpi-v { color: #22c55e; }
.zt-t-block .zt-kpi-v { color: #f59e0b; }
@media (max-width: 560px) { .zt-kpis { grid-template-columns: repeat(2, 1fr); } }

/* play card */
.zt-head { display: flex; align-items: baseline; gap: 12px; margin: 2px 0 2px; }
.zt-side { font-size: .72rem; font-weight: 700; letter-spacing: .06em; padding: 3px 10px; border-radius: 8px; }
.zt-long { background: rgba(34, 197, 94, .16); color: #22c55e; }
.zt-short { background: rgba(239, 68, 68, .16); color: #ef4444; }
.zt-px { font-size: 1.6rem; font-weight: 700; letter-spacing: -.02em; }
.zt-pill { display: inline-block; font-size: .7rem; font-weight: 700; letter-spacing: .07em; padding: 3px 10px; border-radius: 999px; }
.zt-take { background: rgba(34, 197, 94, .18); color: #22c55e; }
.zt-block { background: rgba(245, 158, 11, .18); color: #f59e0b; }
.zt-skip { background: rgba(148, 163, 184, .2); color: #94a3b8; }
.zt-verdict { border-radius: 12px; padding: 12px 14px; margin: 10px 0; border: 1px solid rgba(128, 128, 128, .22); }
.zt-v-take { background: rgba(34, 197, 94, .09); border-color: rgba(34, 197, 94, .35); }
.zt-v-block { background: rgba(245, 158, 11, .08); border-color: rgba(245, 158, 11, .35); }
.zt-v-skip { background: rgba(148, 163, 184, .08); }
.zt-vh { display: flex; align-items: center; gap: 10px; font-weight: 600; }
.zt-fail { margin-top: 10px; padding-left: 10px; border-left: 3px solid rgba(239, 68, 68, .6); }
.zt-fail-d { font-size: .88rem; font-weight: 600; }
.zt-fail-w { font-size: .78rem; opacity: .65; margin-top: 2px; }

/* check list */
.zt-details { margin: 8px 0 12px; border: 1px solid rgba(128, 128, 128, .2); border-radius: 12px; padding: 4px 12px; }
.zt-details summary { cursor: pointer; font-weight: 600; font-size: .85rem; padding: 8px 0; }
.zt-row { display: flex; gap: 10px; align-items: flex-start; padding: 7px 0; border-top: 1px solid rgba(128, 128, 128, .14); }
.zt-st { flex: 0 0 auto; min-width: 42px; text-align: center; font-size: .65rem; font-weight: 700; letter-spacing: .06em; padding: 3px 0; border-radius: 6px; margin-top: 2px; }
.zt-st-pass { background: rgba(34, 197, 94, .18); color: #22c55e; }
.zt-st-fail { background: rgba(239, 68, 68, .18); color: #ef4444; }
.zt-st-off { background: rgba(148, 163, 184, .18); color: #94a3b8; }
.zt-rl { font-size: .85rem; font-weight: 600; }
.zt-rd { font-size: .78rem; opacity: .65; }
.zt-chips { display: flex; flex-wrap: wrap; gap: 6px; padding: 10px 0 6px; border-top: 1px solid rgba(128, 128, 128, .14); }
.zt-chip { font-size: .72rem; padding: 3px 10px; border-radius: 999px; background: rgba(128, 128, 128, .14); opacity: .7; }
.zt-chip.zt-on { background: rgba(34, 197, 94, .16); color: #22c55e; opacity: 1; }

/* single ticker quote and footer */
.zt-quote { display: flex; align-items: baseline; gap: 12px; margin: 6px 0 2px; }
.zt-q-name { font-size: 1.2rem; font-weight: 700; }
.zt-q-px { font-size: 1.2rem; font-weight: 500; opacity: .75; }
.zt-foot { margin-top: 28px; padding-top: 14px; border-top: 1px solid rgba(128, 128, 128, .2); font-size: .75rem; opacity: .55; line-height: 1.5; }
</style>
"""


def esc(s):
    """Escape text for the HTML snippets. Headlines and reasons come from outside, so never skip this.
    Dollar signs become entities because Streamlit would otherwise read a pair of them as maths."""
    return _html.escape(str(s), quote=True).replace("$", "&#36;")


def pill(text, kind):
    return f'<span class="zt-pill zt-{kind}">{esc(text)}</span>'


def section(title, count=None):
    badge = "" if count is None else f'<span class="zt-count">{esc(count)}</span>'
    st.markdown(f'<div class="zt-sec"><span>{esc(title)}</span>{badge}</div>', unsafe_allow_html=True)


def kpi_row(items):
    cells = "".join(
        f'<div class="zt-kpi zt-t-{tone}"><div class="zt-kpi-v">{esc(val)}</div>'
        f'<div class="zt-kpi-l">{esc(lab)}</div></div>' for lab, val, tone in items)
    st.markdown(f'<div class="zt-kpis">{cells}</div>', unsafe_allow_html=True)


NUMBER_LABELS = {"Passes the screen", "Zone score", "Reward : risk", "Affordable size",
                 "Leverage within limit", "Margin under 50%"}


def verdict_html(p, why):
    """TAKE, BLOCKED or SKIP banner, with every failing check and the reason it matters."""
    fails = [(lab, det) for lab, st_, det in (p.get("checks") or []) if st_ is False]
    if not fails and why:
        fails = [("", w) for w in why]
    if not fails:
        return ('<div class="zt-verdict zt-v-take"><div class="zt-vh">' + pill("TAKE", "take")
                + '<span>Every check passes</span></div></div>')
    skip = any(lab in NUMBER_LABELS for lab, _ in fails)
    kind, name = ("skip", "SKIP") if skip else ("block", "BLOCKED")
    rows = "".join(
        f'<div class="zt-fail"><div class="zt-fail-d">{esc(det)}</div>'
        + (f'<div class="zt-fail-w">{esc(CHECK_WHY[lab])}</div>' if lab in CHECK_WHY else "") + "</div>"
        for lab, det in fails)
    n = len(fails)
    return (f'<div class="zt-verdict zt-v-{kind}"><div class="zt-vh">{pill(name, kind)}'
            f'<span>{n} check{"s" if n != 1 else ""} failing</span></div>{rows}</div>')


def checks_html(p):
    """Collapsible list of every check with PASS, FAIL or OFF."""
    checks = p.get("checks")
    if not checks:
        return ""
    n_pass = sum(1 for _, s_, _ in checks if s_ is True)
    n_fail = sum(1 for _, s_, _ in checks if s_ is False)
    names = {True: ("pass", "PASS"), False: ("fail", "FAIL"), None: ("off", "OFF")}
    rows = ""
    for label, state, detail in checks:
        kind, word = names[state]
        rows += (f'<div class="zt-row"><span class="zt-st zt-st-{kind}">{word}</span>'
                 f'<div><div class="zt-rl">{esc(label)}</div><div class="zt-rd">{esc(detail)}</div></div></div>')
    chips = "".join(
        f'<span class="zt-chip{" zt-on" if ok else ""}">{"✓" if ok else "–"} {esc(lab)}</span>'
        for lab, ok in zone_points(p))
    return (f'<details class="zt-details"><summary>All checks · {n_pass} pass · {n_fail} fail</summary>'
            f'{rows}<div class="zt-chips">{chips}</div></details>')


def play_body(p, price, s, why, S, key):
    long = p["side"] == "long"
    nd = p.get("nd", 2)
    dist = abs(price - p["prox"]) / price * 100
    st.markdown(
        f'<div class="zt-head"><span class="zt-side {"zt-long" if long else "zt-short"}">'
        f'{"BUY" if long else "SELL SHORT"}</span><span class="zt-px">{esc(fmt(p["prox"], nd))}</span></div>',
        unsafe_allow_html=True)
    st.caption(f"{'Demand' if long else 'Supply'} zone, {dist:.1f}% from price. Use a limit order.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Stop", fmt(s["stop"], nd))
    c2.metric("Sell target" if long else "Cover target", fmt(p["tgt"], nd))
    c3.metric("Reward : risk", f"{p['rr']:.1f} : 1")
    c4, c5, c6 = st.columns(3)
    c4.metric("Units", f"{s['units']:,}")
    cur = s["cur"]
    c5.metric("Margin", f"{cur}{s['margin']:,.0f}")
    c6.metric("Loss at stop", f"{cur}{s['loss']:,.0f}")
    extra = f" · about {s['units'] / 100000:.2f} standard lots" if p.get("kind") == "forex" else ""
    st.write(md(f"Zone score {p['score']}/5 · gain at target {cur}{s['gain']:,.0f} · leverage used {s['lev']:.1f}x{extra}"))
    if s["approx"]:
        st.caption(f"Could not read the {s['ccy']} to {s['acct']} rate, so size and margin are approximate.")
    lines = context_lines(p, S)
    if lines:
        st.caption(md("\n\n".join(lines)))
    st.markdown(verdict_html(p, why), unsafe_allow_html=True)
    chk = checks_html(p)
    if chk:
        st.markdown(chk, unsafe_allow_html=True)
    if st.button("Log this trade", key=f"log_{key}", disabled=s["units"] < 1,
                 help="Adds it to the Journal tab so it counts toward your limits and results."):
        log_trade(p, s)
        st.rerun()


def render_banner(ctx, S, lim=None):
    now = datetime.now(timezone.utc)
    idx = ctx.get("index")
    if idx is None:
        st.caption("SPY and QQQ could not be read, so the index check is skipped.")
    elif not idx["open"]:
        st.caption("US market is closed. Volume, VWAP and index checks use the last session.")
    elif idx["down"]:
        st.warning(f"{'/'.join(idx['down'])} selling off. Buy signals on US stocks are held back.")
    elif idx["up"]:
        st.warning(f"{'/'.join(idx['up'])} spiking. Short signals on US stocks are held back.")
    m = S["ev_min"]
    if S["f_events"]:
        for e in sorted(ctx.get("events", []), key=lambda x: x["t"]):
            mins = (e["t"] - now).total_seconds() / 60
            when = f"{e['name']} at {e['t'].astimezone(SYD):%a %H:%M} Sydney"
            if -m <= mins <= m:
                st.error(f"Blackout active: {when} ({mins:+.0f} min). Signals for affected markets are suppressed.")
            elif m < mins <= 180:
                st.warning(f"Coming up: {when}, in {int(mins // 60)}h {int(mins % 60)}m.")
    if lim and S["f_limits"]:
        if lim["halt"]:
            st.error(md("Trading limits: " + "; ".join(lim["halt"]) + ". New signals are held back."))
        else:
            st.caption(md(f"Today {money(lim['cur'], lim['day_pnl'])} · {lim['open_n']} of {lim['max_open']} trades open · "
                          f"open risk {money(lim['cur'], lim['open_risk'])} of {money(lim['cur'], lim['risk_cap'])}"))
    st.caption(ctx["status"])


def make_cfg():
    return dict(equity=st.session_state["equity"], risk_pct=st.session_state["risk_pct"],
                levs={m: st.session_state[f"lev_{m}"] for m in MARKETS}, acct=st.session_state["acct"],
                cur="A$" if st.session_state["acct"] == "AUD" else "$",
                alerts=st.session_state["alerts_on"], topic=ntfy_topic())


def rows_for(plays, cfg, S, ctx):
    lim = limit_state(jload(), S, cfg["equity"], cfg["cur"])
    rows = evaluate(plays, cfg["equity"], cfg["risk_pct"], cfg["levs"], cfg["acct"], S, ctx, acct_rate, lim)
    return rows, lim


def maybe_alert(scan, S, cfg):
    """Push one notification for TAKE plays that have not been announced yet today."""
    rows, _ = rows_for(scan["plays"], cfg, S, scan["ctx"])
    takes = [r for r in rows if not r[2] and not r[3]]
    day = datetime.now(SYD).date()
    seen = st.session_state.setdefault("alerted", set())
    fresh = [r for r in takes if (r[0]["tkr"], r[0]["side"], r[0]["prox"], day) not in seen]
    if not fresh:
        return
    try:
        send_ntfy(cfg["topic"], f"Zone Ticket: {len(fresh)} TAKE", "\n".join(alert_lines(fresh)), "high")
        seen.update((r[0]["tkr"], r[0]["side"], r[0]["prox"], day) for r in fresh)
        scan["alert"] = f"Alert sent for {len(fresh)} play{'s' if len(fresh) != 1 else ''}."
    except Exception as e:
        scan["alert"] = f"Alert failed: {e}"


def do_scan(selected, watchlist, extras, S, cfg, max_dist=None):
    max_dist = max_dist or S["max_dist"]
    jobs, seen = [], set()

    def add(sym, label, m, kind):
        if sym and sym not in seen:
            seen.add(sym)
            jobs.append((sym, label, m, kind))

    for m in selected:
        kind = MARKETS[m][0]
        if m == "Stocks":
            for sym in watchlist.replace(",", " ").upper().split():
                add(sym, sym, m, kind)
        else:
            for sym, label in MARKETS[m][2]:
                add(sym, label, m, kind)
    for sym in extras.replace(",", " ").upper().split():
        label, kind, m = infer(sym)
        add(sym, label, m, kind)

    t0 = time.time()
    ctx = get_ctx(S)
    store = data_store()
    skipped_us = 0
    idx = ctx.get("index")
    if S["f_stale"] and idx is not None and not idx["open"]:
        # the stale-data filter would block every US instrument anyway, so do not spend requests on them
        kept = [j for j in jobs if j[2] not in US_MARKETS]
        skipped_us = len(jobs) - len(kept)
        jobs = kept

    def work(job):
        sym, label, m, kind = job
        us = m in US_MARKETS
        try:
            d15, d1h, dd, _ = cached(store, ("px", sym), SCAN_CACHE_SECS, lambda: polite(fetch, sym, with_info=False))
            if len(d15) < 60 or len(d1h) < 10 or len(dd) < 25:
                return sym, [], "not enough data"
            ps = build_plays(sym, d15, d1h, dd, None, max_dist, kind=kind)[:int(S["per_inst"])]
            news, earn, news_ok = [], [], True
            if us and ps and ps[0]["score"] >= 3:
                # only worth the extra requests for instruments with a real candidate
                spr = cached(store, ("spr", sym), 3600, lambda: polite(fetch_spread, sym))
                for p in ps:
                    p["spr"] = None if spr is None else round(spr, 3)
                try:
                    news = cached(store, ("news", sym), 600, lambda: polite(fetch_news, sym))
                except Exception:
                    news_ok = False
                if m == "Stocks":
                    try:
                        earn = cached(store, ("earn", sym), 21600, lambda: polite(fetch_earnings, sym))
                    except Exception:
                        pass
            for p in ps:
                p["label"], p["market"] = label, m
                enrich(p, us, m == "Stocks", news, earn, news_ok)
            return sym, ps, None
        except Exception as e:
            return sym, [], str(e)[:80]

    plays, errors = [], []
    if jobs:
        bar = st.progress(0.0, text=f"Scanning {len(jobs)} instruments...")
        with ThreadPoolExecutor(max_workers=SCAN_WORKERS) as ex:
            futs = [ex.submit(work, j) for j in jobs]
            for i, fu in enumerate(as_completed(futs), 1):
                sym, ps, err = fu.result()
                plays += ps
                if err:
                    errors.append(f"{sym}: {err}")
                bar.progress(i / len(jobs), text=f"Scanned {i} of {len(jobs)}")
        bar.empty()
    scan = dict(ts=time.time(), plays=plays, errors=errors, n=len(jobs), ctx=ctx, secs=time.time() - t0,
                skipped_us=skipped_us)
    st.session_state["scan"] = scan
    if cfg["alerts"] and valid_topic(cfg["topic"]):
        maybe_alert(scan, S, cfg)


def render_scan(cfg, S):
    s = st.session_state.get("scan")
    if not s:
        st.info("Pick your markets and tap Scan now.")
        return
    ctx = s["ctx"]
    when = datetime.fromtimestamp(s["ts"], SYD).strftime("%a %H:%M")
    st.caption(f"Last scan {when} Sydney time · {s['n']} instruments in {s.get('secs', 0):.0f}s · prices delayed")
    if s.get("skipped_us"):
        st.caption(f"Skipped {s['skipped_us']} US stocks and ETFs because the US market is closed. "
                   "They are scanned again once it opens.")
    if s.get("alert"):
        st.caption(s["alert"])
    rows, lim = rows_for(s["plays"], cfg, S, ctx)
    render_banner(ctx, S, lim)
    now = datetime.now(timezone.utc)
    stale = [p for p in s["plays"] if p.get("bar_ts") is not None
             and (now - p["bar_ts"]).total_seconds() / 60 > S["stale_min"]]
    if s["plays"] and len(stale) >= 0.6 * len(s["plays"]):
        st.warning(f"{len(stale)} of {len(s['plays'])} plays use candles older than {S['stale_min']:.0f} minutes, "
                   "so those markets are probably closed." +
                   (" The stale-data filter is holding them back." if S["f_stale"] else ""))
    skipped = sorted({p["tkr"] for p in s["plays"] if p.get("news_ok") is False})
    if skipped and S["f_news"]:
        st.warning("Headlines could not be read for " + ", ".join(skipped[:8]) +
                   (" and more" if len(skipped) > 8 else "") + ". The headline check was skipped for them.")

    view = (st.session_state["sort_by"], st.session_state["dir_f"], st.session_state["min_rr"])
    takes = apply_view([r for r in rows if not r[2] and not r[3]], *view)
    blocked = apply_view([r for r in rows if not r[2] and r[3]], *view)
    close = apply_view([r for r in rows if r[2] and r[0]["score"] >= S["min_score"] - 1
                        and r[0]["rr"] >= S["min_rr_take"] - 1], *view)

    def show(items, tag):
        for i, (p, sz, tech, flt) in enumerate(items):
            long = p["side"] == "long"
            colour = {"TAKE": "green", "BLOCKED": "orange"}.get(tag, "gray")
            title = (f":{colour}[**{tag}**] · {'▲ BUY' if long else '▼ SELL SHORT'} {p['label']} @ {fmt(p['prox'], p['nd'])}"
                     f" · {p['rr']:.1f}R · {p['score']}/5")
            if tag == "BLOCKED":
                title += f" · {flt[0][:48]}"
            with st.expander(md(title)):
                play_body(p, p["px"], sz, tech + flt, S, key=f"{tag}_{i}_{p['tkr']}_{p['side']}_{p['prox']}")

    kpi_row([("Take", len(takes), "take"), ("Blocked", len(blocked), "block"),
             ("Scanned", s["n"], ""), ("Last scan", when, "")])
    section("Take", len(takes))
    if takes:
        show(takes, "TAKE")
    else:
        st.info("Nothing passes every check right now. That is normal. Waiting is part of the strategy.")
    if blocked:
        section("Blocked by filters", len(blocked))
        st.caption("These pass the zone rules but a filter says wait. Open one to see why.")
        show(blocked, "BLOCKED")
    if st.session_state["show_close"]:
        section("Close calls", len(close))
        show(close, "WATCH")
    if s["errors"]:
        with st.expander(f"{len(s['errors'])} instruments could not load"):
            st.write("\n".join(s["errors"]))


def snapshot(tk, d15, d1h, dd, spread, kind, S):
    price = float(d15.Close.iloc[-1])
    atr_pct = float(atr_series(dd).iloc[-1] / dd.Close.iloc[-1] * 100)
    lo, hi = ATR_RANGE[kind]
    nd = decimals(price, tk)
    if kind != "stock":
        return price, nd, [(f"Daily range {lo}-{hi}%", f"{atr_pct:.2f}%", lo <= atr_pct <= hi)]
    adv = float(dd.Volume.iloc[-21:-1].mean())
    rvd = float(dd.Volume.iloc[-1] / adv) if adv else 0.0
    _, cum = intraday_rvol(completed(d15))
    rv = cum if cum is not None else rvd
    rv_label = (f"Relative volume {SCREEN_RVOL:g}x+ (same time of day)" if cum is not None
                else f"Relative volume {SCREEN_RVOL:g}x+ (daily)")
    return price, nd, [
        ("Price above $5", f"${price:,.2f}", price > 5),
        ("Avg volume 1M+", f"{adv:,.0f}", adv >= 1e6),
        (rv_label, f"{rv:.1f}x", rv >= SCREEN_RVOL),
        (f"Daily range {lo}-{hi}%", f"{atr_pct:.1f}%", lo <= atr_pct <= hi),
        (f"Spread {S['max_spread']:.3f}% or less", "n/a" if spread is None else f"{spread:.3f}%",
         None if spread is None else spread <= S["max_spread"]),
    ]


def render_journal(cfg, S):
    trades = jload()
    cur = cfg["cur"]
    lim = limit_state(trades, S, cfg["equity"], cur)
    st.caption("Log trades from any play with the Log this trade button. Close them here when you exit. "
               "Results and your trading limits are calculated from this list.")
    if S["f_limits"]:
        if lim["halt"]:
            st.error(md("Trading limits: " + "; ".join(lim["halt"]) + "."))
    a, b, c = st.columns(3)
    a.metric("Today", money(cur, lim["day_pnl"]))
    b.metric("Open trades", f"{lim['open_n']} / {lim['max_open']}")
    c.metric("Open risk", money(cur, lim["open_risk"]))

    stats = journal_stats(trades)
    st.subheader("Results")
    if not stats["n"]:
        st.info("No closed trades yet. Results appear here once you close some.")
    else:
        d1, d2, d3 = st.columns(3)
        d1.metric("Closed trades", stats["n"])
        d2.metric("Win rate", f"{stats['win_rate']:.0f}%")
        d3.metric("Average R", f"{stats['avg_r']:+.2f}")
        e1, e2, e3 = st.columns(3)
        e1.metric("Total P&L", money(cur, stats["total_pnl"]))
        e2.metric("Profit factor", "n/a" if stats["profit_factor"] is None else f"{stats['profit_factor']:.2f}")
        e3.metric("Max drawdown", f"{stats['max_dd_r']:.1f}R")
        st.caption(f"Average win {stats['avg_win_r']:+.2f}R · average loss {stats['avg_loss_r']:+.2f}R · "
                   f"total {stats['total_r']:+.1f}R")
        if stats["n"] >= 2:
            st.line_chart(pd.DataFrame({"Cumulative R": stats["curve"]}))
        if stats["n"] < 30:
            st.caption("Under 30 closed trades is too few to trust win rate or profit factor. Keep logging.")

    st.subheader("Open trades")
    open_t = [t for t in trades if t["status"] == "open"]
    if not open_t:
        st.info("Nothing open.")
    for t in open_t:
        nd = t.get("nd", 2)
        long = t["side"] == "long"
        with st.container(border=True):
            st.markdown(md(f"**{'BUY' if long else 'SELL SHORT'} {t['label']}** at {fmt(t['entry'], nd)}"))
            logged = datetime.fromisoformat(t["logged"]).astimezone(SYD)
            st.caption(md(f"Stop {fmt(t['stop'], nd)} · target {fmt(t['target'], nd)} · {t['units']:,.0f} units · "
                          f"risk {money(cur, t['risk'])} · logged {logged:%a %H:%M} Sydney"))
            x, y, z = st.columns([2, 1, 1])
            x.number_input("Exit price", value=float(t["entry"]), min_value=0.0, step=float(10 ** -nd),
                           format=f"%.{nd}f", key=f"x_{t['id']}", label_visibility="collapsed")
            y.button("Close", key=f"c_{t['id']}", on_click=close_cb, args=(t["id"],), use_container_width=True,
                     help="Closes the trade at the exit price in the box.")
            z.button("Not filled", key=f"n_{t['id']}", on_click=cancel_cb, args=(t["id"],), use_container_width=True,
                     help="Removes it without counting it, for limit orders that never triggered.")

    closed = sorted((t for t in trades if t["status"] == "closed"), key=lambda t: t["closed"], reverse=True)
    if closed:
        st.subheader("Closed trades")
        df = pd.DataFrame([{
            "Closed": datetime.fromisoformat(t["closed"]).astimezone(SYD).strftime("%d %b %H:%M"),
            "Instrument": t["label"], "Side": t["side"], "Entry": t["entry"], "Exit": t["exit"],
            "R": t["r"], "P&L": t["pnl"], "Score": t["score"]} for t in closed])
        st.dataframe(df, hide_index=True, use_container_width=True)

    st.subheader("Backup")
    st.caption("The journal is saved in journal.json next to app.py. Hosted services such as Streamlit Community "
               "Cloud wipe that file when the app restarts, so download a backup regularly.")
    if trades:
        st.download_button("Download journal (CSV)", data=trades_to_csv(trades), mime="text/csv",
                           file_name=f"zone-ticket-journal-{datetime.now(SYD):%Y-%m-%d}.csv",
                           use_container_width=True)
    up = st.file_uploader("Restore from a journal CSV", type="csv", key="journal_upload")
    if up is not None and st.button("Import file", use_container_width=True):
        try:
            incoming = trades_from_csv(up.getvalue().decode("utf-8-sig"))
            merged, added = merge_trades(trades, incoming)
            jsave(merged)
            flash(f"Imported {added} new trade{'s' if added != 1 else ''}"
                  f"{'' if added == len(incoming) else f', {len(incoming) - added} already there'}.")
        except Exception:
            flash("Could not read that file.")
        st.rerun()


# ---------------- page ----------------
for _m in st.session_state.pop("_flash", []):
    st.toast(_m)

st.markdown(STYLE, unsafe_allow_html=True)
st.markdown('<div class="zt-hero"><div class="zt-logo">ZT</div><div><div class="zt-title">Zone Ticket</div>'
            '<div class="zt-sub">Supply and demand day-trade levels. Scan the markets or look up one ticker.</div>'
            '</div></div>', unsafe_allow_html=True)
_dirty = current_settings() != st.session_state.get("_saved")
st.button("Save settings" + (" (unsaved changes)" if _dirty else ""), type="primary" if _dirty else "secondary",
          disabled=not _dirty, on_click=save_settings, use_container_width=True, key="save_btn")


def num(label, key, step, **kw):
    lo, hi = BOUNDS[key]
    return st.number_input(label, min_value=lo, max_value=hi, step=step, key=key, **kw)


with st.expander("Settings"):
    st.selectbox("Account currency", ["AUD", "USD"], key="acct")
    num("Account equity", "equity", 500.0)
    num("Risk per trade (%)", "risk_pct", 0.25)
    st.caption("Leverage limits. Defaults are typical Australian retail limits. Use your broker's real numbers.")
    for m in MARKETS:
        num(f"{m} max leverage (x)", f"lev_{m}", 1.0)
    st.text_area("Stock watchlist (tickers separated by spaces)", key="watch")
    st.text_area("Extra symbols to scan every time", key="extras",
                 help="Any Yahoo Finance symbol, for example AUDNZD=X, NG=F, BTC-USD or a stock like UBER.")
    st.markdown("**Phone alerts**")
    st.toggle("Send a push alert when a new TAKE appears", key="alerts_on")
    st.text_input("ntfy topic", key="ntfy_topic", type="password",
                  help="Install the free ntfy app, subscribe to a long random topic name, and enter the same name here. "
                       "Anyone who knows the topic can read your alerts, so treat it like a password. "
                       "To keep it out of the page address, add NTFY_TOPIC in the app's Secrets settings instead.")
    st.button("Send test alert", on_click=test_alert, disabled=not valid_topic(ntfy_topic()))
    st.caption("Alerts are sent after each scan, so they only arrive while this page is open with hourly rescans on.")
    st.markdown("**Move settings to another device**")
    st.caption("Copy this code, then paste it on the other device and tap Apply. It includes your ntfy topic, "
               "so keep it private.")
    st.code(settings_to_code(current_settings()), language=None)
    st.text_input("Paste a settings code", key="import_code")
    st.button("Apply code", on_click=apply_code, disabled=not st.session_state.get("import_code", "").strip())
    st.button("Reset to defaults", on_click=reset_settings)
    st.caption("Tap Save settings at the top to keep your changes. They are stored in this page's web address, "
               "so bookmark the page or add it to your home screen again after saving.")

with st.expander("Filters"):
    st.markdown("**What counts as a TAKE**")
    num("Minimum zone score (out of 5)", "min_score", 1.0, format="%.0f")
    num("Minimum reward : risk", "min_rr_take", 0.5)
    num("How far from price to look for zones (15m ATR)", "max_dist", 1.0, format="%.0f")
    num("Plays kept per instrument", "per_inst", 1.0, format="%.0f")
    st.markdown("**Filters**")
    st.caption("Each filter can hold a play back. Turn one off to see what it was hiding.")
    st.toggle("Event blackouts (FOMC, CPI, jobs, your own times)", key="f_events")
    num("Blackout window, minutes either side", "ev_min", 5.0)
    st.text_area("Your own blackout times, Sydney time, one per line", key="manual_ev",
                 placeholder="2026-10-22 23:30 US CPI",
                 help="Format: YYYY-MM-DD HH:MM then a name. Affects every market.")
    st.text_input("FMP API key for CPI, PPI and jobs data (this visit only)", type="password", key="fmp_key_input",
                  help="Not saved. To keep it, add FMP_KEY in the app's Secrets settings on Streamlit.")
    st.toggle("Block stale data (market closed or feed delayed)", key="f_stale")
    num("Stale after (minutes since the last candle started)", "stale_min", 5.0, format="%.0f")
    st.toggle("Headline check for US stocks and ETFs", key="f_news")
    num("Headline window, hours", "news_hrs", 0.5)
    st.toggle("Block earnings day and the day after", key="f_earn")
    st.toggle("Block weak volume at the zone", key="f_rvol")
    num("Weak volume below (x normal for the time of day)", "rv_weak", 0.1)
    num("Max bid-ask spread for US stocks and ETFs (%)", "max_spread", 0.01, format="%.3f")
    st.toggle("Session VWAP side", key="f_vwap")
    num("How far on the wrong side of VWAP is allowed (15m ATR)", "vwap_tol", 0.25)
    st.toggle("Block counter-trend plays (1h structure)", key="f_htf")
    st.toggle("Hold plays while SPY or QQQ is selling off or spiking", key="f_index")
    st.toggle("Zone must have left strongly", key="f_depart")
    num("Minimum departure (ATR)", "zq_depart", 0.5)
    st.toggle("Block old zones", key="f_age")
    num("Oldest zone allowed (hours of trading)", "zq_age", 4.0)
    st.toggle("Block zones inside a bigger opposing zone", key="f_nest")
    st.toggle("Only take after a rejection candle at the zone", key="f_react",
              help="Price must have tagged the zone and closed back out with a long wick or engulfing candle. "
                   "With this on, plays waiting for price to arrive show under Blocked until it happens.")
    st.markdown("**Trading limits** (use the Journal to track them)")
    st.toggle("Enforce trading limits", key="f_limits",
              help="Holds back every signal once you hit a limit. Limits come from the trades you log in the Journal.")
    num("Daily loss limit (% of equity, Sydney day)", "day_loss_pct", 0.5)
    num("Max open trades", "max_open", 1.0, format="%.0f")
    num("Max total open risk (% of equity)", "max_open_risk", 0.5)

with st.expander("Why a play is TAKE or BLOCKED"):
    st.markdown("A play is **TAKE** when every check passes, **BLOCKED** when its numbers are fine but a filter "
                "says wait, and **SKIP** when it fails on its own numbers. Open any play to see each check "
                "marked PASS, FAIL or OFF. These are the reasons behind them:")
    for _label, _why in CHECK_WHY.items():
        st.markdown(md(f"**{_label}**: {_why}"))

S = {k: st.session_state[k] for k in FILTERS}
cfg = make_cfg()
acct, equity, risk_pct, levs = cfg["acct"], cfg["equity"], cfg["risk_pct"], cfg["levs"]

tab_scan, tab_one, tab_journal = st.tabs(["Scan markets", "One ticker", "Journal"])

with tab_scan:
    selected = st.multiselect("Markets", list(MARKETS), key="sel")
    watchlist, extras = st.session_state["watch"], st.session_state["extras"]
    st.toggle("Show close calls", key="show_close")
    auto = st.toggle("Rescan every hour while this page is open", key="auto")
    v1, v2, v3 = st.columns([3, 3, 2])
    v1.selectbox("Sort by", SORTS, key="sort_by")
    v2.radio("Direction", DIRS, horizontal=True, key="dir_f")
    v3.number_input("Min R:R", min_value=BOUNDS["min_rr"][0], max_value=BOUNDS["min_rr"][1], step=0.5,
                    key="min_rr")
    can_scan = bool(selected) or bool(extras.strip())
    if st.button("Scan now", type="primary", use_container_width=True, disabled=not can_scan):
        do_scan(selected, watchlist, extras, S, cfg)

    @st.fragment(run_every=3600 if auto else None)
    def results():
        s = st.session_state.get("scan")
        if auto and s and time.time() - s["ts"] > 3300 and can_scan:
            do_scan(selected, watchlist, extras, S, cfg)
        render_scan(cfg, S)

    results()
    if auto:
        st.caption("Auto scan only runs while this page is open on your phone. It does not run in the background.")

with tab_one:
    tk = st.text_input("Ticker", placeholder="NVDA, EURUSD=X, GC=F, BTC-USD").strip().upper()
    if tk:
        label, kind, market = infer(tk)
        us = market in US_MARKETS
        with st.spinner(f"Reading {tk}..."):
            try:
                d15, d1h, dd, spread = load(tk, us)
                ok = len(d15) >= 60 and len(d1h) >= 10 and len(dd) >= 25
            except Exception:
                ok = False
        if not ok:
            st.error(f"Could not get enough data for {tk}. Check the ticker.")
        else:
            ctx = get_ctx(S)
            price, nd, rows1 = snapshot(tk, d15, d1h, dd, spread, kind, S)
            st.markdown(f'<div class="zt-quote"><span class="zt-q-name">{esc(label)}</span>'
                        f'<span class="zt-q-px">{esc(fmt(price, nd))}</span></div>', unsafe_allow_html=True)
            st.caption("Prices are delayed. Confirm on your chart before ordering.")
            lim1 = limit_state(jload(), S, equity, cfg["cur"])
            render_banner(ctx, S, lim1)
            with st.expander("Screen", expanded=any(r[2] is False for r in rows1)):
                for lab, val, res in rows1:
                    mark = ":gray[n/a]" if res is None else (":green[PASS]" if res else ":red[FAIL]")
                    st.write(md(f"{mark}  {lab}: **{val}**"))
            plays = build_plays(tk, d15, d1h, dd, spread, max_dist=S["max_dist"], kind=kind)
            news, earn, news_ok = [], [], True
            if us and plays:
                try:
                    news = fetch_news(tk)
                except Exception:
                    news_ok = False
                if market == "Stocks":
                    try:
                        earn = fetch_earnings(tk)
                    except Exception:
                        pass
            for head, side in (("Buy levels", "long"), ("Sell short levels", "short")):
                section(head)
                items = [p for p in plays if p["side"] == side][:int(S["per_inst"])]
                if not items:
                    st.info("No zone with a usable target right now.")
                for i, p in enumerate(items):
                    enrich(p, us, market == "Stocks", news, earn, news_ok)
                    p["label"], p["market"] = label, market
                    row = evaluate([p], equity, risk_pct, levs, acct, S, ctx, acct_rate, lim1)[0]
                    with st.container(border=True):
                        play_body(p, price, row[1], row[2] + row[3], S, key=f"one_{tk}_{side}_{i}")

with tab_journal:
    render_journal(cfg, S)

st.markdown('<div class="zt-foot">Leveraged products lose money for most retail traders. Test on a demo '
            'account first. This is a calculator, not financial advice.</div>', unsafe_allow_html=True)
