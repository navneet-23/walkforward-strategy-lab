"""
pit_reconstruct.py - full point-in-time index membership from the official
NSE Indices press-release archive (event-accurate, replaces the sparse
web-archive-snapshot reconstruction).

Run on your machine, from the Citadel folder (needs internet):

    pip install pdfplumber requests pandas pyarrow
    python pit_reconstruct.py --all          # download -> parse -> build
    python pit_reconstruct.py --download     # stages individually
    python pit_reconstruct.py --parse
    python pit_reconstruct.py --build

Outputs:
    pit_events/pdfs/           cached press-release PDFs
    pit_events/txt/            cached extracted text
    pit_events/events.csv      parsed events: date,index,action,symbol,...
    pit_events/parse_report.txt  validation: balance checks, flagged docs
    data_pit_wf/membership_v2.parquet   the new membership matrix
    pit_events/reconcile.txt   diff vs the old membership.parquet

Method: anchor at today's official constituent lists, then walk the
announced inclusion/exclusion events backward in time. Indices covered:
NIFTY 100 (CNX 100) and NIFTY Midcap 100 (CNX Midcap / Free Float
Midcap 100) - the union is the strategy universe.
"""

import argparse
import os
import re
import sys
from io import StringIO

import pandas as pd

BASE = "https://www.niftyindices.com/Press_Release/"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/120.0 Safari/537.36"),
    "Accept": "*/*",
}
ANCHOR_LISTS = {
    "N100": "https://www.niftyindices.com/IndexConstituent/ind_nifty100list.csv",
    "MC100": "https://www.niftyindices.com/IndexConstituent/ind_niftymidcap100list.csv",
}
# heading-core -> target index key
ALIAS = {
    "NIFTY 100": "N100", "CNX 100": "N100",
    "NIFTY MIDCAP 100": "MC100", "CNX MIDCAP": "MC100",
    "NIFTY FREE FLOAT MIDCAP 100": "MC100",
}

PDF_FILES = """
ind_prs15092026 ind_prs04092026 ind_prs10082026 ind_prs10082026_3 ind_prs04082026_3
ind_prs20072026 ind_prs17072026_1 ind_prs13072026_1 ind_prs24062026_1 ind_prs22062026
ind_prs19062026 ind_prs17062026 ind_prs17062026_2 ind_prs10062026 ind_prs21052026_1
ind_prs20052026_1 ind_prs08052026 ind_prs04052026 ind_prs23042026 ind_prs12032026
ind_prs02032026_2 ind_prs23022026 ind_prs20022026_1 ind_prs20022026_2 ind_prs20012026_2
ind_prs23122025 ind_prs11122025 ind_prs11122025_1 ind_prs01122025 ind_prs26112025
ind_prs17112025_1 ind_prs13112025 ind_prs20102025 ind_prs17102025 ind_prs22092025
ind_prs15092025 ind_prs15092025_1 ind_prs22082025 ind_prs22082025_1 ind_prs24072025
ind_prs08072025 ind_prs25062025_2 ind_prs20062025 ind_prs06062025 ind_prs21042025_1
ind_prs04042025_2 ind_prs17032025 ind_prs17032025_1 ind_prs13032025 ind_prs06032025
ind_prs21022025 ind_prs18022025 ind_prs06022025_1 ind_prs05022025_1
ind_prs11122024 ind_prs22112024 ind_prs10102024 ind_prs10102024_1 ind_prs04102024
ind_prs25092024 ind_prs23092024 ind_prs19092024 ind_prs13092024 ind_prs27082024
ind_prs23082024 ind_prs23082024_1 ind_prs24072024 ind_prs21062024_1 ind_prs07062024
ind_prs22052024 ind_prs24042024 ind_prs24042024_1 ind_prs19032024 ind_prs14032024
ind_prs28022024 ind_prs30012024 ind_prs19012024 ind_prs10012024
ind_prs07122023 ind_prs09112023 ind_prs17102023 ind_prs15092023 ind_prs05092023
ind_prs23082023 ind_prs17082023 ind_prs24072023 ind_prs04072023 ind_prs27062023
ind_prs09062023 ind_prs19042023_1 ind_prs06032023 ind_prs21022023 ind_prs17022023_1
ind_prs09022023_1
ind_prs22122022 ind_prs06122022 ind_prs20102022 ind_prs20102022_1 ind_prs16092022
ind_prs01092022 ind_prs23082022 ind_prs11072022 ind_prs15062022 ind_prs24052022
ind_prs06042022 ind_prs05042022 ind_prs05042022_1 ind_prs08032022 ind_prs24022022_1
ind_prs08122021 ind_prs22102021 ind_prs20092021 ind_prs15092021 ind_prs23082021
ind_prs15062021 ind_prs22042021 ind_prs10032021 ind_prs23022021
ind_prs11122020 ind_prs18112020 ind_prs26102020 ind_prs30092020 ind_prs07092020
ind_prs20082020 ind_prs02072020_1 ind_prs10062020 ind_prs13052020 ind_prs19032020
ind_prs16032020 ind_prs12032020 ind_prs18022020 ind_prs09012020
ind_prs19122019 ind_prs18122019 ind_prs16122019 ind_prs17092019 ind_prs17092019_1
ind_prs28082019 ind_prs20082019 ind_prs13062019 ind_prs08042019 ind_prs20032019
ind_prs13032019 ind_prs25022019 ind_prs20022019 ind_prs21012019
ind_prs14122018 ind_prs24092018 ind_prs31082018 ind_prs28082018 ind_prs01082018
ind_prs20062018 ind_prs15062018 ind_prs24052018 ind_prs06032018 ind_prs21022018
ind_prs08012018
ind_prs18122017 ind_prs28112017 ind_prs10112017 ind_prs03112017 ind_prs16102017
ind_prs29082017 ind_prs28082017 ind_prs15062017 ind_prs09052017 ind_prs27042017
ind_prs07042017 ind_prs07032017 ind_prs16022017 ind_prs16012017 ind_prs03012017
ind_prs09112016 ind_prs17102016 ind_prs12082016 ind_prs18072016 ind_prs28042016
ind_prs22042016 ind_prs22022016_2 ind_prs11012016
ind_prs07122015 ind_prs18092015 ind_prs24082015 ind_prs29042015 ind_prs18032015_2
ind_prs20022015 ind_prs23012015 ind_prs21012015
ind_prs18112014 ind_prs26082014 ind_prs20082014 ind_prs27022014 ind_prs21022014
ind_prs07112013 ind_prs29102013 ind_prs09102013 ind_prs22072013 ind_prs02072013
ind_prs21062013 ind_prs26022013 ind_prs25012013
ind_prs27112012 ind_prs31102012 ind_prs03092012
""".split()

DIR = "pit_events"
PDFD, TXTD = f"{DIR}/pdfs", f"{DIR}/txt"

MONTHS = ("January February March April May June July August September "
          "October November December").split()
DATE_PAT = re.compile(
    r"effective\s+(?:from|on)?\s*(" + "|".join(MONTHS) +
    r")\s+(\d{1,2})\s*,?\s*(\d{4})", re.I)


def heading_core(s):
    s = re.sub(r"\s+", " ", s).strip().rstrip(".:")
    s = re.sub(r"\s+index$", "", s, flags=re.I)
    return s.upper()


# ------------------------------------------------------------------ download
def download():
    import requests
    os.makedirs(PDFD, exist_ok=True)
    sess = requests.Session()
    sess.headers.update(HEADERS)
    ok = skip = bad = 0
    for i, f in enumerate(PDF_FILES):
        path = f"{PDFD}/{f}.pdf"
        if os.path.exists(path) and os.path.getsize(path) > 1000:
            skip += 1
            continue
        try:
            r = sess.get(BASE + f + ".pdf", timeout=40)
            r.raise_for_status()
            open(path, "wb").write(r.content)
            ok += 1
            print(f"  [{i+1}/{len(PDF_FILES)}] {f}.pdf  {len(r.content)//1024}KB")
        except Exception as e:
            bad += 1
            print(f"  !! {f}.pdf  {e}")
    for key, url in ANCHOR_LISTS.items():
        import requests as rq
        r = sess.get(url, timeout=40)
        r.raise_for_status()
        open(f"{DIR}/anchor_{key}.csv", "wb").write(r.content)
        print(f"  anchor {key}: saved")
    print(f"download done: {ok} new, {skip} cached, {bad} failed")


# ------------------------------------------------------------------ parse
def pdf_text(f):
    os.makedirs(TXTD, exist_ok=True)
    tpath = f"{TXTD}/{f}.txt"
    if os.path.exists(tpath):
        return open(tpath, encoding="utf-8").read()
    import pdfplumber
    with pdfplumber.open(f"{PDFD}/{f}.pdf") as pdf:
        txt = "\n".join((p.extract_text() or "") for p in pdf.pages)
    open(tpath, "w", encoding="utf-8").write(txt)
    return txt


ROW_PAT = re.compile(
    r"^\s*(?:\d{1,3}[.)]?\s+)?([A-Z].{2,70}?)\s+([A-Z0-9][A-Z0-9&\-]{1,19})\s*$")
NUM_PRE = re.compile(r"^\s*(?:\(?\d{1,2}[.)]\)?|[A-Za-z][.)])\s+(.+)$")
IDXWORD = re.compile(r"(nifty|cnx|index|indices)", re.I)


def _sym_ok(sym, company):
    if sum(c.isalpha() for c in sym) < 2:
        return False
    if IDXWORD.search(company):
        return False
    if not re.search(r"[a-z]", company):
        return False
    return sym not in ("NSE", "IISL", "LTD", "LIMITED", "INDIA", "EQ")


def _mkdate(m):
    return pd.Timestamp(f"{m.group(3)}-{MONTHS.index(m.group(1).capitalize())+1:02d}-{int(m.group(2)):02d}")


def parse_doc(f, report):
    txt = pdf_text(f)
    raw_lines = txt.splitlines()
    # drop orphan serial-number lines (pdf extraction splits them off rows;
    # ROW_PAT treats serials as optional, so they carry no information)
    lines = [ln for ln in raw_lines if not re.match(r"^\s*\d{1,3}\s*$", ln)]

    events = []
    eff = None
    gm = DATE_PAT.search(re.sub(r"\s+", " ", txt))
    if gm:
        eff = _mkdate(gm)
    cur_idx, mode = None, None
    for line in (l.strip() for l in lines):
        low = line.lower()
        m = DATE_PAT.search(line)
        if m:
            eff = _mkdate(m)
        nm = NUM_PRE.match(line)
        rest = nm.group(1) if nm else line
        core = heading_core(rest)
        if core in ALIAS:                       # a target-index section begins
            cur_idx, mode = ALIAS[core], None
            continue
        if "being excluded" in low or "are excluded" in low:
            mode = "exc"
            continue
        if "being included" in low or "are included" in low:
            mode = "inc"
            continue
        rm = ROW_PAT.match(line)
        valid = bool(rm) and _sym_ok(rm.group(2), rm.group(1))
        if valid and cur_idx and mode:
            events.append((eff, cur_idx, mode, rm.group(2), rm.group(1)))
            continue
        if nm and not valid:                    # a different section heading
            cur_idx = None

    # ---- ad-hoc single-company docs (Exclusion of X from ... indices) ----
    if not events:
        head = " ".join(lines[:8])
        am = re.search(r"Exclusion of (.{3,60}?)(?:Ltd\.?|Limited)", head, re.I)
        if am:
            syms = set()
            for raw in lines:
                rm = ROW_PAT.match(raw.strip())
                if rm and _sym_ok(rm.group(2), rm.group(1)):
                    syms.add(rm.group(2))
            idxs = set()
            for raw in lines:
                s = raw.strip()
                nm = NUM_PRE.match(s)
                for cand in (nm.group(1) if nm else s,
                             re.sub(r"^\d{1,3}\s+", "", s)):
                    core = heading_core(cand)
                    if core in ALIAS:
                        idxs.add(ALIAS[core])
            if syms and idxs and eff is not None and len(syms) <= 4:
                for ix in idxs:
                    for s in syms:
                        events.append((eff, ix, "exc", s, am.group(1).strip()))
                report.append(f"ADHOC {f}: {sorted(syms)} excluded from "
                              f"{sorted(idxs)} eff {eff.date()}")
            else:
                report.append(f"SKIPPED-ADHOC {f}: company={am.group(1)!r} "
                              f"syms={sorted(syms)[:6]} n={len(syms)} "
                              f"target_idx={sorted(idxs)} eff={eff}")
    if any(e[0] is None for e in events):
        report.append(f"NO-DATE {f}: {sum(e[0] is None for e in events)} rows dropped")
        events = [e for e in events if e[0] is not None]
    return events


def parse_all():
    os.makedirs(DIR, exist_ok=True)
    report, allev = [], []
    files = [f for f in PDF_FILES
             if os.path.exists(f"{PDFD}/{f}.pdf") or os.path.exists(f"{TXTD}/{f}.txt")]
    print(f"parsing {len(files)} PDFs...")
    for f in files:
        try:
            ev = parse_doc(f, report)
        except Exception as e:
            report.append(f"ERROR {f}: {e}")
            continue
        for row in ev:
            allev.append({"eff": row[0], "index": row[1], "action": row[2],
                          "symbol": row[3], "company": row[4], "src": f})
    E = pd.DataFrame(allev).drop_duplicates(
        subset=["eff", "index", "action", "symbol"])
    E = E.sort_values(["eff", "index", "action"]).reset_index(drop=True)
    E.to_csv(f"{DIR}/events.csv", index=False)
    # balance report per index per year
    rep = [f"events: {len(E)} rows from {E['src'].nunique()} documents", ""]
    for ix in ("N100", "MC100"):
        sub = E[E["index"] == ix]
        inc = sub[sub.action == "inc"].groupby(sub.eff.dt.year).size()
        exc = sub[sub.action == "exc"].groupby(sub.eff.dt.year).size()
        bal = pd.DataFrame({"inc": inc, "exc": exc}).fillna(0).astype(int)
        rep.append(f"--- {ix} events per year (inc vs exc)")
        rep.append(bal.to_string())
        rep.append("")
    rep += ["--- flags ---"] + report
    open(f"{DIR}/parse_report.txt", "w", encoding="utf-8").write("\n".join(rep))
    print("\n".join(rep[:40]))
    print(f"\nsaved: {DIR}/events.csv, {DIR}/parse_report.txt")


# ------------------------------------------------------------------ build
def build():
    E = pd.read_csv(f"{DIR}/events.csv", parse_dates=["eff"])
    close = pd.read_parquet("data_pit_wf/close.parquet")
    dates = close.index
    # anchors
    members = {}
    for key in ("N100", "MC100"):
        a = pd.read_csv(f"{DIR}/anchor_{key}.csv")
        a.columns = [c.strip() for c in a.columns]
        members[key] = set(a["Symbol"].str.strip())
        print(f"anchor {key}: {len(members[key])} names")
    # snap effective dates to next trading date
    def snap(d):
        pos = dates.searchsorted(d)
        return dates[min(pos, len(dates) - 1)]
    E["eff_t"] = E["eff"].map(snap)
    ev_by_date = {k: v for k, v in E.groupby("eff_t")}
    # walk backward
    rows = {}
    for dt in dates[::-1]:
        rows[dt] = {k: frozenset(v) for k, v in members.items()}
        if dt in ev_by_date:
            g = ev_by_date[dt]
            for _, r in g.iterrows():
                if r["action"] == "inc":
                    members[r["index"]].discard(r["symbol"])
                else:
                    members[r["index"]].add(r["symbol"])
    # to matrix: union universe, aligned to price panel columns
    try:
        sys.path.insert(0, ".")
        from walkforward_pit import ALIAS as ALIASES
        ALIASES = dict(ALIASES)
    except Exception:
        ALIASES = {}
    ALIASES.update({                    # renames found during reconciliation
        "GMRINFRA": "GMRAIRPORT", "AMARAJABAT": "ARE&M",
        "MOTHERSUMI": "MOTHERSON", "LTI": "LTM", "MINDTREE": "LTM",
        "LTIM": "LTM", "CADILAHC": "ZYDUSLIFE", "TATAMOTORS": "TMCV",
        "NIITTECH": "COFORGE", "PVR": "PVRINOX", "MAX": "MFSL",
        "STRTECH": "STLTECH",
    })
    def to_col(sym):
        s = ALIASES.get(sym, sym)
        return s + ".NS" if not s.endswith(".NS") else s
    allsyms = set()
    for dt in dates:
        for k in rows[dt]:
            allsyms |= set(rows[dt][k])
    colmap = {s: to_col(s) for s in allsyms}
    present = {s: c for s, c in colmap.items() if c in close.columns}
    missing = sorted(set(colmap) - set(present))
    import numpy as np
    colpos = {}
    for i, c in enumerate(close.columns):
        colpos.setdefault(c, i)                 # first occurrence wins
    arr = np.zeros((len(dates), len(close.columns)), dtype=bool)
    for r, dt in enumerate(dates):
        syms = set()
        for k in rows[dt]:
            syms |= set(rows[dt][k])
        for s in syms:
            c = present.get(s)
            if c is not None:
                arr[r, colpos[c]] = True
    M = pd.DataFrame(arr, index=dates, columns=close.columns)
    M.to_parquet("data_pit_wf/membership_v2.parquet")
    # reconciliation
    rep = [f"membership_v2: {M.shape}, names/day mean "
           f"{M.sum(axis=1).mean():.1f} (min {M.sum(axis=1).min()}, "
           f"max {M.sum(axis=1).max()})",
           f"symbols never matched to price panel: {len(missing)}"]
    rep += [f"  missing: {m} (last member "
            f"{max(dt for dt in dates if m in (set().union(*rows[dt].values())))})"
            for m in missing[:40]]
    if os.path.exists("data_pit_wf/membership.parquet"):
        old = pd.read_parquet("data_pit_wf/membership.parquet")
        old = old.reindex(index=M.index, columns=M.columns).fillna(False)
        both = (M & old).sum(axis=1)
        either = (M | old).sum(axis=1)
        jac = (both / either.replace(0, pd.NA)).astype(float)
        rep.append("\nagreement vs old Wayback membership (Jaccard/day, by year):")
        rep.append(jac.groupby(jac.index.year).mean().round(3).to_string())
        rep.append("\nnames/day old vs new, by year:")
        cmpdf = pd.DataFrame({"old": old.sum(axis=1), "new": M.sum(axis=1)})
        rep.append(cmpdf.groupby(cmpdf.index.year).mean().round(1).to_string())
    txt = "\n".join(map(str, rep))
    open(f"{DIR}/reconcile.txt", "w", encoding="utf-8").write(txt)
    print(txt)
    print(f"\nsaved: data_pit_wf/membership_v2.parquet, {DIR}/reconcile.txt")



# ------------------------------------------------------------- fetch-missing
def fetch_missing():
    """Download price history for membership symbols absent from the panel
    (runs on a machine with internet; extends close/volume parquets)."""
    import numpy as np
    import yfinance as yf
    E = pd.read_csv(f"{DIR}/events.csv", parse_dates=["eff"])
    import requests
    for key, url in ANCHOR_LISTS.items():          # anchors: download if absent
        p = f"{DIR}/anchor_{key}.csv"
        if not os.path.exists(p):
            r = requests.get(url, headers=HEADERS, timeout=40)
            r.raise_for_status()
            open(p, "wb").write(r.content)
            print(f"  fetched anchor {key}")
    close = pd.read_parquet("data_pit_wf/close.parquet")
    volume = pd.read_parquet("data_pit_wf/volume.parquet")
    try:
        sys.path.insert(0, ".")
        from walkforward_pit import ALIAS as AL
        AL = dict(AL)
    except Exception:
        AL = {}
    AL.update({"GMRINFRA": "GMRAIRPORT", "AMARAJABAT": "ARE&M",
               "MOTHERSUMI": "MOTHERSON", "LTI": "LTM", "MINDTREE": "LTM",
               "LTIM": "LTM", "CADILAHC": "ZYDUSLIFE", "TATAMOTORS": "TMCV",
               "NIITTECH": "COFORGE", "PVR": "PVRINOX", "MAX": "MFSL",
               "STRTECH": "STLTECH"})
    syms = set(E["symbol"])
    for key in ("N100", "MC100"):
        a = pd.read_csv(f"{DIR}/anchor_{key}.csv")
        a.columns = [c.strip() for c in a.columns]
        syms |= set(a["Symbol"].str.strip())
    need = sorted({AL.get(s, s) + ".NS" for s in syms} - set(close.columns))
    print(f"missing from panel: {len(need)} -> downloading")
    got, fail = [], []
    for i in range(0, len(need), 25):
        batch = need[i:i + 25]
        raw = yf.download(batch, start="2013-01-01", end="2026-10-01",
                          auto_adjust=True, group_by="ticker",
                          threads=True, progress=False)
        for t in batch:
            try:
                sub = raw[t] if len(batch) > 1 else raw
                c = sub["Close"].dropna()
                if len(c) < 60:
                    fail.append(t)
                    continue
                close[t] = sub["Close"].reindex(close.index)
                volume[t] = sub["Volume"].reindex(close.index)
                got.append(t)
            except Exception:
                fail.append(t)
    print(f"recovered {len(got)}; unrecoverable (likely delisted): {len(fail)}")
    print("  " + ", ".join(fail))
    close.to_parquet("data_pit_wf/close.parquet")
    volume.to_parquet("data_pit_wf/volume.parquet")
    print("panel extended -> rebuilding membership_v2")
    build()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--parse", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--fetch-missing", action="store_true")
    ap.add_argument("--all", action="store_true")
    a = ap.parse_args()
    if a.all or a.download:
        download()
    if a.all or a.parse:
        parse_all()
    if a.all or a.build:
        build()
    if a.fetch_missing:
        fetch_missing()
    if not any([a.all, a.download, a.parse, a.build]):
        print(__doc__)
