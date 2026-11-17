"""More style data. Extends the over-the-board roster (pgnmentor) with solid positional players, and
pulls Lichess rapid and classical games for a curated solid roster. Only rapid and classical, since
bullet and blitz games are noise for a style head. Sharp attacking players are left out on purpose,
since they would not fit a defensive style. Network and light CPU only.

  python chess/pull_more.py
Output: chess/gm_pgn/{Player}.pgn (OTB)  and  chess/gm_pgn/lichess_{handle}.pgn (rapid+classical)
"""
import os, time, urllib.request, zipfile, io
HERE = os.path.dirname(os.path.abspath(__file__))
GM_DIR = os.path.join(HERE, "gm_pgn"); os.makedirs(GM_DIR, exist_ok=True)
UA = "Mozilla/5.0 (mayfly-chess-research)"

# over-the-board (pgnmentor) additions, solid positional players by last name (404 misses are logged)
OTB_MORE = ["Giri", "So", "Aronian", "Svidler", "Navara", "Almasi", "Nisipeanu", "Shankland",
            "Radjabov", "Vitiugov", "Harikrishna", "Eljanov", "Tomashevsky", "Jakovenko",
            "Wojtaszek", "Kasimdzhanov", "Movsesian", "Onischuk", "Bologan", "Avrukh", "Andreikin",
            "Sargissian", "Malakhov", "Bacrot", "Fressinet", "Gelfand", "Dominguez", "Ding",
            "Gujrathi", "Inarkiev", "Matlakov", "Najer", "Smirin", "Kozul", "Fridman", "Berkes",
            "Lupulescu", "Bu", "Yu", "Wang", "Gawain"]

# Lichess handles to try (rated rapid and classical), a curated list of solid players.
LICHESS = ["TRadjabov", "AnishGiri", "AnishOnYoutube", "GMWSO", "GMWesleySo", "Vitiugov_N",
           "LeinierDP", "Zhigao", "SergeyKarjakin", "ChokoBars", "VladimirKramnik", "LevonAronian",
           "HarikrishnaP", "ViditChess", "YuYangyi", "WangHao", "BuXiangzhi", "Eljanov",
           "Tomashevsky", "Jakovenko", "Malakhov", "Sargissian", "Inarkiev", "Matlakov",
           "GMGeorgMeier", "MeierG", "Shankland", "GiriA", "Svidler", "Wojtaszek", "Kasimdzhanov",
           "Bacrot", "Fressinet", "ShortN", "AdamsM", "MickeyAdams", "McShane", "HowellD",
           "GawainJones", "LekoP", "PeterLeko", "Bologan", "DmitryAndreikin", "Andreikin",
           "Fedoseev", "Sjugirov", "Najer", "Sutovsky", "Rodshtein", "Smirin", "Nabaty", "Avrukh",
           "Roiz", "Movsesian", "Navara", "Lupulescu", "Nisipeanu", "Berkes", "Almasi", "Saric",
           "Kozul", "Fridman", "Dominguez", "Iturrizaga", "Bachmann", "Onischuk"]

def get(url, accept=None, timeout=120):
    h = {"User-Agent": UA}
    if accept:
        h["Accept"] = accept
    return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout).read()

ok = 0
for p in OTB_MORE:
    if os.path.exists(os.path.join(GM_DIR, f"{p}.pgn")):
        continue
    try:
        data = get(f"https://www.pgnmentor.com/players/{p}.zip")
        zipfile.ZipFile(io.BytesIO(data)).extractall(GM_DIR)
        print(f"OTB OK   {p}", flush=True); ok += 1
    except Exception as e:
        print(f"OTB MISS {p}: {e}", flush=True)
print(f"OTB additions: {ok}", flush=True)

lok = 0
for h in LICHESS:
    try:
        url = (f"https://lichess.org/api/games/user/{h}"
               f"?perfType=rapid,classical&rated=true&max=1500&clocks=false&evals=false&opening=false")
        data = get(url, accept="application/x-chess-pgn")
        if data and data.strip():
            with open(os.path.join(GM_DIR, f"lichess_{h}.pgn"), "wb") as f:
                f.write(data)
            print(f"LICHESS OK   {h}: {data.count(b'[Event ')} games", flush=True); lok += 1
        else:
            print(f"LICHESS none {h}", flush=True)
    except Exception as e:
        print(f"LICHESS MISS {h}: {e}", flush=True)
    time.sleep(2)   # polite to the lichess API
print(f"lichess hits: {lok}", flush=True)
print("pull_more DONE -> chess/gm_pgn/", flush=True)
