"""
Play the models on a board in your browser.

Runs a small local server, and the model runs on your GPU. Click a piece and its legal moves light
up, click one to play it, or drag.

  python chess/play_web.py            then open http://localhost:8000
  env: VERSION=mayfly_v1.pt   which frozen version to play, or a checkpoint filename
       PASS=               a shared passphrase, required once the board is public
       BIND=0.0.0.0 PONDER=1 TTT=1 OPEN=1
  python chess/play_web.py 8080       different port
  env: VISITS=400   tree search per move, 0 plays the bare network

Every frozen version is available from a menu, each player has their own game, tree and opponent
adapter, and the page shows what the models want in every position.
"""
import os, sys, json, time, secrets, threading, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import chess, chess.pgn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from play_race import load, think, cp_of
from tropical_search import search
from mcts import MCTS, pinned_search, policy_move
import timectl
import paths


def game_over(board):
    """Whether the game is over under the rules the harnesses and online boards use.

    Checkmate, stalemate, insufficient material, and automatic draws at threefold repetition and at
    fifty moves without a capture or pawn move, as the tree's tablebase reading assumes.
    is_repetition(3) is the position occurring for the third time, not a move that could make it so,
    which python-chess's claim_draw would also count."""
    return board.is_game_over() or board.is_repetition(3) or board.is_fifty_moves()


def game_result(board):
    if board.is_checkmate():
        return board.result()
    return "1/2-1/2" if game_over(board) else "*"
from ttt import OpponentModel

E = os.environ.get
# Only if it looks like a port. This runs at import time, and other tools import play_web for
# load_version with arguments of their own.
PORT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 8000
# A cache of loaded weights for every frozen version. Which version a player faces is chosen per
# session.
MODELS = {}
MODELS_LOCK = threading.Lock()
LOADING = {}                 # one lock per version being loaded, so it loads once (model_for)
CATALOGUE = []            # [{key, file, label}], newest version first, filled at startup
DEFAULT_VERSION = ""
# The alpha-beta shortlist search, kept as a diagnostic but no longer reachable from the page.
search_depth = 0
# The tree is on, since a policy head alone misses simple mates the tree finds. VISITS=0 plays the
# bare network.
VISITS = int(E("VISITS", 400))
# Pondering. The tree survives between moves, so a search started after the engine moves keeps
# working on the position the player is looking at.
PONDER = E("PONDER", "1") == "1"
PONDER_CAP = int(E("PONDER_CAP", 20000))
# Where finished games are saved. Their positions can be queued for labelling (LABEL_QUEUE), which
# gives training data from positions humans steer games into, which self-play never reaches.
GAMES_DIR = E("GAMES", paths.under("games"))
# The player's name is stored against the browser cookie (player_names.json), so it survives
# server restarts.
# The modules this process imported, with their mtimes, recorded at startup. Editing a file on disk
# does not change a running server, so the board can report which of its sources are stale.
SOURCE_MTIMES = {}


def record_sources():
    """The mtime of every module in this directory that is already imported."""
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", "") or ""
        f = os.path.abspath(f)
        if f.endswith(".py") and os.path.dirname(f) == HERE:
            try:
                SOURCE_MTIMES[f] = os.path.getmtime(f)
            except OSError:
                pass


def stale_sources():
    """Files that have changed on disk since this process imported them.

    A second of slack, because copy tools round timestamps and a file copied unchanged should not
    read as an edit.
    """
    out = []
    for f, t in SOURCE_MTIMES.items():
        try:
            if os.path.getmtime(f) > t + 1.0:
                out.append(os.path.basename(f))
        except OSError:
            out.append(os.path.basename(f) + " (gone)")
    return sorted(out)


NAMES_FILE = os.path.join(HERE, "player_names.json")
NAMES = {}
NAMES_LOCK = threading.Lock()


def load_names():
    global NAMES
    try:
        with open(NAMES_FILE, encoding="utf-8-sig") as fh:
            NAMES = json.load(fh)
    except Exception:
        NAMES = {}


def remember_name(sid, name):
    """Write through immediately, so a name is never lost to a crash."""
    with NAMES_LOCK:
        if not name:
            NAMES.pop(sid, None)
        else:
            NAMES[sid] = name
        try:
            tmp = NAMES_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(NAMES, fh)
            os.replace(tmp, NAMES_FILE)
        except Exception as e:
            print(f"  could not save names: {type(e).__name__}: {e}", flush=True)
LABEL_QUEUE = E("LABEL_QUEUE", paths.under("selfplay", "positions.txt"))
# Give the GPU back when nobody is playing. After this many idle minutes the board drops every tree,
# moves the network to system memory and empties the CUDA cache. The next request wakes it in about
# a second.
IDLE_MINUTES = float(E("IDLE_MINUTES", 60))
LAST_ACTIVE = time.time()
# touched on every real request, watched by the trainer, which yields while it is fresh
PLAYING_FILE = E("PLAYING_FILE", os.path.join(E("CKPT_DIR", paths.under("ckpt")), "playing.flag"))
SLEEPING = False
WAKE_LOCK = threading.Lock()
# Test-time training. A small adapter on the policy head learns the player's moves during play, and
# the tree consults it only at nodes where the player is to move. It is not reset between games, so
# a plan repeated across games is learned. TTT=0 turns it off.
TTT = E("TTT", "1") == "1"
# How many people can have a game going at once, and how long an abandoned one is kept. Each costs
# a tree and a cache.
MAX_SESSIONS = int(E("MAX_SESSIONS", 8))
SESSION_TTL = float(E("SESSION_TTL", 6 * 3600))
CACHE = int(E("CACHE", 8000))
# Forgiven on every human move, since the request's trip to the server is not thinking time. The
# engine gets none, its clock is read on the machine it runs on.
LAG = float(E("LAG", 0.2))


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>Mayfly chess</title>
<style>
  :root { --light:#ebecd0; --dark:#779556; --sel:#f6f669; --dot:rgba(0,0,0,.22);
          --ink:#1b1b19; --muted:#6b6a63; --line:#dcdbd2; --card:#fbfbf9; }
  * { box-sizing:border-box; }
  body { margin:0; background:#f4f3ee; color:var(--ink); font:14px/1.5 system-ui,-apple-system,Segoe UI,sans-serif; }
  .wrap { display:flex; gap:22px; padding:22px; align-items:flex-start; flex-wrap:wrap; }
  .boardwrap { display:flex; gap:8px; }
  .ranks { display:flex; flex-direction:column; }
  .ranks div, .files div { color:var(--muted); font-size:11px; display:flex; align-items:center; justify-content:center; }
  .ranks div { height:72px; width:14px; }
  .files { display:flex; margin-left:22px; }
  .files div { width:72px; height:14px; }
  #board { display:grid; grid-template-columns:repeat(8,72px); grid-template-rows:repeat(8,72px);
           border:1px solid #b9b8ae; border-radius:3px; overflow:hidden; user-select:none; }
  .sq { position:relative; display:flex; align-items:center; justify-content:center;
        font-size:52px; line-height:1; cursor:default; }
  .sq.l { background:var(--light); } .sq.d { background:var(--dark); }
  .sq.sel { background:var(--sel) !important; }
  .sq.last { box-shadow: inset 0 0 0 4px rgba(246,246,105,.75); }
  .sq.can { cursor:pointer; }
  .sq .dot { position:absolute; width:22px; height:22px; border-radius:50%; background:var(--dot); pointer-events:none; }
  .sq .ring { position:absolute; inset:4px; border:4px solid var(--dot); border-radius:50%; pointer-events:none; }
  /* both colours use the solid glyph and are told apart by fill and outline, since the outline
     white pieces look almost like the black ones in most fonts. */
  .sq .pc { position:relative; z-index:2; }
  .sq .pc.w { color:#fdfdfb; text-shadow:0 1px 0 #57564f,0 -1px 0 #57564f,1px 0 0 #57564f,
              -1px 0 0 #57564f,1px 1px 0 #57564f,-1px -1px 0 #57564f,1px -1px 0 #57564f,
              -1px 1px 0 #57564f; }
  .sq .pc.b { color:#2a2925; text-shadow:0 1px 0 rgba(255,255,255,.35); }
  .sq .pc.mine { cursor:grab; }
  .panel { width:380px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; margin-bottom:12px; }
  .card h3 { margin:0 0 8px; font-size:13px; font-weight:600; letter-spacing:.02em; }
  .row { display:flex; justify-content:space-between; gap:8px; padding:2px 0; }
  .mv { font-variant-numeric:tabular-nums; }
  .bar { height:12px; background:#2b2b28; border-radius:6px; overflow:hidden; margin:6px 0 4px; }
  .bar i { display:block; height:100%; background:#f2f1ea; }
  button, select { font:inherit; padding:5px 10px; border:1px solid var(--line); background:#fff;
                   border-radius:6px; cursor:pointer; }
  button:hover { background:#f0efe9; }
  .controls { display:flex; gap:6px; flex-wrap:wrap; }
  .moves { max-height:180px; overflow:auto; font-variant-numeric:tabular-nums; font-size:13px; }
  .status { font-weight:600; }
  .clocks { display:flex; gap:10px; margin-bottom:10px; }
  .clk { flex:1; text-align:center; padding:10px 6px; border:1px solid var(--line); border-radius:8px;
         font-variant-numeric:tabular-nums; font-size:26px; font-weight:600; letter-spacing:.5px; }
  .clk .who { display:block; font-size:11px; font-weight:400; color:var(--muted); letter-spacing:0; }
  .clk.live { border-color:#2f6b1d; box-shadow:0 0 0 1px #2f6b1d inset; }
  .clk.low { color:#b0413e; }
  .clk.flag { text-decoration:line-through; opacity:.6; }
  .sub { color:var(--muted); font-size:12px; }
  .thinking { opacity:.45; }
</style></head><body>
<div class="wrap">
  <div>
    <div class="boardwrap">
      <div class="ranks" id="ranks"></div>
      <div id="board"></div>
    </div>
    <div class="files" id="files"></div>
  </div>
  <div class="panel">
    <div class="card" id="clockcard" style="display:none">
      <div class="clocks">
        <div class="clk" id="clk_them"><span class="who">engine</span><b id="t_them">-</b></div>
        <div class="clk" id="clk_you"><span class="who">you</span><b id="t_you">-</b></div>
      </div>
      <div class="sub" id="tcname"></div>
    </div>
    <div class="card">
      <div class="status" id="status">loading</div>
      <div class="bar"><i id="evalbar" style="width:50%"></i></div>
      <div class="sub" id="evaltext"></div>
      <div class="sub" id="ttt"></div>
    </div>
    <div class="card">
      <h3>What each model wants here</h3>
      <div id="think"></div>
      <div id="searched"></div>
    </div>
    <div class="card">
      <h3>Moves</h3>
      <div class="moves" id="moves"></div>
    </div>
    <div class="card">
      <div class="controls">
        <button onclick="api('new')">New game</button>
        <button onclick="api('undo')">Undo</button>
        <button onclick="api('swapsides')">Swap colours</button>
        <button onclick="flip()">Flip board</button>
        <select id="opp" onchange="api('opponent',{which:this.value})" title="which frozen version you are playing"></select>
        <select id="depth" onchange="api('depth',{d:parseInt(this.value)})">
          <option value="0">no search, the bare network</option>
          <option value="100">100 visits</option>
          <option value="400" selected>400 visits</option>
          <option value="1600">1600 visits</option>
          <option value="6400">6400 visits, slow</option>
        </select>
        <select id="tc" onchange="api('tc',{v:this.value})" title="a real clock, both sides; the engine allocates its own time">
          <option value="">no clock, fixed visits</option>
          <option value="60+0">1+0 bullet</option>
          <option value="120+1">2+1 bullet</option>
          <option value="180+2">3+2 blitz</option>
          <option value="300+0">5+0 blitz</option>
          <option value="600+0">10+0 rapid</option>
          <option value="900+10">15+10 rapid</option>
          <option value="1500+10">25+10 classical</option>
          <option value="1800+0">30+0 classical</option>
          <option value="2700+30">45+30 classical</option>
          <option value="5400+30">90+30 classical, the FIDE control</option>
        </select>
        <select id="adapt" onchange="api('think',{v:this.value})" title="think longer when the top two moves are close, bank time when the choice is obvious, and spend a lead on your clock down to level with you on a position that will not resolve. v3 onwards; v1 and v2 keep the search they were released with">
          <option value="sealed">adaptive time: as released</option>
          <option value="on">adaptive time: on (v3 onwards)</option>
          <option value="off">adaptive time: off</option>
        </select>
        <select id="temp" onchange="api('temp',{t:parseFloat(this.value)})" title="with a search running this varies its OWN opening choices for the first twelve plies, drawn from its own visit counts and never from a move it rates worse; with no search it is the old policy temperature">
          <option value="0">its top move always</option>
          <option value="1" selected>vary its own openings (as it weighs them)</option>
          <option value="2">vary its own openings (more)</option>
        </select>
      </div>
      <div class="controls" style="margin-top:8px">
        <input id="fen" placeholder="paste a position (FEN)" style="flex:1;min-width:14rem;
               font:inherit;padding:.4rem;border:1px solid var(--line);border-radius:6px">
        <button onclick="setfen()">Set</button>
        <button onclick="copyfen()">Copy this one</button>
      </div>
      <div class="controls" style="margin-top:8px">
        <input id="who" placeholder="your name, so your games are yours" style="flex:1;min-width:14rem;
               font:inherit;padding:.4rem;border:1px solid var(--line);border-radius:6px">
        <button onclick="setname()">Save</button>
      </div>
      <div class="controls" style="margin-top:8px">
        <label style="display:flex;align-items:center;gap:.4rem" title="tick it for games you are really trying in; only these count toward a rating">
          <input type="checkbox" id="serious" onchange="setserious()"> serious game, count it</label>
        <label style="display:flex;align-items:center;gap:.4rem" title="master practice for the first sixteen plies, sampled so two games are not the same game. Never used in any measurement.">
          <input type="checkbox" id="book" onchange="api('book',{on:this.checked})"> opening book</label>
        <input id="elo" type="number" min="100" max="3500" placeholder="rating of whoever is playing"
               style="flex:1;min-width:12rem;font:inherit;padding:.4rem;border:1px solid var(--line);border-radius:6px">
        <button onclick="setserious()">Save</button>
      </div>
    </div>
  </div>
</div>
<script>
const GLYPH={p:'♟',n:'♞',b:'♝',r:'♜',q:'♛',k:'♚'};
let S=null, sel=null, flipped=false, busy=false;

function sqName(f,r){ return 'abcdefgh'[f]+(r+1); }

function draw(){
  const b=document.getElementById('board'); b.innerHTML='';
  const ranks=flipped?[...Array(8).keys()]:[...Array(8).keys()].reverse();
  const files=flipped?[...Array(8).keys()].reverse():[...Array(8).keys()];
  document.getElementById('ranks').innerHTML=ranks.map(r=>`<div>${r+1}</div>`).join('');
  document.getElementById('files').innerHTML=files.map(f=>`<div>${'abcdefgh'[f]}</div>`).join('');
  const targets = sel ? (S.legal[sel]||[]) : [];
  for(const r of ranks) for(const f of files){
    const name=sqName(f,r), pc=S.pieces[name];
    const d=document.createElement('div');
    d.className='sq '+(((f+r)%2)?'l':'d');
    if(name===sel) d.classList.add('sel');
    if(S.last && (S.last.slice(0,2)===name||S.last.slice(2,4)===name)) d.classList.add('last');
    if(targets.includes(name)){
      d.classList.add('can');
      const m=document.createElement('span'); m.className=pc?'ring':'dot'; d.appendChild(m);
    }
    if(pc){
      const s=document.createElement('span');
      s.className='pc '+(pc===pc.toUpperCase()?'w':'b');
      s.textContent=GLYPH[pc.toLowerCase()];
      if(S.legal[name]&&S.legal[name].length){ s.classList.add('mine'); s.draggable=true;
        s.ondragstart=e=>{ sel=name; draw(); e.dataTransfer.setData('text/plain',name); }; }
      d.appendChild(s);
    }
    d.ondragover=e=>{ if(targets.includes(name)) e.preventDefault(); };
    d.ondrop=e=>{ e.preventDefault(); const from=e.dataTransfer.getData('text/plain'); tryMove(from,name); };
    d.onclick=()=>click(name);
    b.appendChild(d);
  }
}

function click(name){
  if(busy) return;
  if(sel && (S.legal[sel]||[]).includes(name)){ tryMove(sel,name); return; }
  sel = (S.legal[name]&&S.legal[name].length && name!==sel) ? name : null;
  draw();
}

function tryMove(from,to){
  if(busy) return;
  let uci=from+to;
  const pc=S.pieces[from];
  // Ask which piece, since the move space carries underpromotions.
  if((pc==='P'&&to[1]==='8')||(pc==='p'&&to[1]==='1')){
    let want=(prompt('Promote to? q, r, b or n','q')||'q').toLowerCase().trim();
    if(!['q','r','b','n'].includes(want)) want='q';
    uci+=want;
  }
  sel=null; busy=true;
  document.getElementById('think').classList.add('thinking');
  api('move',{uci});
}

function fmt(t){
  if(t===null||t===undefined) return '-';
  t=Math.max(0,t);
  if(t>=20) { const m=Math.floor(t/60), s=Math.floor(t%60); return m+':'+String(s).padStart(2,'0'); }
  return t.toFixed(1);
}
let CK=null;      // {you, them, turn_is_you, at} the last reading, ticked locally between polls
function paintClocks(){
  const card=document.getElementById('clockcard');
  if(!CK){ card.style.display='none'; return; }
  card.style.display='';
  const gone=CK.running?(Date.now()-CK.at)/1000:0;
  const you=CK.you-(CK.turn_is_you&&!CK.over?gone:0);
  const them=CK.them-(!CK.turn_is_you&&!CK.over?gone:0);
  document.getElementById('t_you').textContent=fmt(you);
  document.getElementById('t_them').textContent=fmt(them);
  const cy=document.getElementById('clk_you'), ct=document.getElementById('clk_them');
  cy.className='clk'+(CK.turn_is_you&&!CK.over?' live':'')+(you<20?' low':'')+(CK.flagged==='you'?' flag':'');
  ct.className='clk'+(!CK.turn_is_you&&!CK.over?' live':'')+(them<20?' low':'')+(CK.flagged==='them'?' flag':'');
}
setInterval(paintClocks,100);

function render(){
  document.getElementById('status').textContent=S.status;
  document.getElementById('tc').value=S.tc||'';
  CK = S.clock ? {you:S.clock.you, them:S.clock.them, turn_is_you:S.turn_is_you,
                  running:S.clock.running, over:S.over, flagged:S.flagged, at:Date.now()} : null;
  document.getElementById('adapt').value=S.adaptive||'sealed';
  document.getElementById('book').checked=!!S.book;
  document.getElementById('temp').value=String(S.temp);
  document.getElementById('tcname').textContent=S.tcname||'';
  paintClocks();
  const wp=S.white_wp*100;
  document.getElementById('evalbar').style.width=wp.toFixed(1)+'%';
  document.getElementById('evaltext').textContent=S.evaltext+'   you are '+S.you_are;
  document.getElementById('ttt').textContent=S.ttt||'';
  document.getElementById('think').innerHTML=S.think.map(t=>
    `<div style="margin-bottom:8px"><div class="sub">${t.name}</div>`+
    t.moves.map(m=>`<div class="row mv"><span>${m.san}</span><span>${(m.p*100).toFixed(0)}%</span></div>`).join('')+
    `<div class="sub">rates ${t.who} at ${(t.wp*100).toFixed(0)}% (${t.cp>0?'+':''}${t.cp.toFixed(0)}cp)</div></div>`).join('');
  // The tree's number is about the position it moved from, one ply ago, while the bar above is
  // about the board as it stands, so the text says which is which.
  const sr=S.searched;
  document.getElementById('searched').innerHTML = sr
    ? `<div class="sub">its last move was <b>${sr.move}</b>, after ${sr.visits.toLocaleString()} visits`
      + (sr.carried>0?` (${sr.asked.toLocaleString()} asked for, ${sr.carried.toLocaleString()} carried over from pondering)`
         :` (a fresh tree: the ponder had not looked at the move you played)`)
      + `<br>it rated <b>the position it moved from</b> at ${sr.cp>0?'+':''}${sr.cp.toFixed(0)}cp for White.`
      + ` The bar above judges the board as it is now.`
      + (sr.spent?`<br>clock: ${sr.spent}`:'')
      + (sr.pv?`<br>expecting ${sr.pv}`:'') + `</div>`
    : '<div class="sub">the tree has not run yet</div>';
  const w=document.getElementById('who');
  if(document.activeElement!==w) w.value=S.name||'';   // never clobber what is being typed
  document.getElementById('moves').innerHTML=S.movelist;
  // The menu is built from what the server says is on disk, so a newly frozen version appears on
  // the next poll. Rebuilt only when the list changes, so an open dropdown is not reset.
  const opp=document.getElementById('opp');
  const sig=(S.opponents||[]).map(o=>o.key).join(',');
  if(opp.dataset.sig!==sig){
    opp.innerHTML=(S.opponents||[]).map(o=>
      '<option value="'+o.key+'">'+o.label+'</option>').join('');
    opp.dataset.sig=sig;
  }
  opp.value=S.opponent;
  document.getElementById('depth').value=S.depth;
  document.getElementById('serious').checked=!!S.serious;
  if(document.activeElement!==document.getElementById('elo'))
    document.getElementById('elo').value=S.elo||'';
  document.getElementById('think').classList.remove('thinking');
  busy=false; draw();
}

async function api(what,body){
  // The move is sent, so the player's clock stops here rather than when the reply comes back,
  // while the engine thinks.
  if(what==='move'&&CK){ CK.you-=CK.running&&CK.turn_is_you?(Date.now()-CK.at)/1000:0;
                          CK.turn_is_you=false; CK.at=Date.now(); paintClocks(); }
  // Show errors. An error reply is {"error": ...} and a dropped connection makes fetch throw, and
  // either would otherwise leave the board stuck with no message.
  let j;
  try {
    const r=await fetch('/api/'+what,{method:'POST',headers:{'Content-Type':'application/json'},
                                      body:JSON.stringify(body||{})});
    j=await r.json();
  } catch(e) { j={error:'the board did not answer, reload to see the game'}; }
  sel=null;
  if(j.error){ busy=false; document.getElementById('think').classList.remove('thinking');
               document.getElementById('status').textContent='The board hit an error: '+j.error;
               if(S) draw(); return; }
  S=j; render();
}
function flip(){ flipped=!flipped; draw(); }
// set a position from a FEN
function setfen(){ const v=document.getElementById('fen').value.trim(); if(v) api('fen',{fen:v}); }
function setname(){ api('name',{n:document.getElementById('who').value.trim()}); }
function setserious(){ const e=parseInt(document.getElementById('elo').value,10);
  api('serious',{on:document.getElementById('serious').checked, elo:isNaN(e)?null:e}); }
function copyfen(){ const v=S.fen||''; document.getElementById('fen').value=v;
  if(navigator.clipboard) navigator.clipboard.writeText(v).catch(()=>{}); }
api('state');
</script></body></html>"""


class Session:
    """One person's game, with their board, their tree and their opponent model.

    The network is shared, since reading it from several threads is safe. Everything else that makes
    up a game belongs to the session. The browser cookie is the whole identity, with no accounts.
    """

    def __init__(self, version=None):
        self.version = version or DEFAULT_VERSION
        self.model = model_for(self.version)[0]
        model = self.model
        self.board = chess.Board()
        self.lock = threading.Lock()
        self.human_white = True
        # Its own varied openings by default. 1.0 is visit-proportional, so the opening is drawn
        # from the tree's own opinion with no book and no move rated worse than its top choice.
        # Measurements run at 0, where it plays its top move.
        self.temperature = 1.0
        self.record = []               # (ply, what the engine saw), written into the saved PGN
        self.visits = VISITS
        self.seen = time.time()
        dev = str(next(model.parameters()).device)
        # A small evaluation cache, not the engine's 400,000. Each entry is a legal-move slice, so
        # this is a lookback depth rather than a memory limit, about twenty moves per person.
        # A version plays with its own sealed search.
        self.tree = MCTS(model, dev, batch=48, cache_size=CACHE).apply(
            pinned_search(self.version + ".pt"))
        self.om = OpponentModel(model, dev, head=_om_head(self.version + ".pt"),
                                castle=pinned_search(self.version + ".pt").get("castle_prior", "king")) if TTT else None
        self.tree.opponent = self.om
        self.tree.opp_colour = chess.WHITE
        self.ponder_thread = None
        # What the tree concluded, as opposed to the bare network's opinion the bar shows.
        self.last_search = None
        # who is playing, so saved games can be told apart
        self.name = ""
        # this session's cookie, set by session_for at creation, so the name handler can file it
        self.sid = ""
        self.logged = False
        # A serious game and the player's rating. Only ticked games count toward a human-scale
        # rating, since casual games flatter the engine. Both stay set across games in the session.
        self.serious = False
        self.elo = None
        # A real clock for both sides, or None for a fixed visit count. since is when the side to
        # move's clock started running, and every path that changes the turn goes through tick
        # first, so time is taken off exactly once.
        # "sealed" plays the version's own time settings (off for v1 to v3, on from v4), and "on" or
        # "off" is an explicit override for v3 onwards, shown on the page.
        self.adaptive_time = "sealed"
        # The opening book belongs to the board, not the version. It is sealed off in every version
        # and off here by default, so the model chooses its own openings. Shown as a checkbox.
        self.use_book = False
        self.tc = None                       # (base seconds, increment)
        self.clock = {chess.WHITE: 0.0, chess.BLACK: 0.0}
        self.since = None
        self.flagged = None                  # the colour that ran out, or None

    # ---------------------------------------------------------------- the clock
    def tick(self, lag=0.0):
        """Take the time the side to move has used off their clock.

        Idempotent between moves, since since is cleared. lag is forgiven, as on online boards,
        because the trip to the server is not the player's thinking time."""
        if not self.tc or self.since is None or self.flagged is not None:
            return
        turn = self.board.turn
        used = max(0.0, (time.monotonic() - self.since) - lag)
        self.clock[turn] = max(0.0, self.clock[turn] - used)
        self.since = None
        if self.clock[turn] <= 0.0:
            self.flagged = turn

    def start_clock(self):
        # The clock starts on the first move, as on other boards, so whoever moves first does it
        # for free.
        if (self.tc and self.flagged is None and self.board.move_stack
                and not game_over(self.board)):
            self.since = time.monotonic()

    def add_increment(self, colour):
        if self.tc and self.flagged is None:
            self.clock[colour] += self.tc[1]

    def set_tc(self, spec):
        """"180+2" starts a 3+2 game, "" goes back to a fixed visit count.

        Either way the game restarts."""
        spec = str(spec or "").strip()
        tc = None
        if spec:
            try:
                base, _, inc = spec.partition("+")
                tc = (max(1.0, float(base)), max(0.0, float(inc or 0)))
            except ValueError:
                return False
        self.stop_pondering()
        self.tc = tc
        self.flagged = None
        self.since = None
        self.clock = {chess.WHITE: tc[0] if tc else 0.0, chess.BLACK: tc[0] if tc else 0.0}
        return True

    def apply_options(self):
        """Put the board's own settings on the tree before a move.

        These are the book, opening variety and adaptive time. With a search running, the
        temperature sets how varied the engine's own opening choices are."""
        self.tree.book = bool(self.use_book)
        self.tree.open_temp = float(self.temperature) if self.visits > 0 else 0.0
        self.apply_adaptive()

    def apply_adaptive(self):
        """Put the adaptive-time settings on the tree for this move.

        "sealed" uses whatever the version was frozen with. An override only applies from v3
        onwards, since v1 and v2 anchor the ladder and keep exactly their released search. It is
        re-applied every move because switching version rebuilds the tree from the sealed block."""
        if self.adaptive_time == "sealed":
            # Re-apply the seal, since an earlier "on" or "off" override may still be on the tree.
            try:
                p = pinned_search(self.version + ".pt")
                self.tree.apply({k: p[k] for k in ("tm_extend", "tm_stop", "tm_parity", "tm_deep",
                                                   "tm_open_spend", "tm_open_plies") if k in p})
            except Exception:
                pass
            return
        try:
            num = int("".join(c for c in self.version.split("_v")[-1] if c.isdigit()) or 0)
        except ValueError:
            num = 0
        if num < 3:
            return
        on = self.adaptive_time == "on"
        self.tree.tm_extend = on
        self.tree.tm_stop = on
        # The long think rides on the same switch. With it on, the engine also sees the opponent's
        # clock and may spend its lead, down to level, on a position that will not resolve. Off, it
        # never looks at the other clock, as v1 to v3 were released.
        self.tree.tm_parity = 1.0 if on else 0.0
        # and the opening ramp
        self.tree.tm_open_spend = 0.4 if on else 1.0

    def left(self, colour):
        """What that clock reads right now, counting the move in progress."""
        t = self.clock.get(colour, 0.0)
        if self.tc and self.since is not None and self.board.turn == colour                 and self.flagged is None:
            t -= time.monotonic() - self.since
        return max(0.0, t)

    def over(self):
        return self.flagged is not None or game_over(self.board)

    def set_version(self, key):
        """Swap which frozen version this player is facing.

        The tree, the eval cache and the adapter all belong to the old weights, so none carries
        over. The board and the move list survive, so a game in progress carries on against the new
        engine.
        """
        if key == self.version or not any(c["key"] == key for c in CATALOGUE):
            return False
        self.stop_pondering()
        # Loaded before anything changes, so a failed load leaves the session on the old engine
        # under the old name.
        model = model_for(key)[0]
        self.version = key
        self.model = model
        dev = str(next(self.model.parameters()).device)
        self.tree = MCTS(self.model, dev, batch=48, cache_size=CACHE).apply(
            pinned_search(key + ".pt"))
        self.om = OpponentModel(self.model, dev, head=_om_head(self.version + ".pt"),
                                castle=pinned_search(self.version + ".pt").get("castle_prior", "king")) if TTT else None
        self.tree.opponent = self.om
        self.tree.opp_colour = chess.WHITE if self.human_white else chess.BLACK
        self.last_search = None
        return True

    # ---------------------------------------------------------------- the search
    def stop_pondering(self):
        """Called before anything touches the board, because the ponder thread is mutating the
        tree."""
        self.tree.stop()
        if self.ponder_thread is not None:
            self.ponder_thread.join(timeout=10)
            if self.ponder_thread.is_alive():
                # A ponder that has not stopped still owns the tree, and the next search clears the
                # abort flag, so wait for it, as uci.py does.
                print("  ponder slow to stop, still waiting", flush=True)
                self.ponder_thread.join()
            self.ponder_thread = None

    def start_pondering(self):
        if not PONDER or self.visits <= 0 or game_over(self.board):
            return
        self.tree.opponent = self.om
        self.tree.opp_colour = chess.WHITE if self.human_white else chess.BLACK
        snap = self.board.copy()
        # Cleared here, in the thread that owns the decision, not inside ponder(), where it could
        # undo a stop() that arrived while the thread was being scheduled.
        self.tree.abort.clear()
        # Capped, so idle browsers do not hold the GPU at full load indefinitely.
        self.ponder_thread = threading.Thread(target=self.tree.ponder, args=(snap, PONDER_CAP),
                                              daemon=True)
        self.ponder_thread.start()

    def engine_reply(self):
        """Let the engine move if it is its turn. Loops so a colour swap is handled too."""
        while not self.over() and \
                self.board.turn != (chess.WHITE if self.human_white else chess.BLACK):
            model = self.model
            self.tree.opponent = self.om
            self.tree.opp_colour = chess.WHITE if self.human_white else chess.BLACK
            # the visit count comes from the page's dropdown
            self.apply_options()
            info = None
            if self.tc:
                # Its own clock. The allocator gets what is left, the increment and the ply. The
                # search may stretch that when the top two moves are close, and timectl already
                # leaves room for the stretch.
                me = self.board.turn
                ply = len(self.board.move_stack)
                budget = timectl.allocate(self.left(me), self.tc[1], ply,
                                          opp=self.left(not me),
                                          parity=getattr(self.tree, "tm_parity", 0.0),
                                          open_spend=getattr(self.tree, "tm_open_spend", 1.0),
                                          open_plies=getattr(self.tree, "tm_open_plies", 16))
                # How long it may think if the position will not resolve, paid for out of its lead
                # on the clock. The ceiling knows both clocks, and the search still stops once the
                # top two moves separate.
                hard = (timectl.ceiling(self.left(me), self.left(not me), self.tc[1], ply,
                                        parity=getattr(self.tree, "tm_parity", 0.0),
                                        deep=getattr(self.tree, "tm_deep", 0.5))
                        if getattr(self.tree, "tm_parity", 0.0) > 0 else None)
                before = self.board.copy()
                if budget < 0.12:
                    # No time for a tree, so the bare network answers, reading castling where the
                    # seal's castle_prior says.
                    seal = pinned_search(self.version + ".pt")
                    mv = policy_move(self.model, self.board,
                                     str(next(self.model.parameters()).device),
                                     seal.get("policy_head", "cls"),
                                     castle=seal.get("castle_prior", "king"))
                    info = {"visits": 0, "q": 0.0, "pv": "", "budget": budget}
                else:
                    mv, info = self.tree.search(self.board, visits=10 ** 7, seconds=budget,
                                                hard_seconds=hard)
                self.tick()
                if mv is None or self.flagged is not None:
                    break
                self.add_increment(me)
                self.start_clock()
                self.last_search = self._searched(before, mv,
                                                  dict(info, budget=budget, ceiling=hard))
            elif self.visits > 0:
                before = self.board.copy()
                mv, info = self.tree.search(self.board, visits=self.visits)
                if mv is None:
                    break
                self.last_search = self._searched(before, mv, info)
            elif search_depth > 0:
                mv, _, _ = search(model, self.board, depth=search_depth, topk=8)
                if mv is None:
                    break
            else:
                moves, _ = think(model, self.board,
                                 castle=pinned_search(self.version + ".pt").get("castle_prior", "king"))
                if not moves:
                    break
                if self.temperature <= 0 or len(moves) == 1:
                    mv = moves[0][0]
                else:
                    import torch
                    w = torch.tensor([p for _, p in moves]) ** (1.0 / self.temperature)
                    mv = moves[int(torch.multinomial(w / w.sum(), 1))][0]
            # What it saw, kept with the game, so a surprising move can be explained afterwards
            # even when it cannot be reproduced from a clean tree.
            try:
                if info:
                    top = ", ".join(f"{u} {n}" for u, n, _q in (info.get("top") or [])[:4])
                    # the one-based ply number of the move about to be made, as log_game counts
                    self.record.append((len(self.board.move_stack) + 1,
                                        f"{int(info.get('visits', 0))}v "
                                        f"q{float(info.get('q', 0.0)):+.2f}"
                                        + (f" carried {int(info['carried'])}" if info.get("carried")
                                           else "")
                                        + (f" [{top}]" if top else "")))
            except (TypeError, ValueError, KeyError):
                pass
            self.board.push(mv)

    def _searched(self, before, mv, info):
        """What the tree found, in a readable form, how hard it looked, what it thought and the
        line it expects. q is from the point of view of the side that just moved."""
        q = float(info.get("q", 0.0))
        wp = (q + 1.0) / 2.0
        white_wp = wp if before.turn == chess.WHITE else 1.0 - wp
        pv, tmp = [], before.copy()
        for u in (info.get("pv") or "").split():
            try:
                m = chess.Move.from_uci(u)
            except Exception:
                break
            if m not in tmp.legal_moves:
                break
            pv.append(tmp.san(m))
            tmp.push(m)
        spent = ""
        if info.get("proof"):
            spent = info["proof"]                      # "opening book", "mate in one", ...
        elif info.get("budget"):
            ceil = info.get("ceiling") or 0.0
            spent = (f"{info['budget']:.2f}s allocated"
                     + (f" of up to {ceil/60:.1f} min" if ceil > info["budget"] * 2 else "")
                     + (f", extended {int(info.get('extended', 0))}x" if info.get("extended") else "")
                     + (", stopped early" if info.get("banked") else ""))
        return {"visits": int(info.get("visits", 0)), "asked": self.visits, "spent": spent,
                "carried": int(info.get("carried", 0)),
                "cp": cp_of(white_wp),
                "move": before.san(mv) if mv in before.legal_moves else mv.uci(),
                "pv": " ".join(pv)}

    # ---------------------------------------------------------------- the page
    def snapshot(self):
        """Everything the page needs to draw itself and everything you need to judge the model."""
        board = self.board
        legal = {}
        for m in board.legal_moves:
            legal.setdefault(chess.square_name(m.from_square), []).append(
                chess.square_name(m.to_square))
        pieces = {chess.square_name(s): p.symbol() for s, p in board.piece_map().items()}

        think_out = []
        for model, name, _ in (model_for(self.version),):
            moves, wp = think(model, board,
                              castle=pinned_search(self.version + ".pt").get("castle_prior", "king"))
            who = "White" if board.turn == chess.WHITE else "Black"
            think_out.append({"name": name, "who": who, "wp": wp, "cp": cp_of(wp),
                              "moves": [{"san": board.san(m), "p": p} for m, p in moves]})

        # The bar is always from White's point of view. At game over there are no legal moves and
        # think() returns a 0.5 placeholder, so the result is shown instead.
        wp_stm = think_out[0]["wp"] if think_out and not game_over(board) else None
        if wp_stm is None:
            white_wp = 1.0 if game_result(board) == "1-0" else 0.0 if game_result(board) == "0-1" else 0.5
        else:
            white_wp = wp_stm if board.turn == chess.WHITE else 1 - wp_stm

        # say who won in terms of the person reading it, not in terms of colours
        over = self.over()
        if self.flagged is not None:
            yours = (self.flagged == chess.WHITE) == self.human_white
            status = "You lost on time" if yours else "The engine lost on time"
            if self.board.has_insufficient_material(not self.flagged):
                status = "Draw: time out against insufficient material"
            evaltext = "flag"
        elif over:
            r = game_result(board)
            you = "1-0" if self.human_white else "0-1"
            if r == "1/2-1/2":
                status = "Draw by " + ("stalemate" if board.is_stalemate()
                                       else "insufficient material"
                                       if board.is_insufficient_material()
                                       else "repetition" if board.is_repetition(3)
                                       else "the fifty-move rule" if board.halfmove_clock >= 100
                                       else "the rules")
            else:
                how = "checkmate" if board.is_checkmate() else "the game ending"
                status = ("You win by " + how) if r == you else ("The engine wins by " + how)
            evaltext = "game over"
        else:
            yours = board.turn == (chess.WHITE if self.human_white else chess.BLACK)
            status = ("Check. " if board.is_check() else "") + ("Your move" if yours
                                                                else "The engine is thinking")
            evaltext = (f"{cp_of(white_wp):+.0f}cp for White, judged by "
                        f"{model_for(self.version)[1]}")

        # From the game's own root, not the standard start, since a game can start from a FEN.
        tmp, rows = board.root(), []
        for i, m in enumerate(board.move_stack):
            san = tmp.san(m); tmp.push(m)
            if i % 2 == 0:
                rows.append(f"<div class='row'><span>{i//2+1}. {san}</span>"
                            f"<span id='b{i//2}'></span></div>")
            else:
                rows[-1] = rows[-1].replace(f"<span id='b{i//2}'></span>", f"<span>{san}</span>")
        return {
            "pieces": pieces, "legal": legal if not game_over(board) else {},
            "last": board.move_stack[-1].uci() if board.move_stack else None,
            "status": status, "white_wp": white_wp, "evaltext": evaltext,
            "you_are": "White" if self.human_white else "Black",
            "turn_is_you": board.turn == (chess.WHITE if self.human_white else chess.BLACK),
            "think": think_out, "movelist": "".join(rows) or "<div class='sub'>no moves yet</div>",
            "opponent": self.version, "opponents": CATALOGUE,
            "fen": board.fen(), "depth": self.visits,
            # a FEN carries no history, so the ply count goes in the payload
            "plies": len(board.move_stack),
            # the tree's verdict on its own move, which is the number it actually played on
            "searched": self.last_search,
            # how well the adapter predicted the player's move before seeing it, early in the
            # session against late, which shows whether it is learning
            "ttt": self.om.summary() if self.om is not None else "",
            "name": self.name,
            "serious": self.serious, "elo": self.elo,
            "tc": (f"{self.tc[0]:g}+{self.tc[1]:g}" if self.tc else ""),
            "tcname": (timectl.describe(*self.tc) + ", the engine allocates its own time"
                       if self.tc else ""),
            "clock": ({"you": self.left(chess.WHITE if self.human_white else chess.BLACK),
                       "them": self.left(chess.BLACK if self.human_white else chess.WHITE),
                       # whether a clock is running, not just whose turn it is, since before the
                       # first move neither runs
                       "running": self.since is not None}
                      if self.tc else None),
            "adaptive": self.adaptive_time, "book": self.use_book, "temp": self.temperature,
            "flagged": (None if self.flagged is None else
                        ("you" if (self.flagged == chess.WHITE) == self.human_white else "them")),
            "over": self.over(),
        }

    def close(self):
        self.stop_pondering()
        self.tree.cache.clear()
        self.tree.reset()


SESSIONS = {}
SESSIONS_LOCK = threading.Lock()


def _loaded():
    """Every model currently loaded. More than one once somebody picks a second version."""
    with MODELS_LOCK:
        return [m for m, _, _ in MODELS.values()]


def sleep_now():
    """Hand the GPU back. The games are kept, and their trees are rebuilt on the next move."""
    global SLEEPING
    with WAKE_LOCK:
        # asked again under the lock, since a request that got in first cancels the sleep
        if SLEEPING or time.time() - LAST_ACTIVE <= IDLE_MINUTES * 60:
            return
        # The games stay, only their trees and caches go, so a player back from a break finds the
        # game where it was.
        with SESSIONS_LOCK:
            live = list(SESSIONS.values())
        for g in live:
            g.close()
        for m in _loaded():
            m.to("cpu")
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        SLEEPING = True
        print(f"  idle for {IDLE_MINUTES:.0f} minutes, gave the card back to training. "
              f"Still listening; the next move wakes it.", flush=True)


def wake():
    """Called by any real request. The health check deliberately does not call this."""
    global SLEEPING, LAST_ACTIVE
    LAST_ACTIVE = time.time()
    # Tell the trainer somebody is playing. On a shared GPU the board's work queues behind the
    # trainer's, so the trainer watches this file's mtime and pauses briefly between steps while it
    # is fresh. A file, so it survives either process restarting.
    try:
        with open(PLAYING_FILE, "w") as _fh:
            _fh.write(str(int(LAST_ACTIVE)))
    except OSError:
        pass
    # Always under the lock, so a request that arrives while the models are moving to the CPU
    # waits for the sleep to finish and then wakes it.
    with WAKE_LOCK:
        if not SLEEPING:
            return
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        for m in _loaded():
            m.to(dev)
        SLEEPING = False
        print("  woken up", flush=True)


def idle_watch():
    while True:
        time.sleep(30)
        try:
            if not SLEEPING and IDLE_MINUTES > 0 and                     time.time() - LAST_ACTIVE > IDLE_MINUTES * 60:
                sleep_now()
        except Exception as e:
            print(f"  idle watch: {type(e).__name__}: {e}", flush=True)


def _om_head(version):
    """The policy head this version's tree reads, as the key the network returns it under.

    The adapter fits a rank-16 delta against one head's logits and the tree adds it to whatever
    policy_head names, so it must be fitted on the head the tree uses.
    """
    try:
        return "policy_att" if pinned_search(version).get("policy_head") == "att" else "policy"
    except Exception:
        return "policy"


def log_game(sess):
    """Keep the game under the player's name, and hand its positions to the labeller."""
    b = sess.board
    if sess.logged or len(b.move_stack) < 8:
        return
    sess.logged = True
    who = "".join(c for c in (sess.name or "") if c.isalnum() or c in "-_ ").strip()[:24]
    who = who or "anonymous"
    engine = model_for(sess.version)[1]
    try:
        os.makedirs(GAMES_DIR, exist_ok=True)
        g = chess.pgn.Game.from_board(b)
        g.headers["Event"] = "MayFly board"
        g.headers["Date"] = time.strftime("%Y.%m.%d")
        g.headers["White"] = who if sess.human_white else engine
        g.headers["Black"] = engine if sess.human_white else who
        # a flag against a side that cannot mate is a draw by the rules, not a loss
        if sess.flagged is not None and b.has_insufficient_material(not sess.flagged):
            g.headers["Result"] = "1/2-1/2"
        elif sess.flagged is not None:
            g.headers["Result"] = "0-1" if sess.flagged == chess.WHITE else "1-0"
        else:
            g.headers["Result"] = game_result(b)
        if sess.tc:
            g.headers["TimeControl"] = f"{sess.tc[0]:g}+{sess.tc[1]:g}"
            g.headers["Termination"] = "time forfeit" if sess.flagged is not None else "normal"
        g.headers["Annotator"] = f"{sess.visits} visits"
        # which games count, ticked on the page, with the human's rating in the standard Elo tag
        g.headers["Serious"] = "yes" if sess.serious else "no"
        g.headers["Book"] = "yes" if sess.use_book else "no"
        # so a saved game says whether the opening was its top move or a draw from its own visits
        g.headers["Openings"] = ("its top move" if sess.temperature <= 0
                                 else f"its own, sampled at {sess.temperature:g}")
        if sess.elo:
            g.headers["WhiteElo" if sess.human_white else "BlackElo"] = str(sess.elo)
        # the engine's own reading, ply by ply, so a surprising move is answerable later
        by_ply = dict(getattr(sess, "record", []))
        node, i = g, 0
        while node.variations:
            node = node.variations[0]
            i += 1
            if i in by_ply:
                node.comment = by_ply[i]
        stamp = time.strftime("%Y%m%d_%H%M%S")
        safe = who.replace(" ", "-")
        with open(os.path.join(GAMES_DIR, f"{safe}_{stamp}.pgn"), "w", encoding="utf-8") as f:
            print(g, file=f)
    except Exception as e:
        print(f"  could not write the game: {type(e).__name__}: {e}", flush=True)

    # Every position the game passed through, appended to the labelling queue in one write, so it
    # cannot interleave mid-line with a self-play generator writing the same file. A torn line would
    # be one unparseable FEN, which deep_label skips.
    try:
        tmp, fens = b.root(), []
        for m in b.move_stack:
            if not tmp.is_game_over():
                fens.append(tmp.fen())
            tmp.push(m)
        os.makedirs(os.path.dirname(LABEL_QUEUE), exist_ok=True)
        with open(LABEL_QUEUE, "a", encoding="utf-8") as f:
            f.write("".join(x + chr(10) for x in fens))
        print(f"  logged {len(b.move_stack)} plies by {who}, "
              f"{len(fens)} positions queued for labelling", flush=True)
    except Exception as e:
        print(f"  could not queue the positions: {type(e).__name__}: {e}", flush=True)


def session_for(sid):
    """The session this cookie names, made if it is new, with the stale ones cleared out first.

    Every session holds a tree and a cache, so they cannot be allowed to accumulate. Anything
    untouched for SESSION_TTL goes, and if there are still too many the longest idle one goes as
    well. The session being asked for is never the one evicted.
    """
    now = time.time()
    with SESSIONS_LOCK:
        dead = [k for k, s in SESSIONS.items() if now - s.seen > SESSION_TTL and k != sid]
        while len(SESSIONS) - len(dead) >= MAX_SESSIONS and sid not in SESSIONS:
            idle = sorted((s.seen, k) for k, s in SESSIONS.items() if k not in dead and k != sid)
            if not idle:
                break
            dead.append(idle[0][1])
        # Taken out of the registry here and shut down below, outside the lock, since close() can
        # wait up to ten seconds on a ponder thread and would stall every other player.
        gone = [SESSIONS.pop(k) for k in dead]
        s = SESSIONS.get(sid)
        if s is None:
            s = SESSIONS[sid] = Session()
            s.sid = sid
            s.name = NAMES.get(sid, "")
            print(f"  a new player, {len(SESSIONS)} on the board", flush=True)
        s.seen = now
    for g in gone:
        # saved first, as a finished game is
        try:
            log_game(g)
        except Exception as e:
            print(f"  could not save an evicted game: {type(e).__name__}: {e}", flush=True)
        g.close()
    return s


# A shared passphrase, off by default, for a board with a public address. Not real security, just
# one secret checked on every request. Serve it over HTTPS if it is public.
PASS = os.environ.get("PASS", "")


def authorised(handler):
    if not PASS:
        return True
    cookie = handler.headers.get("Cookie") or ""
    for part in cookie.split(";"):
        if part.strip().startswith("mf="):
            # the gate page stores it through encodeURIComponent, so decode it before comparing
            from urllib.parse import unquote
            return unquote(part.strip()[3:]) == PASS
    return handler.headers.get("X-Pass") == PASS


GATE = """<!doctype html><meta charset=utf-8><title>MayFly</title>
<style>body{font:15px system-ui;margin:16vh auto;max-width:22rem;text-align:center;color:#1b1b19}
input{font:inherit;padding:.55rem;width:12rem;border:1px solid #ccc;border-radius:6px}
button{font:inherit;padding:.55rem 1rem;margin-left:.4rem;border-radius:6px;border:1px solid #779556;
background:#779556;color:#fff}</style>
<h3>MayFly</h3><p>This board needs a passphrase.</p>
<input id=p type=password autofocus><button onclick="go()">Enter</button>
<script>function go(){document.cookie='mf='+encodeURIComponent(p.value)+';path=/;max-age=2592000';
location.reload()}
p.addEventListener('keydown',e=>{if(e.key==='Enter')go()})</script>"""


class Handler(BaseHTTPRequestHandler):
    # HTTP/1.1 with keep alive, so clients do not reopen a socket for every poll. Content-Length is
    # set on every reply below, which makes that safe.
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _sid(self):
        """This browser's session id, from its cookie, made now if it has not been here before.

        Returns (id, is_new). Anything that does not look like an issued id is treated as absent
        rather than trusted, since it comes from the client and indexes a dictionary.
        """
        cookie = self.headers.get("Cookie") or ""
        for part in cookie.split(";"):
            part = part.strip()
            if part.startswith("mfs="):
                v = part[4:]
                if v.isalnum() and 8 <= len(v) <= 40:
                    return v, False
        return secrets.token_hex(12), True

    def _send(self, code, ctype, body, sid=None, fresh=False):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if fresh and sid:
            self.send_header("Set-Cookie", f"mfs={sid}; Path=/; Max-Age=2592000; SameSite=Lax")
        self.end_headers()
        self.wfile.write(body)

    def _deny(self):
        self._send(401, "text/html; charset=utf-8", GATE.encode("utf-8"))

    def do_GET(self):
        if not authorised(self):
            return self._deny()
        # No wake() here. This serves a static page, and monitors, link previews and crawlers
        # should not take the GPU back. A real visitor wakes it a moment later, since the page's
        # first act is api('state'), a POST.
        sid, fresh = self._sid()
        self._send(200, "text/html; charset=utf-8", PAGE.encode("utf-8"), sid, fresh)

    def do_POST(self):
        if not authorised(self):
            return self._deny()
        sid, fresh = self._sid()
        what = self.path.rsplit("/", 1)[-1]
        # The health check must not count as somebody playing, or the board would never go idle.
        if what == "health":
            body = json.dumps(self._health()).encode("utf-8")
            return self._send(200, "application/json", body)
        wake()
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            req = {}
        if not isinstance(req, dict):
            req = {}
        sess = session_for(sid)
        # One lock per session, not one for the server, so several people can play at once.
        with sess.lock:
            try:
                out = json.dumps(self._act(sess, what, req)).encode("utf-8")
            except Exception as e:
                # A handler that raises would close the socket with no reply, so report the error.
                print(f"  {what} failed: {type(e).__name__}: {e}", flush=True)
                out = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode("utf-8")
        self._send(200, "application/json", out, sid, fresh)

    def _health(self):
        """Alive, and able to think unless it is asleep.

        Sleeping is healthy, since the board is listening and the next real request wakes it."""
        # status is liveness (awake or sleeping), and staleness is its own field, so existing
        # readers of status are unaffected.
        stale = stale_sources()
        if SLEEPING:
            return {"status": "sleeping", "healthy": not stale, "stale": stale, "players": 0,
                    "idle_minutes": (time.time() - LAST_ACTIVE) / 60.0}
        try:
            think(model_for(DEFAULT_VERSION)[0], chess.Board())
            return {"status": "awake", "healthy": not stale, "stale": stale,
                    "players": len(SESSIONS),
                    "idle_minutes": (time.time() - LAST_ACTIVE) / 60.0}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def _act(self, sess, what, req):
        """One action on one person's game. Returns the snapshot they should be looking at."""
        if what == "move":
            uci = str(req.get("uci", ""))
            mv = None
            for cand in (uci, uci[:4]):
                try:
                    m = chess.Move.from_uci(cand)
                except Exception:
                    continue
                if m in sess.board.legal_moves:
                    mv = m
                    break
            if mv is not None and sess.board.turn != (chess.WHITE if sess.human_white else chess.BLACK):
                mv = None                 # the engine's move to make, not a move for the page
            if mv is not None and sess.over():
                mv = None                 # a finished game takes no more moves, nor teaches the adapter
            if mv is not None:
                # The player's clock stops the instant the move arrives, before the server's own
                # housekeeping (stopping the ponder, the adapter's update).
                sess.tick(lag=LAG)
                if sess.flagged is None:
                    sess.add_increment(sess.board.turn)
                # The ponder thread is walking the tree, so it stops before the adapter changes
                # what the tree would think. On nobody's clock, since since is already cleared.
                sess.stop_pondering()
                if sess.om is not None:
                    # the position before the move and the move they chose, a labelled example
                    sess.om.observe(sess.board, mv)
                    sess.tree.cache.clear()     # cached priors predate what it just learned
                if sess.flagged is not None:
                    log_game(sess)              # saved now, as it stands
                    return sess.snapshot()      # they ran out before the move landed
                sess.board.push(mv)
                sess.start_clock()           # and the engine's starts
                sess.engine_reply()
                if sess.over():
                    log_game(sess)
                sess.start_pondering()
        elif what == "new":
            sess.stop_pondering()
            log_game(sess)                 # the one you are abandoning is still worth keeping
            sess.last_search = None
            sess.logged = False
            sess.board = chess.Board()
            sess.record = []
            sess.tree.reset()
            sess.tree.cache.clear()
            if sess.tc:
                sess.set_tc(f"{sess.tc[0]:g}+{sess.tc[1]:g}")
            sess.engine_reply()
            sess.start_clock()
            sess.start_pondering()
        elif what == "undo":
            if sess.flagged is not None:
                return sess.snapshot()      # a game lost on time is over, start a new one or paste a position
            sess.stop_pondering()
            sess.tree.reset()
            sess.last_search = None
            # Back to the human's turn, taking back the engine's reply and the human's move, or only
            # the engine's first move when the human plays Black and has not moved yet.
            human = chess.WHITE if sess.human_white else chess.BLACK
            human_moves = 0
            while sess.board.move_stack:
                sess.board.pop()
                if sess.board.turn == human:
                    human_moves += 1         # that was the human's move
                    break
            if sess.om is not None:
                for _ in range(human_moves):
                    sess.om.forget_last()    # exactly the human moves taken back
            # engine readings for plies that no longer exist leave the record
            sess.record = [r for r in sess.record if r[0] <= len(sess.board.move_stack)]
            if sess.logged and not sess.over():
                sess.logged = False          # a finished game reopened is saved again at its end
            if sess.board.turn != human and not sess.over():
                sess.engine_reply()
                sess.start_pondering()
        elif what == "swapsides":
            sess.stop_pondering()
            sess.tree.reset()               # its cache holds priors shifted by the human's adapter
            sess.human_white = not sess.human_white
            sess.engine_reply()
            sess.start_pondering()
        elif what == "fen":
            # Paste a position in, for example to show the engine where it went wrong.
            try:
                b = chess.Board(str(req.get("fen", "")))
            except Exception:
                return sess.snapshot()
            if not b.is_valid():
                return sess.snapshot()
            sess.stop_pondering()
            log_game(sess)                  # the game being replaced is still worth keeping
            sess.logged = False
            sess.record = []                # its readings belong to the old game
            sess.flagged = None             # a flag belongs to the old game too
            sess.last_search = None
            sess.board = b
            sess.tree.reset()
            sess.human_white = (b.turn == chess.WHITE)   # whoever is to move is you
            if sess.tc:
                sess.set_tc(f"{sess.tc[0]:g}+{sess.tc[1]:g}")   # fresh clocks for the new position
                sess.start_clock()
            sess.start_pondering()
        elif what == "name":
            sess.name = str(req.get("n", ""))[:40]
            remember_name(sess.sid, sess.name)
        elif what == "book":
            sess.use_book = bool(req.get("on", True))
        elif what == "think":
            v = str(req.get("v", "sealed"))
            if v in ("sealed", "on", "off"):
                sess.adaptive_time = v
        elif what == "tc":
            was = sess.tc
            if sess.set_tc(req.get("v", "")):
                new, sess.tc = sess.tc, was  # saved under the control it was played at
                log_game(sess)               # the game being abandoned is still worth keeping
                sess.tc = new
                sess.logged = False
                sess.last_search = None
                sess.record = []
                sess.board = chess.Board()
                sess.tree.reset()
                sess.tree.cache.clear()
                sess.engine_reply()
                sess.start_clock()
                sess.start_pondering()
        elif what == "serious":
            sess.serious = bool(req.get("on", False))
            try:
                e = int(req.get("elo")) if req.get("elo") not in (None, "") else None
            except (TypeError, ValueError):
                e = None
            sess.elo = e if e is not None and 100 <= e <= 3500 else None
        elif what == "temp":
            sess.temperature = max(0.0, min(5.0, float(req.get("t", 0))))
        elif what == "opponent":
            # If it is the engine's turn when the swap lands, the new engine makes the move.
            if sess.set_version(str(req.get("which", ""))):
                sess.engine_reply()
        elif what == "depth":
            sess.stop_pondering()
            sess.visits = max(0, min(100_000, int(req.get("d", 400))))
            sess.tree.reset()              # a new visit count wants a fresh tree
        return sess.snapshot()


def version_catalogue():
    """Every frozen version on disk, newest first, read from versions.json.

    Read rather than hard-coded, so a newly frozen version appears on the menu. A version whose
    .pt has been moved or deleted is skipped rather than offered.
    """
    vdir = E("VERSION_DIR", paths.under("versions"))
    try:
        with open(os.path.join(vdir, "versions.json"), encoding="utf-8-sig") as fh:
            rows = json.load(fh)
    except Exception:
        rows = []
    out = []
    for v in sorted(rows, key=lambda r: -int(r.get("version", 0))):
        f = v.get("file", "")
        if not f or not os.path.exists(os.path.join(vdir, f)):
            continue
        bits = ["MayFly v%s" % v.get("version")]
        if v.get("hours"):
            bits.append("%.0fh" % v["hours"])
        if v.get("top1"):
            bits.append("top-1 %.1f%%" % (100 * v["top1"]))
        out.append({"key": f[:-3] if f.endswith(".pt") else f, "file": f,
                    "label": " · ".join(bits)})
    return out


def model_for(key):
    """The weights for a version key, loaded on first use and then kept.

    Lazy, because each version is several hundred MB of weights, and loading all of them up front
    would take GPU memory a trainer may need. A model loaded while the board is asleep goes
    straight to the CPU.
    """
    # The load happens outside MODELS_LOCK, so one player's load does not stall everyone else.
    # A per-version lock still loads each version once.
    with MODELS_LOCK:
        if key in MODELS:
            return MODELS[key]
        f = next((c["file"] for c in CATALOGUE if c["key"] == key), None)
        if f is None:
            raise KeyError(f"no such version: {key}")
        gate = LOADING.setdefault(key, threading.Lock())
    with gate:
        with MODELS_LOCK:
            if key in MODELS:
                return MODELS[key]
        got = load_version(f)
        with MODELS_LOCK:
            MODELS[key] = got
            if SLEEPING:
                got[0].to("cpu")
            return got


def load_version(name):
    """A frozen MayFly version, or the live best checkpoint if none is named."""
    import torch as _t
    from chess_net import ChessNet as _CN
    dev = "cuda" if _t.cuda.is_available() else "cpu"
    vdir = os.environ.get("VERSION_DIR", paths.under("versions"))
    path = name if os.path.isabs(name) else os.path.join(vdir, name)
    if not os.path.exists(path):
        path = os.path.join(HERE, name)
    if not os.path.exists(path):
        raise SystemExit(f"cannot find {name}")
    ck = _t.load(path, map_location=dev)
    c = ck.get("cfg", {})
    m = _CN(d=c.get("d", 1024), nhead=c.get("nhead", 16), layers=c.get("layers", 16),
            dim_ff=int(c.get("ff_mult", 2.0) * c.get("d", 1024)),
            qhead=bool(c.get("qhead", "qvalue.weight" in ck["model"])),
            att=bool(c.get("att", "att.q.weight" in ck["model"])),
            dist=int(c.get("dist", 64 if "vdist.weight" in ck["model"] else 0)),
            mlh=bool(c.get("mlh", "mleft.0.weight" in ck["model"]))).to(dev).eval()
    missing, _ = m.load_state_dict(ck["model"], strict=False)
    if missing:
        print(f"  {len(missing)} tensor(s) at their init: {', '.join(missing)}")
    v = ck.get("version")
    label = f"MayFly v{v}" if v else os.path.basename(path).replace(".pt", "")
    val = ck.get("val", {})
    print(f"  {label}: {m.num_params()/1e6:.1f}M params, iteration {ck.get('iter', 0):,}"
          + (f", {ck['hours']:.1f}h trained" if "hours" in ck else "")
          + (f", top-1 {100*val['top1']:.1f}%" if "top1" in val else ""))
    return m, label, float(val.get("policy", float("nan")))


if __name__ == "__main__":
    from pidfile import claim
    claim("play_web")
    want = os.environ.get("VERSION", "mayfly_v1.pt")
    CATALOGUE = version_catalogue()
    DEFAULT_VERSION = want[:-3] if want.endswith(".pt") else want
    if not any(c["key"] == DEFAULT_VERSION for c in CATALOGUE):
        # a hand-named checkpoint that is not in versions.json still has to be playable
        CATALOGUE.insert(0, {"key": DEFAULT_VERSION, "file": want, "label": DEFAULT_VERSION})
    load_names()
    record_sources()
    model_for(DEFAULT_VERSION)          # load the default now, so a cold first move is not slow
    print(f"  {len(CATALOGUE)} version(s) on the menu: "
          + ", ".join(c["label"] for c in CATALOGUE))

    if TTT:
        _m = model_for(DEFAULT_VERSION)[0]
        _om = OpponentModel(_m, "cpu")
        print(f"  test-time training on: a rank {_om.rank} adapter, "
              f"{_om.A.numel() + _om.B.numel():,} parameters against "
              f"{_m.num_params()/1e6:.0f}M, one per player, learns their moves as they play")
    print(f"  up to {MAX_SESSIONS} people at once, each with their own board, tree and adapter")

    url = f"http://localhost:{PORT}"
    print(f"\nboard at {url}   (ctrl+c to stop)")
    # On a machine with no desktop, opening a browser can hang, so set OPEN=0 there.
    if os.environ.get("OPEN", "1") == "1":
        try:
            webbrowser.open(url)
        except Exception:
            pass
    # Every interface by default, so the board is reachable from a phone or another machine without
    # a tunnel. BIND=127.0.0.1 makes it local again.
    host = os.environ.get("BIND", "0.0.0.0")
    srv = ThreadingHTTPServer((host, PORT), Handler)
    # The address is printed from getsockname, not from the value asked for, so the log is true.
    if IDLE_MINUTES > 0:
        threading.Thread(target=idle_watch, daemon=True).start()
        print(f"  idles after {IDLE_MINUTES:.0f} minutes with nobody playing, and gives the GPU "
              f"back to training until someone returns")
    ip, bound = srv.socket.getsockname()[:2]
    print(f"listening on {ip}:{bound}, "
          + ("passphrase required" if PASS else "no passphrase"), flush=True)
    try:
        srv.serve_forever()
    except BaseException as e:
        # if the server stops, it says so and why
        print(f"server stopped: {type(e).__name__}: {e}", flush=True)
        raise
