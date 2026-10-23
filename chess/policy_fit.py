r"""Measure how well the policy head fits its target, and how much mass it puts on illegal moves.

The policy target has at most five moves, so below rank five it is zero, not flat. If the model
puts real probability on moves whose target is zero, it is underfit to a peaked target, and
widening the target would not help.

Three measurements, because the training objective and the search read different distributions.

  mass on the 5 target moves, over the full 4,162-way softmax
        what the training loss optimises. Near 1.0 means the head is well fit.

  mass on legal moves, over the full softmax
        probability spent on moves that cannot be played, which the loss does not see.

  mass on the 5 target moves after renormalising over legal moves only
        what _expand hands to PUCT as priors.

It also reports top-1 with and without a legality mask, and checks that the target's own moves are
legal on the decoded board, which validates the measurement itself.

  python chess/policy_fit.py [POSITIONS]
  env: VERSION=mayfly_v2.pt  SHARDS=<data>/shards
"""
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)

import paths

E = os.environ.get
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
SHARDS = E("SHARDS", paths.SHARDS)
BATCH = 64


if __name__ == "__main__":
    import numpy as np
    import torch
    import chess
    from data_packed import PackedLoader
    from play_web import load_version
    from moves import move_to_idx, N_MOVES

    model, label, _ = load_version(E("VERSION", "mayfly_v2.pt"))
    dev = str(next(model.parameters()).device)
    print(f"{label} on {dev}, {N_POS} positions from the policy corpus", flush=True)

    # The board comes back out of the record, so legality can be asked of the real position.
    # ids are already in the side-to-move frame, same as board_to_ids produces, so they go on the
    # board as white to move and move_to_idx is called with flip=False.
    SYM = {1: chess.PAWN, 2: chess.KNIGHT, 3: chess.BISHOP, 4: chess.ROOK,
           5: chess.QUEEN, 6: chess.KING}

    def to_board(ids):
        b = chess.Board(None)
        for sq in range(64):
            v = int(ids[sq])
            if 1 <= v <= 6:
                b.set_piece_at(sq, chess.Piece(SYM[v], chess.WHITE))
            elif 7 <= v <= 12:
                b.set_piece_at(sq, chess.Piece(SYM[v - 6], chess.BLACK))
        b.turn = chess.WHITE
        b.clear_stack()
        return b

    ld = PackedLoader(SHARDS, batch=BATCH, split="val", device=dev)
    on_target, on_legal, on_target_legal, kls, ents, tents = [], [], [], [], [], []
    tgt_legal = []
    illegal_top = unmasked_top1 = masked_top1 = 0
    seen = 0
    with torch.no_grad():
        while seen < N_POS:
            ids, cast, hm, ep, pol, val, *_ = ld.next()
            out = model(ids, cast, hm, ep)
            logp = torch.log_softmax(out["policy"].float(), -1)
            p = logp.exp()
            tgt = pol.float()
            # the objective's own measure, how much mass sits on the moves the target names
            support = tgt > 0
            m_t = (p * support).sum(1)
            # KL(target || model), the loss minus the target's own entropy
            kl = (tgt * (torch.log(tgt.clamp_min(1e-12)) - logp)).sum(1)
            ent = -(p * logp).sum(1)
            tent = -(tgt * torch.log(tgt.clamp_min(1e-12))).sum(1)

            idn = ids.detach().cpu().numpy()
            pn = p.detach().cpu().numpy()
            sn = support.detach().cpu().numpy()
            tn = tgt.argmax(1).detach().cpu().numpy()
            for i in range(idn.shape[0]):
                if seen >= N_POS:
                    break
                b = to_board(idn[i])
                if not b.is_valid():
                    continue
                legal = np.zeros(N_MOVES, bool)
                for mv in b.legal_moves:
                    j = move_to_idx(mv, False)
                    if 0 <= j < N_MOVES:
                        legal[j] = True
                # The instrument check. The target's own moves come from this same record, so
                # they must be legal on the board it encodes, or the board reconstruction or the
                # move_to_idx frame is wrong and every legality number here is noise.
                tmoves = np.where(sn[i])[0]
                if len(tmoves):
                    tgt_legal.append(float(legal[tmoves].mean()))
                lm = float(pn[i][legal].sum())
                on_legal.append(lm)
                on_target.append(float(m_t[i]))
                # what the search sees, renormalised over legal moves only
                if lm > 1e-9:
                    on_target_legal.append(float(pn[i][sn[i] & legal].sum() / lm))
                # Top-1 both ways. train_av's validate takes argmax over all 4,162 moves with no
                # legality mask, while _expand always renormalises over legal moves.
                top = int(np.argmax(pn[i]))
                illegal_top += int(not legal[top])
                want = int(np.argmax(tgt[i].detach().cpu().numpy())) if False else int(tn[i])
                unmasked_top1 += int(top == want)
                masked = np.where(legal, pn[i], -1.0)
                masked_top1 += int(int(np.argmax(masked)) == want)
                kls.append(float(kl[i])); ents.append(float(ent[i])); tents.append(float(tent[i]))
                seen += 1

    f = lambda v: (sum(v) / max(len(v), 1))
    print("")
    print(f"  {seen} positions")
    print(f"  {'CHECK: target moves legal on the decoded board':<48} "
          f"{100*f(tgt_legal):>6.1f}%   (must be ~100)")
    if f(tgt_legal) < 0.95:
        print("")
        print("  THE INSTRUMENT IS BROKEN. The target's own moves are not legal on the board this")
        print("  script decoded, so the legality figures below mean nothing. Fix the frame first.")
    print(f"  {'mass on the 5 target moves (full softmax)':<48} {100*f(on_target):>6.1f}%")
    print(f"  {'mass on LEGAL moves (full softmax)':<48} {100*f(on_legal):>6.1f}%")
    print(f"  {'mass on target moves after legal renormalisation':<48} "
          f"{100*f(on_target_legal):>6.1f}%")
    print(f"  {'the model top move is ILLEGAL':<48} "
          f"{100*illegal_top/max(seen,1):>6.1f}% of positions")
    print("")
    print(f"  {'top-1 as train_av reports it (no legality mask)':<48} "
          f"{100*unmasked_top1/max(seen,1):>6.1f}%")
    print(f"  {'top-1 over LEGAL moves, which is what plays':<48} "
          f"{100*masked_top1/max(seen,1):>6.1f}%")
    print("")
    print(f"  {'KL(target || model)':<48} {f(kls):>6.3f} nats")
    print(f"  {'entropy of the model distribution':<48} {f(ents):>6.3f} nats")
    print(f"  {'entropy of the target distribution':<48} {f(tents):>6.3f} nats")
    print("")
    m = f(on_target)
    if m > 0.90:
        print("  THE POLICY IS WELL FIT to its target. The smear is not an underfitting problem,")
        print("  and whatever explains the rook-pawn habit is somewhere else.")
    elif m > 0.40:
        print(f"  UNDERFIT, as predicted: {100*m:.0f}% of the mass sits on the target's five moves")
        print("  and the rest is spread over four thousand others. The model has learned WHICH")
        print("  moves are good without learning to CONCENTRATE on them. Widening PV_TAKE would")
        print("  not help: the target is already 5-support, which is as peaked as it gets.")
    else:
        print(f"  BADLY UNDERFIT: only {100*m:.0f}% on the target moves. The head is closer to")
        print("  uniform than to its objective.")
    print("")
    print(f"  entropy gap: model {f(ents):.2f} nats against target {f(tents):.2f}. A well-fit head")
    print(f"  would sit near the target; the difference is the temperature the smear lives in.")
