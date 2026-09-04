"""Pre-flight checks for the nine-run room comparison.

Nine runs cost about eighteen hours of GPU time, so the properties the
comparison depends on are checked here first, mechanically, rather than assumed.
Each check either passes or explains what it found.

The checks that need a dataset build a tiny synthetic one in a temporary
directory, so this runs anywhere without the real split and without a GPU.

    python verify_rooms_wen.py
    python verify_rooms_wen.py --data-root /content/dataset_split   # real data
"""
from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT / "fer_wen"), str(ROOT / "vit_s16_baseline" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

FAILS: List[str] = []
PARTIAL: List[str] = []


def check(label: str, ok: Optional[bool], detail: str = "") -> None:
    """ok True = confirmed, False = refuted, None = could not be checked here."""
    tag = {True: "CONFIRMAT", False: "INFIRMAT ", None: "PARTIAL  "}[ok]
    print(f"  [{tag}] {label}" + (f"\n             {detail}" if detail else ""))
    if ok is False:
        FAILS.append(label)
    elif ok is None:
        PARTIAL.append(label)


def head(n: int, title: str) -> None:
    print()
    print("=" * 78)
    print(f"{n}. {title}")
    print("=" * 78)


# ----------------------------------------------------------------------
def tiny_dataset(classes: List[str], per: Dict[str, int]) -> Path:
    """A synthetic split, so the RNG and loader checks need no real data."""
    from PIL import Image
    root = Path(tempfile.mkdtemp(prefix="rooms_wen_verify_"))
    rng = np.random.default_rng(0)
    for part, n in per.items():
        for c in classes:
            d = root / part / c
            d.mkdir(parents=True)
            for i in range(n):
                arr = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
                Image.fromarray(arr).save(d / f"{i:04d}.jpg")
    return root


def config_for(data_root: Path, cfg: Dict[str, Any], mode: str, seed: int,
               workers: int = 0) -> Dict[str, Any]:
    import copy
    c = copy.deepcopy(cfg)
    c["data"]["train_dir"] = str(data_root / "train")
    c["data"]["val_dir"] = str(data_root / "val")
    c["data"]["eval_dir"] = str(data_root / "eval")
    c["training"]["center_mode"] = mode
    c["training"]["seed"] = seed
    c["training"]["num_workers"] = workers
    c["training"]["batch_size"] = 8
    return c


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None,
                    help="use a real split instead of a synthetic one")
    args = ap.parse_args()

    base = yaml.safe_load(open(HERE / "config_rooms_wen.yaml", encoding="utf-8"))
    classes = list(base["data"]["class_names"])

    # ---------------------------------------------------------------
    head(1, "Toate fisierele Python se compileaza")
    files = sorted(HERE.glob("*.py")) + [ROOT / "fer_wen" / "wen_center_loss.py"]
    for f in files:
        try:
            ast.parse(f.read_text(encoding="utf-8"))
            check(f.name, True)
        except SyntaxError as e:
            check(f.name, False, f"linia {e.lineno}: {e.msg}")

    # ---------------------------------------------------------------
    head(2, "Regula Wen coincide cu formula din articol, calculata naiv")
    from wen_center_loss import CenterLossVariant
    torch.manual_seed(0)
    worst = 0.0
    for _ in range(3):
        C, D, B = 6, 16, 32
        m = CenterLossVariant(num_classes=C, feat_dim=D, mode="wen")
        with torch.no_grad():
            m.centers.copy_(torch.randn(C, D))
        c0 = m.centers.clone()
        feats, labels, alpha = torch.randn(B, D), torch.randint(0, C, (B,)), 0.5
        m.update_centers(feats, labels, alpha=alpha)
        exp = c0.clone()
        for j in range(C):
            num, n = torch.zeros(D), 0
            for i in range(B):
                if labels[i].item() == j:
                    num += (c0[j] - feats[i]); n += 1
            exp[j] = c0[j] - alpha * (num / (1 + n))
        worst = max(worst, float((m.centers - exp).abs().max()))
    check("ecuatia (4), eroare maxima pe 3 probe", worst < 1e-5, f"{worst:.2e}")

    # ---------------------------------------------------------------
    head(3, "Varianta gradient coincide cu implementarea din teza")
    from center_loss import CenterLoss
    worst_l = worst_f = worst_c = 0.0
    for _ in range(3):
        C, D, B = 6, 16, 32
        init = torch.randn(C, D)
        a = CenterLossVariant(num_classes=C, feat_dim=D, mode="gradient")
        b = CenterLoss(num_classes=C, feat_dim=D)
        with torch.no_grad():
            a.centers.copy_(init); b.centers.copy_(init)
        f = torch.randn(B, D)
        fa = f.clone().requires_grad_(True); fb = f.clone().requires_grad_(True)
        y = torch.randint(0, C, (B,))
        la, lb = a(fa, y), b(fb, y)
        la.backward(); lb.backward()
        worst_l = max(worst_l, abs(la.item() - lb.item()))
        worst_f = max(worst_f, float((fa.grad - fb.grad).abs().max()))
        worst_c = max(worst_c, float((a.centers.grad - b.centers.grad).abs().max()))
    check("loss, gradient pe trasaturi si pe centre",
          max(worst_l, worst_f, worst_c) == 0.0,
          f"loss {worst_l:.2e} | feat {worst_f:.2e} | centre {worst_c:.2e}")

    # ---------------------------------------------------------------
    head(4, "Aliniere RNG intre brate: model, centre si ordinea imaginilor")
    from rooms_wen_data import build_dataloaders
    from rooms_wen_model import build_model_from_config
    from train_rooms_wen import set_seed

    data_root = Path(args.data_root) if args.data_root else tiny_dataset(
        classes, {"train": 6, "val": 2, "eval": 2})
    print(f"  (set folosit: {data_root})")

    def simulate(mode: str, seed: int = 42, n_batches: int = 3):
        """Reproduce the training loop's RNG consumption, without training."""
        cfg = config_for(data_root, base, mode, seed)
        set_seed(seed)
        train_loader, _, _ = build_dataloaders(cfg, pin=False)
        model = build_model_from_config(cfg["model"])
        w = model.patch_embed.proj.weight.detach().flatten()[:4].clone()
        centres = None
        if mode != "none":
            g = torch.get_rng_state()
            cl = CenterLossVariant(num_classes=cfg["model"]["num_classes"],
                                   feat_dim=cfg["model"]["embed_dim"], mode=mode)
            centres = cl.centers.detach().flatten()[:4].clone()
            torch.set_rng_state(g)
        order: List[int] = []
        for i, (_, targets) in enumerate(train_loader):
            order.extend(targets.tolist())
            if i + 1 >= n_batches:
                break
        return w, centres, order

    res = {m: simulate(m) for m in ("none", "gradient", "wen")}
    w0 = res["none"][0]
    check("modelul e identic la toate trei modurile",
          all(torch.equal(res[m][0], w0) for m in res))
    check("centrele initiale identice intre gradient si wen",
          torch.equal(res["gradient"][1], res["wen"][1]))
    same_none_grad = res["none"][2] == res["gradient"][2]
    same_grad_wen = res["gradient"][2] == res["wen"][2]
    check("ordinea imaginilor identica: none == gradient", same_none_grad,
          f"none {res['none'][2][:8]}\n             grad {res['gradient'][2][:8]}")
    check("ordinea imaginilor identica: gradient == wen", same_grad_wen)

    # ---------------------------------------------------------------
    head(5, "Schedulerul produce exact ratele intentionate")
    from train_rooms_wen import build_scheduler
    t = dict(base["training"])
    lin = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(lin.parameters(), lr=float(t["learning_rate"]))
    sch = build_scheduler(opt, t)
    seq = []
    for _ in range(int(t["epochs"])):
        seq.append(opt.param_groups[0]["lr"])
        sch.step()
    base_lr, warm = float(t["learning_rate"]), int(t["warmup_epochs"])
    want_warm = [base_lr * (i + 1) / warm for i in range(warm)]
    check(f"epocile 1..{warm} sunt warmup liniar",
          all(abs(a - b) < 1e-12 for a, b in zip(seq[:warm], want_warm)),
          f"observat {[f'{x:.2e}' for x in seq[:warm]]}")
    check("prima epoca este lr/warmup = 6.00e-05",
          abs(seq[0] - base_lr / warm) < 1e-12, f"{seq[0]:.2e}")
    check("dupa warmup rata scade monoton (cosinus)",
          all(seq[i] >= seq[i + 1] - 1e-15 for i in range(warm, len(seq) - 1)),
          f"epoca {warm+1}: {seq[warm]:.2e} ... ultima: {seq[-1]:.2e}")

    # ---------------------------------------------------------------
    head(6, "Un pas AMP sarit nu misca centrele Wen")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        check("pas AMP sarit", None,
              "fara GPU aici; GradScaler nu sare pasi cu AMP dezactivat")
    else:
        net = torch.nn.Linear(8, 6).to(dev)
        o = torch.optim.SGD(net.parameters(), lr=0.1)
        scaler = torch.amp.GradScaler(dev, enabled=True)
        cl = CenterLossVariant(num_classes=6, feat_dim=6, mode="wen").to(dev)
        c_before = cl.centers.detach().clone()
        w_before = net.weight.detach().clone()

        o.zero_grad(set_to_none=True)
        x = torch.randn(16, 8, device=dev)
        y = torch.randint(0, 6, (16,), device=dev)
        with torch.autocast(dev, dtype=torch.float16, enabled=True):
            out = net(x)
            loss = out.pow(2).mean()
        scaler.scale(loss).backward()
        net.weight.grad[0, 0] = float("inf")        # gradient neterminat
        feats_finite = bool(torch.isfinite(out).all())
        s_before = scaler.get_scale()
        scaler.step(o); scaler.update()
        step_taken = scaler.get_scale() >= s_before
        if step_taken:                               # protectia din bucla
            cl.update_centers(out.float(), y, alpha=0.5)
        check("embeddingurile pot fi finite cand gradientul nu e", feats_finite)
        check("pasul a fost detectat ca sarit", not step_taken,
              f"scale {s_before:g} -> {scaler.get_scale():g}")
        check("reteaua nu s-a miscat", torch.equal(w_before, net.weight.detach()))
        check("centrele Wen nu s-au miscat",
              torch.equal(c_before, cl.centers.detach()))

    # ---------------------------------------------------------------
    head(7, "Centrele nefinite sunt detectate imediat")
    src = (HERE / "train_rooms_wen.py").read_text(encoding="utf-8")
    check("garda pe embeddinguri inainte de actualizare",
          "torch.isfinite(feats).all()" in src)
    check("garda pe centre dupa actualizare",
          "torch.isfinite(center_loss.centers).all()" in src)
    check("garda pe loss la fiecare epoca", "math.isfinite" in src)

    # ---------------------------------------------------------------
    head(8, "Testul nu este construit si nu este atins de antrenare")
    data_src = (HERE / "rooms_wen_data.py").read_text(encoding="utf-8")
    check("build_dataloaders nu citeste eval_dir",
          "eval_dir" not in data_src.split("def build_dataloaders")[1]
          .split("def build_eval_loader")[0])
    check("bucla de antrenare nu importa build_eval_loader",
          "build_eval_loader" not in src)
    check("selectia checkpointului foloseste doar validarea",
          "improved = vacc_m.avg > best_acc" in src)
    check("best_acc porneste sub zero", "best_acc = -1.0" in src)

    # ---------------------------------------------------------------
    head(9, "Reluarea pastreaza toate cele noua brate, si run_matches respinge "
            "o configuratie schimbata")
    import compare_rooms_wen as CR

    fake = Path(tempfile.mkdtemp(prefix="rooms_wen_runs_"))

    def make_run(v, cfg, best_val, commit="a" * 40, dirty=False):
        d = fake / f"2026-01-01_00-00-{v['seed'] % 60:02d}_{v['run_name']}"
        (d / "logs").mkdir(parents=True)
        ev = d / "outputs" / "eval_2026-01-01_00-00-00"
        ev.mkdir(parents=True)
        (d / "checkpoints").mkdir()
        with open(d / "logs" / "config_used.yaml", "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        with open(d / "logs" / "provenance.json", "w", encoding="utf-8") as fh:
            json.dump({"git_commit": commit, "git_dirty": dirty,
                       "torch": "2.12.1+cu126", "cuda": "12.6",
                       "gpu": "Tesla T4"}, fh)
        for f in ("classification_report.txt", "confusion_matrix.csv",
                  "predictions.csv"):
            (ev / f).write_text("x", encoding="utf-8")
        for f in ("embeddings.npy", "labels.npy"):
            np.save(ev / f.replace(".npy", ""), np.zeros(2))
        (ev / "metrics.txt").write_text(
            f"Overall accuracy: {0.5 + v['seed'] / 1000:.4f}" + chr(10), encoding="utf-8")
        torch.save({"best_val_accuracy": best_val},
                   d / "checkpoints" / "best_model.pt")
        return d

    all_v = CR.variants()
    for v in all_v:
        make_run(v, CR.variant_config(base, v, workers=2), best_val=0.55)

    records = [CR.record_for(str(fake), base, v) for v in all_v]
    ok_rows = [r for r in records if r is not None]
    check("toate cele 9 brate sunt regasite dupa o reluare", len(ok_rows) == 9,
          f"{len(ok_rows)}/9")
    check("fiecare (mod, seed) apare o singura data",
          len({(r["center_mode"], r["seed"]) for r in ok_rows}) == 9)
    check("acuratetea de validare e recuperata din checkpoint, nu din log",
          all(r["best_val_accuracy"] == 0.55 for r in ok_rows),
          str(sorted({r["best_val_accuracy"] for r in ok_rows})))
    check("modurile si seed-urile sunt cele asteptate",
          {(r["center_mode"], r["seed"]) for r in ok_rows}
          == {(v["center_mode"], v["seed"]) for v in all_v})

    # o configuratie schimbata intr-un singur camp trebuie respinsa
    v0 = all_v[0]
    for field, path in [("weight_decay", ("training", "weight_decay")),
                        ("dropout", ("model", "dropout")),
                        ("val_dir", ("data", "val_dir")),
                        ("num_classes", ("model", "num_classes")),
                        ("num_workers", ("training", "num_workers")),
                        ("center_loss_lr", ("training", "center_loss_lr")),
                        ("image_size", ("model", "image_size"))]:
        import copy as _c
        bad = _c.deepcopy(base)
        node = bad
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = "SCHIMBAT" if isinstance(node[path[-1]], str) else 12345
        d = fake / "changed"
        if d.exists():
            import shutil as _s
            _s.rmtree(d)
        # workers=None so the deliberately broken value survives instead of
        # being overwritten by the argument -- the first version of this test
        # passed workers=2 and so could never fail on num_workers.
        run = make_run({**v0, "run_name": "changed", "seed": 1},
                       CR.variant_config(bad, v0, workers=None), 0.5)
        ok, why = CR.run_matches(str(run), base, v0)
        check(f"respinge o rulare cu {field} schimbat", not ok, why[:100])
        import shutil as _s
        _s.rmtree(run)

    # o rulare produsa dintr-un arbore murdar nu poate fi identificata
    run = make_run({**v0, "run_name": "dirtyrun", "seed": 2},
                   CR.variant_config(base, v0, workers=2), 0.5, dirty=True)
    ok, why = CR.run_matches(str(run), base, v0)
    check("respinge o rulare produsa dintr-un arbore murdar", not ok, why[:90])

    # coduri diferite intre brate trebuie semnalate
    comp = CR.consistent_provenance(
        [{"provenance": {"git_commit": "a" * 40, "torch": "2.12.1", "gpu": "T4"}},
         {"provenance": {"git_commit": "b" * 40, "torch": "2.12.1", "gpu": "T4"}}])
    check("semnaleaza brate produse de commit-uri diferite", bool(comp),
          comp[0][:80] if comp else "")

    # ---------------------------------------------------------------
    head(10, "Formulele Center Loss nu au fost modificate")
    def sh(*a):
        r = subprocess.run(a, cwd=str(ROOT), capture_output=True, text=True)
        return r.stdout if r.returncode == 0 else ""
    diff = sh("git", "diff", "--", "fer_wen/wen_center_loss.py",
              "vit_s16_baseline/src/center_loss.py")
    check("wen_center_loss.py si center_loss.py nemodificate",
          diff.strip() == "", diff[:200] if diff.strip() else "")

    # ---------------------------------------------------------------
    head(11, "Notebook-ul: fixare pe commit si fara jeton in remote")
    nb = json.loads((HERE / "run_rooms_wen_colab.ipynb").read_text(encoding="utf-8"))
    nbtxt = "".join("".join(c["source"]) for c in nb["cells"])
    check("fixeaza un commit explicit", "PINNED_COMMIT" in nbtxt)
    check("curata jetonul din remote dupa autentificare",
          "set-url" in nbtxt and "PUBLIC_URL" in nbtxt)
    check("verifica setul cu RoomDataset, nu cu un glob",
          "RoomDataset" in nbtxt)

    # ---------------------------------------------------------------
    print()
    print("=" * 78)
    if FAILS:
        print(f"VERDICT: {len(FAILS)} VERIFICARI AU ESUAT")
        for f in FAILS:
            print("   -", f)
    else:
        print("VERDICT: TOATE VERIFICARILE AUTOMATE TREC")
    if PARTIAL:
        print(f"\n{len(PARTIAL)} nu au putut fi verificate aici:")
        for f in PARTIAL:
            print("   -", f)
    print("=" * 78)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
