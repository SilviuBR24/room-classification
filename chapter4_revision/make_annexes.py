"""Build the code annexes as a single, self-contained LaTeX file.

The output compiles on its own -- one file uploaded to Overleaf produces the
PDF -- and carries instructions for merging it into the thesis afterwards. The
code is embedded rather than pulled in with \\lstinputlisting, so nothing else
has to be uploaded alongside it.

Formatting follows the ETTI regulation: fixed-width font, body size within the
permitted 8-12 points, single line spacing, and a single column, which at 8pt
leaves roughly 100 characters per line -- enough for every line in this project
except a handful.

    python make_annexes.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

REPO = Path(__file__).resolve().parent.parent
OUT = (REPO.parent / "Versions" / "Chapter_4" /
       "Lupu_Silviu-George_Anexe_Cod.tex")

# (file, caption, one line saying what it contributes)
ANNEXES = [
    ("vit_s16_baseline/make_split.py", "make\\_split.py",
     "Deterministic partitioning of the labelled pool into training, "
     "validation and unlabelled subsets."),
    ("vit_s16_baseline/src/dataset.py", "dataset.py",
     "Folder-based dataset with a fixed, explicit class order, and the "
     "augmentation pipeline."),
    ("vit_s16_baseline/src/vit.py", "vit.py",
     "The ViT-S/16 architecture, implemented from scratch."),
    ("vit_s16_baseline/src/center_loss.py", "center\\_loss.py",
     "The Center Loss objective."),
    ("vit_s16_baseline/src/trainer.py", "trainer.py",
     "The supervised training engine: schedule, mixed precision, "
     "checkpoint selection on validation."),
    ("vit_s16_baseline/src/pseudo_label.py", "pseudo\\_label.py",
     "Selection of pseudo-labels by confidence threshold or target coverage."),
    ("vit_s16_baseline/src/semi_trainer.py", "semi\\_trainer.py",
     "The online self-training loop, which labels unannotated images during "
     "training."),
    ("vit_s16_baseline/evaluate.py", "evaluate.py",
     "Evaluation on the held-out test set: accuracy, per-class accuracy and "
     "confusion matrices."),
    ("vit_s16_baseline/analyze_embeddings.py", "analyze\\_embeddings.py",
     "Embedding-space measurements: within- and between-class distances, "
     "separation ratio, silhouette, Davies--Bouldin, and the projections."),
    ("vit_s16_baseline/mnist_center_loss.py", "mnist\\_center\\_loss.py",
     "Validation of the Center Loss implementation on MNIST with a "
     "two-dimensional embedding."),
    ("vit_s16_baseline/run_colab.ipynb", "run\\_colab.ipynb",
     "The notebook that ran the experiments, including the coefficient sweep."),
]

PREAMBLE = r"""% ==========================================================================
%  CODE ANNEXES  --  Lupu Silviu-George
%  "Room Type Recognition Using Clustering Cost Functions for the Embedding
%   Space"
%
%  HOW TO USE THIS FILE
%
%  1. On its own. Upload this single file to Overleaf and compile. Nothing
%     else is needed: every listing is embedded, not read from disk.
%
%  2. Merged into the thesis. Delete everything from \documentclass down to
%     \begin{document}, and everything from \end{document} up. Paste what
%     remains at the end of the thesis, after the bibliography. Move the
%     \lstset block into the thesis preamble and add, next to the other
%     package imports:
%
%         \usepackage{listings}
%         \usepackage{xcolor}
%
%     The thesis class may already number sections differently; if so, replace
%     \section*{...} with whatever the class uses for an unnumbered heading,
%     as the Conclusions section does.
%
%  FORMATTING, against the ETTI regulation, Chapter 3 section 2
%
%     Fixed-width font for source code                        Courier
%     Body size 8..12 points, reduction recommended in annexes 8pt
%     Line spacing single, at most 1.2                        single
%     Single column (two columns are permitted, not required)
%
%     One column at 8pt leaves about 100 characters per line, which fits
%     every line in this project bar a handful; those few are wrapped and
%     marked with an arrow. Two columns would have left 49 characters and
%     wrapped several hundred lines.
%
%  EACH ANNEX MUST BE CITED AT LEAST ONCE IN THE BODY OF THE THESIS.
%  See the companion file 03_REFERINTE_ANEXE.md for the eleven sentences and
%  where each one goes. This is the requirement the bachelor thesis missed.
% ==========================================================================

\documentclass[11pt,a4paper]{article}

\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[a4paper,margin=2cm]{geometry}
\usepackage{listings}
\usepackage{xcolor}
\usepackage{hyperref}
\hypersetup{colorlinks=true,linkcolor=black,urlcolor=black}

\definecolor{codecomment}{rgb}{0.35,0.40,0.45}
\definecolor{codekeyword}{rgb}{0.10,0.25,0.55}
\definecolor{codestring}{rgb}{0.55,0.20,0.15}

\lstset{
  language=Python,
  basicstyle=\ttfamily\fontsize{8}{9.2}\selectfont,
  commentstyle=\color{codecomment}\itshape,
  keywordstyle=\color{codekeyword}\bfseries,
  stringstyle=\color{codestring},
  showstringspaces=false,
  breaklines=true,
  breakatwhitespace=false,
  postbreak=\mbox{\textcolor{codecomment}{$\hookrightarrow$}\space},
  tabsize=4,
  columns=fullflexible,
  keepspaces=true,
  upquote=true,
  frame=none,
  xleftmargin=0pt,
  aboveskip=4pt,
  belowskip=4pt,
  literate={ă}{{\u a}}1 {â}{{\^a}}1 {î}{{\^\i}}1 {ș}{{\c s}}1 {ț}{{\c t}}1
           {Ă}{{\u A}}1 {Â}{{\^A}}1 {Î}{{\^I}}1 {Ș}{{\c S}}1 {Ț}{{\c T}}1
           {–}{{-}}1 {—}{{-}}1 {’}{{'}}1 {“}{{"}}1 {”}{{"}}1 {…}{{...}}3
}

\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}

\title{Code Annexes}
\author{Lupu Silviu-George}
\date{}

\begin{document}
\maketitle
\thispagestyle{empty}

The annexes below contain the source code written for this project. They
follow the order in which the pipeline runs: the data are partitioned and
loaded, the model and the objective are defined, the supervised and
semi-supervised training engines are applied, the results are measured, the
implementation is validated on a reference task, and the whole is reproduced
from a notebook.

Code that is not the author's own contribution is not reproduced here.
Standard library and third-party imports appear only where the project's own
code calls them.

\tableofcontents
\newpage

"""


# Characters that appear in comments and markdown but have no glyph in the
# T1-encoded typewriter font `listings` uses. Mapping them here is more robust
# than extending the \lstset literate table for every one: prose in a code
# annex does not need Greek letters or emoji, and a missing glyph is a
# compilation failure rather than a visible defect.
NON_ASCII = {
    "λ": "lambda", "μ": "mu", "σ": "sigma", "α": "alpha",
    "β": "beta", "×": "x", "→": "->", "←": "<-",
    "≤": "<=", "≥": ">=", "≈": "~=", "≠": "!=",
    "–": "-", "—": "--", "‘": "'", "’": "'",
    "“": '"', "”": '"', "…": "...", "·": ".",
    "°": "deg", "±": "+/-", "•": "*", " ": " ",
}


def sanitise(text: str) -> str:
    """Replace characters the code font cannot render, and report the rest."""
    for bad, good in NON_ASCII.items():
        text = text.replace(bad, good)
    leftover = sorted({c for c in text if ord(c) > 127})
    if leftover:
        # Romanian diacritics are handled by the literate table in \lstset;
        # anything else is dropped rather than left to fail at compile time.
        keep = set("ăâîșțĂÂÎȘȚ")
        drop = [c for c in leftover if c not in keep]
        if drop:
            print("    dropped unrenderable: " +
                  " ".join(f"U+{ord(c):04X}" for c in drop))
            for c in drop:
                text = text.replace(c, "")
    return text


def read_python(path: Path) -> str:
    return sanitise(path.read_text(encoding="utf-8").rstrip()) + "\n"


def read_notebook(path: Path) -> str:
    """Flatten a notebook into readable Python: markdown headings become
    banner comments, code cells follow in order. Outputs are dropped."""
    nb = json.loads(path.read_text(encoding="utf-8"))
    out: list[str] = []
    for cell in nb["cells"]:
        src = "".join(cell["source"]).rstrip()
        if not src:
            continue
        if cell["cell_type"] == "markdown":
            title = src.lstrip("# ").splitlines()[0].strip()
            out.append("")
            out.append("# " + "=" * 74)
            out.append("# " + title)
            out.append("# " + "=" * 74)
        else:
            out.append(src)
    return sanitise("\n".join(out).strip()) + "\n"


def main() -> None:
    parts = [PREAMBLE]
    total = 0
    print(f"{'anexa':>5s}  {'fisier':34s} {'linii':>6s}")
    print("-" * 50)
    for i, (rel, caption, blurb) in enumerate(ANNEXES, start=1):
        path = REPO / rel
        if not path.is_file():
            raise SystemExit(f"lipseste: {path}")
        code = read_notebook(path) if path.suffix == ".ipynb" else read_python(path)
        n = len(code.splitlines())
        total += n
        print(f"{i:5d}  {rel.split('/')[-1]:34s} {n:6d}")

        parts.append(f"\\section*{{Annex {i} --- {caption}}}\n")
        parts.append(f"\\addcontentsline{{toc}}{{section}}"
                     f"{{Annex {i} --- {caption}}}\n")
        parts.append(f"\\label{{annex:{path.stem.replace('_','')}}}\n\n")
        parts.append(blurb + "\n\n")
        parts.append("\\begin{lstlisting}\n")
        parts.append(code)
        parts.append("\\end{lstlisting}\n\n\\newpage\n\n")

    parts.append("\\end{document}\n")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(parts), encoding="utf-8")
    print("-" * 50)
    print(f"{'TOTAL':41s} {total:6d} linii")
    print(f"\nscris: {OUT}")
    print(f"marime: {OUT.stat().st_size/1024:.0f} KB")


if __name__ == "__main__":
    main()
