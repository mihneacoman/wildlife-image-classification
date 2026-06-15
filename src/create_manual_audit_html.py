from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from html import escape
from pathlib import Path


REQUIRED_COLUMNS = {
    "audit_id",
    "audit_source",
    "image_path",
    "true_label",
    "pred_label",
    "confidence",
    "top2_label",
    "top2_confidence",
    "prediction_margin",
}

AUDIT_TAGS = [
    "clear_model_error",
    "ambiguous",
    "possible_label_noise",
    "animal_too_small",
    "animal_hidden_or_occluded",
    "non_target_animal",
    "visually_similar_species",
    "partial_body",
    "bad_lighting_or_night_image",
    "background_confusion",
]

VISIBILITY_VALUES = [
    "clear",
    "barely_visible",
    "not_visible",
    "uncertain",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export manual audit candidates to HTML.")
    parser.add_argument(
        "--input",
        default="reports/manual_audit_candidates.csv",
        help="CSV file containing audit candidates.",
    )
    parser.add_argument(
        "--output",
        default="reports/manual_audit_candidates.html",
        help="HTML file to write.",
    )
    return parser.parse_args()


def display_value(value: str | None) -> str:
    return escape(value) if value not in (None, "") else "&mdash;"


def format_number(value: str | None) -> str:
    if value in (None, ""):
        return "&mdash;"
    try:
        return f"{float(value):.4f}"
    except ValueError:
        return escape(value)


def image_src(image_path: str, project_root: Path, output_file: Path) -> str:
    path = Path(image_path)
    if not path.is_absolute():
        path = project_root / path
    try:
        rel_path = path.resolve().relative_to(project_root.resolve())
    except ValueError:
        rel_path = path
    src = Path("..") / rel_path if not rel_path.is_absolute() else rel_path
    try:
        src = path.resolve().relative_to(output_file.parent.resolve())
    except ValueError:
        pass
    return escape(src.as_posix())


def read_rows(csv_file: Path) -> list[dict[str, str]]:
    with csv_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing == {"audit_id"} and "image_id" in (reader.fieldnames or []):
            missing = set()
        if missing:
            raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
        rows = list(reader)

    for index, row in enumerate(rows, start=1):
        row.setdefault("audit_id", row.get("image_id") or str(index))
        if not row.get("audit_id"):
            row["audit_id"] = str(index)
    return rows


def render_chips(values: list[str]) -> str:
    return "\n".join(f'<span class="chip">{escape(value)}</span>' for value in values)


def render_card(row: dict[str, str], project_root: Path, output_file: Path) -> str:
    src = image_src(row["image_path"], project_root, output_file)
    return f"""
        <article class="card">
          <img src="{src}" alt="Audit image {display_value(row.get("audit_id"))}" loading="lazy">
          <div class="content">
            <div class="audit-id">#{display_value(row.get("audit_id"))}</div>
            <dl class="meta">
              <div><dt>True</dt><dd>{display_value(row.get("true_label"))}</dd></div>
              <div><dt>Pred</dt><dd>{display_value(row.get("pred_label"))}</dd></div>
              <div><dt>Conf</dt><dd>{format_number(row.get("confidence"))}</dd></div>
              <div><dt>Top 2</dt><dd>{display_value(row.get("top2_label"))} ({format_number(row.get("top2_confidence"))})</dd></div>
              <div><dt>Margin</dt><dd>{format_number(row.get("prediction_margin"))}</dd></div>
            </dl>
            <div class="annotation">
              <div><span>audit_tag</span><strong>{display_value(row.get("audit_tag"))}</strong></div>
              <div><span>animal_visible</span><strong>{display_value(row.get("animal_visible"))}</strong></div>
              <div><span>notes</span><strong>{display_value(row.get("notes"))}</strong></div>
            </div>
          </div>
        </article>
    """


def render_html(rows: list[dict[str, str]], project_root: Path, output_file: Path) -> str:
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("audit_source") or "unknown"].append(row)

    sections = []
    for source in sorted(groups):
        cards = "\n".join(render_card(row, project_root, output_file) for row in groups[source])
        sections.append(
            f"""
            <section class="source-group">
              <h2>{escape(source)} <span>{len(groups[source])} examples</span></h2>
              <div class="grid">{cards}</div>
            </section>
            """
        )

    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Manual Audit Candidates</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --text: #18202a;
      --muted: #657181;
      --line: #d9dee6;
      --accent: #1d6f5f;
      --soft: #edf5f3;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 24px 28px 18px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    h1, h2, p {{ margin: 0; }}
    h1 {{ font-size: 24px; }}
    .summary {{
      margin-top: 6px;
      color: var(--muted);
    }}
    .reference {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      padding: 16px 28px;
      border-bottom: 1px solid var(--line);
      background: #fbfcfd;
    }}
    .reference h3 {{
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: .04em;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
    }}
    .chip {{
      padding: 3px 8px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: var(--panel);
      white-space: nowrap;
      font-size: 12px;
    }}
    main {{ padding: 22px 28px 34px; }}
    .source-group + .source-group {{ margin-top: 30px; }}
    h2 {{
      display: flex;
      align-items: baseline;
      gap: 10px;
      margin-bottom: 12px;
      font-size: 18px;
    }}
    h2 span {{
      color: var(--muted);
      font-size: 13px;
      font-weight: 500;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
      gap: 14px;
    }}
    .card {{
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    img {{
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      object-fit: contain;
      background: #111820;
    }}
    .content {{ padding: 10px 12px 12px; }}
    .audit-id {{
      margin-bottom: 8px;
      color: var(--accent);
      font-weight: 700;
      overflow-wrap: anywhere;
    }}
    .meta {{
      display: grid;
      gap: 5px;
      margin: 0;
    }}
    .meta div {{
      display: grid;
      grid-template-columns: 58px 1fr;
      gap: 8px;
    }}
    dt {{
      color: var(--muted);
      font-size: 12px;
    }}
    dd {{
      margin: 0;
      overflow-wrap: anywhere;
    }}
    .annotation {{
      display: grid;
      gap: 6px;
      margin-top: 10px;
      padding-top: 10px;
      border-top: 1px solid var(--line);
    }}
    .annotation div {{
      display: grid;
      grid-template-columns: 92px 1fr;
      gap: 8px;
      min-height: 24px;
      align-items: center;
    }}
    .annotation span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .annotation strong {{
      min-height: 24px;
      padding: 3px 7px;
      border: 1px dashed #adb6c2;
      border-radius: 6px;
      background: var(--soft);
      font-weight: 500;
    }}
    @media (max-width: 680px) {{
      header, .reference, main {{ padding-left: 14px; padding-right: 14px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>Manual Audit Candidates</h1>
    <p class="summary">{len(rows)} examples generated from {escape(str(output_file.parent / "manual_audit_candidates.csv"))} on {generated_at}</p>
  </header>
  <aside class="reference">
    <div>
      <h3>Suggested audit_tag values</h3>
      <div class="chips">{render_chips(AUDIT_TAGS)}</div>
    </div>
    <div>
      <h3>Suggested animal_visible values</h3>
      <div class="chips">{render_chips(VISIBILITY_VALUES)}</div>
    </div>
  </aside>
  <main>
    {"".join(sections)}
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    input_file = (project_root / args.input).resolve()
    output_file = (project_root / args.output).resolve()

    rows = read_rows(input_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(render_html(rows, project_root, output_file), encoding="utf-8")
    print(f"Wrote {output_file} ({len(rows)} examples)")


if __name__ == "__main__":
    main()
