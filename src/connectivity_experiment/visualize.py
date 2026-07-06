from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from connectivity_experiment.difficulty import sort_by_numeric


def make_contact_sheet(
    records: list[dict],
    repo_root: str | Path,
    out_path: str | Path,
    title: str,
    count: int = 16,
    columns: int = 4,
) -> None:
    if not records:
        return
    repo_root = Path(repo_root)
    chosen = records[:count]
    images = []
    font = ImageFont.load_default()

    for record in chosen:
        image = Image.open(repo_root / record["image_path"]).convert("RGB")
        caption = _build_caption(record)
        tile = Image.new("RGB", (image.width, image.height + 24), color=(12, 12, 12))
        tile.paste(image, (0, 0))
        draw = ImageDraw.Draw(tile)
        draw.text((4, image.height + 4), caption, fill=(240, 240, 240), font=font)
        images.append(tile)

    rows = (len(images) + columns - 1) // columns
    tile_width, tile_height = images[0].size
    canvas = Image.new("RGB", (columns * tile_width, rows * tile_height + 28), color=(20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 6), title, fill=(255, 255, 255), font=font)

    for index, tile in enumerate(images):
        row = index // columns
        col = index % columns
        canvas.paste(tile, (col * tile_width, row * tile_height + 28))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_visualization_sets(records: list[dict]) -> dict[str, list[dict]]:
    visuals = {}
    positives = [record for record in records if int(record["connected"]) == 1]
    negatives = [record for record in records if int(record["connected"]) == 0]
    visuals["sample_grid_by_label_positive"] = positives[:16]
    visuals["sample_grid_by_label_negative"] = negatives[:16]
    if any("same_source_pair" in record for record in records):
        visuals["sample_grid_same_source_positive"] = [record for record in positives if int(record.get("same_source_pair", 0)) == 1][:16]
        visuals["sample_grid_cross_source_positive"] = [record for record in positives if int(record.get("same_source_pair", 0)) == 0][:16]
    visuals["difficulty_low"] = sort_by_numeric(records, "difficulty_proxy", reverse=False)[:16]
    visuals["difficulty_high"] = sort_by_numeric(records, "difficulty_proxy", reverse=True)[:16]
    families = sorted({record["shape_family"] for record in records})
    for family in families[:8]:
        visuals[f"sample_grid_{family}"] = [record for record in records if record["shape_family"] == family][:16]
    if any(record.get("loss") not in ("", None) for record in records):
        visuals["loss_high"] = sort_by_numeric(records, "loss", reverse=True)[:16]
    return visuals


def _build_caption(record: dict) -> str:
    sample_id = record["sample_id"]
    difficulty = _fmt(record.get("difficulty_proxy"))
    loss = _fmt(record.get("loss"))
    family = record.get("shape_family", "-")
    return f"{sample_id} {family} d={difficulty} l={loss}"


def _fmt(value) -> str:
    if value in (None, "", "NA", "None"):
        return "-"
    return f"{float(value):.2f}"
