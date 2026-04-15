import argparse
import json
from pathlib import Path

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract captions from a JSON file and write them to per-image TXT files."
    )
    parser.add_argument(
        "--json-file",
        default="./dataset/EquiFashion_DB/test.json",
        help="Path to the JSON file that contains items with 'gt' and 'caption' fields.",
    )
    # Keep the original default (including the typo) to avoid breaking existing scripts.
    parser.add_argument(
        "--txt-folder",
        default="./text_foleder/lle_a5",
        help="Output folder where TXT files will be written.",
    )
    parser.add_argument(
        "--max-tags",
        type=int,
        default=10,
        help="Max number of comma-separated caption tags to keep (0 = keep all).",
    )
    parser.add_argument(
        "--suffix",
        default="_0.txt",
        help="Suffix appended to each base filename (default: '_0.txt').",
    )
    return parser.parse_args()


def normalize_caption(caption: str, max_tags: int) -> str:
    if max_tags == 0:
        return caption
    tags = (tag.strip() for tag in caption.split(","))
    trimmed = [tag for tag in tags if tag][:max_tags]
    return ",".join(trimmed)


def main() -> None:
    args = parse_args()

    txt_folder = Path(args.txt_folder)
    txt_folder.mkdir(parents=True, exist_ok=True)

    json_file = Path(args.json_file)
    with json_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    for entry in tqdm(data):
        gt_filename = Path(entry["gt"]).name
        base = Path(gt_filename).stem
        txt_file_path = txt_folder / f"{base}{args.suffix}"

        caption = normalize_caption(entry.get("caption", ""), args.max_tags)
        txt_file_path.write_text(caption, encoding="utf-8")


if __name__ == "__main__":
    main()
