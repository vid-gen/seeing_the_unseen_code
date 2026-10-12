#!/usr/bin/env python3
"""Generate metadata_dual_control.json for the miradata2 dataset.

Scans <root>/tracking/ for subdirectories containing exactly two *.mp4
files, matches each folder name (e.g. "00296") against the basename of
entries in <root>/videos.txt to look up the corresponding prompt in
<root>/prompt.txt (strategy A: name-based map), and emits JSON entries
consumable by videox_fun.data.dataset_image_video.ImageVideoControlDataset.

Canonical tracking filenames (miradata2):
    first_frame_tracking_video.mp4      -> control_file_path   (tracking 1)
    future_visible_tracking_video.mp4   -> control_file_path_2 (tracking 2)

Legacy pair (off by default; enable with --include-legacy):
    visible_tracking_das.mp4            -> control_file_path
    visible_tracking_pred_track.mp4     -> control_file_path_2
"""
import argparse
import json
import os
import sys
from pathlib import Path


CANONICAL_PAIR = ("first_frame_tracking_video.mp4", "future_visible_tracking_video.mp4")
LEGACY_PAIR = ("visible_tracking_das.mp4", "visible_tracking_pred_track.mp4")


def build_prompt_map(root: Path):
    videos_txt = root / "videos.txt"
    prompts_txt = root / "prompt.txt"
    v_lines = videos_txt.read_text().splitlines()
    p_lines = prompts_txt.read_text().splitlines()
    if len(v_lines) != len(p_lines):
        raise ValueError(
            f"videos.txt ({len(v_lines)}) and prompt.txt ({len(p_lines)}) "
            "line counts differ"
        )
    prompt_map = {}
    for vline, pline in zip(v_lines, p_lines):
        vline = vline.strip()
        if not vline:
            continue
        base = os.path.basename(vline)
        if base.lower().endswith(".mp4"):
            base = base[:-4]
        if base in prompt_map and prompt_map[base] != pline.strip():
            print(f"[warn] duplicate basename {base!r} in videos.txt with "
                  "different prompts; keeping the first")
            continue
        prompt_map[base] = pline.strip()
    return prompt_map, len(v_lines)


def verify_video(path: Path, min_frames: int):
    import decord
    vr = decord.VideoReader(str(path))
    n = len(vr)
    del vr
    return n, n >= min_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", required=True,
                        help="dataset root containing tracking/, videos/, images/, videos.txt, prompt.txt")
    parser.add_argument("--tracking-subdir", default="tracking",
                        help="tracking subdirectory name under <root> (e.g. 'tracking', 'tracking_1000')")
    parser.add_argument("--out", default=None,
                        help="output JSON path (default: <root>/metadata_dual_control.json)")
    parser.add_argument("--report-json", default=None,
                        help="audit report path (default: <root>/metadata_dual_control.report.json)")
    parser.add_argument("--strict", action="store_true",
                        help="raise on any missing file / prompt instead of skipping")
    parser.add_argument("--verify-videos", action="store_true",
                        help="open each mp4 with decord and drop samples with <min-frames")
    parser.add_argument("--min-frames", type=int, default=81,
                        help="minimum frame count when --verify-videos is set (default 81)")
    parser.add_argument("--include-legacy", action="store_true",
                        help="also include folders using the legacy visible_tracking_das pair")
    parser.add_argument("--exclude-folders", default="",
                        help="comma-separated list of tracking folder names to skip "
                             "(e.g. validation samples that should not appear in the training set)")
    parser.add_argument("--exclude-file", default=None,
                        help="path to a file with one folder name per line to exclude "
                             "(merged with --exclude-folders)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print stats only, do not write any file")
    args = parser.parse_args()

    exclude_set = {s.strip() for s in args.exclude_folders.split(",") if s.strip()}
    if args.exclude_file:
        with open(args.exclude_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    exclude_set.add(line)
    if exclude_set:
        print(f"[exclude] {len(exclude_set)} folder(s): {sorted(exclude_set)}")

    root = Path(args.root).resolve()
    if not root.exists():
        print(f"[error] root not found: {root}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out) if args.out else root / "metadata_dual_control.json"
    report_path = Path(args.report_json) if args.report_json else root / "metadata_dual_control.report.json"

    print(f"[root] {root}")
    print(f"[out]  {out_path}")

    prompt_map, total_lines = build_prompt_map(root)
    print(f"[scan] videos.txt / prompt.txt lines : {total_lines}")
    print(f"[scan] unique prompt-map entries     : {len(prompt_map)}")

    tracking_dir = root / args.tracking_subdir
    if not tracking_dir.exists():
        print(f"[error] tracking dir not found: {tracking_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[scan] tracking dir                  : {tracking_dir}")
    subdirs = sorted(p.name for p in tracking_dir.iterdir() if p.is_dir())
    print(f"[scan] tracking subdirs              : {len(subdirs)}")

    entries = []
    skipped = []

    stats = {
        "mp4_eq2": 0,
        "canonical": 0,
        "legacy_included": 0,
        "excluded": 0,
    }

    for sub in subdirs:
        if sub in exclude_set:
            skipped.append({"folder": sub, "reason": "excluded_by_user"})
            stats["excluded"] += 1
            continue
        sub_dir = tracking_dir / sub
        mp4s = sorted(p.name for p in sub_dir.glob("*.mp4"))

        if len(mp4s) != 2:
            skipped.append({"folder": sub, "reason": f"mp4_count={len(mp4s)}"})
            continue
        stats["mp4_eq2"] += 1

        name_set = set(mp4s)
        if name_set == set(CANONICAL_PAIR):
            ctrl1, ctrl2 = CANONICAL_PAIR
            stats["canonical"] += 1
        elif name_set == set(LEGACY_PAIR) and args.include_legacy:
            ctrl1, ctrl2 = LEGACY_PAIR
            stats["legacy_included"] += 1
        else:
            skipped.append({"folder": sub, "reason": f"unexpected_names={mp4s}"})
            continue

        main_video = root / "videos" / f"{sub}.mp4"
        if not main_video.exists():
            msg = f"main_video_missing: videos/{sub}.mp4"
            if args.strict:
                raise FileNotFoundError(msg)
            skipped.append({"folder": sub, "reason": msg})
            continue

        if sub not in prompt_map:
            msg = f"prompt_not_found_for_basename={sub}"
            if args.strict:
                raise KeyError(msg)
            skipped.append({"folder": sub, "reason": msg})
            continue

        first_img = root / "images" / f"{sub}.png"
        first_img_missing = not first_img.exists()

        if args.verify_videos:
            bad = None
            for name, p in [("main", main_video),
                            ("ctrl1", sub_dir / ctrl1),
                            ("ctrl2", sub_dir / ctrl2)]:
                try:
                    n, ok = verify_video(p, args.min_frames)
                except Exception as e:
                    bad = f"decord_error[{name}]={e}"
                    break
                if not ok:
                    bad = f"{name}_frames<{args.min_frames}(got={n})"
                    break
            if bad is not None:
                if args.strict:
                    raise ValueError(f"{sub}: {bad}")
                skipped.append({"folder": sub, "reason": bad})
                continue

        entry = {
            "file_path":           f"videos/{sub}.mp4",
            "text":                prompt_map[sub],
            "type":                "video",
            "control_file_path":   f"{args.tracking_subdir}/{sub}/{ctrl1}",
            "control_file_path_2": f"{args.tracking_subdir}/{sub}/{ctrl2}",
        }
        entries.append(entry)

        if first_img_missing:
            skipped.append({
                "folder": sub,
                "reason": "first_frame_png_missing (entry still INCLUDED; only validation CLI uses the PNG)",
                "included": True,
            })

    reason_counts = {}
    for s in skipped:
        key = s["reason"].split(":")[0].split("=")[0].split("(")[0].strip()
        reason_counts[key] = reason_counts.get(key, 0) + 1

    print()
    print(f"[stats] mp4==2 subdirs               : {stats['mp4_eq2']}")
    print(f"[stats] canonical naming             : {stats['canonical']}")
    print(f"[stats] legacy naming included       : {stats['legacy_included']}")
    print(f"[stats] excluded by --exclude-*      : {stats['excluded']}")
    print(f"[final] entries                      : {len(entries)}")
    print(f"[skip] total skip records            : {len(skipped)}")
    for k, v in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f"  - {k:45s}: {v}")

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"\n[write] {len(entries)} entries -> {out_path}")

    report = {
        "root": str(root),
        "total_tracking_subdirs": len(subdirs),
        "mp4_eq2_subdirs": stats["mp4_eq2"],
        "canonical_subdirs": stats["canonical"],
        "legacy_included": stats["legacy_included"],
        "entries_count": len(entries),
        "skipped_count": len(skipped),
        "excluded": sorted(exclude_set),
        "skipped": skipped,
        "args": vars(args),
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[write] report  -> {report_path}")


if __name__ == "__main__":
    main()
