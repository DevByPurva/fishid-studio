import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from rastrear_peixes_auto import (
    FISH_PROMPTS,
    PALETTE_BGR,
    _build_resumo,
    _class_name,
    _color_for_id,
    _write_yolo_dataset_meta,
    parse_args,
)


def test_yolo_dataset_meta_files(tmp_path):
    labels_dir = tmp_path / "labels_yolo"
    labels_dir.mkdir()
    _write_yolo_dataset_meta(tmp_path, labels_dir)

    assert (labels_dir / "classes.txt").read_text(encoding="utf-8").splitlines() == ["fish"]
    assert (tmp_path / "data.yaml").read_text(encoding="utf-8").splitlines() == [
        f"path: {json.dumps(tmp_path.resolve().as_posix())}",
        "train: frames_autolabel",
        "val: frames_autolabel",
        "nc: 1",
        "names:",
        "  0: fish",
    ]


def test_track_colors_use_bgr_palette_and_wrap():
    size = len(PALETTE_BGR)
    for track_id in (0, 1, size - 1, size, size + 1, -1):
        color = _color_for_id(track_id)
        assert isinstance(color, tuple)
        assert len(color) == 3
        assert all(isinstance(channel, int) and 0 <= channel <= 255 for channel in color)
        assert color == PALETTE_BGR[track_id % size]


def test_tracking_cli_defaults():
    args = parse_args([])
    assert args.tracker == "bytetrack.yaml"
    assert args.no_labels is False
    assert args.max_frames is None


def test_tracking_cli_flags_ignore_extra_notebook_arguments():
    args = parse_args(["--no-labels", "--max-frames", "12", "-f", "kernel.json"])
    assert args.no_labels is True
    assert args.max_frames == 12


def test_class_name_maps_prompt_index_and_falls_back_for_unknown():
    assert _class_name(0) == FISH_PROMPTS[0]
    assert _class_name(4) == "tuna"
    assert _class_name(len(FISH_PROMPTS)) == str(len(FISH_PROMPTS))


def test_resumo_uses_most_frequent_class_per_track():
    traj_df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 0, 1],
            "tempo_s": [0.0, 0.04, 0.08, 0.0, 0.04],
            "track_id": [1, 1, 1, 2, 2],
            "confianca": [0.9, 0.8, 0.7, 0.6, 0.5],
            "largura_px": [10, 12, 14, 20, 22],
            "classe_prompt": ["tuna", "fish", "tuna", "shark", "shark"],
        }
    )
    resumo = _build_resumo(traj_df).set_index("track_id")
    assert resumo.loc[1, "classe_prevista"] == "tuna"
    assert resumo.loc[2, "classe_prevista"] == "shark"
