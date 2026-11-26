"""Rastreamento automático (ByteTrack) e Auto-Labeling (formato YOLO) de peixes em vídeo.

Funciona tanto via terminal quanto colando direto em uma célula do Google Colab (com GPU)!
Instalação no Colab:
    !pip install -q ultralytics opencv-python-headless pandas tqdm

O que este script gera automaticamente em `--out-dir`:
  1. video_rastreado.mp4          -> Vídeo anotado com caixas, ID único por peixe e trilha de movimento
  2. trajetorias.csv              -> Posição (x, y), dimensões em pixels e ID de cada peixe frame a frame
  3. resumo_individuos.csv        -> Contagem única de peixes detectados, tempo em cena e espécie prevista
  4. labels_yolo/frame_XXXXXX.txt -> Anotações automáticas no formato YOLO (prontas para Roboflow/CVAT)
  5. labels_yolo/classes.txt e data.yaml -> Nomes de classe e config do dataset YOLO
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# =============================================================================
# Configurações padrão (edite aqui se for colar direto numa célula do Colab)
# =============================================================================
DEFAULT_VIDEO_PATH: str = os.getenv("FISHID_VIDEO_PATH", "./data/video_teste.mp4")
DEFAULT_OUT_DIR: str = os.getenv("FISHID_TRACK_OUT_DIR", "./data/rastreamento_saida")
DEFAULT_KERAS_MODEL: str = ""  # Opcional: caminho para o modelo.keras (classifica cada peixe rastreado)

FISH_PROMPTS: list[str] = [
    "fish",
    "marlin",
    "billfish",
    "swordfish",
    "tuna",
    "shark",
]

PALETTE_BGR: list[tuple[int, int, int]] = [
    (110, 118, 15),   # Verde-petróleo (#0F766E)
    (217, 151, 10),   # Azul oceano (#0A97D9)
    (11, 158, 245),   # Âmbar dourado
    (166, 184, 20),   # Turquesa
    (129, 185, 16),   # Esmeralda
    (246, 130, 59),   # Azul royal
    (153, 72, 236),   # Coral/Magenta
]


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    return PALETTE_BGR[int(track_id) % len(PALETTE_BGR)]


def _write_yolo_dataset_meta(out_dir: Path, labels_dir: Path) -> None:
    """Gera labels_yolo/classes.txt e <out_dir>/data.yaml (todas as anotações usam a classe 0 = peixe)."""
    (labels_dir / "classes.txt").write_text("fish\n", encoding="utf-8")
    (out_dir / "data.yaml").write_text(
        f"path: {json.dumps(out_dir.resolve().as_posix())}\n"  # string JSON = YAML válido mesmo com ':' ou '#'
        "train: frames_autolabel\n"
        "val: frames_autolabel\n"
        "nc: 1\n"
        "names:\n"
        "  0: fish\n",
        encoding="utf-8",
    )


def format_yolo_bbox_line(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    class_id: int = 0,
) -> str:
    """Formata coordenadas de bounding box (x1, y1, x2, y2) para linha de anotação YOLO.

    Calcula coordenadas normalizadas do centro (nx, ny) e dimensões (nw, nh),
    aplicando clamping independente para o intervalo [0.0, 1.0] com exatamente 6 casas decimais.
    """
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    img_w = max(width, 1)
    img_h = max(height, 1)

    raw_nx = (x1 + bw / 2.0) / img_w
    raw_ny = (y1 + bh / 2.0) / img_h
    raw_nw = bw / img_w
    raw_nh = bh / img_h

    nx = 0.0 if raw_nx <= 0.0 else (1.0 if raw_nx >= 1.0 else raw_nx)
    ny = 0.0 if raw_ny <= 0.0 else (1.0 if raw_ny >= 1.0 else raw_ny)
    nw = 0.0 if raw_nw <= 0.0 else (1.0 if raw_nw >= 1.0 else raw_nw)
    nh = 0.0 if raw_nh <= 0.0 else (1.0 if raw_nh >= 1.0 else raw_nh)

    return f"{class_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}"


def _class_name(cls_idx: int) -> str:
    idx = int(cls_idx)
    return FISH_PROMPTS[idx] if 0 <= idx < len(FISH_PROMPTS) else str(idx)


def _build_resumo(traj_df: pd.DataFrame) -> pd.DataFrame:
    resumo = (
        traj_df.groupby("track_id", as_index=False)
        .agg(
            frames_visiveis=("frame", "count"),
            primeiro_segundo=("tempo_s", "min"),
            ultimo_segundo=("tempo_s", "max"),
            confianca_media=("confianca", "mean"),
            largura_media_px=("largura_px", "mean"),
            classe_prevista=("classe_prompt", lambda s: s.value_counts().idxmax()),
        )
        .sort_values("frames_visiveis", ascending=False)
    )
    resumo["duracao_em_cena_s"] = (resumo["ultimo_segundo"] - resumo["primeiro_segundo"]).round(2)
    return resumo


def run_tracking(
    video_path: Path,
    out_dir: Path,
    yolo_weights: str,
    tracker_cfg: str,
    conf_thresh: float,
    save_yolo_labels: bool,
    max_frames: int | None,
) -> None:
    from ultralytics import YOLO

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Vídeo não encontrado em: {video_path}\n"
            "Ajuste DEFAULT_VIDEO_PATH no topo do script (ou passe --video /caminho/video.mp4)."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = out_dir / "labels_yolo"
    frames_dir = out_dir / "frames_autolabel"
    if save_yolo_labels:
        labels_dir.mkdir(parents=True, exist_ok=True)
        frames_dir.mkdir(parents=True, exist_ok=True)
        _write_yolo_dataset_meta(out_dir, labels_dir)

    print(f"Carregando detector open-vocabulary: {yolo_weights}")
    model = YOLO(yolo_weights)
    model.set_classes(FISH_PROMPTS)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Não foi possível abrir o vídeo: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0.0 or np.isnan(fps):
        fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    raw_video_out = out_dir / "video_rastreado_raw.mp4"
    final_video_out = out_dir / "video_rastreado.mp4"
    writer = cv2.VideoWriter(str(raw_video_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=32))
    traj_rows: list[dict] = []
    frame_idx = 0

    print(f"Iniciando rastreamento ({tracker_cfg}) em {video_path.name} ({width}x{height} @ {fps:.1f} fps)...")

    try:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            results = model.track(
                frame_bgr,
                persist=True,
                tracker=tracker_cfg,
                conf=conf_thresh,
                verbose=False,
            )[0]

            annotated = frame_bgr.copy()
            yolo_lines: list[str] = []

            if results.boxes is not None and len(results.boxes) > 0:
                boxes_xyxy = results.boxes.xyxy.cpu().numpy()
                confs = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.int().cpu().numpy()
                ids = (
                    results.boxes.id.int().cpu().numpy()
                    if results.boxes.id is not None
                    else np.arange(1, len(boxes_xyxy) + 1)
                )

                for box, conf, tid, cls_idx in zip(boxes_xyxy, confs, ids, classes):
                    x1, y1, x2, y2 = [int(round(v)) for v in box]
                    bw = max(1, x2 - x1)
                    bh = max(1, y2 - y1)
                    cx = x1 + bw // 2
                    cy = y1 + bh // 2
                    tid_int = int(tid)
                    color = _color_for_id(tid_int)

                    trails[tid_int].append((cx, cy))

                    # Desenha trilha de movimento do peixe
                    pts = list(trails[tid_int])
                    for k in range(1, len(pts)):
                        thickness = max(1, int(3 * (k / len(pts))))
                        cv2.line(annotated, pts[k - 1], pts[k], color, thickness, cv2.LINE_AA)

                    # Caixa e badge do ID
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
                    tag = f"Peixe #{tid_int} ({conf * 100:.0f}%)"
                    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    ty0 = max(0, y1 - th - 10)
                    cv2.rectangle(annotated, (x1, ty0), (x1 + tw + 12, ty0 + th + 10), color, -1)
                    cv2.putText(
                        annotated,
                        tag,
                        (x1 + 6, ty0 + th + 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    traj_rows.append(
                        {
                            "frame": frame_idx,
                            "tempo_s": round(frame_idx / fps, 3),
                            "track_id": tid_int,
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "cx": cx,
                            "cy": cy,
                            "largura_px": bw,
                            "altura_px": bh,
                            "confianca": round(float(conf), 4),
                            "classe_prompt": _class_name(cls_idx),
                        }
                    )

                    # Linha de anotação automática em formato YOLO (classe 0 = peixe)
                    if save_yolo_labels:
                        yolo_lines.append(
                            format_yolo_bbox_line(
                                x1=x1,
                                y1=y1,
                                x2=x2,
                                y2=y2,
                                width=width,
                                height=height,
                                class_id=0,
                            )
                        )

            # Salva rótulo YOLO automático a cada 5 frames com detecções (evita frames redundantes)
            if save_yolo_labels and yolo_lines and frame_idx % 5 == 0:
                stem = f"frame_{frame_idx:06d}"
                (labels_dir / f"{stem}.txt").write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")
                cv2.imwrite(str(frames_dir / f"{stem}.jpg"), frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            # Contador discreto no topo esquerdo
            unique_count = len(trails)
            hud = f"Peixes unicos rastreados: {unique_count} | Frame {frame_idx}"
            cv2.rectangle(annotated, (12, 12), (430, 48), (15, 23, 42), -1)
            cv2.putText(annotated, hud, (22, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(annotated)
            frame_idx += 1
            if total_frames > 0 and frame_idx % 50 == 0:
                print(f"Progresso: {frame_idx}/{total_frames} frames ({unique_count} peixes únicos)")
    finally:
        cap.release()
        writer.release()

    # Converte para H.264 compatível com navegador
    proc = subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(raw_video_out),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(final_video_out),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0 and final_video_out.exists():
        raw_video_out.unlink(missing_ok=True)
    else:
        final_video_out = raw_video_out

    traj_df = pd.DataFrame(traj_rows)
    traj_csv = out_dir / "trajetorias.csv"
    traj_df.to_csv(traj_csv, index=False)

    if not traj_df.empty:
        _build_resumo(traj_df).to_csv(out_dir / "resumo_individuos.csv", index=False)

    print(f"\nRastreamento concluído! Total de peixes únicos: {len(trails)}")
    print(f"Vídeo anotado: {final_video_out}")
    print(f"Trajetórias CSV: {traj_csv}")
    if save_yolo_labels:
        print(f"Auto-labels YOLO salvos em: {labels_dir}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rastreamento automático (ByteTrack) + Auto-Labeling YOLO para vídeos de peixes."
    )
    parser.add_argument("--video", type=Path, default=Path(DEFAULT_VIDEO_PATH), help="Caminho do vídeo MP4/MOV")
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR), help="Pasta de saída")
    parser.add_argument("--yolo-weights", type=str, default="yolov8l-worldv2.pt", help="Peso YOLO-World")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="bytetrack.yaml ou botsort.yaml")
    parser.add_argument("--conf", type=float, default=0.22, help="Confiança mínima de detecção")
    parser.add_argument("--no-labels", action="store_true", help="Não salvar auto-labels YOLO")
    parser.add_argument("--max-frames", type=int, default=None, help="Limita número de frames para teste rápido")
    args, _ = parser.parse_known_args(argv)
    return args


if __name__ == "__main__":
    try:
        from google.colab import drive
        if not Path("/content/drive/MyDrive").exists():
            drive.mount("/content/drive")
    except ImportError:
        pass

    args = parse_args()
    run_tracking(
        video_path=args.video,
        out_dir=args.out_dir,
        yolo_weights=args.yolo_weights,
        tracker_cfg=args.tracker,
        conf_thresh=args.conf,
        save_yolo_labels=not args.no_labels,
        max_frames=args.max_frames,
    )
