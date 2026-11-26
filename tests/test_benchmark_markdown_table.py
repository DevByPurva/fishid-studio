import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rodar_suite_benchmark as benchmark  # noqa: E402

HEADER = [
    "# Comparativo Consolidado — Suite de 10 Experimentos (`30_06_2023.mp4`)",
    "",
    "| Experimento | Objetivo | Frames | Contados | Tam. Mediano (px) | Erro Médio (±%) | Tempo (s) | Pasta |",
    "| :--- | :--- | :-: | :-: | :-: | :-: | :-: | :--- |",
]


def test_format_benchmark_markdown_table_renders_completed_and_missing_runs():
    records = [
        {
            "tag": "A1",
            "description": "Cena clara",
            "status": "OK",
            "frames_processados": 120,
            "peixes_contados": 7,
            "comprimento_mediano_px": 42.5,
            "erro_medio_pct": 3.25,
            "tempo_execucao_s": 61.4,
            "pasta": "runs/2026_06_30_A1",
        },
        {
            "tag": "A2",
            "description": "Sem dados",
            "status": "ERRO(1)",
            "frames_processados": None,
            "peixes_contados": None,
            "comprimento_mediano_px": None,
            "erro_medio_pct": None,
            "tempo_execucao_s": None,
            "pasta": None,
        },
    ]

    lines = benchmark.format_benchmark_markdown_table(records).splitlines()

    assert lines[:4] == HEADER
    assert lines[4] == (
        "| `A1` | Cena clara | 120 | **7** | 42.5 | ±3.2% | 61.4s | `2026_06_30_A1` |"
    )
    assert lines[5] == "| `A2` | Sem dados | - | ERRO(1) | - | - | - | - |"
    assert len(lines) == 6


def test_format_benchmark_markdown_table_returns_header_only_without_records():
    assert benchmark.format_benchmark_markdown_table([]).splitlines() == HEADER
