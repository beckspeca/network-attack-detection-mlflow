#!/usr/bin/env python3
"""Build the self-contained competition code.ipynb from the reviewed script."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_DIR / "scripts" / "v15_full_refit_submission.py"
TARGET = PROJECT_DIR / "notebooks" / "code.ipynb"


def main() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    notebook = {
        "cells": [
            {
                "id": "v15-overview",
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 네트워크 침입 분류 — v15 대회 최종 제출\n",
                    "\n",
                    "전체 학습 풀에서 v13 RandomForest를 재학습하고 Infilteration 확률을 "
                    "1.225배 보정해 `submission.csv`를 생성한다. 테스트 Label은 검증용으로도 "
                    "사용하지 않으며, 테스트 rolling 피처는 과거 학습 이벤트만 참조한다.\n",
                ],
            },
            {
                "id": "v15-pipeline",
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": source.splitlines(keepends=True),
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(TARGET)


if __name__ == "__main__":
    main()
