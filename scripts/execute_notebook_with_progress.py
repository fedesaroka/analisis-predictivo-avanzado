"""Execute the final notebook while reporting progress by cell."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


ROOT = Path(__file__).resolve().parents[1]

NOTEBOOK_PATH = (
    ROOT
    / "notebooks"
    / "TP2_sistema_recomendacion_anime.ipynb"
)


class ProgressExecutePreprocessor(
    ExecutePreprocessor
):
    """Execute cells while printing timing information."""

    def preprocess_cell(
        self,
        cell,
        resources,
        index,
    ):
        if cell.cell_type != "code":
            return super().preprocess_cell(
                cell,
                resources,
                index,
            )

        meaningful_lines = [
            line.strip()
            for line in cell.source.splitlines()
            if line.strip()
            and not line.strip().startswith("%")
        ]

        preview = (
            meaningful_lines[0]
            if meaningful_lines
            else "<empty code cell>"
        )

        print(
            f"[START] Cell {index}: "
            f"{preview[:100]}",
            flush=True,
        )

        started = perf_counter()

        try:
            result = super().preprocess_cell(
                cell,
                resources,
                index,
            )
        except Exception:
            elapsed = perf_counter() - started

            print(
                f"[FAILED] Cell {index} "
                f"after {elapsed:.1f} seconds.",
                flush=True,
            )

            print(
                "Cell source:",
                flush=True,
            )

            print(
                cell.source,
                flush=True,
            )

            raise

        elapsed = perf_counter() - started

        print(
            f"[DONE] Cell {index} "
            f"in {elapsed:.1f} seconds.",
            flush=True,
        )

        return result


def main() -> None:
    if not NOTEBOOK_PATH.exists():
        raise FileNotFoundError(
            f"Notebook not found: {NOTEBOOK_PATH}"
        )

    notebook = nbformat.read(
        NOTEBOOK_PATH,
        as_version=4,
    )

    executor = ProgressExecutePreprocessor(
        timeout=7200,
        kernel_name="apa-tp2-final",
        allow_errors=False,
        record_timing=True,
    )

    started = perf_counter()

    print(
        f"Executing: {NOTEBOOK_PATH}",
        flush=True,
    )

    print(
        f"Working directory: {ROOT}",
        flush=True,
    )

    executor.preprocess(
        notebook,
        resources={
            "metadata": {
                "path": str(ROOT),
            }
        },
    )

    nbformat.write(
        notebook,
        NOTEBOOK_PATH,
    )

    elapsed = perf_counter() - started

    print(
        f"Notebook completed in "
        f"{elapsed:.1f} seconds.",
        flush=True,
    )


if __name__ == "__main__":
    main()
