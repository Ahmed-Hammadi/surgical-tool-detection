# Project instructions

Start by reading `HANDOVER.md`: it has the current state, the owner's machine setup, the next
steps in order, and decisions already made. Then `README.md` and `docs/adding_instruments.md`.

- The owner works on Windows PowerShell with a GTX 1650 Ti (4 GB). Give PowerShell commands.
- Run `python -m pytest -q` after changing anything under `src/`.
- `data/`, `runs/`, `mobile/` and model weights are git-ignored; never commit them.
- Synthetic-data results say nothing about real accuracy. Report real-data numbers only from
  `src/evaluate.py` and `src/export.py`.
- Dataset facts in `HANDOVER.md` are unverified until someone opens the pages; check the
  license and class names before merging a dataset.
- Update `HANDOVER.md` when the state changes.
