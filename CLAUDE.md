# Project instructions

Start by reading `HANDOVER.md`: it has the current state, the Kaggle training setup, the next
steps in order, and decisions already made. Then `README.md` and `docs/adding_instruments.md`.

- The owner trains on Kaggle notebooks, not a local PC. Give commands as Kaggle notebook cells
  (`!command`, Linux paths under `/kaggle/input` and `/kaggle/working`), and keep
  `notebooks/kaggle_train.ipynb` in step with any change to the scripts' options.
- Run `python -m pytest -q` after changing anything under `src/`.
- `data/`, `runs/`, `mobile/` and model weights are git-ignored; never commit them.
- Synthetic-data results say nothing about real accuracy. Report real-data numbers only from
  `src/evaluate.py` and `src/export.py`.
- Dataset facts in `HANDOVER.md` are unverified until someone opens the pages; check the
  license and class names before merging a dataset.
- Update `HANDOVER.md` when the state changes.
