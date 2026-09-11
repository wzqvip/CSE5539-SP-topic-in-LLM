# TODO

- [X] Install `requirements.txt` in a local virtual environment.
- [X] Run a small end-to-end smoke test and confirm that both modes complete (`runs/smoke/`).
- [X] Run `run_lm_experiments.py` and review `runs/language-model/samples.md`.
- [X] Run `visualize_sgd.py` and include `runs/sgd/minimum.png` and `maximum.png`.
- [X] Run the full GPU experiment with a fixed seed (`runs/modernbert-sst2/`, RTX 5060).
- [X] Complete the final experiment on the local GPU rather than using a reduced CPU run.
- [X] Check `parameter_counts.json`; LoRA is in the same order-of-magnitude parameter ballpark as head tuning.
- [X] Use the best validation checkpoint; SST-2 test labels are unavailable, so test accuracy is `null`.
- [X] Copy the validation accuracies from `runs/modernbert-sst2/table1.md` into the homework Table 1.
- [ ] Add the GitHub or Colab link to `HOMEWORK.md` and the submitted report.
