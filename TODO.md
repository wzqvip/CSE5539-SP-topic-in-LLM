# TODO

- [ ] Install `requirements.txt` in a local virtual environment.
- [x] Run a small end-to-end smoke test and confirm that both modes complete (`runs/smoke/`).
- [ ] Run the full GPU experiment with a fixed seed.
- [ ] If only CPU is available, run a documented reduced-size experiment or use a GPU/Colab for final numbers.
- [ ] Check `parameter_counts.json`; adjust `--lora-r` if LoRA is not in the same parameter-count ballpark as head tuning.
- [ ] Use the best validation checkpoint and record the test result when SST-2 test labels are available.
- [ ] Copy the validation accuracies from `table1.md` into the homework Table 1.
- [ ] Add the GitHub or Colab link to the report.
- [ ] Briefly discuss freezing, LoRA parameter count, accuracy, and training cost.