"""Visualize SGD trajectories for minimization and maximization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=0.08)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/sgd"))
    return parser.parse_args()


def run_trajectory(function, start, momentum, weight_decay, maximize, args):
    point = torch.tensor(start, dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.SGD(
        [point], lr=args.learning_rate, momentum=momentum,
        weight_decay=weight_decay, maximize=maximize,
    )
    trajectory = [point.detach().tolist()]
    for _ in range(args.steps):
        optimizer.zero_grad()
        function(point).backward()
        optimizer.step()
        trajectory.append(point.detach().tolist())
    return trajectory


def plot_objective(name, maximize, trajectories, output):
    grid = np.linspace(-4, 4, 200)
    x_grid, y_grid = np.meshgrid(grid, grid)
    values = x_grid**2 + y_grid**2
    if maximize:
        values = -values
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.contour(x_grid, y_grid, values, levels=16, cmap="viridis")
    for label, trajectory in trajectories.items():
        points = np.asarray(trajectory)
        axis.plot(points[:, 0], points[:, 1], marker="o", markersize=2, label=label)
    axis.set_title(name)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.legend()
    axis.set_aspect("equal")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    minimum = lambda point: point.square().sum()
    maximum = lambda point: -point.square().sum()
    all_trajectories = {}
    for name, function, maximize in (("minimum", minimum, False), ("maximum", maximum, True)):
        trajectories = {}
        for momentum in (0.0, 0.9):
            label = f"momentum={momentum}"
            trajectories[label] = run_trajectory(
                function, (3.0, 2.0), momentum, 0.0, maximize, args
            )
        label = "momentum=0, weight_decay=0.1"
        trajectories[label] = run_trajectory(
            function, (3.0, 2.0), 0.0, 0.1, maximize, args
        )
        all_trajectories[name] = trajectories
        plot_objective(name, maximize, trajectories, args.output_dir / f"{name}.png")

    with (args.output_dir / "trajectories.json").open("w", encoding="utf-8") as file:
        json.dump(all_trajectories, file, indent=2)
    print(f"Wrote SGD artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()