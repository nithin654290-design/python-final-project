"""
Genetic Algorithm search over CNN architecture and hyperparameters (PyGAD).

Chromosome (6 genes, all integer-indexed for stable crossover/mutation):

    [0] backbone index        -> custom | resnet18 | efficientnet_b0
    [1] filters/dense index   -> base conv filters + dense head width
    [2] dropout index         -> 0.2 ... 0.6
    [3] learning rate index   -> 1e-4 ... 3e-3
    [4] optimizer index       -> adam | adamw | sgd | rmsprop
    [5] batch size index      -> 16 | 32 | 64

Fitness = validation macro F1 after a short warm-up training run
(default 3 epochs) on a subset of the data, which keeps the search affordable.

Run:
    python ga_optimizer.py --data-dir data --generations 5 --population 8
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, List

import numpy as np
import pygad
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from model import build_model, build_optimizer
from preprocess import build_dataloaders, compute_class_weights

# ---------------------------------------------------------------------------
# Gene space
# ---------------------------------------------------------------------------

BACKBONES: List[str] = ["custom", "resnet18", "efficientnet_b0"]
WIDTHS: List[Dict[str, int]] = [
    {"base_filters": 16, "dense_units": 128},
    {"base_filters": 32, "dense_units": 256},
    {"base_filters": 48, "dense_units": 384},
    {"base_filters": 64, "dense_units": 512},
]
DROPOUTS: List[float] = [0.2, 0.3, 0.4, 0.5, 0.6]
LEARNING_RATES: List[float] = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
OPTIMIZERS: List[str] = ["adam", "adamw", "sgd", "rmsprop"]
BATCH_SIZES: List[int] = [16, 32, 64]

GENE_SPACE = [
    list(range(len(BACKBONES))),
    list(range(len(WIDTHS))),
    list(range(len(DROPOUTS))),
    list(range(len(LEARNING_RATES))),
    list(range(len(OPTIMIZERS))),
    list(range(len(BATCH_SIZES))),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode(chromosome) -> Dict:
    """Turn a chromosome into a model/training configuration dictionary."""
    genes = [int(round(float(g))) for g in chromosome]
    genes = [min(max(g, 0), len(space) - 1) for g, space in zip(genes, GENE_SPACE)]
    width = WIDTHS[genes[1]]
    return {
        "backbone": BACKBONES[genes[0]],
        "base_filters": width["base_filters"],
        "dense_units": width["dense_units"],
        "num_blocks": 4,
        "dropout": DROPOUTS[genes[2]],
        "learning_rate": LEARNING_RATES[genes[3]],
        "optimizer": OPTIMIZERS[genes[4]],
        "batch_size": BATCH_SIZES[genes[5]],
        "weight_decay": 1e-4,
        "freeze_backbone": False,
    }


# ---------------------------------------------------------------------------
# Warm-up training used as the fitness signal
# ---------------------------------------------------------------------------


def evaluate_macro_f1(model: nn.Module, loader) -> float:
    model.eval()
    predictions: List[int] = []
    targets: List[int] = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            logits = model(images)
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    if not targets:
        return 0.0
    return float(f1_score(targets, predictions, average="macro", zero_division=0))


def warmup_fitness(config: Dict, args, class_weights: torch.Tensor) -> float:
    """Train a candidate briefly and return its validation macro F1."""
    train_loader, val_loader, _ = build_dataloaders(
        args.data_dir,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        subset_fraction=args.subset_fraction,
        clean=not args.no_clean,
    )

    model = build_model(config).to(DEVICE)
    optimizer = build_optimizer(model, config)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    for _ in range(args.warmup_epochs):
        model.train()
        for images, labels in train_loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    score = evaluate_macro_f1(model, val_loader)
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return score


def run_ga(args) -> Dict:
    class_weights = compute_class_weights(args.data_dir)
    cache: Dict[str, float] = {}
    history: List[Dict] = []

    def fitness_func(ga_instance, solution, solution_idx):  # noqa: ARG001
        config = decode(solution)
        key = json.dumps(config, sort_keys=True)
        if key in cache:
            return cache[key]
        started = time.time()
        try:
            score = warmup_fitness(config, args, class_weights)
        except (RuntimeError, ValueError) as error:  # OOM or invalid combo
            print(f"  candidate failed ({error}); fitness 0")
            score = 0.0
        cache[key] = score
        history.append({"config": config, "macro_f1": score, "seconds": round(time.time() - started, 1)})
        print(f"  {config['backbone']:16s} lr={config['learning_rate']:<7} "
              f"opt={config['optimizer']:<8} drop={config['dropout']} "
              f"bs={config['batch_size']} -> macro F1 {score:.4f}")
        return score

    def on_generation(ga_instance):
        best_solution, best_fitness, _ = ga_instance.best_solution()
        print(f"Generation {ga_instance.generations_completed}: best macro F1 {best_fitness:.4f} "
              f"-> {decode(best_solution)['backbone']}")

    ga_instance = pygad.GA(
        num_generations=args.generations,
        num_parents_mating=max(2, args.population // 2),
        fitness_func=fitness_func,
        sol_per_pop=args.population,
        num_genes=len(GENE_SPACE),
        gene_space=GENE_SPACE,
        gene_type=int,
        parent_selection_type="tournament",
        K_tournament=3,
        keep_elitism=1,
        crossover_type="two_points",
        crossover_probability=0.8,
        mutation_type="random",
        mutation_probability=0.25,
        on_generation=on_generation,
        random_seed=args.seed,
        save_best_solutions=False,
        suppress_warnings=True,
    )

    print(f"Starting GA search on {DEVICE} "
          f"({args.population} individuals x {args.generations} generations, "
          f"{args.warmup_epochs} warm-up epochs each)")
    ga_instance.run()

    best_solution, best_fitness, _ = ga_instance.best_solution()
    best_config = decode(best_solution)
    best_config["ga_val_macro_f1"] = round(float(best_fitness), 4)
    best_config["ga_settings"] = {
        "generations": args.generations,
        "population": args.population,
        "warmup_epochs": args.warmup_epochs,
        "subset_fraction": args.subset_fraction,
    }

    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(best_config, handle, indent=2)
    with open("ga_search_history.json", "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)

    print("\nBest configuration:")
    print(json.dumps(best_config, indent=2))
    print(f"Saved to {args.output} (search log: ga_search_history.json)")
    return best_config


def parse_args():
    parser = argparse.ArgumentParser(description="GA search for the best CNN configuration")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--generations", type=int, default=5)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--subset-fraction", type=float, default=0.25,
                        help="Fraction of the dataset used during the search (keeps it fast)")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-clean", action="store_true",
                        help="Skip DullRazor/CLAHE during the search for extra speed")
    parser.add_argument("--output", default="best_model_config.json")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    run_ga(arguments)
