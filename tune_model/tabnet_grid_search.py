import numpy as np
import pandas as pd
import time
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score

start_time = time.time()

# load training and validation data
X_train = pd.read_csv("/global/cfs/projectdirs/m1532/Projects_MVP/_members/Daniel/2025/test env/tabnet/data/X_train.csv").values
y_train = pd.read_csv("/global/cfs/projectdirs/m1532/Projects_MVP/_members/Daniel/2025/test env/tabnet/data/y_train.csv").values.ravel()
X_valid = pd.read_csv("/global/cfs/projectdirs/m1532/Projects_MVP/_members/Daniel/2025/test env/tabnet/data/X_valid.csv").values
y_valid = pd.read_csv("global/cfs/projectdirs/m1532/Projects_MVP/_members/Daniel/2025/test env/tabnet/data/y_valid.csv").values.ravel()

# grid of hyperparameters to check in the search
param_grid = {
    'n_d': [64, 128],
    'n_a': [64, 128],
    'n_steps': [3, 5],
    'gamma': [1.2, 1.5],
    'lr': [0.01, 0.02],
    'batch_size': [256, 512]
}
grid = list(ParameterGrid(param_grid))

results = []
best_acc = 0
best_params = None

# run grid search
for i, params in enumerate(grid):
    print(f"\nRunning configuration {i+1}/{len(grid)}")
    print(params)

    model = TabNetClassifier(
        n_d=params['n_d'],
        n_a=params['n_a'],
        n_steps=params['n_steps'],
        gamma=params['gamma'],
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=params['lr']),
        seed=42,
        verbose=0
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        max_epochs=100,
        patience=10,
        batch_size=params['batch_size'],
        virtual_batch_size=min(128, params['batch_size'] // 2)
    )

    y_pred = model.predict(X_valid)
    acc = accuracy_score(y_valid, y_pred)
    results.append((params, acc))

    if acc > best_acc:
        best_acc = acc
        best_params = params

# Save results
df = pd.DataFrame(results, columns=["params", "accuracy"])
df.to_csv("tabnet_grid_results.csv", index=False)

# Print best hyperparameters
print("\n optimal hyperparameters:")
print(best_params)
print(f"Best Accuracy: {best_acc:.6f}")


end_time = time.time()
runtime = end_time - start_time
print(f"\nTotal Runtime: {runtime:.6f} seconds")
