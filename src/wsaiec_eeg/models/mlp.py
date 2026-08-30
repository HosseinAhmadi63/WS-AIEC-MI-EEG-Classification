"""Scikit-compatible MLP for the frozen deterministic completion protocol."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from wsaiec_eeg.utils.randomness import set_global_seed


class TorchMLPClassifier(ClassifierMixin, BaseEstimator):
    """One or more hidden layers, Adam, L2, dropout, and early stopping."""

    def __init__(
        self,
        hidden_layer_sizes: tuple[int, ...] = (150,),
        activation: str = "relu",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0001,
        dropout: float = 0.5,
        batch_size: int = 32,
        max_epochs: int = 300,
        validation_fraction: float = 0.15,
        early_stopping_patience: int = 30,
        random_state: int = 42,
        device: str = "cpu",
        verbose: bool = False,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.validation_fraction = validation_fraction
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

    @staticmethod
    def _torch() -> Any:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "The paper's dropout MLP requires PyTorch. Install with: pip install -e '.[full]'"
            ) from exc
        return torch

    def _build_network(self, n_features: int, n_classes: int) -> Any:
        torch = self._torch()
        if self.activation.lower() != "relu":
            raise ValueError("The paper configuration specifies ReLU activation")
        layers: list[Any] = []
        previous = n_features
        for width in tuple(self.hidden_layer_sizes):
            layers.extend(
                [
                    torch.nn.Linear(previous, int(width)),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(float(self.dropout)),
                ]
            )
            previous = int(width)
        layers.append(torch.nn.Linear(previous, n_classes))
        return torch.nn.Sequential(*layers)

    def fit(self, X: np.ndarray, y: np.ndarray) -> TorchMLPClassifier:
        torch = self._torch()
        X_array = np.asarray(X, dtype=np.float32)
        y_array = np.asarray(y)
        if X_array.ndim != 2 or len(X_array) != len(y_array):
            raise ValueError("X must be a 2D matrix aligned with y")
        self.classes_, encoded = np.unique(y_array, return_inverse=True)
        if len(self.classes_) < 2:
            raise ValueError("MLP training requires at least two classes")
        self.n_features_in_ = X_array.shape[1]
        set_global_seed(int(self.random_state))

        n_samples = len(X_array)
        n_valid = max(1, int(round(n_samples * float(self.validation_fraction))))
        n_train = n_samples - n_valid
        if n_train < len(self.classes_):
            raise ValueError("Not enough samples for the configured validation fraction")

        device = torch.device(self.device)
        self.model_ = self._build_network(self.n_features_in_, len(self.classes_)).to(device)
        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=float(self.learning_rate),
            weight_decay=float(self.weight_decay),
        )
        criterion = torch.nn.CrossEntropyLoss()

        train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_array[:n_train]), torch.from_numpy(encoded[:n_train]).long()
        )
        generator = torch.Generator().manual_seed(int(self.random_state))
        loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(int(self.batch_size), n_train),
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        valid_X = torch.from_numpy(X_array[n_train:]).to(device)
        valid_y = torch.from_numpy(encoded[n_train:]).long().to(device)

        best_loss = float("inf")
        best_state: dict[str, Any] | None = None
        patience = 0
        self.training_history_ = []
        for epoch in range(int(self.max_epochs)):
            self.model_.train()
            batch_losses: list[float] = []
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(self.model_(batch_X), batch_y)
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))

            self.model_.eval()
            with torch.no_grad():
                validation_loss = float(criterion(self.model_(valid_X), valid_y).cpu())
            self.training_history_.append(
                {
                    "epoch": epoch + 1,
                    "train_loss": float(np.mean(batch_losses)),
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_loss - 1e-8:
                best_loss = validation_loss
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.model_.state_dict().items()
                }
                self.best_epoch_ = epoch + 1
                patience = 0
            else:
                patience += 1
                if patience >= int(self.early_stopping_patience):
                    break

        if best_state is None:
            raise RuntimeError("MLP optimization failed to produce a finite validation loss")
        self.model_.load_state_dict(best_state)
        self.model_.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise RuntimeError("TorchMLPClassifier must be fitted before prediction")
        torch = self._torch()
        features = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(self.device)
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(features), dim=1).cpu().numpy()

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
