"""
ASUS-21 Laplace implementation.

Protocol:
- diagonal empirical Fisher from per-sample squared training-loss gradients
- lambda = 1e-4 * mean(F)
- covariance = alpha^2 * diag(F + lambda)^(-1)
- alpha selected from {0.25, 0.50, 1.00, 2.00} by validation NLL
- 30 posterior parameter samples
- predictive variance is the primary epistemic uncertainty

This module is intentionally standalone.
It does not modify ASUS_21_TRAINING.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np
import torch
from torch import nn


ALPHA_GRID = (0.25, 0.50, 1.00, 2.00)
POSTERIOR_SAMPLES = 30
DAMPING_FACTOR = 1.0e-4


@dataclass
class LaplaceResult:
    alpha: float
    damping: float
    parameter_count: int
    fisher_mean: float
    validation_nll: dict[float, float]


def binary_nll(
    y_true: np.ndarray,
    probability: np.ndarray,
) -> float:

    y_true = np.asarray(
        y_true,
        dtype=np.float64,
    )

    probability = np.clip(
        np.asarray(
            probability,
            dtype=np.float64,
        ),
        1.0e-7,
        1.0 - 1.0e-7,
    )

    return float(
        -np.mean(
            y_true * np.log(probability)
            +
            (1.0 - y_true)
            * np.log(1.0 - probability)
        )
    )


def flatten_parameters(
    parameters: Iterable[torch.Tensor],
) -> torch.Tensor:

    return torch.cat(
        [
            parameter.detach()
            .to(dtype=torch.float64)
            .reshape(-1)
            for parameter in parameters
        ]
    )


def restore_parameters(
    parameters: Iterable[torch.Tensor],
    flat: torch.Tensor,
) -> None:

    offset = 0

    with torch.no_grad():

        for parameter in parameters:

            count = parameter.numel()

            value = (
                flat[
                    offset:
                    offset + count
                ]
                .reshape(parameter.shape)
                .to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
            )

            parameter.copy_(
                value
            )

            offset += count


def parameter_count(
    parameters: Iterable[torch.Tensor],
) -> int:

    return sum(
        parameter.numel()
        for parameter in parameters
    )


def empirical_fisher_diagonal(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    parameters: list[torch.Tensor],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Compute the diagonal empirical Fisher:

        F_j = mean_i[
            (d L_i / d theta_j)^2
        ]

    using training data only.
    """

    X = torch.as_tensor(
        X_train,
        dtype=dtype,
        device=device,
    )

    y = torch.as_tensor(
        y_train,
        dtype=torch.float32,
        device=device,
    )

    fisher = torch.zeros(
        parameter_count(parameters),
        dtype=torch.float64,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss()

    model.eval()

    for index in range(
        len(X)
    ):

        model.zero_grad(
            set_to_none=True
        )

        logits = model(
            X[index:index + 1]
        )

        loss = criterion(
            logits.reshape(-1),
            y[index:index + 1].reshape(-1),
        )

        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=False,
            create_graph=False,
        )

        flat_gradient = torch.cat(
            [
                gradient.detach()
                .to(dtype=torch.float64)
                .reshape(-1)
                for gradient in gradients
            ]
        )

        fisher += (
            flat_gradient
            * flat_gradient
        )

    fisher /= float(
        len(X)
    )

    return fisher


def covariance_standard_deviation(
    fisher: torch.Tensor,
    alpha: float,
    damping: float,
) -> torch.Tensor:
    """
    From

        Sigma =
            alpha^2 diag(F + lambda)^(-1)

    return the diagonal posterior standard deviation.
    """

    return (
        float(alpha)
        /
        torch.sqrt(
            fisher
            +
            float(damping)
        )
    )


def sample_parameter_vector(
    mean: torch.Tensor,
    fisher: torch.Tensor,
    alpha: float,
    damping: float,
    generator: torch.Generator,
) -> torch.Tensor:

    noise = torch.randn(
        mean.shape,
        dtype=mean.dtype,
        device=mean.device,
        generator=generator,
    )

    std = covariance_standard_deviation(
        fisher,
        alpha,
        damping,
    )

    return (
        mean
        +
        noise * std
    )


def posterior_predictive_samples(
    model: nn.Module,
    X: np.ndarray,
    mean_parameters: torch.Tensor,
    fisher: torch.Tensor,
    alpha: float,
    damping: float,
    parameters: list[torch.Tensor],
    predictor: Callable[
        [nn.Module, np.ndarray],
        np.ndarray,
    ],
    generator: torch.Generator,
    samples: int = POSTERIOR_SAMPLES,
) -> np.ndarray:

    predictions = []

    for _ in range(
        samples
    ):

        sampled_parameters = (
            sample_parameter_vector(
                mean_parameters,
                fisher,
                alpha,
                damping,
                generator,
            )
        )

        restore_parameters(
            parameters,
            sampled_parameters,
        )

        probability = predictor(
            model,
            X,
        )

        predictions.append(
            np.asarray(
                probability,
                dtype=np.float64,
            )
        )

    restore_parameters(
        parameters,
        mean_parameters,
    )

    return np.stack(
        predictions,
        axis=0,
    )


def select_alpha(
    model: nn.Module,
    X_val: np.ndarray,
    y_val: np.ndarray,
    mean_parameters: torch.Tensor,
    fisher: torch.Tensor,
    damping: float,
    parameters: list[torch.Tensor],
    predictor: Callable[
        [nn.Module, np.ndarray],
        np.ndarray,
    ],
    generator: torch.Generator,
) -> tuple[float, dict[float, float]]:

    scores: dict[float, float] = {}

    for alpha in ALPHA_GRID:

        predictions = (
            posterior_predictive_samples(
                model=model,
                X=X_val,
                mean_parameters=mean_parameters,
                fisher=fisher,
                alpha=alpha,
                damping=damping,
                parameters=parameters,
                predictor=predictor,
                generator=generator,
                samples=POSTERIOR_SAMPLES,
            )
        )

        mean_probability = (
            predictions.mean(
                axis=0
            )
        )

        score = binary_nll(
            y_val,
            mean_probability,
        )

        scores[
            float(alpha)
        ] = float(score)

    selected = min(
        scores,
        key=scores.get,
    )

    return (
        float(selected),
        scores,
    )


def fit_laplace(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    predictor: Callable[
        [nn.Module, np.ndarray],
        np.ndarray,
    ],
    parameters: list[torch.Tensor],
    seed: int,
    dtype: torch.dtype = torch.float32,
) -> LaplaceResult:

    count = parameter_count(
        parameters
    )

    mean_parameters = (
        flatten_parameters(
            parameters
        )
        .to(device)
    )

    fisher = empirical_fisher_diagonal(
        model=model,
        X_train=X_train,
        y_train=y_train,
        parameters=parameters,
        device=device,
        dtype=dtype,
    )

    fisher_mean = max(
        float(
            fisher.mean()
            .detach()
            .cpu()
            .item()
        ),
        1.0e-12,
    )

    damping = (
        DAMPING_FACTOR
        * fisher_mean
    )

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        int(seed)
    )

    alpha, validation_nll = (
        select_alpha(
            model=model,
            X_val=X_val,
            y_val=y_val,
            mean_parameters=mean_parameters,
            fisher=fisher,
            damping=damping,
            parameters=parameters,
            predictor=predictor,
            generator=generator,
        )
    )

    restore_parameters(
        parameters,
        mean_parameters,
    )

    return LaplaceResult(
        alpha=alpha,
        damping=damping,
        parameter_count=count,
        fisher_mean=fisher_mean,
        validation_nll=validation_nll,
    )


def sample_test_predictions(
    model: nn.Module,
    X_test: np.ndarray,
    fitted: LaplaceResult,
    fisher: torch.Tensor,
    mean_parameters: torch.Tensor,
    parameters: list[torch.Tensor],
    predictor: Callable[
        [nn.Module, np.ndarray],
        np.ndarray,
    ],
    device: torch.device,
    seed: int,
) -> np.ndarray:

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        int(seed) + 100000
    )

    predictions = (
        posterior_predictive_samples(
            model=model,
            X=X_test,
            mean_parameters=mean_parameters,
            fisher=fisher,
            alpha=fitted.alpha,
            damping=fitted.damping,
            parameters=parameters,
            predictor=predictor,
            generator=generator,
            samples=POSTERIOR_SAMPLES,
        )
    )

    return predictions


def predictive_statistics(
    posterior_predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:

    mean_probability = (
        posterior_predictions.mean(
            axis=0
        )
    )

    predictive_variance = (
        posterior_predictions.var(
            axis=0
        )
    )

    return (
        mean_probability,
        predictive_variance,
    )


if __name__ == "__main__":

    print("=" * 72)
    print("ASUS-21 LAPLACE MODULE")
    print("=" * 72)
    print(
        "Empirical Fisher: per-sample squared training-loss gradients"
    )
    print(
        "Damping: lambda = 1e-4 * mean(F)"
    )
    print(
        "Covariance: alpha^2 * diag(F + lambda)^(-1)"
    )
    print(
        "Alpha grid:",
        ALPHA_GRID,
    )
    print(
        "Posterior samples:",
        POSTERIOR_SAMPLES,
    )
    print("=" * 72)