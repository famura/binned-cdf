# binned-cdf

[![License: CC-BY-4.0][license-badge]][license-url]
![python][python-badge]<space>
[![Docs][docs-badge]][docs]
[![CI][ci-badge]][ci]
[![CD][cd-badge]][cd]
[![Coverage][coverage-badge]][coverage]
[![Tests][tests-badge]][tests]
[![mkdocs-material][mkdocs-material-badge]][mkdocs-material]
[![mypy][mypy-badge]][mypy]
[![pre-commit][pre-commit-badge]][pre-commit]
[![pytest][pytest-badge]][pytest]
[![Ruff][ruff-badge]][ruff]
[![uv][uv-badge]][uv]

A PyTorch-based distribution parametrized by the logits of CDF bins

## Background

The Cumulative Distribution Function (CDF) is a fundamental concept in probability theory and statistics that describes
the probability that a random variable $X$ takes on a value less than or equal to a given threshold $x$.
Formally, the CDF is defined as $F(x) = P(X \leq x)$, where $F(x)$ ranges from 0 to 1 as $x$ varies from negative to
positive infinity.
The CDF provides a complete characterization of the probability distribution of a random variable:
for continuous distributions, it is the integral of the probability density function (PDF), while
for discrete distributions, it is the sum of probabilities up to and including $x$.
Key properties of any CDF are the monotonicity and the boundary conditions
$\lim_{x \to -\infty} F(x) = 0$ and $\lim_{x \to \infty} F(x) = 1$.
CDFs are particularly useful for computing probabilities of intervals, quantiles, and for statistical inference.

## Application to Machine Learning

This repository uses the CDF to model and learn flexible probability distributions in machine learning tasks.
By parameterizing the CDF with binned logits, it enables differentiable training and efficient sampling, making it
suitable for uncertainty estimation, probabilistic prediction, and distributional modeling in neural networks.

## Implementation

The `BinnedLogitCDF` class inherits directly from `torch.distributions.Distribution`, implementing all necessary
methods plus some convenience functions.
It supports multi-dimensional batch shapes and CUDA devices.
The bins can be initialized linearly or log-spaced.

`torch>=2.7` it the only non-dev dependency of this repo.

:point_right: **Please have a look at the [documentation][docs] to get started.**

<!-- URLs -->
[cd-badge]: https://github.com/famura/binned-cdf/actions/workflows/cd.yaml/badge.svg
[cd]: https://github.com/famura/binned-cdf/actions/workflows/cd.yaml
[ci-badge]: https://github.com/famura/binned-cdf/actions/workflows/ci.yaml/badge.svg
[ci]: https://github.com/famura/binned-cdf/actions/workflows/ci.yaml
[coverage-badge]: https://famura.github.io/binned-cdf/latest/exported/coverage/badge.svg
[coverage]: https://famura.github.io/binned-cdf/latest/exported/coverage/index.html
[docs-badge]: https://img.shields.io/badge/Docs-gh--pages-informational
[docs]: https://famura.github.io/binned-cdf
[license-badge]: https://img.shields.io/badge/License-CC--BY--4.0%20-blue.svg
[license-url]: https://creativecommons.org/licenses/by/4.0
[mkdocs-material-badge]: https://img.shields.io/badge/Material_for_MkDocs-526CFE?logo=MaterialForMkDocs&logoColor=white
[mkdocs-material]: https://github.com/squidfunk/mkdocs-material
[mypy-badge]: https://www.mypy-lang.org/static/mypy_badge.svg
[mypy]: https://github.com/python/mypy
[pre-commit-badge]: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
[pre-commit]: https://github.com/pre-commit/pre-commit
[pytest-badge]: https://img.shields.io/badge/Pytest-green?logo=pytest
[pytest]: https://github.com/pytest-dev/pytest
[python-badge]: https://img.shields.io/badge/python-3.11%20|3.12%20|%203.13-informational?logo=python&logoColor=ffdd54
[ruff-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
[ruff]: https://docs.astral.sh/ruff
[tests-badge]: https://famura.github.io/binned-cdf/latest/exported/tests/badge.svg
[tests]: https://famura.github.io/binned-cdf/latest/exported/tests/index.html
[uv-badge]: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json
[uv]: https://docs.astral.sh/uv
