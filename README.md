# Deep Learning (Computational Graph)

## About

This project was completed as part of the **Deep Learning (DeLearn)** module in the
[Master of Science in Engineering](https://www.msengineering.ch/) (MSE) program. The goal is to implement a minimal,
NumPy-free computational graph with automatic differentiation from scratch. The `computational_graph` package provides the core
building blocks — value nodes, operator nodes, and a graph runner — to perform forward and backward passes by hand,
building intuition for how autodiff frameworks work under the hood.

## Authors

All authors contributed equally (alphabetical order by last name).

- Jakob Fender [GitHub](https://github.com/jakobfender)
- Jahnavi Patil
- Orlaith Quinn

## Setup

Developed with **Python 3.11+**. It is recommended to use a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

Then install the package in editable mode:

```bash
pip install -e .
```

For development tools (ruff, pre-commit):

```bash
pip install -e ".[dev]"
pre-commit install
```

## Usage

```python
from computational_graph import ValueNode, CompGraph, MultiplyNode

x1, x2, out = ValueNode(), ValueNode(), ValueNode()
MultiplyNode(x1, x2, out)

g = CompGraph([x1, x2], [out])
g.forward([[3.0, 4.0]])   # out.v == 12.0
g.backward()             # x1.grad_v == 4.0, x2.grad_v == 3.0
g.reset_values()
```
