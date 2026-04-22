from __future__ import annotations

from collections import defaultdict, deque

from ..core import CompGraph
from ..nodes.add import AddNode
from ..nodes.multiply import MultiplyNode
from ..nodes.square import SquareNode
from ..nodes.value import ValueNode

_OPERATOR_SYMBOLS: dict[type, str] = {
    MultiplyNode: r"$\times$",
    AddNode: r"$+$",
    SquareNode: r"\^{}2",
}

_PREAMBLE = r"""\begin{tikzpicture}[
    node distance=1.5cm,
    every node/.style={circle, minimum size=0.8cm, draw, thick},
    input/.style={fill=yellow!40},
    operator/.style={fill=red!30},
    intermediary/.style={fill=green!30},
    output/.style={fill=cyan!30},
    arrow/.style={-{Stealth}, thick}
]"""

_X_STEP = 2.0  # horizontal spacing between columns (cm)
_Y_STEP = 1.2  # vertical spacing between rows within a column (cm)


def _bfs_nodes(graph: CompGraph) -> list:
    """Return all nodes reachable from graph inputs in BFS order.

    Args:
        graph (CompGraph): The graph to traverse.

    Returns:
        list: All reachable nodes in BFS order.
    """
    visited: dict[int, object] = {}
    queue: deque = deque()
    for node in graph.in_nodes:
        visited[id(node)] = node
        queue.append(node)
    while queue:
        node = queue.popleft()
        for child in node.children:
            if id(child) not in visited:
                visited[id(child)] = child
                queue.append(child)
    return list(visited.values())


def _assign_levels(graph: CompGraph, all_nodes: list) -> dict[int, int]:
    """Assign a column level to each node via Kahn's topological sort.

    Uses longest-path from sources so that nodes with multiple incoming paths
    (e.g. an operator fed by both an early and a late node) are placed at
    the deepest possible column.

    Input nodes (``graph.in_nodes``) are then repositioned to
    ``min(child_level) - 1`` so that late-joining inputs appear adjacent to
    the operator they feed rather than at column 0.

    Args:
        graph (CompGraph): The graph whose nodes are to be levelled.
        all_nodes (list): All reachable nodes (from ``_bfs_nodes``).

    Returns:
        dict[int, int]: Mapping of ``id(node)`` to column level.
    """
    node_ids = {id(n) for n in all_nodes}

    # Build in-degree restricted to the visible subgraph
    in_degree: dict[int, int] = {id(n): 0 for n in all_nodes}
    for node in all_nodes:
        for child in node.children:
            if id(child) in node_ids:
                in_degree[id(child)] += 1

    # Kahn's algorithm — process nodes only once all parents are done,
    # accumulating the longest path seen so far
    levels: dict[int, int] = {}
    queue: deque = deque()
    for node in all_nodes:
        if in_degree[id(node)] == 0:
            levels[id(node)] = 0
            queue.append(node)

    while queue:
        node = queue.popleft()
        for child in node.children:
            if id(child) not in node_ids:
                continue
            levels[id(child)] = max(levels.get(id(child), 0), levels[id(node)] + 1)
            in_degree[id(child)] -= 1
            if in_degree[id(child)] == 0:
                queue.append(child)

    # Reposition input nodes to sit one column before their earliest child
    for node in graph.in_nodes:
        child_levels = [levels[id(c)] for c in node.children if id(c) in levels]
        if child_levels:
            levels[id(node)] = min(child_levels) - 1

    return levels


def _node_style(node, graph: CompGraph) -> str:
    """Return the TikZ style name for a node.

    Args:
        node: The node to classify.
        graph (CompGraph): The enclosing graph (used to identify boundary nodes).

    Returns:
        str: One of ``"input"``, ``"output"``, ``"intermediary"``, ``"operator"``.
    """
    if isinstance(node, ValueNode):
        if node in graph.in_nodes:
            return "input"
        if node in graph.out_nodes:
            return "output"
        return "intermediary"
    return "operator"


def _node_label(node) -> str:
    """Return the LaTeX label string for a node.

    Args:
        node: The node to label.

    Returns:
        str: A LaTeX string suitable for use inside a TikZ node.
    """
    if isinstance(node, ValueNode):
        safe_name = node.name.replace("_", r"\_")
        return f"${safe_name}$"
    for cls, symbol in _OPERATOR_SYMBOLS.items():
        if isinstance(node, cls):
            return symbol
    return "?"


def _format_coord(v: float) -> str:
    """Format a coordinate value compactly (no trailing zeros)."""
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}".rstrip("0")


def _y_positions(level_nodes: list, in_node_set: set) -> list[float]:
    """Compute y coordinates for nodes in one column.

    Computed (non-input) nodes are placed from ``(n-1)*Y_STEP`` down to ``0``.
    Input nodes that were repositioned into this column are placed below ``0``
    at ``-Y_STEP``, ``-2*Y_STEP``, etc.

    For input-only columns, nodes are placed from ``(n-1)*Y_STEP`` down to ``0``.

    Args:
        level_nodes (list): Nodes in this column, in BFS discovery order.
        in_node_set (set): Set of ``id``\\s of graph input nodes.

    Returns:
        list[float]: y coordinate for each node in ``level_nodes`` order.
    """
    computed = [n for n in level_nodes if id(n) not in in_node_set]
    inputs = [n for n in level_nodes if id(n) in in_node_set]

    positions: dict[int, float] = {}

    if not computed:
        # Input-only column: top-down, bottom node at y=0
        count = len(inputs)
        for row, node in enumerate(inputs):
            positions[id(node)] = (count - 1 - row) * _Y_STEP
    else:
        # Computed nodes: top-down, bottom computed node at y=0
        n_c = len(computed)
        for i, node in enumerate(computed):
            positions[id(node)] = (n_c - 1 - i) * _Y_STEP
        # Repositioned inputs: below the computed nodes
        for i, node in enumerate(inputs):
            positions[id(node)] = -(i + 1) * _Y_STEP

    return [positions[id(n)] for n in level_nodes]


def generate_tikz(graph: CompGraph) -> str:
    """Generate a TikZ ``tikzpicture`` for the given computational graph.

    Nodes are laid out on a grid: columns correspond to topological depth
    (longest path from an input). Input nodes that feed into later operators
    are placed adjacent to those operators rather than at column 0.

    Node styles:

    - **input** (yellow): graph input ``ValueNode`` objects.
    - **operator** (red): ``MultiplyNode``, ``AddNode``, ``SquareNode``.
    - **intermediary** (green): internal ``ValueNode`` objects.
    - **output** (cyan): graph output ``ValueNode`` objects.

    Args:
        graph (CompGraph): The computational graph to visualize.

    Returns:
        str: A self-contained TikZ ``tikzpicture`` environment as a string.
             Requires ``\\usepackage{tikz}`` and
             ``\\usetikzlibrary{arrows.meta, positioning}`` in the document
             preamble.
    """
    all_nodes = _bfs_nodes(graph)
    levels = _assign_levels(graph, all_nodes)
    in_node_set = {id(n) for n in graph.in_nodes}

    by_level: dict[int, list] = defaultdict(list)
    for node in all_nodes:
        by_level[levels[id(node)]].append(node)

    tikz_id: dict[int, str] = {id(n): f"n{i}" for i, n in enumerate(all_nodes)}

    lines = [_PREAMBLE, ""]

    for level in sorted(by_level):
        level_nodes = by_level[level]
        x = level * _X_STEP
        ys = _y_positions(level_nodes, in_node_set)
        for node, y in zip(level_nodes, ys):
            nid = tikz_id[id(node)]
            style = _node_style(node, graph)
            label = _node_label(node)
            xc, yc = _format_coord(x), _format_coord(y)
            lines.append(f"\\node[{style}] ({nid}) at ({xc},{yc}) {{{label}}};")

    lines.append("")

    for node in all_nodes:
        src = tikz_id[id(node)]
        for child in node.children:
            dst = tikz_id[id(child)]
            lines.append(f"\\draw[arrow] ({src}) -- ({dst});")

    lines += ["", r"\end{tikzpicture}"]
    return "\n".join(lines)
