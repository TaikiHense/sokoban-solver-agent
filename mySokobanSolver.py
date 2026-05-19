"""
Sokoban assignment — mySokobanSolver.py
"""

import search
import sokoban
import collections

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def my_team():
    return [
        (12026875, 'Noel',  'Vaikath'),
        (11906910, 'Riyat', 'Sreng'),
        (12031020, 'Taiki', 'Hense'),
    ]

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                          HELPER FUNCTIONS
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def _inside_cells(warehouse):
    """
    Flood-fill from worker to find all interior reachable cells,
    ignoring boxes (only walls block).
    """
    walls    = set(warehouse.walls)
    visited  = set()
    frontier = [warehouse.worker]
    while frontier:
        cell = frontier.pop()
        if cell in visited or cell in walls:
            continue
        visited.add(cell)
        x, y = cell
        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
            nb = (x+dx, y+dy)
            if nb not in visited and nb not in walls:
                frontier.append(nb)
    return visited


def _compute_taboo(warehouse):
    """
    Compute taboo cells using Rule 1 (corners) and Rule 2 (wall segments).
    """
    walls   = set(warehouse.walls)
    targets = set(warehouse.targets)
    inside  = _inside_cells(warehouse)

    taboo = set()

    # Rule 1: non-target corners
    for (x, y) in inside:
        if (x, y) in targets:
            continue
        blocked_h = (x-1, y) in walls or (x+1, y) in walls
        blocked_v = (x, y-1) in walls or (x, y+1) in walls
        if blocked_h and blocked_v:
            taboo.add((x, y))

    taboo_corners = set(taboo)

    # Rule 2: horizontal runs between corners
    for y in range(warehouse.nrows):
        row_tc = sorted(x for (x, yy) in taboo_corners if yy == y)
        for i in range(len(row_tc)):
            x1 = row_tc[i]
            for j in range(i+1, len(row_tc)):
                x2 = row_tc[j]
                run = [(xx, y) for xx in range(x1+1, x2)]
                if not run:
                    continue
                if not all(c in inside for c in run):
                    continue
                full_span = [(xx, y) for xx in range(x1, x2+1)]
                if any(c in targets for c in full_span):
                    continue
                wall_above = all((xx, y-1) in walls for xx, _ in run)
                wall_below = all((xx, y+1) in walls for xx, _ in run)
                if wall_above or wall_below:
                    taboo.update(run)

    # Rule 2: vertical runs between corners
    for x in range(warehouse.ncols):
        col_tc = sorted(y for (xx, y) in taboo_corners if xx == x)
        for i in range(len(col_tc)):
            y1 = col_tc[i]
            for j in range(i+1, len(col_tc)):
                y2 = col_tc[j]
                run = [(x, yy) for yy in range(y1+1, y2)]
                if not run:
                    continue
                if not all(c in inside for c in run):
                    continue
                full_span = [(x, yy) for yy in range(y1, y2+1)]
                if any(c in targets for c in full_span):
                    continue
                wall_left  = all((x-1, yy) in walls for _, yy in run)
                wall_right = all((x+1, yy) in walls for _, yy in run)
                if wall_left or wall_right:
                    taboo.update(run)

    return frozenset(taboo)


def _min_cost_assignment(cost_rows):
    """
    Optimal assignment via bitmask DP.
    cost_rows[i][j] = cost of assigning box i to target j.
    Returns minimum total assignment cost.
    O(2^n * n) — fast for n <= 8 boxes.
    """
    n = len(cost_rows)
    if n == 0:
        return 0
    m = len(cost_rows[0])
    if m == 0:
        return 0

    INF = float('inf')
    dp = [INF] * (1 << n)
    dp[0] = 0

    for mask in range(1 << n):
        if dp[mask] == INF:
            continue
        # target index = number of boxes assigned so far
        t = bin(mask).count('1')
        if t >= m:
            continue
        for b in range(n):
            if mask & (1 << b):
                continue
            new_mask = mask | (1 << b)
            c = dp[mask] + cost_rows[b][t]
            if c < dp[new_mask]:
                dp[new_mask] = c

    # best complete assignment (min(n,m) boxes assigned)
    k = min(n, m)
    best = INF
    for mask in range(1 << n):
        if bin(mask).count('1') == k:
            best = min(best, dp[mask])

    return best if best != INF else 0


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                          taboo_cells
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def taboo_cells(warehouse):
    """
    Return a string with '#' for walls and 'X' for taboo cells only.
    """
    walls = set(warehouse.walls)
    taboo = _compute_taboo(warehouse)

    X, Y   = zip(*walls)
    x_size = 1 + max(X)
    y_size = 1 + max(Y)

    vis = [[" "] * x_size for _ in range(y_size)]
    for (x, y) in walls:
        vis[y][x] = "#"
    for (x, y) in taboo:
        vis[y][x] = "X"

    return "\n".join("".join(row) for row in vis)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                          SokobanPuzzle
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

class SokobanPuzzle(search.Problem):
    """
    Sokoban modelled as a search.Problem.

    State: (worker_pos, boxes_tuple)
        worker_pos  — (x, y)
        boxes_tuple — sorted tuple of (x, y, w) triples
                      weight travels with the box, so path_cost is always right.

    Optimisations:
      - Tight heuristic: optimal box-to-target assignment weighted by
        (1 + box_weight) * manhattan_dist, plus worker-to-nearest-box distance
      - Freeze deadlock pruning in actions()
      - box_positions set built once per actions() call and reused
    """



    _DIRS = {
        'Left':  (-1,  0),
        'Right': ( 1,  0),
        'Up':    ( 0, -1),
        'Down':  ( 0,  1),
    }

    def __init__(self, warehouse):
        self.warehouse       = warehouse
        self.walls           = frozenset(warehouse.walls)
        self.targets         = frozenset(warehouse.targets)
        self._taboo          = _compute_taboo(warehouse)
        self._goal_positions = frozenset(warehouse.targets)

        initial_boxes = tuple(sorted(
            (x, y, w)
            for (x, y), w in zip(warehouse.boxes, warehouse.weights)
        ))
        super().__init__((warehouse.worker, initial_boxes), goal=None)

    def _box_pos_set(self, boxes):
        return frozenset((x, y) for x, y, _ in boxes)

    def _is_frozen(self, pos, box_positions):
        x, y = pos
        h_blocked = ((x-1, y) in self.walls or (x-1, y) in box_positions) and \
                    ((x+1, y) in self.walls or (x+1, y) in box_positions)
        v_blocked = ((x, y-1) in self.walls or (x, y-1) in box_positions) and \
                    ((x, y+1) in self.walls or (x, y+1) in box_positions)
        return h_blocked and v_blocked

    def actions(self, state):
        worker, boxes = state
        box_positions = self._box_pos_set(boxes)
        valid = []
        for action, (dx, dy) in self._DIRS.items():
            nw = (worker[0] + dx, worker[1] + dy)
            if nw in self.walls:
                continue
            if nw in box_positions:
                nb = (nw[0] + dx, nw[1] + dy)
                if nb in self.walls or nb in box_positions or nb in self._taboo:
                    continue
                if nb not in self.targets:
                    sim = (box_positions - {nw}) | {nb}
                    if self._is_frozen(nb, sim):
                        continue
            valid.append(action)
        return valid

    def result(self, state, action):
        worker, boxes = state
        dx, dy = self._DIRS[action]
        nw     = (worker[0] + dx, worker[1] + dy)

        new_boxes = list(boxes)
        for i, (x, y, w) in enumerate(new_boxes):
            if (x, y) == nw:
                new_boxes[i] = (x + dx, y + dy, w)
                break

        return (nw, tuple(sorted(new_boxes)))

    def goal_test(self, state):
        _, boxes = state
        return self._box_pos_set(boxes) == self._goal_positions

    def path_cost(self, c, state1, action, state2):
        worker1, boxes1 = state1
        dx, dy          = self._DIRS[action]
        pushed_xy       = (worker1[0] + dx, worker1[1] + dy)
        extra           = 0
        for x, y, w in boxes1:
            if (x, y) == pushed_xy:
                extra = w
                break
        return c + 1 + extra

    def h(self, node):
        """
        Admissible heuristic: optimal weighted box-to-target assignment.
        cost[i][j] = manhattan(box_i, target_j) * (1 + box_weight_i)
        Solved with bitmask DP for exact minimum — never overestimates.
        """
        _, boxes = node.state

        unplaced = [(x, y, w) for x, y, w in boxes
                    if (x, y) not in self._goal_positions]

        if not unplaced:
            return 0

        placed_xy    = self._box_pos_set(boxes)
        open_targets = [t for t in self.targets if t not in placed_xy]

        cost_rows = []
        for (bx, by, bw) in unplaced:
            row = [(abs(bx - tx) + abs(by - ty)) * (1 + bw)
                   for tx, ty in open_targets]
            cost_rows.append(row)

        return _min_cost_assignment(cost_rows)


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                       check_elem_action_seq
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def check_elem_action_seq(warehouse, action_seq):
    """
    Simulate action_seq on the warehouse.
    Returns 'Impossible' if any action is illegal, otherwise the
    Warehouse.__str__() of the resulting state.
    Pushing onto a taboo cell is legal (not an illegal move).
    """
    _DIRS = {
        'Left':  (-1,  0),
        'Right': ( 1,  0),
        'Up':    ( 0, -1),
        'Down':  ( 0,  1),
    }

    walls  = set(warehouse.walls)
    worker = warehouse.worker
    boxes  = set(warehouse.boxes)

    for action in action_seq:
        dx, dy     = _DIRS[action]
        new_worker = (worker[0]+dx, worker[1]+dy)

        if new_worker in walls:
            return 'Impossible'

        if new_worker in boxes:
            new_box = (new_worker[0]+dx, new_worker[1]+dy)
            if new_box in walls or new_box in boxes:
                return 'Impossible'
            boxes.discard(new_worker)
            boxes.add(new_box)

        worker = new_worker

    return str(warehouse.copy(worker=worker, boxes=list(boxes)))


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
#                      solve_weighted_sokoban
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def solve_weighted_sokoban(warehouse):
    puzzle = SokobanPuzzle(warehouse)

    if puzzle.goal_test(puzzle.initial):
        return [], 0

    node = search.astar_graph_search(puzzle)

    if node is None:
        return 'Impossible', None

    return node.solution(), node.path_cost