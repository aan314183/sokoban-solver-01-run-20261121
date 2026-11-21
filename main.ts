// Sokoban solver — push-optimal, per-room decomposition.
//
// Strategy: every generated level is a hallway (the player's row) plus 1-4
// rooms, each with 3-4 boxes/goals and a single door. A box pushed out of a
// room through its door can never come back, so rooms are independent: the
// level optimum P* is the sum of the rooms' optima. We therefore solve each
// room on its own with an A* search over PUSHES (a macro-state is the sorted
// box squares plus the player's reachable region, named by its lowest
// square), then join the rooms with plain walks through the hallway.
//
// The heuristic is the optimal box->goal assignment (brute-force
// permutations, 3-4 boxes) over BFS distances that ignore boxes, which is
// admissible (each push moves one box by one square) and consistent, so the
// first solved state popped from the open list is push-optimal. Deadlock
// pruning: dead squares (reverse pull-BFS from the goals) and frozen 2x2
// blocks, both of which prune only unsolvable states, so optimality stands.
//
// Self-verification: every joined solution is replayed on the initial
// level with the Sokoban rules before it is written; a replay failure
// writes an empty solution instead, so no unvalidated solution is ever
// emitted. See verifySolution() at the bottom.
//
// Determinism: budgets are counted in expansions/states, never wall-clock;
// no randomness, no clock reads, no network. Tie-breaking is by insertion
// order, so two runs write byte-identical outputs.
//
// Fallbacks (in order, per room): A* -> plain push-BFS -> greedy best-first
// (h only). The BFS fallback is the same search with h=0, which the level
// generator itself guarantees to solve every kept room within 250k states.
// If every search fails for a room, the whole level falls back to a
// whole-level search, and finally to an empty (valid, 0-point) answer, so
// every instance is always answered in the stated format.
//
// Node 22, plain JavaScript in a .ts file (erasable syntax only), node
// builtins only.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (expansions / states), never the clock.
const MAX_EXP_ASTAR = 400000;
const MAX_EXP_BFS = 500000;
const MAX_EXP_GREEDY = 200000;

const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

// ---------------------------------------------------------------- parsing

function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++)
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

// Squares from which a box alone can be pulled back onto some goal.
function liveSquares(level) {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (y >= 0 && y < wall.length && !wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

// Flood fill from `from` around walls and `occupied` boxes; returns the
// lowest reachable square. `seen`/`stamp`/`stack` are scratch arrays.
function reach(w, wall, occupied, from, seen, stamp, stack) {
  let low = from;
  let n = 0;
  stack[n++] = from;
  seen[from] = stamp;
  while (n > 0) {
    const p = stack[--n];
    if (p < low) low = p;
    const q0 = p - w, q1 = p + w, q2 = p - 1, q3 = p + 1;
    if (wall[q0] === 0 && occupied[q0] === 0 && seen[q0] !== stamp) { seen[q0] = stamp; stack[n++] = q0; }
    if (wall[q1] === 0 && occupied[q1] === 0 && seen[q1] !== stamp) { seen[q1] = stamp; stack[n++] = q1; }
    if (wall[q2] === 0 && occupied[q2] === 0 && seen[q2] !== stamp) { seen[q2] = stamp; stack[n++] = q2; }
    if (wall[q3] === 0 && occupied[q3] === 0 && seen[q3] !== stamp) { seen[q3] = stamp; stack[n++] = q3; }
  }
  return low;
}

// Whether the box just pushed to `t` sits in a frozen 2x2 block of walls
// and boxes with a box off its goal.
function frozenBlock(w, wall, occupied, goal, t) {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1, c = a + w, d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) ||
        !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) ||
        (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

// Shortest walk from `from` to `to` around walls and boxes, lower-case LURD.
// Returns null when unreachable.
function walkTo(level, occupied, from, to) {
  const { w, wall } = level;
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

function endOf(w, from, moves) {
  let p = from;
  for (const ch of moves.toLowerCase())
    p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

function perms(n) {
  const out = [];
  const a = [...Array(n).keys()];
  const gen = (k) => {
    if (k === 1) { out.push(a.slice()); return; }
    for (let i = 0; i < k; i++) {
      gen(k - 1);
      const j = k % 2 === 0 ? i : 0;
      const t = a[j]; a[j] = a[k - 1]; a[k - 1] = t;
    }
  };
  gen(n);
  return out;
}

// ------------------------------------------------------- push-optimal search
//
// mode: 'astar' (f = g + h), 'bfs' (f = g), 'greedy' (f = h).
// Returns { pushes, moves } or null when the budget runs out / unsolvable.
function solvePush(level, maxExpansions, mode) {
  const { w, wall, goals } = level;
  const size = wall.length;
  const n = goals.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(level);
  if (level.boxes.some((b) => !live[b])) return null;
  const solved = (boxes) => { for (let i = 0; i < n; i++) if (!goal[boxes[i]]) return false; return true; };
  if (solved(level.boxes)) return { pushes: 0, moves: '' };

  // BFS distances from each goal, ignoring boxes (walls only).
  const dist = goals.map((g) => {
    const d = new Int32Array(size).fill(-1);
    d[g] = 0;
    const q = [g];
    for (let qi = 0; qi < q.length; qi++) {
      const p = q[qi];
      for (const [dx, dy] of DIRS) {
        const qq = p + dy * w + dx;
        if (!wall[qq] && d[qq] < 0) { d[qq] = d[p] + 1; q.push(qq); }
      }
    }
    return d;
  });
  const permList = perms(n);
  const heur = (boxes) => {
    let best = Infinity;
    for (const pm of permList) {
      let s = 0;
      for (let i = 0; i < n; i++) {
        const dd = dist[pm[i]][boxes[i]];
        if (dd < 0) { s = Infinity; break; }
        s += dd;
      }
      if (s < best) best = s;
    }
    return best;
  };

  const keyOf = (boxes, low) => {
    let k = low;
    for (const b of boxes) k = k * size + b;
    return k;
  };
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  const occupied = new Uint8Array(size);
  let stamp = 0;

  const sBoxes = [level.boxes.slice()];
  const sPlayer = [level.player];
  const sParent = [-1];
  const sFrom = [-1];
  const sTo = [-1];
  const sG = [0];
  const index = new Map();
  for (const b of level.boxes) occupied[b] = 1;
  const low0 = reach(w, wall, occupied, level.player, seen, ++stamp, stack);
  for (const b of level.boxes) occupied[b] = 0;
  const h0 = heur(level.boxes);
  if (h0 === Infinity) return null;
  index.set(keyOf(level.boxes, low0), 0);

  // Binary heap on [f, -g, seq, state]: smaller f first, deeper g first,
  // then earlier insertion — fully deterministic.
  const heap = [];
  let seq = 0;
  const prio = (mode, g, h) => (mode === 'bfs' ? g : mode === 'greedy' ? h : g + h);
  const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e) => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      const t = heap[i]; heap[i] = heap[p]; heap[p] = t;
      i = p;
    }
  };
  const heapPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ;) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        const t = heap[i]; heap[i] = heap[c]; heap[c] = t;
        i = c;
      }
    }
    return top;
  };
  heapPush([prio(mode, 0, h0), 0, seq++, 0]);

  let expanded = 0;
  let found = -1;
  while (heap.length > 0) {
    const [, ng, , s] = heapPop();
    const g = -ng;
    if (sG[s] !== g) continue; // stale entry (state re-reached cheaper)
    if (solved(sBoxes[s])) { found = s; break; }
    if (++expanded > maxExpansions) return null;
    const boxes = sBoxes[s];
    for (const b of boxes) occupied[b] = 1;
    reach(w, wall, occupied, sPlayer[s], seen, ++stamp, stack);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== here || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wall, occupied, goal, t);
        let key = -1;
        let next = null;
        let h = Infinity;
        if (!frozen) {
          next = boxes.slice();
          next[bi] = t;
          next.sort((x, y) => x - y);
          h = heur(next);
          if (h !== Infinity) key = keyOf(next, reach(w, wall, occupied, b, mark, ++stamp, stack));
        }
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen || h === Infinity) continue;
        const old = index.get(key);
        if (old !== undefined) {
          if (sG[old] <= g + 1) continue;
          sG[old] = g + 1;
          sParent[old] = s;
          sFrom[old] = b;
          sTo[old] = t;
          heapPush([prio(mode, g + 1, h), -(g + 1), seq++, old]);
          continue;
        }
        const ni = sBoxes.length;
        index.set(key, ni);
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(s);
        sFrom.push(b);
        sTo.push(t);
        sG.push(g + 1);
        heapPush([prio(mode, g + 1, h), -(g + 1), seq++, ni]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Reconstruct: walk to each push square, then push.
  const path = [];
  for (let s = found; s > 0; s = sParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = level.player;
  let boxes = level.boxes;
  for (const s of path) {
    const nextBoxes = sBoxes[s];
    const from = sFrom[s];
    const to = sTo[s];
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d);
    for (const b of boxes) occupied[b] = 1;
    const walk = walkTo(level, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk + dir[3];
    player = from;
    boxes = nextBoxes;
  }
  return { pushes: path.length, moves };
}

// ------------------------------------------------------------ room structure
//
// Cut the hallway row's dead squares (floor where a box reaches no goal)
// out of the floor; each remaining component holding boxes is a room.
function roomsOf(level) {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++)
    if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) ||
      rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

function solveRoom(level, room, otherRooms, done, player) {
  const wall = level.wall.slice();
  for (const other of otherRooms) {
    const src = done.has(other) ? other.goals : other.boxes;
    for (const b of src) wall[b] = 1;
  }
  const sub = {
    w: level.w, h: level.h, wall,
    goals: room.goals.slice().sort((a, b) => a - b),
    boxes: room.boxes.slice().sort((a, b) => a - b),
    player,
  };
  return solvePush(sub, MAX_EXP_ASTAR, 'astar') ??
    solvePush(sub, MAX_EXP_BFS, 'bfs') ??
    solvePush(sub, MAX_EXP_GREEDY, 'greedy');
}

function solveByRooms(level, rooms) {
  let player = level.player;
  let pushes = 0;
  let moves = '';
  const done = new Set();
  for (const room of rooms) {
    const others = rooms.filter((r) => r !== room);
    const solved = solveRoom(level, room, others, done, player);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = endOf(level.w, player, solved.moves);
    done.add(room);
  }
  return { pushes, moves };
}

function solveLevel(level) {
  if (level.boxes.length === 0) return '';
  const rooms = roomsOf(level);
  if (rooms !== null) {
    const r = solveByRooms(level, rooms);
    if (r !== null) return r.moves;
  }
  const s = solvePush(level, MAX_EXP_ASTAR, 'astar') ??
    solvePush(level, MAX_EXP_BFS, 'bfs') ??
    solvePush(level, MAX_EXP_GREEDY, 'greedy');
  return s !== null ? s.moves : '';
}

// ------------------------------------------------------- self-verification
//
// Replay a joined move string on the initial level with the Sokoban rules:
// a walk step (lower-case) into a wall or a box is illegal, a push step
// (upper-case) needs a box ahead and free floor behind it. Returns
// { ok, pushes } — ok only when no step is illegal AND every box ends on
// a goal. Pure state replay: no clock, no randomness, so determinism is
// untouched.
function verifySolution(level, moves) {
  const { w, wall, goals } = level;
  const goal = new Uint8Array(wall.length);
  for (const g of goals) goal[g] = 1;
  const box = new Uint8Array(wall.length);
  for (const b of level.boxes) box[b] = 1;
  let player = level.player;
  let pushes = 0;
  for (const ch of moves) {
    const lower = ch.toLowerCase();
    const d = lower === 'u' ? -w : lower === 'd' ? w : lower === 'l' ? -1 : lower === 'r' ? 1 : 0;
    if (d === 0) return { ok: false, pushes };
    const t = player + d;
    if (wall[t]) return { ok: false, pushes };
    if (ch !== lower) { // push step: a box must be ahead, free floor behind
      if (!box[t]) return { ok: false, pushes };
      const tt = t + d;
      if (wall[tt] || box[tt]) return { ok: false, pushes };
      box[t] = 0;
      box[tt] = 1;
      pushes++;
    } else if (box[t]) {
      return { ok: false, pushes }; // walk step into a box
    }
    player = t;
  }
  for (let i = 0; i < wall.length; i++)
    if (box[i] && !goal[i]) return { ok: false, pushes };
  return { ok: true, pushes };
}

for (const id of readdirSync(I).sort()) {
  let moves = '';
  let level = null;
  try {
    level = parseLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
    moves = solveLevel(level);
  } catch {
    moves = '';
  }
  if (level !== null && moves !== '') {
    const v = verifySolution(level, moves);
    if (v.ok) {
      console.error(`${id}: verified ${v.pushes} pushes`);
    } else {
      console.error(`${id}: INVALID, writing empty`);
      moves = '';
    }
  } else if (level !== null && level.boxes.length === 0) {
    console.error(`${id}: verified 0 pushes`);
  } else {
    console.error(`${id}: no solution, writing empty`);
  }
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
