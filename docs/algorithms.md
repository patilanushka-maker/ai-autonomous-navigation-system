# Navigation Algorithms Specification

This document presents the mathematical formulations of every algorithm implemented in the **AI-Based Autonomous Navigation System**. All equations match the actual source code.

---

## 1. 2D Occupancy Grid Representation

The workspace is a 2D matrix $M \in \{0, 1\}^{H \times W}$ where:
- $M[r, c] = 0$: Free traversable cell
- $M[r, c] = 1$: Occupied obstacle cell

### World ↔ Grid Coordinate Conversion

Given resolution $\delta$ (meters/cell) and origin $(o_x, o_y)$:

**World → Grid** (floor discretization):
$$c = \left\lfloor \frac{x - o_x}{\delta} \right\rfloor, \quad r = \left\lfloor \frac{y - o_y}{\delta} \right\rfloor$$

**Grid → World** (cell center):
$$x = o_x + (c + 0.5) \cdot \delta, \quad y = o_y + (r + 0.5) \cdot \delta$$

**Grid → World** (cell lower corner):
$$x = o_x + c \cdot \delta, \quad y = o_y + r \cdot \delta$$

---

## 2. A* Path Planning Algorithm

### Problem Definition
Find the minimum-cost 4-connected path from start cell $S = (r_s, c_s)$ to goal cell $G = (r_g, c_g)$ on occupancy grid $M$, avoiding all cells where $M[r, c] = 1$.

### Cost Function
$$f(n) = g(n) + h(n)$$
- $g(n)$: Exact accumulated cost from $S$ to node $n$
- $h(n)$: Admissible heuristic estimate from $n$ to $G$

### Heuristic: Manhattan Distance
$$h(n) = |r_n - r_g| + |c_n - c_g|$$

**Admissibility proof**: Under 4-connected movement with uniform step cost 1.0, the true shortest path length $h^*(n) \ge |r_n - r_g| + |c_n - c_g|$ always. Thus $h(n) \le h^*(n)$, guaranteeing optimality.

**Consistency**: $h(n) \le 1 + h(n')$ for any neighbor $n'$, since Manhattan distance changes by at most 1 per step.

### Motion Primitives (4-Connected)
| Direction | $(\Delta r, \Delta c)$ | Cost |
| :--- | :--- | :--- |
| Up | $(-1, 0)$ | 1.0 |
| Down | $(+1, 0)$ | 1.0 |
| Left | $(0, -1)$ | 1.0 |
| Right | $(0, +1)$ | 1.0 |

Diagonal movements are **not** permitted.

### Algorithm Steps (as implemented)
1. Initialize open priority queue with $(f(S), h(S), S)$
2. Initialize `g_score = {S: 0}`, `came_from = {}`, `closed_set = ∅`
3. While open queue is non-empty:
   a. Pop node with lowest $f$-score (ties broken by lower $h$)
   b. If already in closed set, skip
   c. Add to closed set
   d. If node is $G$: reconstruct path via `came_from` chain → return
   e. For each 4-connected neighbor $n'$:
      - Skip if out-of-bounds, occupied, or in closed set
      - `tentative_g = g(current) + 1`
      - If `tentative_g < g(n')`: update `g_score`, `came_from`, push to queue
4. Return `None` (no path exists)

---

## 3. Unicycle Robot Kinematics

### State Vector
$$\mathbf{s} = \begin{bmatrix} x \\ y \\ \theta \end{bmatrix}, \quad \mathbf{u} = \begin{bmatrix} v \\ \omega \end{bmatrix}$$

### Continuous-Time Dynamics
$$\dot{x} = v \cos(\theta), \quad \dot{y} = v \sin(\theta), \quad \dot{\theta} = \omega$$

### Discrete Forward Euler Integration (as implemented)
$$x_{k+1} = x_k + v_k \cos(\theta_k) \cdot \Delta t$$
$$y_{k+1} = y_k + v_k \sin(\theta_k) \cdot \Delta t$$
$$\theta_{k+1} = \text{normalize}(\theta_k + \omega_k \cdot \Delta t)$$

### Angle Normalization
Maps any angle to $[-\pi, \pi)$:
$$\text{normalize}(\theta) = ((\theta + \pi) \mod 2\pi) - \pi$$

### Heading Convention
| $\theta$ Value | Direction |
| :--- | :--- |
| $0$ | Positive $x$ (right) |
| $+\pi/2$ | Positive $y$ (up) |
| $\pm\pi$ | Negative $x$ (left) |
| $-\pi/2$ | Negative $y$ (down) |

---

## 4. Heading-Based Proportional Controller

### Control Law

Given robot pose $(x, y, \theta)$ and target waypoint $(x_w, y_w)$:

**Step 1: Position error**
$$dx = x_w - x, \quad dy = y_w - y$$

**Step 2: Distance**
$$d = \sqrt{dx^2 + dy^2}$$

**Step 3: Arrival check**
If $d \le \text{goal\_tolerance}$: return $(v=0, \omega=0)$

**Step 4: Desired heading**
$$\theta_{\text{desired}} = \text{atan2}(dy, dx)$$

**Step 5: Heading error** (wrapped to $[-\pi, \pi)$)
$$e_\theta = \text{normalize}(\theta_{\text{desired}} - \theta)$$

**Step 6: Angular velocity** (proportional + clamped)
$$\omega_{\text{raw}} = K_p \cdot e_\theta$$
$$\omega_{\text{cmd}} = \text{clamp}(\omega_{\text{raw}}, -\omega_{\max}, \omega_{\max})$$

**Step 7: Linear velocity** (heading-modulated + slowdown)
- If $|e_\theta| \ge \theta_{\text{threshold}}$: $v = 0$ (rotate in place)
- Otherwise:
$$v_{\text{cmd}} = v_{\max} \cdot \cos(e_\theta) \cdot \min\left(1, \frac{d}{d_{\text{slow}}}\right)$$
where $d_{\text{slow}} = \max(2 \cdot \text{goal\_tolerance}, 1.0)$

---

## 5. Pure Pursuit Path Tracking Algorithm

Pure Pursuit calculates the curvature $\kappa$ of a continuous circular arc connecting the robot's current pose $(x, y, \theta)$ to a forward lookahead waypoint $(x_{la}, y_{la})$ located at lookahead distance $L_d$ along the planned path.

### Mathematical Formulation (Unicycle Mobile Robot)

1. **Lookahead Point Selection**:
   The controller projects the robot onto the path segments, advances monotonically along the path index, and determines the forward-most intersection of a circle of radius $L_d$ centered at $(x, y)$ with the piecewise-linear path segments. When within $L_d$ of the final waypoint, the lookahead target clamps to the goal $(x_g, y_g)$.

2. **Chord Vector & Relative Bearing**:
   $$\Delta x = x_{la} - x, \quad \Delta y = y_{la} - y$$
   $$D = \sqrt{\Delta x^2 + \Delta y^2}$$
   $$\theta_{la} = \text{atan2}(\Delta y, \Delta x)$$
   $$\alpha = \text{normalize\_angle}(\theta_{la} - \theta) \in [-\pi, \pi)$$

3. **Lateral Offset in Robot Coordinate Frame**:
   The lateral displacement $y_{\text{local}}$ perpendicular to the robot's forward axis is:
   $$y_{\text{local}} = -\Delta x \sin\theta + \Delta y \cos\theta = D \sin\alpha$$

4. **Circular Arc Curvature**:
   From the geometric relationship $2 R_c y_{\text{local}} = D^2$, where $R_c$ is the turning radius:
   $$\kappa = \frac{1}{R_c} = \frac{2 \sin\alpha}{D}$$

5. **Unicycle Control Mapping**:
   Unlike Ackermann vehicle models (which command front-wheel steering angle $\delta = \text{atan}(\kappa \cdot L)$), a differential-drive / unicycle robot directly controls rotational rate:
   $$\omega = v \cdot \kappa = \frac{2 v \sin\alpha}{D}$$
   $$\omega_{\text{cmd}} = \text{clamp}(\omega, -\omega_{\max}, \omega_{\max})$$

6. **Velocity Scheduling & Goal Arrival**:
   - **Goal Arrival**: If Euclidean distance to goal $d_{\text{goal}} \le \text{goal\_tolerance}$, commands halt immediately:
     $$v_{\text{cmd}} = 0.0, \quad \omega_{\text{cmd}} = 0.0$$
   - **Large Heading Deviation**: If $|\alpha| \ge \theta_{\text{threshold}}$ (default $\pi/2$), forward translation halts ($v = 0.0$) and the robot rotates in place:
     $$\omega_{\text{cmd}} = \text{clamp}(K_p \cdot \alpha, -\omega_{\max}, \omega_{\max})$$
   - **Nominal Forward Motion & Deceleration**:
     $$v_{\text{cmd}} = v_{\max} \cdot \cos\alpha \cdot \min\left(1.0, \frac{d_{\text{goal}}}{d_{\text{slow}}}\right)$$
     $$v_{\text{cmd}} = \text{clamp}(v_{\text{cmd}}, 0.0, v_{\max})$$

### Lookahead Distance ($L_d$) Tuning & Corner-Cutting Tradeoff

The lookahead distance $L_d$ governs the trade-off between path-tracking fidelity and motion smoothness:

1. **Small Lookahead ($L_d \to 0$, e.g., $L_d = 0.25$ m)**:
   - High path fidelity: the robot tightly hugs the reference trajectory.
   - Reduced damping: steering reacts aggressively to small lateral errors ($\kappa \propto \frac{1}{L_d}$), inducing subtle lateral oscillations and increased travel distance.
2. **Intermediate Lookahead ($L_d = 0.50$ m - Default)**:
   - Balanced response: smooth circular arcs on corners while maintaining sufficient clearance around obstacles.
   - Achieves 100% success rate across all navigable maps with zero collisions and zero timeouts.
3. **Large Lookahead ($L_d \to \infty$, e.g., $L_d = 1.00$ m)**:
   - Over-smoothing & Corner Cutting: When navigating sharp corners with corridor width $W \approx L_d$ (such as in `maze_like_map` with 1.0 m grid cells), the lookahead point jumps across the corner to the next orthogonal segment.
   - Collision Risk: The resulting pursuit arc cuts inside the obstacle corner. With robot radius $R = 0.20$ m and clearance margin $< R$, the robot's physical footprint intersects the obstacle boundary, causing a collision.

---

## 6. Circle-Rectangle Collision Checking

### Robot Footprint
Circle centered at $(x, y)$ with radius $R > 0$.

### Occupied Cell Geometry
Each occupied cell $(r, c)$ forms an axis-aligned bounding box (AABB):
$$[x_{\min}(c), x_{\max}(c)] \times [y_{\min}(r), y_{\max}(r)]$$
where:
$$x_{\min} = o_x + c \cdot \delta, \quad x_{\max} = o_x + (c+1) \cdot \delta$$
$$y_{\min} = o_y + r \cdot \delta, \quad y_{\max} = o_y + (r+1) \cdot \delta$$

### Distance Computation
Closest point on rectangle to circle center:
$$x_{\text{clamp}} = \max(x_{\min}, \min(x_{\max}, x))$$
$$y_{\text{clamp}} = \max(y_{\min}, \min(y_{\max}, y))$$
$$d = \sqrt{(x - x_{\text{clamp}})^2 + (y - y_{\text{clamp}})^2}$$

### Collision Rule (Conservative)
Collision if $d \le R + \epsilon$ where $\epsilon = 10^{-9}$.
Touching the obstacle boundary counts as collision.

### Map Boundary Rule
The entire footprint must be within map bounds:
$$x - R \ge X_{\min}, \quad x + R \le X_{\max}, \quad y - R \ge Y_{\min}, \quad y + R \le Y_{\max}$$

---

## 7. Simulation Evaluation Metrics

| Metric | Formula |
| :--- | :--- |
| **Success Flag** | $\mathbb{I}(\text{dist}(p_{\text{final}}, p_{\text{goal}}) \le \text{tolerance} \land \text{no collision})$ |
| **Path Length** | $L = \sum_{k=1}^N \|p_k - p_{k-1}\|_2$ |
| **Step Count** | Total discrete time steps executed |
| **Termination Status** | SUCCESS, COLLISION, TIMEOUT, or NO_PATH |
