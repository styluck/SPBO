# SPBO

MATLAB solver for **Error Bounds and a Smoothing Penalty Method for Binary Orthogonal Optimization**, by Ruikai Lv and Lianghai Xiao.

Contact: Lianghai Xiao, xiaolh@jnu.edu.cn.

SPBO minimizes a smooth objective over binary orthogonal matrices:

```text
minimize f(X), subject to X in {-1,+1}^{n x r} and X'X = n I_r.
```

The implementation applies Huber smoothing to an orthogonality-residual penalty over the box [-1,1], with spectral projected-gradient subproblem solves. The solver file is copied unchanged from the research implementation.

## Files

- `solve_bop_exact_penalty_box.m`: self-contained solver, including optional balanced-switch local search.
- `demo_projection.m`: one synthetic projection example; no datasets or baseline methods are needed.

## Quick start

Open MATLAB, change the current folder to this directory, and run:

```matlab
out = demo_projection;
B = out.X_round;
```

The demo minimizes `norm(X-C,'fro')^2` for a Gaussian matrix C, using n=128, r=8 and seed=101. It reports the objective, elapsed time, orthogonality residual and feasibility of the binary output. It creates no files or figures and restores the caller's random-number state.

To change the size and seed:

```matlab
out = demo_projection(256, 8, 102);
```

The demo accepts power-of-two n and 1 <= r <= n, which ensures the binary orthogonal feasible set is nonempty. This restriction belongs to the demo, not to the solver. This is a small usage example, not the paper's complete projection benchmark or a reproduction of its tables.

## Use your own objective

Provide a real, finite n-by-r initial matrix and a function that returns a scalar objective and its n-by-r Euclidean gradient. The function must also support being called with just one output.

```matlab
X0 = min(1, max(-1, randn(128, 8)));
opts = struct('verbose', 1);
out = solve_bop_exact_penalty_box(@my_objective, X0, opts);

% Save this function in my_objective.m:
function [f, g] = my_objective(X)
    f = sum(X(:).^2);
    if nargout > 1
        g = 2 * X;
    end
end
```

## Interpret the output

- `out.X`: continuous box-constrained iterate.
- `out.X_sign`: sign rounding, with zeros mapped to +1.
- `out.X_round`: final binary output, including optional local search.
- `out.F_fro` and `out.F_round`: continuous and final binary orthogonality residuals.
- `out.rounded_feasible`: whether the final binary residual meets the numerical feasibility tolerance.
- `out.status` and `out.hist`: termination reason and continuation history.

Termination does not certify a global minimum, and the solver may terminate without a feasible binary output. Check `out.rounded_feasible` and `out.F_round`. A continuous residual below 1 supplies the sign-rounding certificate discussed in the paper; the implementation also checks rounded feasibility numerically.

The solver disables optional local search by default. The projection demo enables it, matching the current projection experiment configuration. `out.X_sign` and `out.f_sign` retain the result before local search; `out.X_round` and `out.f_round` report the final result.
 
