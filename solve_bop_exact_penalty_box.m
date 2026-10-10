function out = solve_bop_exact_penalty_box(obj_fun, X0, opts)
%SOLVE_BOP_EXACT_PENALTY_BOX Huber smoothing method for the exact-penalty model
%
%   min f(X) + mu * ||X' * X - n * I||_F,   -1 <= X <= 1.
%
% The nonsmooth Frobenius norm is replaced during the computation by its
% Huber--Moreau smoothing
%
%               { s^2/(2*delta),  s <= delta,
%   h_delta(s) = {
%               { s-delta/2,      s >  delta,
%
% where s = ||X' * X - n * I||_F.  A continuation loop increases mu and
% decreases delta.  Each smooth box-constrained subproblem is solved by a
% spectral projected-gradient method with nonmonotone Armijo line search.
%
% Required input
%   obj_fun  function handle with interface [f,g] = obj_fun(X)
%   X0       initial n-by-r matrix
%
% Main options
%   eps_feas       target Frobenius orthogonality residual       [1e-3]
%   eps_opt        projected-gradient tolerance                 [1e-5]
%   max_stages     maximum continuation stages                  [20]
%   max_inner      maximum SPG iterations per stage             [500]
%   mu_auto        balance the initial penalty gradient         [true]
%   mu0            initial penalty parameter                    [1]
%   mu_balance     target initial penalty/objective grad ratio   [0.1]
%   mu_mult        penalty growth factor                        [2]
%   mu_max         maximum penalty parameter                    [1e8]
%   delta0         initial smoothing parameter                  [max(1,F0/10)]
%   delta_decay    smoothing reduction factor                   [0.2]
%   delta_min      minimum smoothing parameter                  [eps_feas/10]
%   stop_on_round  stop after certified feasible sign rounding  [true]
%   round_tol      numerical tolerance for rounded feasibility   [1e-10*max(1,n*r)]
%   local_search_enabled
%                  apply balanced-switch local search after a
%                  certified feasible sign rounding             [false]
%   local_search_max_iters
%                  maximum accepted balanced switches           [50]
%   local_search_tol
%                  relative objective-decrease tolerance         [1e-12]
%   nonmonotone_M  line-search memory                           [10]
%   verbose        0 or 1                                       [1]
%
% Output
%   out.X          final continuous box-constrained point
%   out.X_sign     entrywise sign rounding (zeros mapped to +1)
%   out.X_round    final binary output; equals out.X_sign unless local
%                  search is enabled and produces an improvement
%   out.f          objective at out.X
%   out.f_sign     objective at out.X_sign
%   out.f_round    objective at out.X_round
%   out.F_fro      ||out.X' * out.X - n*I||_F
%   out.F_sign     sign-rounded Frobenius orthogonality residual
%   out.F_round    final binary-output orthogonality residual
%   out.local_search
%                  diagnostics for the optional local search
%   out.status     termination status
%   out.hist       one record per continuation stage

    if nargin < 3 || isempty(opts)
        opts = struct();
    end
    if ~isa(obj_fun, 'function_handle')
        error('ExactPenalty:InvalidObjective', ...
            'obj_fun must be a function handle with interface [f,g]=obj_fun(X).');
    end
    if ~isnumeric(X0) || ~ismatrix(X0) || isempty(X0) || any(~isfinite(X0(:)))
        error('ExactPenalty:InvalidInitialPoint', ...
            'X0 must be a nonempty finite numeric matrix.');
    end

    [n, r] = size(X0);
    X = project_box(X0);

    eps_feas = getopt(opts, 'eps_feas', 1e-3);
    eps_opt = getopt(opts, 'eps_opt', 1e-5);
    max_stages = getopt(opts, 'max_stages', 20);
    max_inner = getopt(opts, 'max_inner', 500);
    mu_auto = getopt(opts, 'mu_auto', true);
    mu = getopt(opts, 'mu0', 1);
    mu_balance = getopt(opts, 'mu_balance', 0.1);
    mu_mult = getopt(opts, 'mu_mult', 2);
    mu_max = getopt(opts, 'mu_max', 1e8);
    delta_decay = getopt(opts, 'delta_decay', 0.2);
    delta_min = getopt(opts, 'delta_min', max(1e-12, eps_feas / 10));
    stop_on_round = getopt(opts, 'stop_on_round', true);
    round_tol = getopt(opts, 'round_tol', 1e-10 * max(1, n * r));
    local_search_enabled = getopt(opts, 'local_search_enabled', false);
    local_search_max_iters = getopt(opts, 'local_search_max_iters', 50);
    local_search_tol = getopt(opts, 'local_search_tol', 1e-12);
    verbose = getopt(opts, 'verbose', 1);

    validate_positive(eps_feas, 'eps_feas');
    validate_positive(eps_opt, 'eps_opt');
    validate_positive(mu, 'mu0');
    validate_positive(mu_mult - 1, 'mu_mult-1');
    validate_positive(mu_max, 'mu_max');
    validate_positive(delta_min, 'delta_min');
    validate_positive(round_tol, 'round_tol');
    validate_logical_scalar(local_search_enabled, 'local_search_enabled');
    validate_nonnegative(local_search_tol, 'local_search_tol');
    if delta_decay <= 0 || delta_decay >= 1
        error('ExactPenalty:InvalidOption', 'delta_decay must lie in (0,1).');
    end
    if max_stages < 1 || max_stages ~= fix(max_stages) || ...
            max_inner < 1 || max_inner ~= fix(max_inner) || ...
            local_search_max_iters < 1 || ...
            local_search_max_iters ~= fix(local_search_max_iters)
        error('ExactPenalty:InvalidOption', ...
            ['max_stages, max_inner, and local_search_max_iters ', ...
             'must be positive integers.']);
    end

    [f0, g0] = obj_fun(X);
    check_objective_output(f0, g0, X);
    [C0, F0] = orthogonality_residual(X, n);
    delta = getopt(opts, 'delta0', max(1, F0 / 10));
    validate_positive(delta, 'delta0');
    delta = max(delta, delta_min);

    if mu_auto && F0 > 0
        % Match a fraction of the initial objective-gradient norm.  This is
        % only a scale heuristic; continuation is responsible for raising
        % mu when the initial value is too small.
        unit_penalty_grad = 2 * X * C0 / max(F0, delta);
        unit_norm = norm(unit_penalty_grad, 'fro');
        if unit_norm > 0
            mu = max(mu, mu_balance * norm(g0, 'fro') / unit_norm);
        end
    end
    mu = min(mu, mu_max);

    hist.stage = zeros(max_stages, 1);
    hist.f = zeros(max_stages, 1);
    hist.f_round = zeros(max_stages, 1);
    hist.penalized_value = zeros(max_stages, 1);
    hist.penalty_value = zeros(max_stages, 1);
    hist.F_fro = zeros(max_stages, 1);
    hist.F_max = zeros(max_stages, 1);
    hist.F_round = zeros(max_stages, 1);
    hist.R_penalty = zeros(max_stages, 1);
    hist.mu = zeros(max_stages, 1);
    hist.delta = zeros(max_stages, 1);
    hist.inner_iters = zeros(max_stages, 1);
    hist.func_evals = zeros(max_stages, 1);
    hist.step_norm = zeros(max_stages, 1);

    status = 'max_stages_reached';
    last = 0;
    for stage = 1:max_stages
        X_before = X;
        inner_tol = max(eps_opt, min(1e-2, sqrt(delta)));
        [X, inner] = solve_smooth_stage( ...
            X, obj_fun, n, mu, delta, inner_tol, max_inner, opts);

        % The smooth-stage solver returns the quantities already computed
        % at its final iterate, avoiding another objective, Gram-matrix,
        % residual, and gradient evaluation here.
        pval = inner.value;
        G = inner.G;
        details = inner.details;
        R_penalty = max(abs(project_box(X - G) - X), [], 'all');
        X_round = sign_binary(X);
        f_round = obj_fun(X_round);
        [~, F_round] = orthogonality_residual(X_round, n);

        last = stage;
        hist.stage(stage) = stage;
        hist.f(stage) = details.f;
        hist.f_round(stage) = f_round;
        hist.penalized_value(stage) = pval;
        hist.penalty_value(stage) = details.penalty_value;
        hist.F_fro(stage) = details.F_fro;
        hist.F_max(stage) = max(abs(details.C), [], 'all');
        hist.F_round(stage) = F_round;
        hist.R_penalty(stage) = R_penalty;
        hist.mu(stage) = mu;
        hist.delta(stage) = delta;
        hist.inner_iters(stage) = inner.iter;
        hist.func_evals(stage) = inner.func_evals;
        hist.step_norm(stage) = norm(X - X_before, 'fro');

        if verbose
            fprintf(['[BOX-EP %2d] f=% .6e F=% .3e Fround=% .3e ', ...
                'Rpen=% .3e mu=% .2e delta=% .2e inner=%d\n'], ...
                stage, details.f, details.F_fro, hist.F_round(stage), ...
                R_penalty, mu, delta, inner.iter);
        end

        if details.F_fro <= eps_feas
            status = 'converged_feasibility';
            break;
        end
        if stop_on_round && details.F_fro < 1 && ...
                hist.F_round(stage) <= round_tol
            status = 'converged_exact_rounding';
            break;
        end
        if mu >= mu_max && delta <= delta_min
            status = 'stopped_mu_max';
            break;
        end

        mu = min(mu_max, mu_mult * mu);
        delta = max(delta_min, delta_decay * delta);
    end

    names = fieldnames(hist);
    for i = 1:numel(names)
        hist.(names{i}) = hist.(names{i})(1:last, :);
    end

    X_sign = sign_binary(X);
    f = details.f;
    f_sign = obj_fun(X_sign);
    C = details.C;
    F_fro = details.F_fro;
    [~, F_sign] = orthogonality_residual(X_sign, n);

    X_round = X_sign;
    local_search = initialize_local_search_info( ...
        logical(local_search_enabled), f_sign);
    if local_search_enabled
        if F_fro >= 1
            local_search.status = 'certificate_not_met';
        elseif F_sign > round_tol
            local_search.status = 'sign_rounding_infeasible';
        else
            [X_round, local_search] = balanced_switch_local_search( ...
                X_sign, obj_fun, local_search_max_iters, local_search_tol);
            if verbose
                fprintf(['[BOX-EP LS] status=%s moves=%d evals=%d ', ...
                    'f_sign=% .6e f_final=% .6e\n'], ...
                    local_search.status, local_search.accepted_moves, ...
                    local_search.func_evals, f_sign, local_search.f_final);
            end
        end
    end

    f_round = obj_fun(X_round);
    [~, F_round] = orthogonality_residual(X_round, n);

    out.X = X;
    out.X_sign = X_sign;
    out.X_round = X_round;
    out.f = f;
    out.f_sign = f_sign;
    out.f_round = f_round;
    out.F_fro = F_fro;
    out.F_max = max(abs(C), [], 'all');
    out.F_sign = F_sign;
    out.F_round = F_round;
    out.rounded_feasible = out.F_round <= round_tol;
    out.local_search = local_search;
    out.mu = hist.mu(end);
    out.delta = hist.delta(end);
    out.status = status;
    out.hist = hist;
    out.method = 'Huber smoothing of the orthogonality exact-penalty model over the box';
end

function [X, info] = solve_smooth_stage(X, obj_fun, n, mu, delta, ...
        tol, maxit, opts)
    armijo = getopt(opts, 'armijo', 1e-4);
    backtrack = getopt(opts, 'backtrack', 0.5);
    max_linesearch = getopt(opts, 'max_linesearch', 40);
    memory = getopt(opts, 'nonmonotone_M', 10);
    alpha_min = getopt(opts, 'alpha_min', 1e-16);
    alpha_max = getopt(opts, 'alpha_max', 1e16);
    alpha = getopt(opts, 'alpha0', 1);

    X = project_box(X);
    [value, G, details] = penalty_value_grad(X, obj_fun, n, mu, delta);
    func_evals = 1;
    values = value;
    for it = 1:maxit
        PG = project_box(X - G) - X;
        residual = max(abs(PG), [], 'all');
        if residual <= tol
            break;
        end

        D = project_box(X - alpha * G) - X;
        gtd = sum(G(:) .* D(:));
        if gtd >= 0 || norm(D, 'fro') == 0
            D = PG;
            gtd = sum(G(:) .* D(:));
        end
        if norm(D, 'fro') == 0
            break;
        end

        ref = max(values(max(1, end - memory + 1):end));
        step = 1;
        accepted = false;
        for ls = 1:max_linesearch
            Xtrial = project_box(X + step * D);
            [trial_value, trial_cache] = penalty_value( ...
                Xtrial, obj_fun, n, mu, delta);
            func_evals = func_evals + 1;
            if trial_value <= ref + armijo * step * gtd
                accepted = true;
                break;
            end
            step = backtrack * step;
        end
        if ~accepted
            break;
        end

        Xold = X;
        Gold = G;
        X = Xtrial;
        [value, G, details] = penalty_value_grad_from_cache( ...
            X, obj_fun, mu, trial_cache);
        func_evals = func_evals + 1;
        values(end + 1, 1) = value; %#ok<AGROW>

        S = X - Xold;
        Y = G - Gold;
        sty = sum(S(:) .* Y(:));
        if sty > 0
            alpha = min(alpha_max, max(alpha_min, ...
                sum(S(:) .* S(:)) / sty));
        else
            alpha = min(alpha_max, max(alpha_min, ...
                max(1, norm(X, 'fro')) / max(norm(G, 'fro'), realmin)));
        end
    end

    PG = project_box(X - G) - X;
    info.iter = it;
    info.func_evals = func_evals;
    info.residual = max(abs(PG), [], 'all');
    info.value = value;
    info.G = G;
    info.details = details;
end

function [value, cache] = penalty_value(X, obj_fun, n, mu, delta)
    f = obj_fun(X);
    [C, F] = orthogonality_residual(X, n);
    [penalty, coeff] = huber_norm_value(F, delta);
    value = f + mu * penalty;

    if nargout > 1
        cache.C = C;
        cache.F_fro = F;
        cache.penalty_value = penalty;
        cache.coeff = coeff;
    end
end

function [value, G, details] = penalty_value_grad(X, obj_fun, n, mu, delta)
    [f, gf] = obj_fun(X);
    check_objective_output(f, gf, X);
    [C, F] = orthogonality_residual(X, n);
    [penalty, coeff] = huber_norm_value(F, delta);
    value = f + mu * penalty;
    XC = X * C;
    G = gf + 2 * mu * coeff * XC;

    if nargout > 2
        details.f = f;
        details.C = C;
        details.F_fro = F;
        details.penalty_value = penalty;
    end
end

function [value, G, details] = penalty_value_grad_from_cache( ...
        X, obj_fun, mu, cache)
%PENALTY_VALUE_GRAD_FROM_CACHE Reuse accepted line-search residual data.
    [f, gf] = obj_fun(X);
    check_objective_output(f, gf, X);
    XC = X * cache.C;
    value = f + mu * cache.penalty_value;
    G = gf + 2 * mu * cache.coeff * XC;

    details.f = f;
    details.C = cache.C;
    details.F_fro = cache.F_fro;
    details.penalty_value = cache.penalty_value;
end

function [C, F] = orthogonality_residual(X, n)
%ORTHOGONALITY_RESIDUAL Form C(X) and its Frobenius norm once.
    r = size(X, 2);
    C = X' * X;
    diagonal = 1:r+1:r*r;
    C(diagonal) = C(diagonal) - n;
    F = norm(C, 'fro');
end

function [value, coeff] = huber_norm_value(F, delta)
    if F <= delta
        value = F^2 / (2 * delta);
        coeff = 1 / delta;
    else
        value = F - delta / 2;
        coeff = 1 / F;
    end
end

function X = project_box(X)
    X = min(1, max(-1, X));
end

function X = sign_binary(X)
    X = sign(X);
    X(X == 0) = 1;
end

function info = initialize_local_search_info(enabled, f_initial)
    info.enabled = enabled;
    info.applied = false;
    info.status = 'disabled';
    info.iterations = 0;
    info.accepted_moves = 0;
    info.func_evals = 0;
    info.f_initial = f_initial;
    info.f_final = f_initial;
    info.improvement = 0;
    info.neighborhood = 'balanced switches of size two';
end

function [B, info] = balanced_switch_local_search( ...
        B, obj_fun, max_iters, decrease_tol)
%BALANCED_SWITCH_LOCAL_SEARCH Best-improvement feasibility-preserving search.
%
% For r >= 2, a move flips two entries in one column.  The two rows are
% admissible when their contributions to every affected column inner
% product cancel.  For r = 1, a single-entry flip is always feasible.

    f_current = obj_fun(B);
    if ~isscalar(f_current) || ~isfinite(f_current)
        error('ExactPenalty:InvalidObjectiveValue', ...
            'obj_fun must return a finite scalar objective value.');
    end

    info = initialize_local_search_info(true, f_current);
    info.applied = true;
    info.status = 'max_iters_reached';
    info.func_evals = 1;
    [n, r] = size(B);
    if n == r && n > 2
        % For a square Hadamard matrix, two distinct rows agree in n/2
        % columns and differ in n/2 columns. A balanced two-row switch in
        % one column would require them to have the opposite relation in
        % every other column, which is impossible when n > 2.
        info.status = 'local_optimum';
        info.f_final = f_current;
        info.improvement = 0;
        return;
    end
    if r == 1
        info.neighborhood = 'single-entry switches';
    end

    for it = 1:max_iters
        best_f = f_current;
        best_rows = [];
        best_col = [];
        required_decrease = decrease_tol * max(1, abs(f_current));

        if r == 1
            for i = 1:n
                B_trial = B;
                B_trial(i, 1) = -B_trial(i, 1);
                f_trial = obj_fun(B_trial);
                info.func_evals = info.func_evals + 1;
                if f_trial < best_f - required_decrease
                    best_f = f_trial;
                    best_rows = i;
                    best_col = 1;
                end
            end
        else
            for p = 1:r
                other_cols = [1:p-1, p+1:r];
                row_contributions = B(:, p) .* B(:, other_cols);
                for i = 1:n-1
                    for j = i+1:n
                        if all(row_contributions(i, :) + ...
                                row_contributions(j, :) == 0)
                            B_trial = B;
                            B_trial([i, j], p) = -B_trial([i, j], p);
                            f_trial = obj_fun(B_trial);
                            info.func_evals = info.func_evals + 1;
                            if f_trial < best_f - required_decrease
                                best_f = f_trial;
                                best_rows = [i, j];
                                best_col = p;
                            end
                        end
                    end
                end
            end
        end

        if isempty(best_col)
            info.status = 'local_optimum';
            break;
        end

        B(best_rows, best_col) = -B(best_rows, best_col);
        f_current = best_f;
        info.accepted_moves = info.accepted_moves + 1;
        info.iterations = it;
    end

    info.f_final = f_current;
    info.improvement = info.f_initial - info.f_final;
end

function check_objective_output(f, g, X)
    if ~isscalar(f) || ~isfinite(f)
        error('ExactPenalty:InvalidObjectiveValue', ...
            'obj_fun must return a finite scalar objective value.');
    end
    if ~isequal(size(g), size(X)) || any(~isfinite(g(:)))
        error('ExactPenalty:InvalidObjectiveGradient', ...
            'obj_fun must return a finite gradient with the same size as X.');
    end
end

function validate_positive(value, name)
    if ~isscalar(value) || ~isfinite(value) || value <= 0
        error('ExactPenalty:InvalidOption', '%s must be positive and finite.', name);
    end
end

function validate_nonnegative(value, name)
    if ~isscalar(value) || ~isfinite(value) || value < 0
        error('ExactPenalty:InvalidOption', ...
            '%s must be nonnegative and finite.', name);
    end
end

function validate_logical_scalar(value, name)
    if ~(islogical(value) || isnumeric(value)) || ~isscalar(value) || ...
            ~isfinite(value) || (value ~= 0 && value ~= 1)
        error('ExactPenalty:InvalidOption', ...
            '%s must be a logical scalar.', name);
    end
end

function value = getopt(options, name, default)
    if isfield(options, name) && ~isempty(options.(name))
        value = options.(name);
    else
        value = default;
    end
end
