function out = demo_projection(n, r, seed)
%DEMO_PROJECTION Project a Gaussian matrix onto the binary orthogonal set.
%   out = demo_projection() uses n=128, r=8 and seed=101.
%   out = demo_projection(n,r,seed) changes the problem size and seed.
%   Uses only base MATLAB and the solver in this directory.

if nargin < 1 || isempty(n), n = 128; end
if nargin < 2 || isempty(r), r = 8; end
if nargin < 3 || isempty(seed), seed = 101; end
validateattributes(n, {'numeric'}, {'scalar','integer','positive','finite'});
validateattributes(r, {'numeric'}, {'scalar','integer','positive','finite','<=',n});
validateattributes(seed, {'numeric'}, {'scalar','integer','nonnegative','finite'});
% Power-of-two row counts admit a Sylvester Hadamard construction, so the
% feasible set is nonempty for every r <= n in this demo.
if 2^round(log2(n)) ~= n
    error('SPBO:DemoSize', 'Use a power-of-two n to ensure a nonempty feasible set.');
end
previous_rng = rng;
restore_rng = onCleanup(@() rng(previous_rng)); %#ok<NASGU>
rng(seed, 'twister');
C = randn(n, r);
X0 = min(1, max(-1, randn(n, r)));
objective = @(X) projection_objective(X, C);
opts = struct('eps_opt', 1e-3, 'eps_feas', 1e-3, ...
    'max_stages', 200, 'max_inner', 250, 'mu_auto', true, ...
    'mu_balance', 1.0, 'mu_mult', 1.5, ...
    'local_search_enabled', true, 'verbose', 0);
start_time = tic;
out = solve_bop_exact_penalty_box(objective, X0, opts);
elapsed = toc(start_time);
fprintf('SPBO projection: n=%d, r=%d, seed=%d\n', n, r, seed);
fprintf('Status: %s\n', out.status);
fprintf('Time: %.3f seconds\n', elapsed);
fprintf('Continuous orthogonality residual: %.6e\n', out.F_fro);
fprintf('Sign-rounded objective: %.9g\n', out.f_sign);
fprintf('Final binary objective: %.9g\n', out.f_round);
fprintf('Final binary orthogonality residual: %.6e\n', out.F_round);
fprintf('Feasible binary output: %d\n', out.rounded_feasible);
fprintf('Local search: %s (%d accepted moves)\n', ...
    out.local_search.status, out.local_search.accepted_moves);
end

function [f, g] = projection_objective(X, C)
D = X - C;
f = sum(D(:).^2);
if nargout > 1, g = 2 * D; end
end
