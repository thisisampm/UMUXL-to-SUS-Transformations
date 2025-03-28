function cost = quick_saturating_cost(p, A, A2, SUS, w)
% Cost = RMSE - w * |slope| for saturating transformation A3 = a*A/(A+c) + b

a = p(1);
c = p(2);
b = p(3);

% Restrict c to a stable, positive range
if c <= 1 || c > 20 || any(A + c <= 0)
    cost = Inf;
    return;
end

A3 = a * A ./ (A + c) + b;

if std(A3) < 1e-3 || any(isnan(A3)) || any(isinf(A3))
    cost = Inf;
    return;
end

rmse = sqrt(mean((A3 - SUS).^2));
diffA = A3 - A2;

if std(diffA) < 1e-3 || all(diffA == diffA(1))
    cost = Inf;
    return;
end

warnState = warning('off', 'all');
lm = fitlm(diffA, SUS);
warning(warnState);
slope = lm.Coefficients.Estimate(2);

cost = rmse - w * abs(slope);
end
