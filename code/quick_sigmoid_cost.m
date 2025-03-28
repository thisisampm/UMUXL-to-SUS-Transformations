function cost = quick_sigmoid_cost(p, A, A2, SUS, w)
% Cost = RMSE - w * |slope| for sigmoid A3 = L / (1 + exp(-k*(A - x0))) + b

L = p(1);
k = p(2);
x0 = p(3);
b = p(4);

if L <= 0 || abs(k) > 10 || isnan(x0) || isnan(b)
    cost = Inf;
    return;
end

A3 = L ./ (1 + exp(-k * (A - x0))) + b;

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
