function logloss = loss(act, pred)
epsilon = 1e-15;
pred = max(epsilon, pred);
pred = min(1-epsilon, pred);
ll = sum(act.*log(pred) + (1-act).*log(1-pred));
ll = ll * -1.0/length(act);
logloss = ll;
end