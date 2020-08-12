function [sc, pc, mse, mae] = four_metrics (y_target, y_predict)
% 多任务 多维y 的平均metrics，四种metrics 都有。
assert(isequal(size(y_target), size(y_predict)), "Size of two arrays not equal. Check size.");
sz = size(y_target);
n_tasks = sz(2);
sc = 0;
pc = 0;
mse = 0;
mae = 0;

for ii = 1:n_tasks
    % Assume prediction doesn't contains NaNs
    nan_map = isnan(y_target(:,ii));
    y1 = y_target(~nan_map,ii);
    y2 = y_predict(~nan_map,ii);
    % 实在懒得test 了，以后再看吧
    [each_sc, ~] = corr(y1, y2,'Type','Spearman');
    [each_pc, ~] = corrcoef(y1', y2');
    each_mse = mean((y1 - y2).^2);
    each_mae = mean(abs(y1 - y2));

    sc = sc + each_sc / n_tasks;
    pc = pc + each_pc(1,2) / n_tasks;
    mse = mse + each_mse / n_tasks;
    mae = mae + each_mae / n_tasks;
    
end

end
