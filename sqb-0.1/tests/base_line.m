data_made = train_data(train_label == 1,:);
data_fail = train_data(train_label ~= 1,:);

counts_made = [];
counts_fail = [];
for m = 1:size(data_made,2)
    mins = min(train_data(:,m));
    maxs = max(train_data(:,m));
    intv = mins:1:maxs;
for n = 1:length(intv)
    counts_made(n,m) = sum(data_made(:,m)==intv(n));
    counts_fail(n,m) = sum(data_fail(:,m)==intv(n));
end
end
P = counts_made ./ (counts_made + counts_fail);
P = P - size(data_made,1)/size(train_data,1);

test_baseline = [];
for n = 1:length(pred_label2)
    bound = [];
    val = data2(n,:);
    for m = 1:size(data2,2)
        mins = min(train_data(:,m));
        maxs = max(train_data(:,m));
        intv = mins:1:maxs;
        a = find(intv == val(m));
        bound(1,m) = P(a,m);
    end
    test_baseline = [test_baseline; bound];
end
test_baseline = [test_baseline, ones(size(test_baseline,1),1)*size(data_made,1)/size(train_data,1)];

baseline = [];
for n = 1:length(pred_label1)
    bound = [];
    val = data1(n,:);
    for m = 1:size(data1,2)
        mins = min(train_data(:,m));
        maxs = max(train_data(:,m));
        intv = mins:1:maxs;
        a = find(intv == val(m));
        bound(1,m) = P(a,m);
    end
    baseline = [baseline; bound];
end
baseline = [baseline, ones(size(baseline,1),1)*size(data_made,1)/size(train_data,1)];
train_baseline = baseline;

opts = [];
opts.loss = 'logloss'; % can be logloss or exploss
opts.shrinkageFactor = 0.01;
opts.subsamplingFactor = 0.5;
opts.maxTreeDepth = uint32(5);  % this was the default before customization
opts.randSeed = uint32(0);

numIters = 300;
tic;
base_model = SQBMatrixTrain(single(train_baseline), label1, uint32(numIters), opts);
toc;

base_label = SQBMatrixPredict(base_model, single(test_baseline));
fts = 1.9:0.001:3;
llin = zeros(size(fts));
llot = llin;
for m = 1:length(fts)
    ft = fts(m);
    base_labelt = (1./(1+exp(-ft*base_label)));
    logloss_out = loss((label2+1)/2, base_labelt);
    llot(m) = logloss_out;
end
a = find(llot == min(llot));
ft = fts(a);
exp_ft = fts(a);
base_labelt = (1./(1+exp(-ft*base_label)));
base_loss = loss((label2+1)/2, base_labelt)
plot(fts,llot,'-s')

% % outErr = sum( (out_label > 0) ~= (label2 > 0))/length(out_label)
% in_label = SQBMatrixPredict(model, single(train1));
%%
llot = [];
fts = -1:0.01:1;
for m = 1:length(fts)
    ft = fts(m);
    for n = 1:length(fts)
        ft2 = fts(n);
        baselinet = ft*base_labelt + ft2*pred_label2t;
        logloss_out = loss((label2+1)/2,baselinet);
        llot(m,n) = logloss_out;
    end
end
[a,b] = find(llot == min(llot(:)));
min(min(llot))
ft = fts(a);
result_ft = ft;
ft2 = fts(b);
test_ft = ft2;
surf(llot)


%%
total_base = [];
for n = 1:length(train_label)
    bound = [];
    val = train_data(n,:);
    for m = 1:size(train_data,2)
        mins = min(train_data(:,m));
        maxs = max(train_data(:,m));
        intv = mins:1:maxs;
        a = find(intv == val(m));
        bound(1,m) = P(a,m);
    end
    total_base = [total_base; bound];
end
total_base = [total_base, ones(size(total_base,1),1)*size(data_made,1)/size(train_data,1)];

opts = [];
opts.loss = 'logloss'; % can be logloss or exploss
opts.shrinkageFactor = 0.01;
opts.subsamplingFactor = 0.5;
opts.maxTreeDepth = uint32(5);  % this was the default before customization
opts.randSeed = uint32(0);

numIters = 300;
tic;
total_model = SQBMatrixTrain(single(total_base), train_label, uint32(numIters), opts);
toc;


result_baseline = [];
for n = 1:length(test_label)
    bound = [];
    val = test_data(n,:);
    for m = 1:size(test_data,2)
        mins = min(train_data(:,m));
        maxs = max(train_data(:,m));
        intv = mins:1:maxs;
        a = find(intv == val(m));
        bound(1,m) = P(a,m);
    end
    result_baseline = [result_baseline; bound];
end
result_baseline = [result_baseline, ones(size(result_baseline,1),1)*size(data_made,1)/size(train_data,1)];

result_label = SQBMatrixPredict(total_model, single(result_baseline));
result_labelt = (1./(1+exp(-exp_ft*result_label)));
last_result = result_ft*result_labelt + test_ft*test_label;

result = readtable('sample_submission.csv');
result = result(:,1);
result.shot_made_flag =  last_result;
writetable(result, 'sub.csv');

