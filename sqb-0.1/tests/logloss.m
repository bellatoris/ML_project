close all; clear; format shortg;

my_data = readtable('data.csv');

rows = isnan(my_data.shot_made_flag);
rows2 = isfinite(my_data.shot_made_flag);

my_data.time_remaining = 60 * my_data.minutes_remaining + my_data.seconds_remaining;
[C, ia, my_data.action_type_num] = unique(my_data.action_type);
[C, ia, my_data.combined_shot_type_num] = unique(my_data.combined_shot_type);
[C, ia, my_data.season_num] = unique(my_data.season);
[C, ia, my_data.shot_type_num] = unique(my_data.shot_type);
[C, ia, my_data.shot_zone_area_num] = unique(my_data.shot_zone_area);
[C, ia, my_data.shot_zone_basic_num] = unique(my_data.shot_zone_basic);
[C, ia, my_data.shot_zone_range_num] = unique(my_data.shot_zone_range);
[C, ia, my_data.opponent_num] = unique(my_data.opponent);

train_data = my_data(rows2,[6,7,10,14,26,27,28,29,30,31,32,33,34]);
test_data = my_data(rows,[6,7,10,14,26,27,28,29,30,31,32,33,34]);
train_label = my_data(rows2, 15);

train_data =table2array(train_data);
test_data =table2array(test_data);
train_label = table2array(train_label);

num = 20000
train1 = train_data(1:num,:);
train2 = train_data(num+1:end,:);
label1 = train_label(1:num,:);
label1 = 2*label1 - 1;
label2 = train_label(num+1:end,:);
label2 = 2*label2 - 1;

MEX_PATH = '../build';
addpath(MEX_PATH);

opts = [];
opts.loss = 'logloss'; % can be logloss or exploss
opts.shrinkageFactor = 0.01;
opts.subsamplingFactor = 0.5;
opts.maxTreeDepth = uint32(5);  % this was the default before customization
opts.randSeed = uint32(0);

numIters = 269;
tic;
model = SQBMatrixTrain(single(train1), label1, uint32(numIters), opts);
toc;

out_label = SQBMatrixPredict(model, single(train2));
% outErr = sum( (out_label > 0) ~= (label2 > 0))/length(out_label)
in_label = SQBMatrixPredict(model, single(train1));
% insampleErr = sum( (in_label > 0) ~= (label1 > 0))/length(in_label)

outErr = loss((label2+1)/2, (out_label+1)/2)
insampleErr = loss((label1+1)/2, (in_label+1)/2)

test_label = SQBMatrixPredict(model, single(test_data));
test_label = (test_label + 1)/2;


result = readtable('sample_submission.csv');
result = result(:,1);
result.shot_made_flag = test_label;
writetable(result, 'sub.csv');