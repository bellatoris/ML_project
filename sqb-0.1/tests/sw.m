close all; clear;
% 0.60192 -> 0.60366
my_data = readtable('data.csv');

rows = isnan(my_data.shot_made_flag);
rows2 = isfinite(my_data.shot_made_flag);

% my_data.time_remaining =60*12*my_data.period + 60 * my_data.minutes_remaining + my_data.seconds_remaining;
[C, ia, my_data.action_type_num] = unique(my_data.action_type);
[C, ia, my_data.combined_shot_type_num] = unique(my_data.combined_shot_type);
[C, ia, my_data.season_num] = unique(my_data.season);
[C, ia, my_data.shot_type_num] = unique(my_data.shot_type);
[C, ia, my_data.shot_zone_area_num] = unique(my_data.shot_zone_area);
[C, ia, my_data.shot_zone_basic_num] = unique(my_data.shot_zone_basic);
[C, ia, my_data.opponent_num] = unique(my_data.matchup);
[C, ia, my_data.shot_zone_range_num] = unique(my_data.shot_zone_range);
% [C, ia, my_data.game_date_num] = unique(my_data.game_date);
%%
action_types = {'Alley Oop','Dunk','Layup','Bank','Driving','Fadeaway',...
                'Finger Roll','Hook','Jump','Cutting','Floating','Reverse',...
                'Pullup','Putback','Running','Tip','Step Back'}';
ndata = size(my_data,1);
actions = zeros(ndata,17);
for m = 1:ndata
    action = my_data{m,1}{1};
for n = 1:size(action_types)
    if ~isempty(strfind(action,action_types{n}))
        actions(m,n) = 1;
    end
end
end
my_data.action1 = actions(:,1);
my_data.action2 = actions(:,2);
my_data.action3 = actions(:,3);
my_data.action4 = actions(:,4);
my_data.action5 = actions(:,5);
my_data.action6 = actions(:,6);
% my_data.action7 = actions(:,7);
my_data.action8 = actions(:,8);
my_data.action9 = actions(:,9);
my_data.action10 = actions(:,10);
my_data.action11 = actions(:,11);
% my_data.action12 = actions(:,12);
my_data.action13 = actions(:,13);
my_data.action14 = actions(:,14);
my_data.action15 = actions(:,15);
my_data.action16 = actions(:,16);
my_data.action17 = actions(:,17);
%%
train_data = my_data(rows2,[9,10,13,14,26:end]);
test_data = my_data(rows,[9,10,13,14,26:end]);
train_label = my_data(rows2, 15);

train_data =table2array(train_data);
test_data =table2array(test_data);
train_label = table2array(train_label);
%%
train_data=train_data(:,1:end-1);
% ndata = size(train_data,1);
% num = randperm(ndata,20000);
mo = 1:1:size(train_data,1);
mo1 = logical(mod(mo, 5));
mo2 = logical(1-mo1);
train1 = train_data(mo1,:);
train2 = train_data(mo2,:);
label1 = train_label(mo1,:);
label2 = train_label(mo2,:);

MEX_PATH = '../build';
addpath(MEX_PATH);

opts = [];
opts.loss = 'squaredloss'; % can be logloss or exploss
opts.shrinkageFactor = 0.05;
opts.subsamplingFactor = 0.5;
opts.maxTreeDepth = uint32(3);  % this was the default before customization
opts.randSeed = uint32(0);

numIters = 400;
tic;
model       = SQBMatrixTrain(single(train1), label1, uint32(numIters), opts);
%model_all   = SQBMatrixTrain(single(train_data), train_label, uint32(numIters), opts);
toc;

r_label = SQBMatrixPredict(model, single(train2));
my_label = SQBMatrixPredict(model, single(train1));

logos = loss(label2,r_label)
logloss_insample = loss(label1,my_label)
Eout = sum((r_label - label2).^2)
Win  = sum((my_label-label1).^2)

test_label = SQBMatrixPredict(model, single(test_data));
result = readtable('sample_submission.csv');
result = result(:,1);
result.shot_made_flag = test_label;
writetable(result, 'sub.csv');