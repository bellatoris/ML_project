close all; clear;clc;

my_data = readtable('data.csv');

rows = isnan(my_data.shot_made_flag);
rows2 = isfinite(my_data.shot_made_flag);

[~, ~, my_data.action_type_num] = unique(my_data.action_type);
[~, ~, my_data.combined_shot_type_num] = unique(my_data.combined_shot_type);
[~, ~, my_data.season_num] = unique(my_data.season);
[~, ~, my_data.shot_type_num] = unique(my_data.shot_type);
[~, ~, my_data.shot_zone_area_num] = unique(my_data.shot_zone_area);
[~, ~, my_data.shot_zone_basic_num] = unique(my_data.shot_zone_basic);
[~, ~, my_data.shot_zone_range_num] = unique(my_data.shot_zone_range);
% [~, ~, my_data.opponent_num] = unique(my_data.opponent);
[~, ~, my_data.matchup_num] = unique(my_data.matchup);
my_data.last_moment = my_data.seconds_remaining < 3;
%% home / away information
ndata = size(my_data,1);
hna = zeros(ndata,1);
for m = 1:ndata
    matchup = my_data{m,{'matchup'}}{1};
    if ~isempty(strfind(matchup,'@'));  hna(m) = 1;
    else hna(m) = 0;
    end
end
my_data.hna = hna;
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
% my_data.action17 = actions(:,17);
%%
train_data_tmp  = my_data(rows2,[9,10,13,14,26:end]);
test_data_tmp   = my_data(rows,[9,10,13,14,26:end]);
train_label_tmp = my_data(rows2, 15);

train_data  = table2array(train_data_tmp);
test_data   = table2array(test_data_tmp);
train_label = table2array(train_label_tmp);
num = 20000;
%% for classification, label : -1, 1
train_label = train_label*2-1;
%%
load('ran');
data1 = train_data(ran(1:num),:);
data2 = train_data(ran(num+1:end),:);
label1 = train_label(ran(1:num),:);
label2 = train_label(ran(num+1:end),:);

MEX_PATH = '../build';
addpath(MEX_PATH);

opts = [];
opts.loss = 'logloss'; % can be logloss or exploss
opts.shrinkageFactor = 0.01;
opts.subsamplingFactor = 0.9;
opts.maxTreeDepth = uint32(2);  % this was the default before customization
opts.randSeed = uint32(0);
opts.mtry = uint32(5);

numIters = 4050;
tic;
model       = SQBMatrixTrain(single(data1), label1, uint32(numIters), opts);
model_all   = SQBMatrixTrain(single(train_data), train_label, uint32(numIters), opts);
toc;

pred_label1 = SQBMatrixPredict(model, single(data1));
pred_label2 = SQBMatrixPredict(model, single(data2));
%%
fts = 1.9:0.001:2.1;
llin = zeros(size(fts));
llot = llin;
for m = 1:length(fts)
    ft = fts(m);
    pred_label1t = (1./(1+exp(-ft*pred_label1)));
    pred_label2t = (1./(1+exp(-ft*pred_label2)));
    logloss_in  = loss((label1+1)/2,pred_label1t);
    logloss_out = loss((label2+1)/2,pred_label2t);
llin(m) = logloss_in;
llot(m) = logloss_out;
end
a = find(llot == min(llot));
ft = fts(a);
% Ein  = sum((pred_label1t - (label1+1)/2).^2)
% Eout = sum((pred_label2t - (label2+1)/2).^2)
plot(fts,llot,'-s')
%%
pred_label1t = (1./(1+exp(-ft*pred_label1)));
pred_label2t = (1./(1+exp(-ft*pred_label2)));
logloss_in  = loss((label1+1)/2,pred_label1t)
logloss_out = loss((label2+1)/2,pred_label2t)
plot(pred_label1t)
%%
test_label = SQBMatrixPredict(model_all, single(test_data));
test_label = 1./(1+exp(-ft*test_label));
% result = readtable('sample_submission.csv');
% result = result(:,1);
% result.shot_made_flag = test_label;
% writetable(result, 'sub.csv');