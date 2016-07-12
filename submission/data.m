close all; clear;
% random set
my_data = readtable('data.csv');

rows = isnan(my_data.shot_made_flag);
rows2 = isfinite(my_data.shot_made_flag);

time_remaining =60 * my_data.minutes_remaining + my_data.seconds_remaining;
[C, ia, my_data.action_type_num] = unique(my_data.action_type);
[C, ia, my_data.combined_shot_type_num] = unique(my_data.combined_shot_type);
[C, ia, my_data.season_num] = unique(my_data.season);
[C, ia, my_data.shot_type_num] = unique(my_data.shot_type);
[C, ia, my_data.shot_zone_area_num] = unique(my_data.shot_zone_area);
[C, ia, my_data.shot_zone_basic_num] = unique(my_data.shot_zone_basic);
[C, ia, my_data.opponent_num] = unique(my_data.matchup);
[C, ia, my_data.shot_zone_range_num] = unique(my_data.shot_zone_range);
my_data.last_moment = time_remaining < 3;
% my_data.shot_distance(my_data.shot_distance>30) = 30;

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
my_data.action7 = actions(:,7);
my_data.action8 = actions(:,8);
my_data.action9 = actions(:,9);
my_data.action10 = actions(:,10);
my_data.action11 = actions(:,11);
my_data.action12 = actions(:,12);
my_data.action13 = actions(:,13);
my_data.action14 = actions(:,14);
my_data.action15 = actions(:,15);
my_data.action16 = actions(:,16);
my_data.action17 = actions(:,17);

writetable(my_data, 'my_data.csv');