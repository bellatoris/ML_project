%function prj()
close all; clear;

my_data = readtable('data.csv');

rows = isnan(my_data.shot_made_flag);
rows2 = isfinite(my_data.shot_made_flag);
train_data = my_data(rows2,:);
test_data = my_data(rows,:);


my_data.time_remaining = 60 * my_data.minutes_remaining + my_data.seconds_remaining;

%end
