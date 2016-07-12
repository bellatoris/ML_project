xgb_data = readtable('submission_xgboost.csv');
xgb = xgb_data.shot_made_flag;
k = 0.1
last_result = -0.13*test_label + 1.17*xgb

result = readtable('sample_submission.csv');
result = result(:,1);
result.shot_made_flag =  last_result;
writetable(result, 'sub.csv');