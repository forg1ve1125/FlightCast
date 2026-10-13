# 测试集评估报告（2023）
- 存档: `20261005_123222_quick`
- 评估时间: 2026-10-05T12:33:17
- 注意: 测试集只能使用一次；调参请用 2022 验证集

## cls_arr_delay (catboost)
- **test_metrics@0.5**: 准确率=0.596, 精确率=0.3086, 召回率=0.6082, F1=0.4095, AUC=0.6403, 混淆矩阵 TP=63654 FP=142599 TN=207156 FN=41005
- **test_metrics@tuned**: 准确率=0.2828, 精确率=0.2393, 召回率=0.9705, F1=0.384, AUC=0.6403, 混淆矩阵 TP=101570 FP=322829 TN=26926 FN=3089

## cls_arr_delay (lightgbm)
- **test_metrics@0.5**: 准确率=0.7697, 精确率=0.0, 召回率=0.0, F1=0.0, AUC=0.6272, 混淆矩阵 TP=0 FP=0 TN=349755 FN=104659
- **test_metrics@tuned**: 准确率=0.5514, 精确率=0.2921, 召回率=0.6659, F1=0.4061, AUC=0.6272, 混淆矩阵 TP=69696 FP=168886 TN=180869 FN=34963

## cls_dep_delay (catboost)
- **test_metrics@0.5**: 准确率=0.6157, 精确率=0.3229, 召回率=0.6269, F1=0.4262, AUC=0.6666, 混淆矩阵 TP=64870 FP=136031 TN=214906 FN=38607
- **test_metrics@tuned**: 准确率=0.3187, 精确率=0.2447, 召回率=0.9543, F1=0.3895, AUC=0.6666, 混淆矩阵 TP=98747 FP=304873 TN=46064 FN=4730

## cls_dep_delay (lightgbm)
- **test_metrics@0.5**: 准确率=0.7723, 精确率=0.0, 召回率=0.0, F1=0.0, AUC=0.6558, 混淆矩阵 TP=0 FP=0 TN=350937 FN=103477
- **test_metrics@tuned**: 准确率=0.5781, 精确率=0.305, 召回率=0.6669, F1=0.4186, AUC=0.6558, 混淆矩阵 TP=69013 FP=157252 TN=193685 FN=34464

## reg_arr_delay (catboost)
- **延误航班回归**: MAE=35.33min, RMSE=88.15min, R²=-0.086
- **两段式合成期望延误 MAE**: 3.62min

## reg_arr_delay (lightgbm)
- **延误航班回归**: MAE=35.41min, RMSE=88.15min, R²=-0.0861
- **两段式合成期望延误 MAE**: 1.97min

## reg_dep_delay (catboost)
- **延误航班回归**: MAE=34.85min, RMSE=87.37min, R²=-0.0769
- **两段式合成期望延误 MAE**: 3.69min

## reg_dep_delay (lightgbm)
- **延误航班回归**: MAE=34.97min, RMSE=87.65min, R²=-0.0838
- **两段式合成期望延误 MAE**: 1.94min
