# lightgbm-binary

基于 LightGBM 训练二分类模型。使用的数据集是 [adult](https://archive.ics.uci.edu/dataset/2/adult)

🚀 本仓库的亮点：

- [x] 使用 Optuna 做超参数寻优
- [x] 使用 Treelite 做推理加速
- [x] 使用 Graphviz 做决策树模型可视化
- [x] 使用 `scale_pos_weight` 参数，解决样本不均衡问题
- [x] 使用自编写的 **自适应学习率衰减技术** 提高 AUC，详见 `util.AdaptiveLearningRate`
- [x] 对由标签概率生成标签的阈值寻优 (`y_pred -> y`)，详见 `util.gen_threshold`
- [x] 一站式评估函数，可一次性输出多种指标，包括 accuracy, precision, recall, f1_score, auc, cross-entropy loss, confusion matrix，详见 `util.eval_binary`

✨ 感谢 [Kimi](https://kimi.moonshot.cn/) 在学习过程中提供的无私帮助～

> **Note**: `requirements.txt` 文件列出了当前依赖的部分库版本。
> 
> 如果你在运行过程中遇到了错误，可以尝试执行以下命令，以使用指定的库版本：
> 
> ```
> pip install -r requirements.txt
> ```

## 一、数据可视化

1. 导入数据
2. 统计描述
    - `describe()` 方法
    - `info()` 方法
    - 统计各字段枚举值数量
    - 查看字段下所有枚举值
    - 查看空值个数
3. 可视化
    - 标号的值的比例
    - 小提琴图 (Violin Plot)

## 二、预处理与特征选择

1. 预处理
    - 标签编码
    - 更好的编码方式？
2. 初次训练
    - 使用 `lgb.LGBMClassifier` 进行训练
    - 使用原生 API 进行训练
3. 简单评估
4. 模型存储与导入
    - 模型存储
    - 模型导入
5. 特征选择
    - 计算特征的重要程度
    - 多次实验求均值

## 三、超参数微调

1. 简单的例子
2. 稍微复杂的例子：随机森林
    - 导入数据
    - one-hot 编码
    - 训练
    - 评估
    - 超参数微调
    - 使用微调后的超参数训练
3. LightGMB 超参数微调
    - 单次训练
    - 超参数寻优
    - 使用微调后的超参数训练
4. 学习率衰减
    - 指数衰减
    - 自适应衰减

## 四、训练与评估

1. 基础模型
2. N 折交叉验证
3. 超参寻优：N 折交叉验证
4. 使用微调后的超参数训练
5. 阈值选择 (`y_pred -> y`)
    - 使用 `np.rint`
    - 尝试：考虑原标号的分布
        - 训练标号的分布情况
        - 预测标号的概率密度函数 (PDF)
        - 预测标号的概率分布函数 (CDF)
        - 当 CDF = `训练数据 0 标号比例` 时反推阈值
    - 使用 optuna 寻优
6. 评估
    - 混淆矩阵
    - 准确率
    - 精确率和召回率
    - `f1` 值
    - `log_loss` 交叉熵损失
    - ROC 曲线 与 AUC
    - 一站式评估函数

## 五、加速推理

1. 导入模型和数据
2. 使用 Treelite 加速推理
    - 推理速度
    - 推理准度
    - 通用函数

## 六、部署

1. 模型训练
2. 模型部署
    - 离线部署
    - 在线部署

## 七、模型可视化

1. 创建可交互的可视化界面
2. 将决策树存成 PNG / PDF

