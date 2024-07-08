import os
import numpy as np
import pandas as pd
import scipy.stats
import sklearn.metrics
import sklearn.preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import swifter
import optuna
import json


# 将 数据目录 和 数据相对地址 合成 数据绝对地址
def gen_abspath(directory_path, relative_path):
    abs_directory = os.path.abspath(directory_path)
    return os.path.join(abs_directory, relative_path)


# 读 CSV 的函数，添加了自己的默认参数
def read_csv(file_path, sep=',', header=0, on_bad_lines='warn', encoding='utf-8', dtype=None):
    return pd.read_csv(file_path,
                    header=header,
                    sep=sep,
                    on_bad_lines=on_bad_lines,
                    encoding=encoding,
                    dtype=dtype)


# 统计各字段枚举值的数量
def value_counts(df):
    val_cnt_list = []
    for col in df.columns:
        val_cnt_list.append(len(df[col].value_counts()))

    # 初始化 DataFrame 以存储统计结果
    return pd.DataFrame({
        'col_name': df.columns,
        'val_cnt': val_cnt_list
    })


# 将类别特征 (String) 编码成整数 (int)
def label_encoder(df):
    cat_feats = [col for col in df.columns if df[col].dtypes == np.dtype('object')]
    for col in cat_feats:
        df[col] = sklearn.preprocessing.LabelEncoder().fit_transform(df[col])
    return df


# 改进的 label_encoder：去除特征值前后空格，返回 特征与标签映射 和 类别特征名
def label_encoder_trim(df):
    kkv_dict = dict()
    cat_feats = [col for col in df.columns if df[col].dtypes == np.dtype('object')]
    for col in cat_feats:
        df[col] = df[col].swifter.apply(lambda e: e.strip())

        le = sklearn.preprocessing.LabelEncoder()
        df[col] = le.fit_transform(df[col])
        kkv_dict[col] = {label: index for index, label in enumerate(le.classes_)}
    return df, kkv_dict, cat_feats


# 将 label_encoder 编码保存成 json 文件
def save_label_encoder(kkv, file_path):
    content = json.dumps(kkv, indent=4)
    with open(file_path, 'w') as f:
        f.write(content)


# 加载 kkv 文件，并用 kkv 对新数据做编码
def load_label_encoder(df, kkv_file_path):
    with open(kkv_file_path, 'r') as f:
        kkv = json.load(f)

    for k, kv in kkv.items():
        df[k] = df[k].swifter.apply(lambda e: kv.get(e.strip(), -1))  

    return df, kkv


# 解决样本数据倾斜
def gen_scale_pos_weight(y_train):
    # assuming dataset is highly imbalanced
    total_positive_samples = sum(y_train)
    total_negative_samples = len(y_train) - sum(y_train)
    scale_pos_weight = total_negative_samples / total_positive_samples
    return scale_pos_weight


# 计算特征的重要程度
def feature_importance(gbm):
    items = [(k, v) for k, v in zip(gbm.feature_name(), gbm.feature_importance())]
    sorted_items = sorted(items, key=lambda e: e[1], reverse=True)
    for i, (k, v) in enumerate(sorted_items):
        print(f'[rank {i+1}] {k}: {v}')


# 自适应学习率衰减
class AdaptiveLearningRate:
    def __init__(self, learning_rate=0.3, decay_rate=0.9, patience=10):
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.patience = patience
        self.best_score = float('inf')
        self.wait_count = 0

    def callback(self, env):
        # 检查当前模型的性能
        score = env.evaluation_result_list[0][2]  # 假设使用 auc

        # auc 向 maximize 方向搜索
        if score > self.best_score:
            self.best_score = score
            self.wait_count = 0  # 重置等待次数
        else:
            self.wait_count += 1  # 增加等待次数

        # 如果连续 patience 次迭代性能没有提升，则衰减学习率
        if self.wait_count >= self.patience:
            pre = self.learning_rate
            self.learning_rate *= self.decay_rate
            if env.params.get('verbose', 0) >= 0:
                print(f"Learning rate ==> {self.learning_rate:.3f} (-{pre - self.learning_rate:.4f})")
            self.wait_count = 0  # 重置等待次数

        # 更新学习率
        env.model.params['learning_rate'] = self.learning_rate


# 用 y_pred 评估线性回归任务
def eval_continuous(y_true, y_pred):
    # evaluate
    mae = sklearn.metrics.mean_absolute_error(y_true, y_pred)
    mse = sklearn.metrics.mean_squared_error(y_true, y_pred)
    r2_score = sklearn.metrics.r2_score(y_true, y_pred)

    print(f'mae: {mae:.5f}')
    print(f'mse: {mse:.5f}')
    print(f'r2_score: {r2_score:.5f}')
    

# 以最大化 f1_score 为目标，寻找最优 threshold
def gen_threshold(y_true, y_pred, n_trials):

    # 临时将报警等级设为 ERROR
    verbose = optuna.logging.get_verbosity()
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    def f1_objective(trial):
        t = trial.suggest_float('threshold', 0.0, 1.0)
        y_label = [1 if e > t else 0 for e in y_pred]
        f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_label)
        return f1
    
    f1_study = optuna.create_study(direction='maximize')
    f1_study.optimize(f1_objective, n_trials=n_trials)
    best_params = f1_study.best_params
    
    # 恢复原先的报警等级
    optuna.logging.set_verbosity(verbose)

    return best_params['threshold']


# 以输出结果中负样本的比例为目标，寻找最优 threshold，参数 rate 为负样本 (label 0) 的比例
def gen_threshold_cdf(y_pred, rate, interval=100):
    xx = np.linspace(min(y_pred), max(y_pred), interval)
    kde = scipy.stats.gaussian_kde(y_pred)
    pdf = kde.evaluate(xx)
    cdf = np.cumsum(pdf) * (xx[1] - xx[0])

    px = 0
    for x, y in zip(xx, cdf):
        if y > rate:
            xa = (px + x) / 2
            break
        px = x
        
    return xa


# 用 y_pred 评估二分类任务
def eval_binary(y_true, y_pred, n_trials=200, ret=False, threshold=None):
    
    # 直接用 y_pred 可以计算的指标
    auc = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_pred)
    log_loss = sklearn.metrics.log_loss(y_true=y_true, y_pred=y_pred)
    
    # 如果阈值不存在，获取阈值
    if threshold is None:
        threshold = gen_threshold(y_true, y_pred, n_trials)

    # 必须用 y_label 计算的指标
    y_label = [1 if e > threshold else 0 for e in y_pred]

    acc = sklearn.metrics.accuracy_score(y_true=y_true, y_pred=y_label)
    precision = sklearn.metrics.precision_score(y_true=y_true, y_pred=y_label)
    recall = sklearn.metrics.recall_score(y_true=y_true, y_pred=y_label)
    f1 = sklearn.metrics.f1_score(y_true=y_true, y_pred=y_label)
    cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_label)
    tn, fp, fn, tp = cm.ravel()

    print(f'threshold: {threshold:.5f}')
    print(f'accuracy: {acc:.5f}')
    print(f'precision: {precision:.5f}')
    print(f'recall: {recall:.5f}')
    print(f'f1_score: {f1:.5f}')
    print(f'auc: {auc:.5f}')
    print(f'cross-entropy loss: {log_loss:.5f}')
    print(f'True Positive (TP): {tp}')
    print(f'True Negative (TN): {tn}')
    print(f'False Positive (FP): {fp}')
    print(f'False Negative (FN): {fn}')
    print(f'confusion matrix:\n{cm}')

    if ret:
        return y_label, threshold


# 绘制 ROC 曲线
def roc_curve(y_true, y_score):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=y_true, y_score=y_score)
    auc = sklearn.metrics.roc_auc_score(y_true=y_true, y_score=y_score)
    print(f'AUC: {auc:.5f}')

    # 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))

    # 设置图形元素
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='dashed', alpha=0.5)

    # 显示图形
    plt.show()


# 绘制混淆矩阵
def confusion_matrix(y_true, y_label):
    cm = sklearn.metrics.confusion_matrix(y_true=y_true, y_pred=y_label)
    cm_matrix = pd.DataFrame(data=cm, columns=['Predict Negative:0', 'Predict Positive:1'], 
                                 index=['Actual Negative:0', 'Actual Positive:1'])
    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')
