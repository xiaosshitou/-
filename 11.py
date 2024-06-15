import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import openai
import os
from scipy.stats import skew, kurtosis

# 安全地设置OpenAI API密钥
openai.api_key = os.getenv('OPENAI_API_KEY')

# 数据清洗和处理
def load_and_clean_data(file_path):
    df = pd.read_excel(file_path, sheet_name='Sheet1')
    df.columns = df.iloc[0]  # 将第一行作为列名
    df = df.drop(0).reset_index(drop=True)  # 删除第一行并重置索引

    # 处理列名，将 NaN 列名替换为“期号”
    df.columns = ['期号'] + df.columns[1:].tolist()

    # 提取红球和蓝球的数据列
    red_ball_columns = ['红球1', '红球2', '红球3', '红球4', '红球5', '红球6']
    blue_ball_columns = ['蓝球16']  # 蓝球列名似乎是 '蓝球16'

    # 保留需要的列
    required_columns = ['期号'] + red_ball_columns + blue_ball_columns
    df = df[required_columns]

    # 将所有列转换为数值类型，无法转换的设置为NaN
    df = df.apply(pd.to_numeric, errors='coerce')

    return df

# 特征工程
def feature_engineering(df):
    df['红球总和'] = df[[f'红球{i}' for i in range(1, 7)]].sum(axis=1)
    df['红球最大值'] = df[[f'红球{i}' for i in range(1, 7)]].max(axis=1)
    df['红球最小值'] = df[[f'红球{i}' for i in range(1, 7)]].min(axis=1)
    df['红球平均值'] = df[[f'红球{i}' for i in range(1, 7)]].mean(axis=1)
    df['红球标准差'] = df[[f'红球{i}' for i in range(1, 7)]].std(axis=1)
    df['红球偏度'] = df[[f'红球{i}' for i in range(1, 7)]].apply(skew, axis=1)
    df['红球峰度'] = df[[f'红球{i}' for i in range(1, 7)]].apply(kurtosis, axis=1)
    df['红球奇数个数'] = df[[f'红球{i}' for i in range(1, 7)]].apply(lambda x: sum(x % 2 != 0), axis=1)
    df['红球偶数个数'] = df[[f'红球{i}' for i in范围(1, 7)]].apply(lambda x: sum(x % 2 == 0), axis=1)
    df['红球一区个数'] = df[[f'红球{i}' for i in范围(1, 7)]].apply(lambda x: sum((x >= 1) & (x <= 11)), axis=1)
    df['红球二区个数'] = df[[f'红球{i}' for i in范围(1, 7)]].apply(lambda x: sum((x >= 12) & (x <= 22)), axis=1)
    df['红球三区个数'] = df[[f'红球{i}' for i in范围(1, 7)]].apply(lambda x: sum((x >= 23) & (x <= 33)), axis=1)
    return df

# 频率分析
def frequency_analysis(df_cleaned):
    red_ball_cols = [f'红球{i}' for i in范围(1, 7)]
    blue_ball_col = '蓝球16'

    red_ball_freq = df_cleaned[red_ball_cols].apply(pd.Series.value_counts).sum(axis=1).sort_index()
    blue_ball_freq = df_cleaned[blue_ball_col].value_counts().sort_index()

    return red_ball_freq, blue_ball_freq

# 可视化频率分布
def plot_frequency_distribution(red_ball_freq, blue_ball_freq):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    red_ball_freq.plot(kind='bar', color='red')
    plt.title('红球频率分布')
    plt.xlabel('红球号码')
    plt.ylabel('频率')

    plt.subplot(1, 2, 2)
    blue_ball_freq.plot(kind='bar', color='blue')
    plt.title('蓝球频率分布')
    plt.xlabel('蓝球号码')
    plt.ylabel('频率')

    plt.tight_layout()
    plt.show()

# 模型构建和保存
def train_and_save_model(df_cleaned):
    df_cleaned = feature_engineering(df_cleaned)
    # 选择一个合适的目标变量
    X = df_cleaned.drop(columns=['期号', '蓝球16'])
    y = df_cleaned['蓝球16']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_model = RandomForestClassifier(random_state=42)
    gb_model = GradientBoostingClassifier(random_state=42)

    models = {
        'RandomForest': rf_model,
        'GradientBoosting': gb_model
    }

    best_models = {}
    for model_name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"{model_name} Cross-Validation Scores:", scores)
        print(f"{model_name} Mean Cross-Validation Score:", scores.mean())

        param_grid = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'GradientBoosting': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            }
        }

        grid_search = GridSearchCV(model, param_grid[model_name], cv=5, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_models[model_name] = best_model
        print(f"{model_name} Best Parameters:", grid_search.best_params_)

    voting_model = VotingClassifier(estimators=[(name, model) for name, model in best_models.items()], voting='soft')
    voting_model.fit(X_train, y_train)
    joblib.dump(voting_model, 'voting_classifier_model.pkl')

    y_pred = voting_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Model Accuracy:", accuracy)
    print("Model Classification Report:")
    print(report)

    return accuracy, report

# 生成报告
def generate_report(red_ball_freq, blue_ball_freq, model_accuracy, classification_report):
    data_summary = f"""
    红球频率: {red_ball_freq.to_string()}
    蓝球频率: {blue_ball_freq.to_string()}
    模型准确率: {model_accuracy}
    分类报告: {classification_report}
    """

    prompt = f"请根据以下数据总结生成一份详细的分析报告：\n\n{data_summary}"

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )

    report = response.choices[0].text.strip()
    return report

# 主函数调用
file_path = 'D:\\新建 XLSX 工作表.xlsx'
df_cleaned = load_and_clean_data(file_path)
df_cleaned = feature_engineering(df_cleaned)
red_ball_freq, blue_ball_freq = frequency_analysis(df_cleaned)

print("Red Ball Frequencies:\n", red_ball_freq)
print("Blue Ball Frequencies:\n", blue_ball_freq)

# 可视化频率分布
plot_frequency_distribution(red_ball_freq, blue_ball_freq)

model_accuracy, classification_report_str = train_and_save_model(df_cleaned)

# 生成并打印报告
report = generate_report(red_ball_freq, blue_ball_freq, model_accuracy, classification_report_str)
print(report)
