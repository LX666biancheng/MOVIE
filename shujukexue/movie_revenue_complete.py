import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv(r'C:\Users\曹中洋\Downloads\tmdb_movies_preprocessed.csv')

print("="*80)
print("TMDB 电影票房预测 - 完整分析版")
print("="*80)
print(f"数据集大小: {df.shape[0]} 部电影, {df.shape[1]} 个特征")

feature_cols = [
    'budget', 'popularity', 'vote_average', 'vote_count', 'runtime',
    'release_year', 'release_month', 'genres_count', 'keywords_count',
    'production_companies_count', 'cast_count', 'crew_count',
    'female_cast_count', 'male_cast_count', 'has_homepage', 'has_tagline'
]

genre_cols = [col for col in df.columns if col.startswith('genre_')]
lang_cols = [col for col in df.columns if col.startswith('lang_')]
feature_cols.extend(genre_cols)
feature_cols.extend(lang_cols)

existing_features = [col for col in feature_cols if col in df.columns]
print(f"\n使用的特征数量: {len(existing_features)}")
print("\n主要特征列表:")
for feature in existing_features[:15]:
    print(f"  - {feature}")
if len(existing_features) > 15:
    print(f"  ... 还有 {len(existing_features) - 15} 个特征")

X = df[existing_features].copy()
y = df['revenue'].copy()

X = X.fillna(X.median())
y = y.fillna(y.median())

print(f"\n目标变量: revenue (票房收入)")
print(f"特征数量: {len(existing_features)}")
print(f"样本数量: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*80)
print("一、模型原理说明")
print("="*80)
print("\n1. 线性回归 (Linear Regression)")
print("   原理: 假设特征与票房呈线性关系，通过最小二乘法拟合")
print("   优点: 计算快、可解释性强、适合初步分析")
print("   缺点: 难以捕捉非线性关系")

print("\n2. 随机森林 (Random Forest)")
print("   原理: 集成多个决策树，通过投票方式预测")
print("   优点: 抗过拟合、能处理非线性关系、特征重要性分析")
print("   缺点: 计算量大")

print("\n3. 梯度提升树 (Gradient Boosting)")
print("   原理: 逐步构建决策树，每棵树学习前一棵树的误差")
print("   优点: 预测精度高、能处理复杂关系")
print("   缺点: 调参复杂")

print("\n" + "="*80)
print("二、模型训练与评估")
print("="*80)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_r2 = -float('inf')
predictions = {}

for model_name, model in models.items():
    print(f"\n训练 {model_name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    predictions[model_name] = y_pred

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results[model_name] = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'model': model
    }

    print(f"{model_name} 评估结果:")
    print(f"  MAE (平均绝对误差): ${mae:,.2f}")
    print(f"  RMSE (均方根误差): ${rmse:,.2f}")
    print(f"  R² (决定系数): {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = model_name

print("\n" + "="*80)
print("三、模型对比")
print("="*80)
comparison_df = pd.DataFrame({
    'MAE': [results[m]['MAE'] for m in models],
    'RMSE': [results[m]['RMSE'] for m in models],
    'R2': [results[m]['R2'] for m in models]
}, index=models.keys())
print(comparison_df)

print(f"\n最佳模型: {best_model_name} (R² = {best_r2:.4f})")

print("\n" + "="*80)
print("四、生成可视化图表")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

ax1 = axes[0, 0]
comparison_df['R2'].plot(kind='bar', color=['#1f77b4', '#2ca02c', '#ff7f0e'], ax=ax1)
ax1.set_title('模型 R² 得分对比', fontsize=14, fontweight='bold')
ax1.set_ylabel('R² Score')
ax1.set_ylim(0, 1)
ax1.tick_params(axis='x', rotation=0)

ax2 = axes[0, 1]
best_pred = predictions[best_model_name]
ax2.scatter(y_test, best_pred, alpha=0.5, color='#1f77b4')
min_val = min(y_test.min(), best_pred.min())
max_val = max(y_test.max(), best_pred.max())
ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax2.set_title(f'{best_model_name} 预测值 vs 真实值', fontsize=14, fontweight='bold')
ax2.set_xlabel('真实票房 ($)')
ax2.set_ylabel('预测票房 ($)')

ax3 = axes[1, 0]
residuals = y_test - best_pred
ax3.scatter(best_pred, residuals, alpha=0.5, color='#2ca02c')
ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
ax3.set_title(f'{best_model_name} 残差分析', fontsize=14, fontweight='bold')
ax3.set_xlabel('预测票房 ($)')
ax3.set_ylabel('残差 (真实 - 预测)')

if hasattr(best_model, 'feature_importances_'):
    ax4 = axes[1, 1]
    importances = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': existing_features,
        'Importance': importances
    }).sort_values('Importance', ascending=True)
    
    top_features = feature_importance.tail(15)
    ax4.barh(range(len(top_features)), top_features['Importance'], color='#ff7f0e')
    ax4.set_yticks(range(len(top_features)))
    ax4.set_yticklabels(top_features['Feature'])
    ax4.set_title('特征重要性 (Top 15)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('重要性得分')
else:
    ax4 = axes[1, 1]
    corr_cols = existing_features[:10] + ['revenue']
    corr_cols = [col for col in corr_cols if col in df.columns]
    sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax4)
    ax4.set_title('特征相关性矩阵', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('movie_revenue_analysis.png', dpi=300, bbox_inches='tight')
print("综合分析图表已保存: movie_revenue_analysis.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(y, bins=30, kde=True, color='#1f77b4')
plt.title('票房收入分布', fontsize=14, fontweight='bold')
plt.xlabel('票房收入 ($)')
plt.ylabel('电影数量')
plt.tight_layout()
plt.savefig('revenue_distribution.png', dpi=300)
print("票房分布图已保存: revenue_distribution.png")
plt.close()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='budget', y='revenue', alpha=0.6, color='#2ca02c')
plt.title('预算 vs 票房关系', fontsize=14, fontweight='bold')
plt.xlabel('预算 ($)')
plt.ylabel('票房 ($)')
plt.tight_layout()
plt.savefig('budget_vs_revenue.png', dpi=300)
print("预算 vs 票房图已保存: budget_vs_revenue.png")
plt.close()

with open('movie_revenue_model_complete.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'features': existing_features,
        'target': 'revenue',
        'results': results
    }, f)

print("\n" + "="*80)
print("五、模型误差来源与局限性分析")
print("="*80)
print("\n误差来源:")
print("  1. 特征缺失: 缺少导演影响力、演员知名度等关键特征")
print("  2. 宣传营销: 未考虑广告投入、市场热度等外部因素")
print("  3. 时效性: 电影市场趋势随时间变化")
print("  4. 异常值: 超级大片或票房惨淡的电影难以准确预测")

print("\n模型局限性:")
print("  1. 数据驱动: 预测完全依赖历史数据，无法预测创新型电影")
print("  2. 无法处理突发事件: 如演员丑闻、社会事件等")
print("  3. 因果关系模糊: 模型能预测相关性，但无法解释因果性")

print("\n" + "="*80)
print("六、示例预测")
print("="*80)

sample = X_test.iloc[:3].copy()
sample_scaled = scaler.transform(sample)
predicted_revenue = best_model.predict(sample_scaled)

for i in range(len(sample)):
    actual = y_test.iloc[i]
    predicted = predicted_revenue[i]
    print(f"\n电影 {i+1}:")
    print(f"  预算: ${df.loc[X_test.index[i], 'budget']:,.0f}")
    print(f"  评分: {df.loc[X_test.index[i], 'vote_average']:.1f}/10")
    print(f"  真实票房: ${actual:,.2f}")
    print(f"  预测票房: ${predicted:,.2f}")
    print(f"  误差: ${abs(actual - predicted):,.2f} ({abs(actual - predicted)/actual*100:.1f}%)")

print("\n" + "="*80)
print("分析完成！")
print("="*80)
print("\n生成的文件:")
print("  - movie_revenue_analysis.png (综合分析图)")
print("  - revenue_distribution.png (票房分布图)")
print("  - budget_vs_revenue.png (预算-票房图)")
print("  - movie_revenue_model_complete.pkl (训练好的模型)")
