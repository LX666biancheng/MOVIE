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

df = pd.read_csv(r'C:\Users\曹中洋\Downloads\tmdb_movies_preprocessed.csv')

print("="*60)
print("TMDB Movie Revenue Prediction")
print("="*60)
print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nTarget variable 'revenue' found in dataset")

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
print(f"\nNumber of features used: {len(existing_features)}")

X = df[existing_features].copy()
y = df['revenue'].copy()

X = X.fillna(X.median())
y = y.fillna(y.median())

print(f"\nTarget variable: revenue")
print(f"Feature count: {len(existing_features)}")
print(f"Sample count: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n" + "="*60)
print("Model Training and Evaluation")
print("="*60)

print("\n" + "="*80)
print("模型原理说明")
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
print("模型训练与评估")
print("="*80)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
best_model = None
best_r2 = -float('inf')

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

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

    print(f"{model_name} Results:")
    print(f"  MAE: ${mae:,.2f}")
    print(f"  MSE: ${mse:,.2f}")
    print(f"  RMSE: ${rmse:,.2f}")
    print(f"  R2: {r2:.4f}")

    if r2 > best_r2:
        best_r2 = r2
        best_model = model
        best_model_name = model_name

print("\n" + "="*60)
print("Model Comparison")
print("="*60)
comparison_df = pd.DataFrame({
    'MAE': [results[m]['MAE'] for m in models],
    'RMSE': [results[m]['RMSE'] for m in models],
    'R2': [results[m]['R2'] for m in models]
}, index=models.keys())
print(comparison_df)

print(f"\nBest Model: {best_model_name} (R2 = {best_r2:.4f})")

plt.figure(figsize=(12, 6))
comparison_df['R2'].plot(kind='bar', color=['blue', 'green', 'orange'])
plt.title('Model R2 Score Comparison')
plt.ylabel('R2 Score')
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig('model_comparison_real.png')
plt.close()

plt.figure(figsize=(10, 10))
corr_cols = existing_features[:10] + ['revenue']
corr_cols = [col for col in corr_cols if col in df.columns]
sns.heatmap(df[corr_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix_real.png')
plt.close()

print("\n" + "="*60)
print("Feature Importance Analysis")
print("="*60)

if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': existing_features,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    plt.figure(figsize=(12, 8))
    top_features = feature_importance.head(15)
    plt.barh(range(len(top_features)), top_features['Importance'], color='orange')
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.title('Feature Importance (Top 15)')
    plt.xlabel('Importance Score')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance_real.png')
    plt.close()
    
    print("\nFeature importance plot saved as feature_importance_real.png")
else:
    print("Note: Feature importance not available for this model type")

with open('movie_revenue_model_real.pkl', 'wb') as f:
    pickle.dump({
        'model': best_model,
        'scaler': scaler,
        'features': existing_features,
        'target': 'revenue'
    }, f)

print("\n" + "="*80)
print("模型误差来源与局限性分析")
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
print("示例预测")
print("="*80)

print("\nModel saved as movie_revenue_model_real.pkl")

sample = X_test.iloc[:1].copy()
sample_scaled = scaler.transform(sample)
predicted_revenue = best_model.predict(sample_scaled)
actual_revenue = y_test.iloc[0]

print(f"\nSample Prediction:")
print(f"  Actual Revenue: ${actual_revenue:,.2f}")
print(f"  Predicted Revenue: ${predicted_revenue[0]:,.2f}")
print(f"  Difference: ${abs(actual_revenue - predicted_revenue[0]):,.2f}")