import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.inspection import PartialDependenceDisplay


file_path = "/Users/goenchang/Consumer-Health-Product-Recommendation_Case-Study/Health_and_Personal_Care.jsonl"
df = pd.read_json(file_path, lines=True)

pd.set_option('display.max_columns', None)
df.info()
df.head()

#----Data Engineering---
user_count_rating = df.groupby('user_id')['rating'].count().rename('user_avg_health_interest')
df=df.merge(user_count_rating, on='user_id', how='left')

product_avg_ratings = df.groupby('asin')['rating'].mean().rename('product_popularity_score')
df=df.merge(product_avg_ratings, on='asin', how='left')

df['timestamp_dt'] = pd.to_datetime(df['timestamp'], unit='ms')
latest_interaction = df['timestamp_dt'].max()
df['days_since_last_user_interaction'] = (latest_interaction - df['timestamp_dt']).dt.days


#---Select Features and Target---
df['verified_purchase'] = df['verified_purchase'].astype(int)

features = [
  'user_avg_health_interest',
  'product_popularity_score',
  'days_since_last_user_interaction'
]

X = df[features]
y = df['verified_purchase']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#---Random Forest---
rf = RandomForestClassifier(random_state=42)
rf_param_grid = {
  'n_estimators':[300],
  'max_depth': [20],
  'min_samples_split':[2],
  'min_samples_leaf': [1],
  'max_features':['sqrt']
}

grid_search_rf = GridSearchCV(estimator=rf, param_grid=rf_param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search_rf.fit(X_train, y_train)
best_rf = grid_search_rf.best_estimator_


#---XGBoost---
xgb_model = XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
xgb_param_dist = {
    'n_estimators': [300],
    'max_depth': [5],
    'learning_rate': [0.1],
    'subsample': [1.0],
    'colsample_bytree': [0.7],
    'gamma': [0],
    'min_child_weight': [1]
}

random_search_xgb = RandomizedSearchCV(
    estimator=xgb_model,
    param_distributions=xgb_param_dist,
    n_iter=1,
    scoring='accuracy',
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=-1
)

random_search_xgb.fit(X_train, y_train)

best_xgb = random_search_xgb.best_estimator_

#---Evaluation---
y_pred_rf = best_rf.predict(X_test)
y_pred_xgb = best_xgb.predict(X_test)

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))

print("\nXGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))


#---Feature Importance---
#Random Forest
importances_rf = best_rf.feature_importances_
features_importance_rf = list(zip(features, importances_rf))
features_sorted_rf, importances_sorted_rf = zip(*sorted(features_importance_rf, key=lambda x: x[1], reverse=True))


plt.figure(figsize=(8.5, 1.5))
bars = plt.barh(features_sorted_rf, importances_sorted_rf, color='#A4A386', height=0.7)
plt.xlabel("")
plt.title("Random Forest: Key Features Influencing Product Recommendations to Consumers", loc='right',fontweight='bold',fontsize=11.5)
plt.gca().invert_yaxis()


for bar, importance in zip(bars, importances_sorted_rf):
    width = bar.get_width()
    plt.text(width / 2, bar.get_y() + bar.get_height() / 2,
             f'{importance:.2f}', ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()

#XGBoost
xgb_importances = best_xgb.feature_importances_
features_importance_xgb = list(zip(features, xgb_importances))
features_sorted_xgb, importances_sorted_xgb = zip(*sorted(features_importance_xgb, key=lambda x: x[1], reverse=True))

plt.figure(figsize=(8, 1.5))
bars = plt.barh(features_sorted_xgb, importances_sorted_xgb, height=0.7, color="#A4A386")
plt.xlabel("")
plt.title("XGBoost: Key Features Influencing Product Recommendations to Consumers", fontsize=11.5, fontweight='bold', loc='right')
plt.gca().invert_yaxis()

for bar, importance in zip(bars, importances_sorted_xgb):
    plt.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2,
             f'{importance:.2f}', va='center', ha='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()



#---Partial Dependence Plots---

Deleting this file 


