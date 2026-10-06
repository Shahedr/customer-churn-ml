from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, RocCurveDisplay, ConfusionMatrixDisplay

DATA_PATH = Path('data/customer_churn.csv')
TARGET = 'churn'

def evaluate(name, model, X_test, y_test):
    pred=model.predict(X_test)
    prob=model.predict_proba(X_test)[:,1]
    return {
        'model':name,
        'roc_auc':roc_auc_score(y_test,prob),
        'accuracy':accuracy_score(y_test,pred),
        'precision':precision_score(y_test,pred),
        'recall':recall_score(y_test,pred),
        'f1':f1_score(y_test,pred),
    }

def main():
    if not DATA_PATH.exists():
        from generate_data import main as generate_data
        generate_data()
    df=pd.read_csv(DATA_PATH)
    X=df.drop(columns=TARGET); y=df[TARGET]
    numeric=['tenure_months','monthly_charges','support_tickets']
    categorical=['contract_type','autopay','internet_service']
    prep=ColumnTransformer([
        ('num',StandardScaler(),numeric),
        ('cat',OneHotEncoder(handle_unknown='ignore'),categorical),
    ])
    models={
        'Logistic Regression': LogisticRegression(max_iter=1000,class_weight='balanced',random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=300,min_samples_leaf=5,class_weight='balanced',random_state=42),
    }
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    rows=[]; fitted={}
    for name, estimator in models.items():
        pipe=Pipeline([('prep',prep),('model',estimator)])
        pipe.fit(X_train,y_train)
        rows.append(evaluate(name,pipe,X_test,y_test)); fitted[name]=pipe
    metrics=pd.DataFrame(rows).sort_values('roc_auc',ascending=False)
    Path('results').mkdir(exist_ok=True); Path('assets').mkdir(exist_ok=True); Path('models').mkdir(exist_ok=True)
    metrics.to_csv('results/model_metrics.csv',index=False)
    best_name=metrics.iloc[0]['model']; best=fitted[best_name]
    joblib.dump(best,'models/churn_model.joblib')
    RocCurveDisplay.from_estimator(best,X_test,y_test)
    plt.title(f'ROC Curve — {best_name}'); plt.tight_layout(); plt.savefig('assets/roc_curve.png',dpi=160); plt.close()
    ConfusionMatrixDisplay.from_estimator(best,X_test,y_test)
    plt.title(f'Confusion Matrix — {best_name}'); plt.tight_layout(); plt.savefig('assets/confusion_matrix.png',dpi=160); plt.close()
    print(metrics.to_string(index=False))

if __name__ == '__main__':
    main()
