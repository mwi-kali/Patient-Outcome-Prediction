import shap

import numpy as np
import pandas as pd
import plotly.express as px

from lime.lime_tabular import LimeTabularExplainer


def lime_explain(pipe_uncal, X_train, X_test, idx=0, num_features=8, random_state=42):
    pre, model = pipe_uncal.named_steps['pre'], pipe_uncal.named_steps['clf']
    X_train_num = pre.transform(X_train)
    X_test_num  = pre.transform(X_test)
    feat_names  = pre.get_feature_names_out()

    explainer = LimeTabularExplainer(
        X_train_num, feature_names=feat_names,
        class_names=['Survived','Died'], mode='classification',
        discretize_continuous=False, random_state=random_state
    )
    exp = explainer.explain_instance(
        data_row=X_test_num[idx], predict_fn=model.predict_proba,
        num_features=num_features
    )
    lime_list = exp.as_list()
    df_lime = pd.DataFrame(lime_list, columns=['feature','weight'])
    fig = px.bar(df_lime, x='weight', y='feature', orientation='h',
                 title=f'LIME Explanation', labels={'weight':'Contribution'})
    fig.update_layout(template='plotly_white', margin=dict(l=200))
    return fig


def shap_summary(pipe_uncal, X_train, X_test, top_n=10):
    pre, model = pipe_uncal.named_steps['pre'], pipe_uncal.named_steps['clf']
    X_train_num = pre.transform(X_train)
    X_test_num  = pre.transform(X_test)
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_test_num)
    arr = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
    imp = np.abs(arr).mean(0)
    feat = pre.get_feature_names_out()
    df_imp = pd.DataFrame({'feature':feat,'importance':imp}).nlargest(top_n,'importance')
    fig = px.bar(df_imp[::-1], x='importance', y='feature', orientation='h',
                 title='Top SHAP Importances', labels={'importance':'Mean |SHAP|','feature':''})
    fig.update_layout(template='plotly_white', margin=dict(l=200))
    return fig


