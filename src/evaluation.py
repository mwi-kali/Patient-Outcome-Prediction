import plotly.graph_objects as go

from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve


def compute_metrics(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)[:,1]
    f1  = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    bri = brier_score_loss(y_test, y_prob)
    prec, rec, _, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
    return {'F1':f1, 'AUROC':auc, 'Brier':bri, 'Precision':prec, 'Recall':rec}


def plot_calibration(pipe, X_test, y_test, n_bins=10):
    y_prob = pipe.predict_proba(X_test)[:,1]
    true, pred = calibration_curve(y_test, y_prob, n_bins=n_bins)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pred, y=true, mode='markers+lines', name='Calibration', marker=dict(size=8)))
    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='gray'))
    fig.update_layout(
        title='Calibration Curve', xaxis_title='Mean Predicted Probability',
        yaxis_title='Observed Frequency', template='plotly_white'
    )
    return fig


def plot_confusion(pipe, X_test, y_test):
    y_pred = pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm, x=['Survived','Died'], y=['Survived','Died'],
        colorscale='Blues', showscale=False, text=cm, texttemplate='%{text}'
    ))
    fig.update_layout(
        title='Confusion Matrix', xaxis_title='Predicted', yaxis_title='Actual',
        template='plotly_white'
    )
    return fig


def plot_roc(pipe, X_test, y_test):
    y_prob = pipe.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC={auc:.2f}', line=dict(color='#1f77b4', width=3)))
    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='gray'))
    fig.update_layout(
        title='ROC Curve', xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
        template='plotly_white'
    )
    return fig





