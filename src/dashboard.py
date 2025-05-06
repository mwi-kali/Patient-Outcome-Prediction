import dash
import dash_bootstrap_components as dbc
import plotly.io as pio

from dash import Dash, html, dcc, Input, Output


pio.templates.default = "plotly_white"

dash_index = '''<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Open+Sans:wght@400;600;700&display=swap');
    :root {
        --font-family: 'Open Sans', sans-serif;
        --bg-color: #f5f7fa;
        --text-color: #333;
        --card-bg: #ffffff;
        --card-radius: 12px;
        --card-shadow: rgba(0, 0, 0, 0.1);
        --card-shadow-hover: rgba(0, 0, 0, 0.2);
        --transition-duration: 0.3s;
    }
    body {
        background-color: var(--bg-color);
        color: var(--text-color);
        font-family: var(--font-family);
        line-height: 1.6;
    }
    .card {
        background-color: var(--card-bg);
        border-radius: var(--card-radius);
        box-shadow: 0 4px 12px var(--card-shadow);
        overflow: hidden;
        transition: 
            transform var(--transition-duration) ease,
            box-shadow var(--transition-duration) ease;
    }
    .card-hover:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 16px 32px var(--card-shadow-hover);
    }
    .card-hover:focus-within,
    .card-hover:focus {
        outline: 2px solid rgba(100, 150, 250, 0.6);
        outline-offset: 4px;
    }
    </style>
</head>
<body>
    {%app_entry%}
    <footer>
        {%config%}
        {%scripts%}
        {%renderer%}
    </footer>
</body>
</html>'''


def calib_insight(brier):
    if brier is None:
        return "Calibration information is not provided, so we cannot comment on the alignment between predicted and observed outcomes."
    
    if brier < 0.1:
        return (
            f"A Brier score of {brier:.3f} reflects excellent calibration: the predicted probabilities closely match "
            "the actual event rates. In practice, this means clinicians can trust the individual risk estimates when "
            "making treatment decisions. The model is both well-discriminated and well-calibrated, "
            "ideal for clinical deployment."
        )
    
    if brier < 0.2:
        return (
            f"The Brier score of {brier:.3f} indicates good overall calibration, though there may be slight under- or "
            "over-prediction in the mid-range risk bins. Consider applying isotonic or polynomial calibration to refine "
            "those areas. This level of calibration is acceptable for most use cases but could be tuned "
            "for optimal reliability."
        )
    
    return (
        f"A Brier score of {brier:.3f} points to suboptimal calibration, predicted risks diverge noticeably from observed "
        "outcomes. Before using this model in a clinical environment, recalibration is strongly recommended, as accurate "
        "risk estimates are critical for patient safety."
    )


def cm_insight(precision, recall):
    if precision is None or recall is None:
        return "Precision or recall metrics are missing, so we cannot analyze classification balance."
    
    if precision > recall:
        return (
            f"The model's precision ({precision:.3f}) exceeds its recall ({recall:.3f}), meaning that when it flags a "
            "high-risk patient, it is usually correct, but it misses some true positives. To capture more at-risk "
            "patients, the decision threshold could be lowered; however, this will increase false alarms. "
            "Certainty is prioritised over coverage, adjust thresholds based on clinical context."
        )
    if recall > precision:
        return (
            f"Recall ({recall:.3f}) is higher than precision ({precision:.3f}), so the model catches most true positives "
            "but also generates more false positives. Slightly raising the threshold may help reduce unnecessary alarms. "
            "This setup is risk-averse, favoring detection over specificity, tune the threshold as "
            "needed for operational constraints."
        )
    return (
        f"Precision and recall are balanced at {precision:.3f}, indicating the current threshold equally values correct "
        "detections and minimized false alerts. This equilibrium is often ideal for general-purpose monitoring. "
        "Maintain or only slightly adjust thresholds if clinical priorities change."
    )


def roc_insight(auroc):
    if auroc is None:
        return "ROC curve data is unavailable, so we cannot assess discrimination performance at this time."
    
    if auroc >= 0.9:
        return (
            f"The model achieves an outstanding AUROC of {auroc:.3f}. This indicates near-perfect separation between "
            "patients who survive and those who do not, meaning the model ranks individuals by risk with great "
            "precision. Such a high level of discrimination is rare and suggests the algorithm is capturing the "
            "key clinical signals needed for prognostic accuracy. The model can confidently be used "
            "to prioritize high-risk patients for immediate intervention."
        )
    elif auroc >= 0.85:
        return (
            f"With an AUROC of {auroc:.3f}, the model demonstrates very good discrimination. It correctly identifies "
            "most high-risk cases while maintaining a reasonable false positive rate, making it well-suited as a "
            "screening tool in clinical settings. This performance level suggests the model is reliable "
            "for guiding risk stratification and resource allocation in patient care."
        )
    else:
        return (
            f"The AUROC of {auroc:.3f} indicates moderate discrimination. While the model can separate cases to some "
            "extent, there is room for improvement. Consider revisiting feature engineering or hyperparameter "
            "tuning, or incorporating additional clinical variables to capture more variance. To deploy "
            "in practice, aiming for at least 0.85 would strengthen confidence in decision-making."
        )


def shap_insight(shap_fig):
    if shap_fig is None or not hasattr(shap_fig, 'data') or len(shap_fig.data) == 0:
        return "Global SHAP summary is unavailable, so feature importance cannot be assessed."
    
    return (
        f"The Global SHAP summary ranks the top 10 features by their average impact on model predictions. "
        "Positive mean SHAP values indicate risk-driving factors, while negative values suggest protective factors. "
        "Use these insights to confirm known clinical predictors and discover unexpected drivers. "
        "SHAP enhances interpretability and helps prioritize features for further investigation."
    )

def survival_insight(km_fig):
    if km_fig is None or not hasattr(km_fig, 'data') or len(km_fig.data) == 0:
        return "Survival analysis plot is unavailable, so we cannot comment on group-level prognosis."
    
    groups = len(km_fig.data)
    if groups > 3:
        return (
            f"The Kaplan–Meier plot displays {groups} ejection fraction strata, each representing a distinct risk group. "
            "Clear early divergence between curves suggests strong prognostic separation, with lower EF groups "
            "experiencing steeper declines. Clinicians should note these patterns for targeted interventions. "
            "EF remains a critical predictor, and the model's survival stratification aligns with clinical expectations."
        )
    return (
        "Kaplan–Meier curves stratify patients by ejection fraction, showing which groups have better or worse survival "
        "trajectories over time. Early curve separation highlights the prognostic importance of EF. "
        "These survival insights validate the model’s utility for risk stratification in patient management."
    )


def run_dashboard(metrics: dict, figs: dict):
    external_stylesheets = figs.get('styles', [dbc.themes.FLATLY, dbc.icons.FONT_AWESOME])
    app = Dash(__name__, external_stylesheets=external_stylesheets)
    app.index_string = dash_index
    server = app.server   

    header = dbc.Container(
        dbc.CardBody([
            html.H1("Heart Failure Patient Outcome Predictor", className="display-4 mb-2"),
            html.H4("Model Evaluation & Explainability Dashboard", className="lead text-secondary mb-3"),
            html.P(
                "This is a toolset for performance assessment, calibration analysis, survival modeling, and interpretability (SHAP) of heart failure risk predictions."
            ),
        ])
    )

    cards = []
    for name, val in metrics.items():
        cards.append(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader([html.I(className="fa fa-chart-line me-2"), html.Span(name)], className="bg-dark text-white"),
                        dbc.CardBody(html.H5(f"{val:.3f}", className="card-title")),
                    ],
                    inverse=False,
                    className="shadow-sm h-100 card-hover"
                ),
                width="auto",
                xs=6, sm=4, md=3, lg=2
            )
        )

    metrics_row = dbc.Row(
        [
        dbc.Col(
            dbc.Card(
            [
                dbc.CardHeader(name, className='bg-dark text-white text-center'),
                dbc.CardBody(html.H5(f"{val:.3f}", className='card-title text-center'))
            ],
            className='shadow-sm mb-3'
            ),
            width=2
        )
        for name, val in metrics.items()
        ],
        className="gx-3 gy-3 mb-5 justify-content-center"
    )

    auroc_val = metrics.get('AUROC')
    brier_val = metrics.get('Brier Score')
    prec_val = metrics.get('Precision')
    rec_val = metrics.get('Recall')

    tab_info = {
        'roc-cal': {'label': 'ROC & Calibration', 'insights': [(figs.get('roc'), roc_insight(auroc_val)), (figs.get('calibration'), calib_insight(brier_val))]},
        'confusion': {'label': 'Confusion Matrix', 'insights': [(figs.get('confusion'), cm_insight(prec_val, rec_val))]},
        'survival': {'label': 'Survival Analysis', 'insights': [(figs.get('km'), survival_insight(figs.get('km')))]},
        'shap': {'label': 'Global SHAP', 'insights': [(figs.get('shap'), shap_insight(figs.get('shap')))]}
    }

    tabs = dbc.Tabs(id="tabs", active_tab="roc-cal", children=[dbc.Tab(label=i['label'], tab_id=k) for k,i in tab_info.items()], className="mb-4  justify-content-center px-5")
    content = html.Div(id='tab-content', className='pt-4')

    app.layout = html.Div([header, metrics_row, tabs, content], className='p-4 bg-light d-flex flex-column')

    @app.callback(Output('tab-content','children'), Input('tabs','active_tab'))
    def render_tab(active_tab):
        info = tab_info.get(active_tab, {})
        insights = info.get('insights', [])
        if len(insights)==2:
            (f1,t1),(f2,t2)=insights
            return dbc.Row([dbc.Col(html.Div([dcc.Graph(figure=f1), html.P(t1,className='mt-3')]),width=6),dbc.Col(html.Div([dcc.Graph(figure=f2), html.P(t2,className='mt-3')]),width=6)])
        if len(insights)==1:
            f,t=insights[0]
            return html.Div([dcc.Graph(figure=f), html.P(t,className='mt-3')])
        return info.get('content', html.P('No content available.'))

    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=8050)
