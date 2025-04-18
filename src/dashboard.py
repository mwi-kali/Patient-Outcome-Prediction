import dash_bootstrap_components as dbc

from dash import Dash, html, dcc, Input, Output


def build_header() -> html.Div:
    return html.Div([
        html.H1("Heart Failure Patient Outcome Predictor", className='display-4 mb-2'),
        html.H4("Model Evaluation & Explainability Dashboard", className='text-secondary mb-3'),
        html.P(
            (
                "A comprehensive toolset for performance assessment, calibration analysis, "
                "survival modeling, and interpretability (SHAP & LIME) of heart failure risk predictions."
            ),
            className='lead'
        )
    ], className='py-4')


def build_metric_cards(metrics: dict) -> dbc.Row:
    cols = []
    for name, val in metrics.items():
        cols.append(
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader(name, className='bg-dark text-white'),
                    dbc.CardBody(html.H5(f"{val:.3f}", className='card-title'))
                ], color=card_color(name, val), inverse=False, className='shadow-sm mb-3'),
                width=2
            )
        )
    return dbc.Row(cols, className='gx-2 mb-4')


def build_tabs() -> dbc.Tabs:
    tab_ids = [
        ('overview', 'Overview'),
        ('roc-cal', 'ROC & Calibration'),
        ('confusion', 'Confusion Matrix'),
        ('survival', 'Survival Analysis'),
        ('shap', 'Global SHAP'),
        ('lime', 'Local LIME')
    ]
    return dbc.Tabs(
        id='tabs', active_tab='overview',
        children=[dbc.Tab(label=label, tab_id=tab_id) for tab_id, label in tab_ids]
    )


def card_color(name: str, val: float) -> str:
    thresholds = {
        'AUROC': (0.85, 'success'),
        'F1-Score': (0.7, 'info')
    }
    if name in thresholds and val >= thresholds[name][0]:
        return thresholds[name][1]
    return 'secondary'


def run_dashboard(metrics, figs: dict):
    external_stylesheets = figs.get('styles', [])
    app = Dash(__name__, external_stylesheets=external_stylesheets)

    header = html.Div([
        html.H1("Heart Failure Patient Outcome Predictor", className='display-4 mb-2'),
        html.H4("Model Evaluation & Explainability Dashboard", className='text-secondary mb-3')
    ], className='py-4')

    def card_color(name, val):
        if name == 'AUROC' and val >= 0.85:
            return 'success'
        if name == 'F1-Score' and val >= 0.7:
            return 'info'
        return 'secondary'

    card_components = [
        dbc.Col(
            dbc.Card([
                dbc.CardHeader(name, className='bg-dark text-white'),
                dbc.CardBody(html.H5(f"{val:.3f}", className='card-title'))
            ], color=card_color(name, val), className='shadow-sm mb-3'),
            width=2
        )
        for name, val in metrics.items()
    ]
    metrics_row = dbc.Row(card_components, className='gx-2 mb-4')

    tab_info = {
        'roc-cal': {
            'label': 'ROC & Calibration',
            'insights': [
                (figs['roc'],
                 "AUROC ≈ 0.882 (blue ROC curve) confirms very good separation between survivors and decedents. "
                 "The steep initial rise (TPR > 0.75 by FPR ≈ 0.15) indicates the model can “catch” a large fraction of true events with relatively few false alarms."),
                (figs['calibration'],
                 "Calibration curve shows predicted probabilities line up reasonably with observed frequencies, though some mid‑range bins "
                 "are under‑ or over‑confident. A Brier score of 0.125 confirms good overall reliability of risk estimates.")
            ]
        },
        'confusion': {
            'label': 'Confusion Matrix',
            'insights': [
                (figs['confusion'],
                 "False negatives (missed death predictions) are more numerous than false positives, so if risk aversion is paramount, "
                 "the decision threshold could be lowered to reduce missed high‑risk patients at the expense of a few extra false alarms. "
                 "True positives (correct death predictions) and false negatives directly impact patient safety.")
            ]
        },
        'survival': {
            'label': 'Survival Analysis',
            'insights': [
                (figs['km'],
                 "Patients with EF ≤ 30 % exhibit a dramatic decline, with survival falling below 25 % by 250 days. "
                 "Those in 31–40 % maintain ~ 78 % survival, and 41–100 % ~ 68 % at the same horizon. "
                 "Ejection fraction remains a powerful risk stratifier; even modest EF improvements translate into substantial survival gains.")
            ]
        },
        'shap': {
            'label': 'Global SHAP',
            'insights': [
                (figs['shap'],
                 "Global SHAP values rank feature importance by their average effect on model predictions. "
                 "High-impact features reveal key clinical drivers of risk. "
                 "Top drivers include follow-up time, serum creatinine, serum sodium, age, and platelet count, aligning with clinical knowledge.")
            ]
        },
        'lime': {
            'label': 'Local LIME',
            'insights': [
                (figs['lime'],
                 "Local LIME explanations display feature contributions for individual patients, enabling case-by-case interpretability. "
                 "Use these insights to understand why the model made a specific prediction for a given patient.")
            ]
        }
    }

    tabs = dbc.Tabs(
        id='tabs', active_tab='roc-cal',
        children=[dbc.Tab(label=info['label'], tab_id=tab_id)
                  for tab_id, info in tab_info.items()]
    )

    content = html.Div(id='tab-content', className='pt-4')

    app.layout = dbc.Container([
        header,
        metrics_row,
        tabs,
        content
    ], fluid=True)

    @app.callback(
        Output('tab-content', 'children'),
        Input('tabs', 'active_tab')
    )
    def render_tab(active_tab):
        info = tab_info.get(active_tab, {})
        insights = info.get('insights', [])
        if len(insights) == 2:
            (fig1, text1), (fig2, text2) = insights
            return dbc.Row([
                dbc.Col(html.Div([dcc.Graph(figure=fig1), html.P(text1, className='mt-3')]), width=6),
                dbc.Col(html.Div([dcc.Graph(figure=fig2), html.P(text2, className='mt-3')]), width=6)
            ])
        elif len(insights) == 1:
            fig, text = insights[0]
            return html.Div([dcc.Graph(figure=fig), html.P(text, className='mt-3')])
        else:
            return html.P('Select a tab to view its contents.')

    app.run(host='0.0.0.0', port=8050)