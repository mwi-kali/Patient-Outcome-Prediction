import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dash import html
from plotly.subplots import make_subplots


def add_age_quartile(df, n_quartiles=4):
    df = df.copy()
    df['age_group'] = pd.qcut(
        df['age'], q=n_quartiles,
        labels=[f'Q{i+1}' for i in range(n_quartiles)]
    )
    return df


def add_log_cpk(df):
    df = df.copy()
    df['cpk_log'] = np.log1p(df['creatinine_phosphokinase'])
    df.drop(columns='creatinine_phosphokinase', inplace=True)
    return df


def build_bootstrap_card(title, content):
    return html.Div([
        html.Div([
            html.H5(title, className='card-title'),
            html.Div(content, className='card-text')
        ], className='card-body')
    ], className='card mb-4 shadow-sm')