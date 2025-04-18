import pandas as pd
import plotly.graph_objects as go

from .config import EF_BINS
from lifelines import KaplanMeierFitter, CoxPHFitter


def cox_model(df, duration_col='time', event_col='death_event'):
    df_cox = pd.get_dummies(df, drop_first=True)
    cph = CoxPHFitter()
    cph.fit(df_cox, duration_col=duration_col, event_col=event_col)
    return cph


def km_plot(df, time_col='time', event_col='death_event', group_col='ejection_fraction'):
    df_km = df.copy()
    df_km['ef_bin'] = pd.cut(df_km[group_col], bins=EF_BINS, labels=['≤30%','31-40%','41-100%'])
    kmf = KaplanMeierFitter()
    fig = go.Figure()
    for grp in df_km['ef_bin'].cat.categories:
        sub = df_km[df_km['ef_bin']==grp]
        kmf.fit(sub[time_col], sub[event_col], label=f'EF {grp}')
        fig.add_trace(go.Scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_[f'EF {grp}'],
            mode='lines', name=f'EF {grp}'
        ))
    fig.update_layout(
        title='Kaplan–Meier Survival by EF',
        xaxis_title='Time (days)', yaxis_title='Survival Probability',
        template='plotly_white'
    )
    return fig


