import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_full_timeline(train_df, test_df, train_pred, test_pred, anomalies=None, unit='MW'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Load'], mode='lines',
                             name='Historical', line=dict(color='lightblue', width=1), opacity=0.7))
    fig.add_trace(go.Scatter(x=train_df.index, y=train_pred, mode='lines',
                             name='Train Fit', line=dict(color='green', width=1, dash='dot'), opacity=0.5))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Load'], mode='lines',
                             name='Actual (Test)', line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_pred, mode='lines',
                             name='Forecast', line=dict(color='red', width=2, dash='dash')))
    if anomalies is not None and anomalies.any():
        fig.add_trace(go.Scatter(x=test_df.index[anomalies], y=test_df['Load'].values[anomalies],
                                 mode='markers', name='Anomalies',
                                 marker=dict(color='orange', size=10, symbol='x')))
    fig.update_layout(title='Complete Timeline: Historical + Forecast', xaxis_title='Date',
                      yaxis_title=f'Load ({unit})', hovermode='x unified', height=600,
                      legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig

def plot_model_comparison(test_df, predictions_dict, unit='MW'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Load'], mode='lines',
                             name='Actual', line=dict(color='black', width=3)))
    colors = {'XGBoost': '#e74c3c', 'LSTM': '#3498db', 'SARIMA': '#2ecc71',
              'Decision Tree': '#f39c12', 'Random Forest': '#9b59b6', 'TFT': '#1abc9c'}
    for name, preds in predictions_dict.items():
        if preds is not None:
            fig.add_trace(go.Scatter(x=test_df.index, y=preds, mode='lines', name=name,
                                     line=dict(color=colors.get(name, 'gray'), width=2, dash='dash')))
    fig.update_layout(title='Model Comparison: All Forecasts vs Actual', xaxis_title='Date',
                      yaxis_title=f'Load ({unit})', hovermode='x unified', height=600)
    return fig

def plot_metrics_comparison(metrics_dict):
    models  = list(metrics_dict.keys())
    metrics = ['MAE', 'RMSE', 'MAPE (%)', 'R²']
    fig = make_subplots(rows=2, cols=2, subplot_titles=metrics)
    palette = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    for idx, metric in enumerate(metrics):
        r, c = idx // 2 + 1, idx % 2 + 1
        vals = [metrics_dict[m][metric] for m in models]
        fig.add_trace(go.Bar(x=models, y=vals, name=metric,
                             marker_color=palette[:len(models)]), row=r, col=c)
        fig.update_yaxes(title_text=metric, row=r, col=c)
    fig.update_layout(height=600, showlegend=False, title_text='Metrics Comparison Across Models')
    return fig

def plot_load_history(df, unit='MW'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Load'], mode='lines',
                             name='Load', line=dict(color='royalblue', width=1)))
    fig.update_layout(title='Historical Load Data', xaxis_title='Date',
                      yaxis_title=f'Load ({unit})', hovermode='x unified', height=400)
    return fig

def plot_residuals(residuals, dates):
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Residuals Over Time', 'Residual Distribution'))
    fig.add_trace(go.Scatter(x=dates, y=residuals, mode='lines',
                             name='Residuals', line=dict(color='purple')), row=1, col=1)
    fig.add_trace(go.Histogram(x=residuals, name='Distribution',
                               marker=dict(color='lightblue')), row=2, col=1)
    fig.update_xaxes(title_text='Date', row=1, col=1)
    fig.update_xaxes(title_text='Residual Value', row=2, col=1)
    fig.update_yaxes(title_text='Residual', row=1, col=1)
    fig.update_yaxes(title_text='Frequency', row=2, col=1)
    fig.update_layout(height=600, showlegend=False)
    return fig

def plot_future_overlay(train_df, test_df, future_dict, unit='MW'):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=train_df.index, y=train_df['Load'], mode='lines',
                             name='Historical', line=dict(color='lightblue', width=1)))
    fig.add_trace(go.Scatter(x=test_df.index, y=test_df['Load'], mode='lines',
                             name='Test Actual', line=dict(color='blue', width=2)))
    colors = {'XGBoost': '#e74c3c', 'LSTM': '#3498db', 'SARIMA': '#2ecc71',
              'Decision Tree': '#f39c12', 'Random Forest': '#9b59b6', 'TFT': '#1abc9c'}
    for name, (dates, preds) in future_dict.items():
        if dates is not None and preds is not None:
            fig.add_trace(go.Scatter(x=dates, y=preds, mode='lines',
                                     name=f'{name} Future',
                                     line=dict(color=colors.get(name, 'gray'), width=2, dash='dash')))
    fig.update_layout(title='Historical + Test + Future Forecasts',
                      xaxis_title='Date', yaxis_title=f'Load ({unit})',
                      hovermode='x unified', height=650)
    return fig

def plot_india_national_overview(all_states_df):
    total = all_states_df.groupby('State')['Load'].mean().sort_values(ascending=False)
    fig = go.Figure(go.Bar(
        x=total.index.tolist(), y=total.values,
        marker=dict(color=total.values, colorscale='Viridis', showscale=True,
                    colorbar=dict(title='Avg Demand (MU)')),
        text=[f'{v:.0f}' for v in total.values], textposition='outside'
    ))
    fig.update_layout(title='Average Daily Demand by State (MU)',
                      xaxis_title='State', yaxis_title='Avg Day Demand (MU)',
                      height=500, xaxis_tickangle=-45)
    return fig


