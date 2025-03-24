## Autor: Raymond Bautista
## Fecha: 23/03/2025
## Dashboard del indice de siniestralidad de las ARS del regimen contributivo del SFS

import dash
from dash import dcc, html, Input, Output, dash_table
import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# Cargado de los datos

# Dataframe de la serie mensual de los gastos, ingresos y siniestralidad de las ARS en general (2007-2023)
serie_general_df = pd.read_csv('IngresosGastosSiniestralidad_SerieMensual.csv')

# Dataframe de los ingresos de las ARS por tipo de ARS
ingresosARS_df = pd.read_csv('Ingresos_TiposARS.csv')

# Dataframe de los gastos de las ARS por tipo de ARS
gastosARS_df = pd.read_csv('Gastos_TiposARS.csv')

# Dataframe del indice de siniestralidad de las ARS por tipo de ARS
sinistARS_df = pd.read_csv('Siniestralidad_TiposARS.csv')

# Preparacion de los datos
# Convierto los indices a formato de fecha
serie_general_df['fecha'] = pd.to_datetime(serie_general_df['año'].astype(str) + '-' + serie_general_df['mes'].astype(str), format='%Y-%m')
serie_general_df.set_index('fecha', inplace=True)
sinistARS_df['año'] = pd.to_datetime(sinistARS_df['año'], format='%Y')

# Modelo SARIMAX para las predicciones
model = SARIMAX(serie_general_df['siniestralidad'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit(disp=False)

# Predicciones y forecast
predictions = model_fit.predict()
forecast = model_fit.forecast(steps=24)

# Crea la Web App y la inicializa
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Diseño del Dashboard
app.layout = dbc.Container([
    html.Div(style={'backgroundColor': '#f0f0f0'}, children=[
        html.H1("Índice de Siniestralidad de las ARS del Régimen Contributivo del SFS", style={'textAlign': 'center', 'color': '#0056b3'}),
        html.P("Período de cobertura: 2007-2023", style={'textAlign': 'center'}),
        dbc.Row([
            dbc.Col([
                html.Label("Filtro General: Año"),
                dcc.Dropdown(
                    id='year-dropdown',
                    options=[{'label': 'All', 'value': 'all'}] + [{'label': str(year), 'value': year} for year in sinistARS_df['año'].dt.year.unique()],
                    value='all'
                ),
            ], width=6),
            dbc.Col([
                html.Label("Filtro General: Tipo de ARS"),
                dcc.Dropdown(
                    id='ars-dropdown',
                    options=[
                        {'label': 'All', 'value': 'all'},
                        {'label': 'Pública', 'value': 'ars_publica'},
                        {'label': 'Privada', 'value': 'ars_privada'},
                        {'label': 'Autogestión', 'value': 'ars_autogestion'}
                    ],
                    value='all'
                ),
            ], width=6),
        ]),
        html.P("Autor: Raymond Bautista", style={'textAlign': 'right'}),
        dbc.Row([
            dbc.Col(dcc.Graph(id='siniestralidad-line-chart'), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='siniestralidad-bar-chart'), width=12),
        ]),
        dbc.Row([
            dbc.Col(dash_table.DataTable(
                id='tendencia-table',
                columns=[
                    {'name': 'Año', 'id': 'Año'},
                    {'name': 'Gasto Promedio', 'id': 'Gasto Promedio'},
                    {'name': 'Ingreso Promedio', 'id': 'Ingreso Promedio'},
                    {'name': 'Siniestralidad Promedio', 'id': 'Siniestralidad Promedio'}
                ],
                data=[],  # Inicialmente sin datos
            ), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='gastos-bar-chart'), width=6),
            dbc.Col(dcc.Graph(id='ingresos-bar-chart'), width=6),
        ]),
        dbc.Row([
            dbc.Col(dcc.Graph(id='sarima-forecast-chart'), width=12),
        ]),
        dbc.Row([
            dbc.Col(dcc.Slider(id='forecast-slider', min=0, max=60, value=24, marks={i: str(i) for i in range(0, 61, 12)}), width=12),
        ]),
        html.P("Datos extraídos de los últimos reportes de Superintendencia de Salud y Riesgos Laborales (SISALRIL), 2025", style={'textAlign': 'center'})
    ])
])

# Callbacks para cada uno de los graficos

# Grafico de lineas del indice de siniestralidad en el tiempo
@app.callback(
    Output('siniestralidad-line-chart', 'figure'),
    Input('year-dropdown', 'value') # Se actualiza con el año seleccionado
)
def update_line_chart(selected_year):
    if selected_year == 'all':
        filtered_df = serie_general_df  # Muestra todos los años
    else:
        filtered_df = serie_general_df[serie_general_df.index.year == selected_year]    # Limita unicamente al año filtrado
    fig = px.line(filtered_df, x=filtered_df.index, y='siniestralidad', title='Serie de Tiempo de Siniestralidad', labels = {'siniestralidad': 'Siniestralidad (%)', 'fecha': 'Año'})  # Grafico de lineas interactivo
    fig.update_layout(plot_bgcolor='#e0f2f7', paper_bgcolor='#e0f2f7')
    return fig

# Grafico de barras del indice de siniestralidad por tipo de ARS en el tiempo
@app.callback(
    Output('siniestralidad-bar-chart', 'figure'),
    Input('year-dropdown', 'value'),    # Se actualiza con el tipo de ARS y el año filtrados
    Input('ars-dropdown', 'value')
)
def update_bar_chart(selected_year, selected_ars):
    if selected_year == 'all':
        filtered_df = sinistARS_df
    else:
        filtered_df = sinistARS_df[sinistARS_df['año'].dt.year == selected_year]    # Filtra por año seleccionado
    if selected_ars == 'all':
        fig = px.bar(filtered_df, x='año', y=['ars_publica', 'ars_privada', 'ars_autogestion'], barmode='group', title='Siniestralidad por Tipo de ARS', 
                                 labels={
                'año': 'Año',
                'value': 'Siniestralidad (%)',
                'variable': 'Tipo de ARS',
                'ars_publica': 'Pública',
                'ars_privada': 'Privada',
                'ars_autogestion': 'Autogestión'
            })    # Grafico de barras interactivo
    else:
        fig = px.bar(filtered_df, x='año', y=selected_ars, title=f'Siniestralidad {selected_ars}',
                                 labels={
                'año': 'Año',
                selected_ars: 'Siniestralidad (%)'
            })
    fig.update_layout(plot_bgcolor='#e0f2f7', paper_bgcolor='#e0f2f7')
    return fig

# Tabla de medidas de tendencia central para cada año
@app.callback(
    Output('tendencia-table', 'data'),
    Input('year-dropdown', 'value')
)
def update_table(selected_year):
    if selected_year == 'all':  # Muestra el ultimo año si no se selecciona nada
        latest_year = sinistARS_df['año'].dt.year.max()
        gastos_year = gastosARS_df[gastosARS_df['año'] == latest_year]
        ingresos_year = ingresosARS_df[ingresosARS_df['año'] == latest_year]
        sinist_year = sinistARS_df[sinistARS_df['año'].dt.year == latest_year]
    else:   # De lo contrario muestra el año filtrado
        gastos_year = gastosARS_df[gastosARS_df['año'] == selected_year]
        ingresos_year = ingresosARS_df[ingresosARS_df['año'] == selected_year]
        sinist_year = sinistARS_df[sinistARS_df['año'].dt.year == selected_year]

    gasto_promedio = gastos_year[['ars_publica', 'ars_privada', 'ars_autogestion']].mean().mean()
    ingreso_promedio = ingresos_year[['ars_publica', 'ars_privada', 'ars_autogestion']].mean().mean()
    siniestralidad_promedio = sinist_year[['ars_publica', 'ars_privada', 'ars_autogestion']].mean().mean()

    data = [{
        'Año': gastos_year['año'].iloc[0],
        'Gasto Promedio': gasto_promedio,
        'Ingreso Promedio': ingreso_promedio,
        'Siniestralidad Promedio': siniestralidad_promedio
    }]
    return data

# Graficos de barras apiladas de gastos e ingresos de las ARS por tipo de ARS anual
@app.callback(
    Output('gastos-bar-chart', 'figure'),   # Se actualiza a partir del año y tipo de ARS
    Output('ingresos-bar-chart', 'figure'),
    Input('year-dropdown', 'value'),
    Input('ars-dropdown', 'value')
)
def update_gastos_ingresos_charts(selected_year, selected_ars):
    if selected_year == 'all':
        gastos_filtered = gastosARS_df
        ingresos_filtered = ingresosARS_df
    else:
        gastos_filtered = gastosARS_df[gastosARS_df['año'] == selected_year]
        ingresos_filtered = ingresosARS_df[ingresosARS_df['año'] == selected_year]
    if selected_ars == 'all':
        gastos_fig = px.bar(gastos_filtered, x='año', y=['ars_publica', 'ars_privada', 'ars_autogestion'], title='Gastos por ARS', barmode='stack',
                                        labels={
                'año': 'Año',
                'value': 'Monto (Billones RD$)',
                'variable': 'Tipo de ARS',
                'ars_publica': 'Pública',
                'ars_privada': 'Privada',
                'ars_autogestion': 'Autogestión'
            })
        ingresos_fig = px.bar(ingresos_filtered, x='año', y=['ars_publica', 'ars_privada', 'ars_autogestion'], title='Ingresos por ARS', barmode='stack',
                                        labels={
                'año': 'Año',
                'value': 'Monto (Billones RD$)',
                'variable': 'Tipo de ARS',
                'ars_publica': 'Pública',
                'ars_privada': 'Privada',
                'ars_autogestion': 'Autogestión'
            })
    else:
        gastos_fig = px.bar(gastos_filtered, x='año', y=selected_ars, title=f'Gastos {selected_ars}',
                                        labels={
                'año': 'Año',
                selected_ars: 'Monto (Billones RD$)'
            })
        ingresos_fig = px.bar(ingresos_filtered, x='año', y=selected_ars, title=f'Ingresos {selected_ars}',
                                          labels={
                'año': 'Año',
                selected_ars: 'Monto (Billones RD$)'
            })
    gastos_fig.update_layout(plot_bgcolor='#e0f2f7', paper_bgcolor='#e0f2f7')
    ingresos_fig.update_layout(plot_bgcolor='#e0f2f7', paper_bgcolor='#e0f2f7')
    return gastos_fig, ingresos_fig  # Devuelve ambos gráficos

# Serie de tiempo com prediccion del indice de siniestralidad
@app.callback(
    Output('sarima-forecast-chart', 'figure'),
    Input('forecast-slider', 'value')
)
def update_forecast_chart(forecast_months):
    forecast = model_fit.forecast(steps=forecast_months)
    fig = px.line(serie_general_df, x=serie_general_df.index, y='siniestralidad', title='Predicción de Siniestralidad Mensual', labels = {'siniestralidad': 'Siniestralidad (%)', 'fecha': 'Año'}) # Muestra un grafico de linea interactivo
    fig.add_scatter(x=forecast.index, y=forecast, mode='lines', name='Predicciones', line=dict(color='orange')) # Añade el forecast en otro color
    fig.update_layout(plot_bgcolor='#e0f2f7', paper_bgcolor='#e0f2f7')
    return fig

if __name__ == "__main__":
    app.run(debug=True)