from dash import Dash, html, dcc
# from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
from consts import *
from utils import read_data_file, make_tensors, resample_P_H_mats, set_random_state
import base64
import io
from copy import deepcopy
import plotly.graph_objects as go
from infer import load_trained_model, make_inference
from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Output, State, DashProxy, Input, MultiplexerTransform
import json, requests
import numpy as np
from simulation.simulation import bulge_simulation

# API_URL = "http://" + 'localhost' + ":8000"
# endpoint_post_exp = API_URL + "/post-exp"
# endpoint_post_checkpoint = API_URL + "/post-checkpoint"
app = DashProxy(prevent_initial_callbacks=True, transforms=[MultiplexerTransform()])
# # app = Dash(__name__)


def mk_lines_fig():
    data_df = read_data_file()
    H_mat, P_mat, C_mat = make_tensors(data_df)
    H_mat, P_mat = resample_P_H_mats(H_mat, P_mat, h_max=600, n_points=25)
    fig = px.line(x=H_mat[0,:], y=P_mat[0,:]*10e3)
    for ii in range(1, H_mat.shape[0], 2):  
        fig.add_scatter(x=H_mat[ii,:], y=P_mat[ii,:]*10e3)
    fig.update_layout(showlegend=False, title="Bulge simulation dataset",
        xaxis_title="bulge height [mm]",
        yaxis_title="pressure [MPa]",
        )
    return fig



def parse_contents(content, filename):
    
    content_type, content_string = content.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            output = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            output = pd.read_excel(io.BytesIO(decoded))
        elif 'pk' in filename:
            decoded = base64.b64decode(content_string)
            output = pd.read_pickle(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])
    return output


def update_checkpoint_selector():
    p = MODEL_PATH.glob('*.pk')
    file_names = [x.name for x in p if x.is_file()]
    return file_names


def create_layout(fig):
    layout = html.Div(children=[
        html.H1(children='FEA Learning'),
        html.H2(children='Bulge Test Predict material coeficients'),
        dcc.Graph(id='lines-chart', figure=fig),
        dcc.Upload(id='upload-exp', children='select EXPERIMENT file',
                       style={
                            # 'display': 'inline-block',
                            'width': '90%',
                            # 'fex': 1,
                            'height': '40px',
                            'lineHeight': '40px', 
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'alignItems': 'center',
                            'margin': '10px'
                            }),
        dcc.Upload(id='upload-check', children='select CHECKPOINT file',
                       style={
                            # 'display': 'inline-block',
                            'width': '90%',
                            # 'fex': 1,
                            'height': '40px',
                            'lineHeight': '40px', 
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'alignItems': 'center',
                            'margin': '10px'
                            }),
        html.Div(id='test-block', children='', style={'textAlign': 'left'}),
        html.Br(),
        html.Button('Infer', id='infer-button'),
        html.H2(id='infer-block', children='', style={'textAlign': 'center', 'height': '20px'}),
        html.Button('Run simulation', id='simulation-button'),
        
        
    ])
    return layout




@app.callback(Output('lines-chart', 'figure'),
              Input('upload-exp', 'contents'),
              State('upload-exp', 'filename'),
              State('lines-chart', 'figure'))
def load_exp(content, fname, fig):
    if content is None:
        raise PreventUpdate
    df = parse_contents(content, fname)
    df.columns = ['heights', 'pressures', 'pressures_atm']
    exp_dict = df.to_dict('list')
    place_holder['exp'] = exp_dict
    # payload = json.dumps(exp_dict)
    # result = requests.post(endpoint_post_exp, data=payload)
    
    fig = go.Figure(fig)
    fig.add_scatter(x=exp_dict['heights'], y=np.array(exp_dict['pressures'])*10e3)
    fig.data[-1].line.width = 5
    fig.data[-1].line.color = 'black'
    fig.data[-1].line.dash = 'dash'
    return fig #result.text


@app.callback(Output('test-block', 'children'),
              Input('upload-check', 'contents'),
              State('upload-check', 'filename'))
def load_check(content, fname):
    if content is None:
        raise PreventUpdate
    checkpoint = parse_contents(content, fname)
    net, config, X_norm, Y_norm = load_trained_model(checkpoint)
    place_holder['net'] = net
    place_holder['config'] = config
    place_holder['X_norm'] = X_norm
    place_holder['Y_norm'] = Y_norm
    return 'checkpoint loaded'
    # app.net = net

@app.callback(Output('infer-block', 'children'),
              Input('infer-button', 'n_clicks'))
def infer(n_clicks):
    if n_clicks is not None:
        if n_clicks > 0:
            p = place_holder['exp']['pressures']
            h = place_holder['exp']['heights']
            C_hat = make_inference(h=np.array(h), p=np.array(p), net=place_holder['net'], config=place_holder['config'], X_norm=place_holder['X_norm'], Y_norm=place_holder['Y_norm'])
            place_holder['C_hat'] = C_hat
            return f"C1={C_hat[0]: 0.3e}  |  C2={C_hat[1]: 0.3e}  |  C3={C_hat[2]: 0.3e}"
    else:
        raise PreventUpdate


@app.callback(Output('lines-chart', 'figure'),
              Input('simulation-button', 'n_clicks'),
              State('lines-chart', 'figure'))
def run_simulation(n_clicks, fig):
    if n_clicks is not None:
        if n_clicks > 0:
            h_hat, p_hat = bulge_simulation(place_holder['C_hat'], p_max=2.3E-4, h=2.0, inc=50, tolerance=15, Radius=50.0)
            infer_sim_dict = {'pressures': list(p_hat), 'heights': list(h_hat)}
            place_holder['infer_sim'] = infer_sim_dict
            
            fig = go.Figure(fig)
            fig.add_scatter(x=h_hat, y=p_hat*10e3)
            fig.data[-1].line.width = 5
            fig.data[-1].line.color = 'black'
            fig.data[-1].line.dash = 'dot'

            return fig
    else:
        raise PreventUpdate    
    



if __name__ == '__main__':
    set_random_state(27)
    place_holder = {}
    checkpoint_file_names = update_checkpoint_selector()
    fig = mk_lines_fig()
    base_layout = create_layout(fig)
    layout = deepcopy(base_layout)
    app.layout = layout 
    # app.run(debug=True)
    app.run()


    print('end')
    
    
