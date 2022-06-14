import dash
import plotly.graph_objs as go
from dash import html, dcc, Output, Input
import nibabel as nib
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from application.data_import import get_demographic_data, import_tabular_data, import_images

# Start application with theme.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE])

# Load data from files.
demographic_data = get_demographic_data()
mean_patient_scores, patient_scores = import_tabular_data()
rids = patient_scores['RID'].unique()
img = import_images()

# Display message when no graph data can be displayed.
no_data_message = {
    "layout": {"xaxis": {"visible": False}, "yaxis": {"visible": False},
               "annotations": [
                   {"text": "No matching data found", "font": {"size": 28},
                    "xref": "paper", "yref": "paper", "showarrow": False}
               ]}
}

############################
# TOP LEFT LEFT CARD - RID #
############################


#  RID card empty table definition
RID_card_table_header = [html.Thead(html.Tr([html.Th("Attribute"), html.Th("Value")]))]
RID_card_table_body = [html.Tbody([
    html.Tr([html.Td("Gender"), html.Td(" ")]),
    html.Tr([html.Td("Race"), html.Td(" ")]),
    html.Tr([html.Td("Year of birth"), html.Td(" ")])
])]

# RID card definition
RID_card = dbc.Card(
    dbc.CardBody(
        [
            # RID dropdown
            html.Label(['RID:'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id='rid_dropdown', options=[{'label': i, 'value': i} for i in rids], value=None),

            # Patient data table
            dbc.Table(RID_card_table_header + RID_card_table_body,
                      bordered=True, id='patient_data', style={'margin-top': '20px'})
        ]), color="success", outline=True)


@app.callback(
    Output('patient_data', 'children'),
    [Input('rid_dropdown', 'value')]
)
def update_patient_table_data(rid):
    # If rid is not selected we return empty table.
    if rid is None:
        return RID_card_table_header + RID_card_table_body

    # Get demographic data for patient with rid.
    patient_data = demographic_data[demographic_data['RID'] == rid].iloc[0]

    # Construct the table with patient data.
    user_data = [html.Tbody([
        html.Tr([html.Td("Gender"), html.Td(patient_data['gender'])]),
        html.Tr([html.Td("Race"), html.Td(patient_data['race'])]),
        html.Tr([html.Td("Year of birth"), html.Td(patient_data['year_of_birth'])])
    ])]

    # Combine table header with constructed patient data table.
    return RID_card_table_header + user_data


######################################
# TOP LEFT RIGHT CARD - Phase + Test #
######################################


#  Phase + Test card empty exam table
test_card_table_body = [html.Tbody([
    html.Tr([html.Td("Baseline exam date"), html.Td("", style={'width': '100px'})]),
    html.Tr([html.Td("Baseline diagnosis"), html.Td("", style={'width': '100px'})])
])]

#  Phase + Test card definition
test_card = dbc.Card(
    dbc.CardBody(
        [
            # Phase dropdown
            html.Label(['Phase:'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id='phase_dropdown', options=[], value=None),

            # Test dropdown
            html.Label(['Test:'], style={'font-weight': 'bold', "text-align": "center", 'margin-top': '20px'}),
            dcc.Dropdown(id='test_dropdown', options=[], value=None),

            # Patient tests data table
            dbc.Table(test_card_table_body, bordered=True, id='patient_test_data', style={'margin-top': '22px'})
        ]), color="success", outline=True)


@app.callback(
    Output('phase_dropdown', 'options'),
    [Input('rid_dropdown', 'value')]
)
def update_phase_dropdown(rid):
    # If no rid is selected we return empty dropdown for Phase selection.
    if rid is None:
        return []

    # Get phases we have for patient with rid
    patient_phases = patient_scores[patient_scores['RID'] == rid].sort_values(by='Phase')['Phase'].unique()
    return [{'label': i, 'value': i} for i in patient_phases]


@app.callback(
    [Output('test_dropdown', 'options'),
     Output('patient_test_data', 'children')],
    [Input('rid_dropdown', 'value'),
     Input('phase_dropdown', 'value')]
)
def update_test_dropdown_and_table(rid, phase):
    # If rid or phase not selected we return empty dropdown and table.
    if rid is None or phase is None:
        return [], test_card_table_body

    # Filter patient data by RID and phase.
    patient_data = patient_scores[(patient_scores['RID'] == rid) & (patient_scores['Phase'] == phase)]

    # Display only the options in the test dropdown for which this RID has test scores.
    options = []
    for test in ['MOCASCORE', "GDTOTAL", 'MMSCORE', 'CDRTOTAL']:
        if not all(patient_data[test].isna()):
            options.append({'label': test, 'value': test})

    # Get patient baseline score for this phase.
    patient_baseline = patient_data[patient_data['VISCODE'] == 'bl']

    # If we have a baseline test construct a user test score table.
    if len(patient_baseline) > 0:
        user_test_data = [html.Tbody([
            html.Tr([html.Td("Baseline exam date"), html.Td(patient_baseline['EXAMDATE'])]),
            html.Tr([html.Td("Baseline diagnosis"), html.Td(patient_baseline['DX'])])
        ])]
        return options, user_test_data

    # If we don't have a baseline test construct a user test score table with "N/A" values.
    else:
        user_test_data = [html.Tbody([
            html.Tr([html.Td("Baseline exam date"), html.Td("N/A")]),
            html.Tr([html.Td("Baseline diagnosis"), html.Td("N/A")])
        ])]
        return options, user_test_data


#######################################
# BOTTOM LEFT CARD - Test score graph #
#######################################


test_score_graph_card = dbc.Card(
    dbc.CardBody([
        dcc.Graph(id='test-score-graph')]),
    color="success", outline=True)


@app.callback(
    dash.dependencies.Output('test-score-graph', 'figure'),
    [dash.dependencies.Input('phase_dropdown', 'value'),
     dash.dependencies.Input('rid_dropdown', 'value'),
     dash.dependencies.Input('test_dropdown', 'value')])
def test_score_graph(phase, rid, test):
    # If no test or no RID is selected we return an empty graph.
    if test is None or rid is None:
        return no_data_message

    # Collect the mean data for this phase.
    phase_data = mean_patient_scores[mean_patient_scores['Phase'] == phase]

    # Collect patient rid data with specific phase.
    user_data = patient_scores[(patient_scores['RID'] == rid) & (patient_scores['Phase'] == phase)] \
        .sort_values(by='VISORDER')

    # Add user data to graph.
    trace1 = go.Scatter(
        x=user_data['VISORDER'], y=user_data[test], name="RID %s" % rid,
        mode='markers', marker=dict(color='rgb(0,0,0)'), marker_size=10
    )

    # Add mean phase data to graph.
    fig = px.scatter(phase_data, x='VISORDER', y=test, color='DX', trendline="lowess",
                     labels={"VISORDER": "Visit sequence number"})
    fig.add_trace(trace1)

    fig.update_layout(title=f"Patient {test} results in phase {phase}")

    return fig


#########################################
# TOP RIGHT LEFT CARD - visit selection #
#########################################


# Select visit for image.
image_visit_card = dbc.Card(
    dbc.CardBody(
        [
            # Visit dropdown
            html.Label(['Acquisition date:'], style={'font-weight': 'bold', "text-align": "center"}),
            dcc.Dropdown(id='image_acquisition_date_dropdown', options=[], value=None),
        ]), color="success", outline=True)


@app.callback(
    Output('image_acquisition_date_dropdown', 'options'),
    [Input('rid_dropdown', 'value')]
)
def update_acquisition_date_dropdown(rid):
    # If no rid is selected we return empty dropdown.
    if rid is None:
        return []

    # Get available images for this patient.
    patient_image_dates = img[img['RID'] == rid]['Acq Date']

    options = []
    for date in patient_image_dates:
        options.append({'label': date, 'value': date})

    return options


####################################
# TOP RIGHT RIGHT CARD - plane selection #
####################################


plane_card = dbc.Card(
    dbc.CardBody([
        html.Label(['Select plane:'], style={'font-weight': 'bold', "text-align": "center"}),
        dbc.RadioItems(
            id="plane-radio", className="btn-group", inputClassName="btn-check",
            labelClassName="btn-sm btn-outline-success", labelCheckedClassName="active",
            options=[{'label': i, 'value': i} for i in ['Coronal', 'Sagittal', 'Transverse']],
            value='Coronal',
            style={'margin-top': '5px'})
    ]),
    color="success", outline=True)

###########################################
# BOTTOM RIGHT CARD - image visualization #
###########################################


image_card = dbc.Card(
    dbc.CardBody([
        dcc.Slider(id='slice-slider', min=0, max=160, value=80, step=5, marks=None),
        dcc.Graph(id='plane-graph')]),
    color="success", outline=True)


@app.callback(
    Output('plane-graph', 'figure'),
    [Input('slice-slider', 'value'),
     Input('plane-radio', 'value'),
     Input('rid_dropdown', 'value'),
     Input('image_acquisition_date_dropdown', 'value')])
def plane_image(slice_index, plane_mode, rid, acq_date):
    # If no RID is selected no image can be shown.
    if rid is None or acq_date is None:
        return no_data_message

    filename_data = img[(img['RID'] == rid) & (img['Acq Date'] == acq_date)]['filename']
    if filename_data.empty:
        return no_data_message

    filename = str(filename_data.iloc[0])
    image_data = nib.load(filename).get_fdata()

    # Slice the image dataset based on the selected plane and slice index (from the slider).
    if plane_mode == 'Coronal':
        fig = px.imshow(image_data[:, slice_index, :])
    elif plane_mode == 'Sagittal':
        fig = px.imshow(image_data[:, :, slice_index])
    elif plane_mode == 'Transverse':
        fig = px.imshow(image_data[slice_index, :, :])

    return fig


###############
# HTML layout #
###############


app.layout = html.Div([
    # Left column in dashboard
    html.Div([
        html.H2('Test scores', style={'textAlign': 'center'}),

        # Top left cards
        html.Div([
            html.Div([RID_card], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([test_card], style={'display': 'inline-block', 'width': '49%', 'margin-left': '10px'}),
        ], style={'width': '100%', 'display': 'inline-block'}),

        # Bottom left test score display.
        html.Div([test_score_graph_card], style={'width': '100%', 'display': 'inline-block', 'margin-top': '10px'})

    ], style={'width': '49%', 'display': 'inline-block', 'padding': '10px', 'padding': '20px 0px 0px 20px'}),

    # Right column in dashboard
    html.Div([
        html.H2('MRI images', style={'textAlign': 'center'}),

        # Top right cards
        html.Div([
            html.Div([image_visit_card], style={'display': 'inline-block', 'width': '49%'}),
            html.Div([plane_card], style={'display': 'inline-block', 'width': '49%', 'margin-left': '10px'}),
        ], style={'width': '100%', 'display': 'inline-block'}),

        # Bottom right card image display.
        html.Div([image_card], style={'width': '100%', 'display': 'inline-block', 'margin-top': '10px'})

    ], style={'width': '49%', 'float': 'right', 'display': 'inline-block', 'padding': '20px 20px 0px 0px'}),
], style={'max-width': '1280px', 'margin': 'auto'})

if __name__ == '__main__':
    app.run_server()
