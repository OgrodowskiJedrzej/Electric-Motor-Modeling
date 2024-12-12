import threading
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import webbrowser
import numpy as np


# Defined Parameters

# Parameters of simulation
referencedRevolutionsPerMinute = 3000

# Simulation time parameters
timeOfSimulation = 1000
timeOfSample = 0.1

# Parameters of crankshaft
brakingMoment = 0.2
loadMoment = 5
momentOfInertia = 1.2
constantOfElectromagneticMoment = 0.4

# Parameters of PI regulator
Kp = 0.007
Ki = 0.00015
Kd = 0.0015

# Constraints
Umax = 24
Umin = 0


# Lists of measured values
timeOfSimulationList = [0.0]
loadMomentList = [0.0]
electromagneticMomentList = [0.0]
adjustmentErrors = [referencedRevolutionsPerMinute]
voltagesList = [0.0]
revolutionsList = [0]
previousrevolutionsList = [0]
previoustimeOfSimulationList = [0]

# Calculations


def calculateNumberOfIterations(timeOfSimulation: int, timeOfSample: float) -> int:
    """ Calculates number of iterations for simulation of process

        @Parameters:
        - timeOfSimulation (int): total time of simulation in seconds
        - timeOfSample (float): time at which we repeat the measurement in seconds

        @Return:
        - int: number of iterations
    """
    return int(timeOfSimulation / timeOfSample) + 1


def calculateAdjustmentError(referencedRevolutionsPerMinute: float, currentRevolutionsPerMinute: float) -> float:
    """ Calculates adjustment error which is difference between referenced value and current one

        @Parameters:
        - referencedRevolutionsPerMinute (float): set value to be obtained by regulator
        - currentRevolutionsPerMinute (float): current value

        @Return:
        - float: error
    """
    return referencedRevolutionsPerMinute - currentRevolutionsPerMinute


def calculateVoltageOfRegulator(errorList: list[float], iteration: int) -> float:
    """ Calculates current voltage of regulator using PID control.

    @Parameters:
    - errorList (list[float]): list of errors at the moment and before
    - iteration (int): information about current simulation iteration

    @Return:
    - float: current voltage of regulator
    """

    proportional = Kp * errorList[iteration]

    integral = Ki * sum(errorList) * timeOfSample

    # Deriative part can be done from second iteration
    if iteration > 0:
        derivative = (errorList[iteration] -
                      errorList[iteration - 1]) / timeOfSample
    else:
        derivative = 0.0

    derivative = Kd * derivative

    voltage = proportional + integral + derivative

    return voltage


def calculateElectromagneticMoment(constant: float, currentVoltage: float) -> float:
    """ Calculates current electromagnetic moment based on voltage of regulator

        @Parameters:
        - constant (float): used to scale moment 
        - currentVoltageOfRegulator (float): voltage of regulator at the moment

        @Return
        - float: current electromagnetic moment
    """
    return constant * currentVoltage


def calculateNormalizedVoltage(voltgeOfRegulator: float) -> float:
    """ Calculates normalized voltage based on predefined constraints <Umin;Umax> [V]

    @Parameters:
    - voltageOfRegulator (float): current voltage of regulator

    @Return:
    - float: normalized voltage used to create electromagnetic moment
    """
    return max(Umin, min(Umax, voltgeOfRegulator))


def convertToAngularVelocity(valueToBeConverted: float) -> float:
    return valueToBeConverted * (2 * np.pi / 60)


def convertToRevolutionsPerMinute(valueToBeConverted: float) -> float:
    return valueToBeConverted * (60 / (2 * np.pi))


def calculateRevolutions(latestRevolution: float, latestElectromagneticMoment: float) -> float:
    """ Calculates the updated revolutions per minute (RPM) based on the system's moments.

        @Parameters:
        - latestRevolution (float): value of revolution in previous iteration
        - latestElectromagneticMoment (float): value of electromagnetic moment in previous iteration

        @Return:
        - float: updated revolutions
    """
    omega = convertToAngularVelocity(latestRevolution)
    acceleration = (
        latestElectromagneticMoment - loadMoment - brakingMoment) / momentOfInertia
    newOmega = omega + timeOfSample * acceleration

    return convertToRevolutionsPerMinute(newOmega)


# Visualizations
app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.Label("Load Moment"),
        dcc.Slider(
            id='slider-loadMoment', min=1, max=5, step=0.1, value=1,
            marks={i: str(i) for i in range(1, 11)}
        ),
        html.Label("Referenced RPMs"),
        dcc.Slider(
            id='slider-referencedRevolutionsPerMinute', min=1000, max=5000, step=1000, value=3000,
            marks={i: str(i*1000) for i in range(1, 6)}
        ),
        html.Label("Kp"),
        dcc.Slider(
            id='slider-Kp', min=0.001, max=0.02, step=0.001, value=0.007,
            marks={round(i, 3): str(round(i, 3))
                   for i in [0.001, 0.005, 0.01, 0.015, 0.02]}
        ),
        html.Label("Ki"),
        dcc.Slider(
            id='slider-Ki',
            min=0.00001, max=0.00025, step=0.00001, value=0.00013,
            marks={round(i, 5): str(round(i, 5))
                   for i in [0.00001, 0.00007, 0.00013, 0.00019, 0.00025]}
        ),

        html.Label("Kd"),
        dcc.Slider(
            id='slider-Kd',
            min=0.0001, max=0.01, step=0.0001, value=0.0015,
            marks={round(i, 4): str(round(i, 4))
                   for i in [0.0001, 0.0026, 0.0051, 0.0076, 0.01]}
        ),
    ], style={'width': '50%', 'margin': 'auto'}),

    html.Div([
        dcc.Graph(id='moments-graph'),
        dcc.Graph(id='revolutions-graph')
    ]),
])


@app.callback(
    [
        Output('moments-graph', 'figure'),
        Output('revolutions-graph', 'figure')
    ],
    [
        Input('slider-loadMoment', 'value'),
        Input('slider-referencedRevolutionsPerMinute', 'value'),
        Input('slider-Kp', 'value'),
        Input('slider-Ki', 'value'),
        Input('slider-Kd', 'value'),
    ]
)
def updateGraphs(newLoadMoment, newReferencedRPM, newKp, newKi, newKd):
    # Update global parameters
    global loadMoment, referencedRevolutionsPerMinute, Kp, Ki, Kd
    loadMoment = newLoadMoment
    referencedRevolutionsPerMinute = newReferencedRPM
    Kp = newKp
    Ki = newKi
    Kd = newKd

    # Reinitialize global lists
    global timeOfSimulationList, loadMomentList, electromagneticMomentList
    global adjustmentErrors, voltagesList, revolutionsList, brakingMomentList, previousrevolutionsList, previoustimeOfSimulationList
    timeOfSimulationList = [0.0]
    loadMomentList = [0.0]
    electromagneticMomentList = [0.0]
    adjustmentErrors = [referencedRevolutionsPerMinute]
    voltagesList = [0.0]
    revolutionsList = [0]
    brakingMomentList = [brakingMoment]

    # Simulation
    for i in range(int(timeOfSimulation / timeOfSample)):
        timeOfSimulationList.append(timeOfSimulationList[i] + timeOfSample)

        voltage = calculateNormalizedVoltage(
            calculateVoltageOfRegulator(adjustmentErrors, i)
        )
        voltagesList.append(voltage)

        electromagneticMoment = calculateElectromagneticMoment(
            constantOfElectromagneticMoment, voltagesList[i]
        )
        electromagneticMomentList.append(electromagneticMoment)

        revolutions = calculateRevolutions(
            revolutionsList[i], electromagneticMomentList[i])
        revolutionsList.append(revolutions)

        adjustmentError = calculateAdjustmentError(
            referencedRevolutionsPerMinute, revolutionsList[i]
        )
        adjustmentErrors.append(adjustmentError)
        loadMomentList.append(loadMoment)
        brakingMomentList.append(brakingMoment)

    # Moments graph (Load, Electromagnetic, and Braking Moment)
    momentsFigure = go.Figure()

    momentsFigure.add_trace(go.Scatter(
        x=timeOfSimulationList,
        y=loadMomentList,
        mode='lines',
        name='Load Moment',
        line=dict(color='blue')
    ))

    momentsFigure.add_trace(go.Scatter(
        x=timeOfSimulationList,
        y=electromagneticMomentList,
        mode='lines',
        name='Electromagnetic Moment',
        line=dict(color='red')
    ))

    momentsFigure.add_trace(go.Scatter(
        x=timeOfSimulationList,
        y=brakingMomentList,
        mode='lines',
        name='Braking Moment',
        line=dict(color='green')
    ))

    momentsFigure.update_layout(
        title=f"Load, Electromagnetic, and Braking Moment ({
            brakingMoment}) Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Moment (Nm)"
    )

    # Revolutions graph
    revolutionsFigure = go.Figure()

    revolutionsFigure.add_trace(go.Scatter(
        x=timeOfSimulationList,
        y=revolutionsList,
        mode='lines',
        name='Revolutions'
    ))

    revolutionsFigure.add_trace(go.Scatter(
        x=previoustimeOfSimulationList,
        y=previousrevolutionsList,
        mode='lines',
        name='previousRevolutions',
        line=dict(color='gray', dash="dash")
    ))

    revolutionsFigure.add_hline(y=referencedRevolutionsPerMinute, line_dash="dot",
                                annotation_text="Target RPM")
    revolutionsFigure.update_layout(
        title="Revolutions Over Time",
        xaxis_title="Time (s)",
        yaxis_title="Revolutions per Minute (RPM)"
    )
    previousrevolutionsList = revolutionsList
    previoustimeOfSimulationList = timeOfSimulationList

    return momentsFigure, revolutionsFigure


def openBrowser():
    """Open the web browser to the Dash app"""
    webbrowser.open("http://127.0.0.1:8050")


if __name__ == '__main__':
    threading.Thread(target=lambda: app.run_server(
        debug=True, use_reloader=False, host='127.0.0.1', port=8050)).start()

    openBrowser()
