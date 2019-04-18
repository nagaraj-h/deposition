
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Global variables
amb_temp = 25
amb_pr = 101325
ref_temp = 273.11
g = 9.807
Q_s = 57 * 0.001 / 60
d_t = 0.0381
v_o = 16.0
D_p = 0.00001
rho_p = 1000
R_u = 8314.471
MW = 28.962
R = R_u / MW
rho = amb_pr / (R * (amb_temp + ref_temp))
k = 1.3807E-23

# calculation of viscosity at ambient temperature
sutherland_constant = 110.56
mu_o = 1.716e-5
mu = mu_o *(((amb_temp + ref_temp) / ref_temp) ** 1.5)*(ref_temp + sutherland_constant)/(amb_temp + ref_temp + sutherland_constant)

# Cunningham correction factor
u = 0.4987445
mean_free_path = math.sqrt(math.pi / 8) * (mu / u) * math.sqrt(1 / (rho * amb_pr))

def area(d):
    # this function returns area as a function of diameter
    a = math.pi * d ** 2 / 4
    return a

def velocity(Q, d):
    ### this function returns velocity as a function of diameter using the sample flow rate
    vel = (Q * 4) / (math.pi * d ** 2)
    return vel

def cunningham_correction(Dp):
    C_c = 1 + (mean_free_path / Dp) * (2.34 + 1.05 * math.exp(-0.39 * Dp / mean_free_path))
    return C_c

def Stokes(Q, Dp, d1, d2):
    # this function returns Stokes number as a function of diameters: d1 & d2 varies for different components
    stk = cunningham_correction(Dp) * rho_p * Dp ** 2 * velocity(Q, d1)/(9 * mu * d2)
    return stk

def shrouded_probe(Q,Dp,v_free,probe):

    # defining default probe dimensions
    dimensions = {'RF-2-111':{'d_sh':0.0538, 'd_b':0.0437, 'd_i':0.0183, 'd_o':0.0381, 'L_pr':0.1293},
    'RF-2-112':{'d_sh':0.0538, 'd_b':0.0475, 'd_i':0.0311, 'd_o':0.0381, 'L_pr':0.1293},
    'RF-2-113':{'d_sh':0.0642, 'd_b':0.0559, 'd_i':0.0447, 'd_o':0.0381, 'L_pr':0.2444},
    'CMR4CFM-HI':{'d_sh':0.0780, 'd_b':0.0680, 'd_i': 0.0220, 'd_o': 0.0381, 'L_pr': 0.2385},
    'CMR4CFM-LO':{'d_sh':0.0780, 'd_b':0.0664, 'd_i': 0.0262, 'd_o': 0.0381, 'L_pr': 0.2385},
    'WIPP5CFM':{'d_sh': 0.1020, 'd_b': 0.0860, 'd_i': 0.0300, 'd_o': 0.0510, 'L_pr': 0.3630}}

    # for custom shrouded probe, the dimensions of RF-2-111 is set as default
    dimensions.setdefault(probe, {'d_sh':0.0538, 'd_b':0.0437, 'd_i':0.0183, 'd_o':0.0381, 'L_pr':0.1293})
    d_sh = dimensions[probe]['d_sh']
    d_b = dimensions[probe]['d_b']
    d_i = dimensions[probe]['d_i']
    d_o = dimensions[probe]['d_o']
    L_pr = dimensions[probe]['L_pr']

    A_gap = area(d_sh)-area(d_b)

    if Q / area(d_sh) > v_free:
        Q_sh1 = Q
    else:
        Q_sh1 = -1

    if (v_free * A_gap + Q) / area(d_sh) < v_free:
        Q_sh2 = v_free * A_gap + Q
    else:
        Q_sh2 = -1

    if Q_sh1 * Q_sh2 > 0:
        Q_sh3 = v_free * A_gap
    else:
        Q_sh3 = -1

    v_sh = max(Q_sh1, Q_sh2, Q_sh3) / area(d_sh)
    v_pr = Q / area(d_i)
    R_sh = v_free / v_sh
    v_sc = v_sh * (1 + 1.45 * (1 - ((1 + math.log(R_sh)) / R_sh)))
    R_pr = v_sc / v_pr
    Stk_sh = Stokes(Q, Dp, d_sh, d_sh) * v_free / velocity(Q, d_sh)
    F = 1 - (R_sh - 1) * 0.861 * Stk_sh / ((2.34 + 0.939 * (R_sh - 1)) * Stk_sh + 1)
    L_w = L_pr / d_i
    Fr = (v_pr ** 2) / (g * d_i)
    Stk_i = Stokes(Q, Dp, d_i, d_i)
    if R_sh > 1:
        Stk_pr = Stk_i * v_sc / v_pr
    else:
        Stk_pr = Stk_i * v_sh / v_pr

    W_L = 0.496 * (1 + L_w/Fr) ** 0.194 * Stk_i ** 0.613 * (v_sh / v_pr) ** 1.191
    alpha_sh = 1.05 * Stk_sh / (1 + 1.05 * Stk_sh)
    aspiration_ratio_sh = 1 + alpha_sh * (R_sh - 1)
    alpha_pr = 1.05 * Stk_pr / (1 + 1.05 * Stk_pr)
    aspiration_ratio_pr = 1 + alpha_pr * (R_pr - 1)
    aspiration_ratio = F * aspiration_ratio_sh * aspiration_ratio_pr
    efficiency = 100* aspiration_ratio * (1 - W_L)

    return efficiency


def nozzle(dia, length):

    v_ni = velocity(dia)
    R_N = v_o / v_ni
    Stk_N = Stokes(dia, dia) * v_o / v_ni
    Fr = 2 * v_ni ** 2 / (g * dia)
    Re = rho * v_ni * dia / mu
    Stk_p = Stokes(dia, dia)
    C1 = 3.08 * math.sqrt(rho * mu)/ (D_p * rho_p)
    k_ref = 4 * v_ni * 2 / dia
    tao_N = 1 / (C1 * math.sqrt(k_ref))
    R_z = 2 * Stk_p / math.sqrt(tao_N * v_ni * 2 / dia)
    W_L = (1.769 * (1 + (2 * length)/(dia * Fr)) ** -9.190) * R_z ** 0.559 * Re ** -0.216
    aspiration_ratio = 1 + (R_N - 1) * 1.05 * Stk_N / (1 + 1.05 * Stk_N)
    efficiency = aspiration_ratio * (1 - W_L)
    return efficiency

def tube(Q,d, Dp, length, angle):
    # function will return efficiency of tube
    # tube efficiency is made of three parts: gravitational settling, turbulent eddy and thermal diffusion

    v_tube = velocity(Q,d)
    Reynolds = rho * v_tube * d / mu
    Stk_p = Stokes(Q,Dp,d, d)

    #This section calculates tube efficiency due to gravitational settling
    v_ts = rho_p * Dp ** 2 * g * C_c / (18 * mu)
    v_ts1 = 0
    Re_p = rho * v_ts * Dp / mu

    if Re_p >= 1:
        while abs(v_ts1 - v_ts) > 0.0001 and v_ts1 != 0:
            Re_p = rho * v_ts * Dp / mu
            C_d = 24 * (1 + 0.15 * Re_p ** 0.687) / Re_p
            v_ts1 = v_ts
            v_ts = math.sqrt(4 * rho_p * D_p ** 2 * g / (3 * C_d * rho))

    t_ = length * v_ts * math.cos(angle * math.pi / 180) / (v_tube * d)
    K = 0.75 * t_
    efficiency_gr_lam = 1 - (2 / math.pi) * (2 * K * math.sqrt(1 - K ** (2 / 3)) - K ** (1 / 3) * math.sqrt(1 - K ** (2 / 3)) + math.asin(K ** (1 / 3)))

    if Reynolds < 2100:
        efficiency_gravitational = efficiency_gr_lam
    elif Reynolds > 2100 and Reynolds < 4000:
        Z = 4 * t_ / math.pi
        efficiency_gr_tur = math.exp(-Z)
        efficiency_gravitational = min (efficiency_gr_lam, efficiency_gr_tur)
    else:
        Z = 4 * t_ / math.pi
        efficiency_gr_tur = math.exp(-Z)
        efficiency_gravitational = efficiency_gr_tur

    # End of section for gravitational settling

    #This section calculates transport efficiency due to turbulent eddy
    t_relax = 0.0395 * Stk_p * Reynolds ** 0.75
    if t_relax < 0.3:
        v_part_dep = 0
    elif t_relax > 12.9:
        v_part_dep = 0.1
    else:
        v_part_dep = 0.0006 * t_relax ** 2 + 2E-8 * Reynolds
    v_t = v_part_dep * v_tube * Reynolds ** (-1/8) / 5.03
    efficiency_turbulent = math.exp(-math.pi * d * length * v_t / Q)
    if Reynolds < 2100:
        efficiency_turbulent = 1.0
    # End of section for turbulent eddy model

    # this section calculates transport efficiency due to thermal diffusion
    D_c = k * (amb_temp + ref_temp) * C_c / (3 * math.pi * mu * Dp)
    epsilon = math.pi * D_c * length / Q
    Sc = mu / (rho * D_c)
    Sh_laminar = 3.66 + 0.2672 / (epsilon + 0.10079 * epsilon ** (1/3))
    Sh_turbulent = 0.0118 * Reynolds ** (7/8) * Sc ** (1/3)
    if Reynolds < 2100:
        efficiency_thermal = math.exp(-epsilon * Sh_laminar)
    elif Reynolds > 4000:
        efficiency_thermal = math.exp(-epsilon * Sh_turbulent)
    else:
        efficiency_thermal = min(math.exp(-epsilon * Sh_laminar), math.exp(-epsilon * Sh_turbulent))
    # end of section for thermal diffusion

    efficiency = efficiency_gravitational * efficiency_turbulent * efficiency_thermal

    return efficiency



def contraction(exit_dia, half_angle):
    A_i = area(d_t)
    A_o = area(exit_dia)
    efficiency = 1 - 1 / (1 + (Stokes(d_t, exit_dia) * (1 - (A_o / A_i) / (3.14 * math.exp(-0.0185 * half_angle)))) ** -1.24)
    return efficiency

def expansion(exit_dia, half_angle):
    A_i = area(d_t)
    A_o = area(exit_dia)
    R_ex = 1 - A_i / A_o
    b1 = math.log(Stokes(exit_dia, exit_dia) * R_ex / 0.5518) / 1.9661
    b2 = math.log(half_angle/12.519)/2.7825
    efficiency = 1 - 1.1358 * R_ex ** 2 * math.exp(-0.5 * (b1 ** 2 + b2 ** 2))
    return efficiency

def splitter (inlet_dia, angle):
    a = -2.635
    b = 2.623
    c = 0.4573
    d = 1.680
    e = 2.291
    f = 56.45
    g = 3.870
    h = -2.288
    stk = Stokes(inlet_dia, inlet_dia)
    efficiency = math.exp(a + b / (1 + (stk/c) ** d) + e / (1 + (angle/f)** g) + h / ((1+(stk/c) ** d)*(1 + (angle/f)**g)))
    return efficiency

def bend(tube_dia, angle, bend_radius):
    Re = rho * velocity (tube_dia) * tube_dia / mu
    stk = Stokes(tube_dia, tube_dia)
    if Re > 4000:
        efficiency = math.exp(-2.823 * stk * angle * math.pi / 180)
        print('turbulent')
        print(Re)
    else:
        efficiency = (1 + (stk/0.171) ** (0.452 * stk / 0.171 + 2.242)) ** (-2 * angle / 180)
        print('laminar')
        print(Re)

    return efficiency


shrouded_probe = np.vectorize(shrouded_probe)
bend = np.vectorize(bend)


# # Visualization

from bokeh.plotting import figure
from bokeh.io import output_file, output_notebook, show, curdoc, reset_output
from bokeh.models import ColumnDataSource, Span
from bokeh.models.widgets import Slider, Select, TextInput, Button, InputWidget
from bokeh.layouts import row, column, widgetbox

def update(attr, old, new):
    Dp = particle_size_slider.value * 1e-6
    Q = sample_flow_rate_slider.value * 28.316 * 0.001 / 60
    probe = shrouded_probe_select.value

    x = np.arange(1,30,1,dtype=float)
    new_y = shrouded_probe(Q,Dp,x,probe)
#     new_y = bend(d_t, 90, 5)

    source.data = {'x':x, 'y':new_y}


#Configure widgets, sliders,plots, etc

output_notebook()

n_particle_size = 10
n_flow_rate = 2

particle_size_slider = Slider(start=1,end=12, step=1, title='Particle Size (microns)', value=n_particle_size)
sample_flow_rate_slider = Slider(start=1, end=6, step=0.5, title='Sample Flow Rate (CFM)', value=n_flow_rate)
shrouded_probe_select = Select(options=['RF-2-111', 'RF-2-112', 'Straight Tube', 'Bend', 'Contraction'], value='RF-2-111', title='Shrouded Probe Model')

add_component = Button()
add_component.label = 'Add Component'

tube_diameter_text_box = InputWidget()
tube_diameter_text_box.title = 'enter tube dia'

probe = shrouded_probe_select.value

free_velocity_vector = np.arange(1,30,1,dtype=float)

source = ColumnDataSource(data={'x':free_velocity_vector,
                                'y':shrouded_probe(n_flow_rate*28.316*0.001/60,n_particle_size*1e-6,free_velocity_vector,probe)})

plot1 = figure(plot_height=400,plot_width=600, tools='box_select, pan, box_zoom, wheel_zoom, save, reset', y_range=(40,140),logo=None)
plot1.xaxis.axis_label ='Free Stream Velocity (m/s)'
plot1.yaxis.axis_label ='Transmission Efficiency'

plot1.toolbar.logo = None


plot1.title.text = 'Transmission Efficiency vs Stack Free Stream Velocity'
plot1.title.align = 'center'
plot1.line(x='x', y='y', source=source, color='blue', legend='Shrouded Probe')

plot1.xgrid.grid_line_color = 'black'
plot1.xgrid.grid_line_alpha = 0.3
plot1.xgrid.minor_grid_line_color = 'black'
plot1.xgrid.minor_grid_line_alpha = 0.3

plot1.ygrid.grid_line_color = 'black'
plot1.ygrid.grid_line_alpha = 0.3
plot1.ygrid.minor_grid_line_color = 'black'
plot1.ygrid.minor_grid_line_alpha = 0.3


plot1.circle(x='x', y='y', source=source, color='blue')
hline1 = Span(location=120, dimension='width', line_color='green', line_width=2)
hline2 = Span(location=80, dimension='width', line_color='red', line_width=2)
hline3 = Span(location=50, dimension='width', line_color='black', line_width=2)


plot1.renderers.extend([hline1, hline2,hline3])


particle_size_slider.on_change('value',update)
shrouded_probe_select.on_change('value',update)
sample_flow_rate_slider.on_change('value', update)

controls = [add_component, tube_diameter_text_box, sample_flow_rate_slider, particle_size_slider, shrouded_probe_select]
layout = row(widgetbox(*controls, sizing_mode='scale_width'), plot1)
curdoc().add_root(layout)
# show(layout)
curdoc().title = 'DEPOSITION BY HI-Q'
