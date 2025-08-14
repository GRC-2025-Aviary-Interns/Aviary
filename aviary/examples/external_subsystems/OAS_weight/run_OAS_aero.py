import openmdao.api as om
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

import aviary.api as av
from aviary.examples.external_subsystems.OAS_weight.OAS_aero_builder import OASAeroBuilder
from aviary.examples.external_subsystems.OAS_weight.OAS_aero_analysis import OASAero
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.aviary_values import AviaryValues

from aviary.variable_info.variables import Aircraft, Dynamic

aero_builder = OASAeroBuilder()

phase_info = {
    'pre_mission': {
        'include_takeoff': False,
        'external_subsystems': [],
        'optimize_mass': True,
    },
    'cruise': {
        'subsystem_options': {'core_aerodynamics': {'method': 'computed'}},
        'user_options': {
            'num_segments': 2,
            'order': 3,
            'mach_optimize': False,
            'mach_polynomial_order': 1,
            'mach_initial': (0.02, 'unitless'),
            'mach_final': (0.04, 'unitless'),
            'mach_bounds': ((0.01, 0.05), 'unitless'),
            'altitude_optimize': False,
            'altitude_polynomial_order': 1,
            'altitude_initial': (400, 'm'),
            'altitude_final': (400, 'm'),
            'altitude_bounds': ((350, 450), 'm'),
            'throttle_enforcement': 'boundary_constraint',
            'time_initial': (0, 's'),
            'time_duration_bounds': ((1.0, 100.0), 's'),
        },
        'initial_guesses': {'time': ([0, 10], 's'),
                            'distance': ([0, 5], 'm')}
    },

    'post_mission': {
        'target_range': (5, 'm'),
        'include_landing': False,
        'external_subsystems': [],
    },
}
 
phase_info['cruise']['external_subsystems'] = [aero_builder]
phase_info['cruise']['subsystem_options']['core_aerodynamics'] = {'method': 'external'}
phase_info['cruise']['user_options']['num_segments'] = 1

max_iter = 21
optimizer = 'IPOPT'

prob = av.AviaryProblem()

prob.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info=phase_info)

prob.aviary_inputs.set_val(Dynamic.Mission.ALTITUDE, 400, units='m') 
prob.aviary_inputs.set_val(Dynamic.Mission.VELOCITY, 36, units='m/s') 

prob.aviary_inputs.set_val(Aircraft.Wing.SPAN, 1.524, units='m')  
prob.aviary_inputs.set_val(Aircraft.Wing.ROOT_CHORD, 0.508, units='m')
prob.aviary_inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.10) 
prob.aviary_inputs.set_val(Aircraft.Wing.MAX_THICKNESS_LOCATION, 0.266) 
prob.aviary_inputs.set_val(Aircraft.Wing.CHARACTERISTIC_LENGTH, 0.508, units='m')
prob.aviary_inputs.set_val(Aircraft.Wing.TAPER_RATIO, 1, units='unitless')
prob.aviary_inputs.set_val(Aircraft.Wing.SWEEP, 0, units='deg')
prob.aviary_inputs.set_val(Aircraft.Wing.INCIDENCE, 0, units='deg')

prob.aviary_inputs.set_val(Aircraft.HorizontalTail.SPAN, 0.711, units='m')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.ROOT_CHORD, 0.232, units='m')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.14)
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 1, units='unitless')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.SWEEP, 0, units='deg')

prob.aviary_inputs.set_val(Aircraft.Wing.CENTER_DISTANCE, 0.511, units='unitless')
prob.aviary_inputs.set_val(Aircraft.Fuselage.LENGTH, 1.189, units='m')

prob.check_and_preprocess_inputs()
prob.add_pre_mission_systems()
prob.add_phases()
prob.add_post_mission_systems()

prob.link_phases()

prob.add_driver(optimizer=optimizer, max_iter=max_iter)

prob.add_design_variables()
prob.model.add_design_var('aircraft:wing:span', lower=0.1, upper=5.0)
# prob.model.add_design_var('aircraft:horizontal_tail:span', lower=0.1, upper=2.0)

# prob.model.add_constraint('traj.cruise.rhs_all.aero_point_0.CL', equals=0.3)
prob.model.add_objective('traj.cruise.rhs_all.aero_point_0.CD', scaler=1)

prob.driver.recording_options['record_desvars'] = False
prob.driver.recording_options['record_responses'] = False
prob.driver.recording_options['record_objectives'] = False
prob.driver.recording_options['record_constraints'] = False

prob.setup()

prob.set_initial_guesses()

prob.run_aviary_problem()


print('Wing span:', prob.get_val(Aircraft.Wing.SPAN))
print('HTail span:', prob.get_val(Aircraft.HorizontalTail.SPAN))
print('Wing root chord:', prob.get_val(Aircraft.Wing.ROOT_CHORD))
print('HTail root chord:', prob.get_val(Aircraft.HorizontalTail.ROOT_CHORD))

print('Lift:', prob.get_val('traj.cruise.rhs_all.lift')) 
print('Drag:', prob.get_val('traj.cruise.rhs_all.drag'))

print('CL:', prob.get_val('traj.cruise.rhs_all.aero_point_0.CL')) 
print('CD:', prob.get_val('traj.cruise.rhs_all.aero_point_0.CD')) 

def plot_meshes(meshes): # from openaerostruct.meshing.section_mesh_generator.py
    """this function plots to plot the mesh"""
    plt.figure(figsize=(8, 4))
    for i, mesh in enumerate(meshes):
        mesh_x = mesh[:, :, 0]
        mesh_y = mesh[:, :, 1]
        color = "k"
        for i in range(mesh_x.shape[0]):
            plt.plot(mesh_y[i, :], mesh_x[i, :], color, lw=1)
            plt.plot(-mesh_y[i, :], mesh_x[i, :], color, lw=1)   # plots the other side of symmetric wing
        for j in range(mesh_x.shape[1]):
            plt.plot(mesh_y[:, j], mesh_x[:, j], color, lw=1)
            plt.plot(-mesh_y[:, j], mesh_x[:, j], color, lw=1)   # plots the other side of symmetric wing
    plt.axis("equal")
    plt.xlabel("y (m)")
    plt.ylabel("x (m)")

wing_mesh = prob.get_val('wing.mesh')
htail_mesh = prob.get_val('htail.mesh')

plot_meshes([wing_mesh, htail_mesh])
plt.show()

