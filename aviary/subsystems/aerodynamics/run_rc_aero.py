import openmdao.api as om
import numpy as np
import matplotlib.pyplot as plt

import aviary.api as av
from aviary.subsystems.aerodynamics.rc_aero_builder import RCAeroBuilder
from aviary.subsystems.aerodynamics.rc_aero import TotalAircraftAero
from aviary.utils.functions import set_aviary_initial_values
from aviary.utils.aviary_values import AviaryValues

from aviary.variable_info.variables import Aircraft, Dynamic

aero_builder = RCAeroBuilder()

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
            'mach_initial': (0.08, 'unitless'),
            'mach_final': (0.09, 'unitless'),
            'mach_bounds': ((0.07, 0.11), 'unitless'),
            'altitude_optimize': False,
            'altitude_polynomial_order': 1,
            'altitude_initial': (520, 'm'),
            'altitude_final': (520, 'm'),
            'altitude_bounds': ((500, 600), 'm'),
            'throttle_enforcement': 'boundary_constraint',
            # 'time_initial_bounds': ((0.0, 0.0), 'min'),
            'time_initial': (0, 's'),
            'time_duration_bounds': ((1.0, 300.0), 's'),
        },
        'initial_guesses': {'time': ([0, 250], 's'),
                            'distance': ([0, 800], 'm')}
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

max_iter = 30 
optimizer = 'IPOPT' 

prob = av.AviaryProblem()

prob.load_inputs('models/test_aircraft/aircraft_for_bench_FwFm.csv', phase_info=phase_info)

prob.aviary_inputs.set_val(Dynamic.Mission.ALTITUDE, 520, units='m') # wichita at 400 ft
prob.aviary_inputs.set_val(Dynamic.Mission.VELOCITY, 36, units='m/s') # buzzair M1

prob.aviary_inputs.set_val(Aircraft.Wing.SPAN, 1.524, units='m')  
prob.aviary_inputs.set_val(Aircraft.Wing.ROOT_CHORD, 0.508, units='m')
prob.aviary_inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.10) # SD7032
prob.aviary_inputs.set_val(Aircraft.Wing.MAX_THICKNESS_LOCATION, 0.266) # SD7032
prob.aviary_inputs.set_val(Aircraft.Wing.TAPER_RATIO, 1, units='unitless')
prob.aviary_inputs.set_val(Aircraft.Wing.SWEEP, 0, units='deg')
prob.aviary_inputs.set_val(Aircraft.Wing.INCIDENCE, 0, units='deg')

prob.aviary_inputs.set_val(Aircraft.HorizontalTail.SPAN, 0.711, units='m')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.ROOT_CHORD, 0.232, units='m')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.14) # NACA 0014 why did we not say what airfoil we used lmfao
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 1, units='unitless')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.SWEEP, 0, units='deg')

prob.aviary_inputs.set_val(Aircraft.VerticalTail.SPAN, 0.3048, units='m')
prob.aviary_inputs.set_val(Aircraft.VerticalTail.ROOT_CHORD, 0.22225, units='m')
prob.aviary_inputs.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.14) # NACA 0014 
prob.aviary_inputs.set_val(Aircraft.VerticalTail.TAPER_RATIO, 1, units='unitless')

prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 0.172, units='m')
prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 0.114, units='m')
prob.aviary_inputs.set_val(Aircraft.Wing.CENTER_DISTANCE, 0.511, units='unitless')
prob.aviary_inputs.set_val(Aircraft.Fuselage.LENGTH, 1.190244, units='m')

prob.aviary_inputs.set_val(Dynamic.Vehicle.MASS, 3.787, units='kg')

prob.check_and_preprocess_inputs()
prob.add_pre_mission_systems()
prob.add_phases()
prob.add_post_mission_systems()

prob.link_phases()

prob.add_driver(optimizer=optimizer, max_iter=max_iter)

prob.add_design_variables()

prob.model.add_design_var('aircraft:wing:span', lower=0.1, upper=2.0)

prob.model.add_design_var('traj.cruise.rhs_all.rc_aero_analysis.alpha', lower=-5.0, upper=15.0)

# prob.model.add_constraint('traj.cruise.rhs_all.rc_aero_analysis.lifting_surface_CL', lower=0.01, upper=0.2)
prob.model.add_objective('traj.cruise.rhs_all.rc_aero_analysis.avg_CD', scaler=1)

prob.driver.recording_options['record_desvars'] = False
prob.driver.recording_options['record_responses'] = False
prob.driver.recording_options['record_objectives'] = False
prob.driver.recording_options['record_constraints'] = False

prob.setup()

prob.set_initial_guesses()

# try: 
# prob.run_model()
# except:
#         with open("variables.txt", "w") as f:
#             prob.model.list_vars(out_stream=f, print_arrays=True, units=True)


prob.run_aviary_problem()

with open("variables.txt", "w") as f:
    prob.model.list_vars(out_stream=f, print_arrays=True, units=True)


print('Lift:', prob.get_val('traj.cruise.rhs_all.lift', units='lbf')) # M1 TOW 8.35
print('Drag:', prob.get_val('traj.cruise.rhs_all.drag', units='lbf'))
print('CL:', prob.get_val('traj.cruise.rhs_all.rc_aero_analysis.lifting_surface_CL'))
print('CD:', prob.get_val('traj.cruise.rhs_all.rc_aero_analysis.CD'))

print('CD_fus:', prob.get_val('traj.cruise.rhs_all.rc_aero_analysis.CD_fus'))
print('CD_vtail:', prob.get_val('traj.cruise.rhs_all.rc_aero_analysis.CD_vtail'))
print('CD_gear:', prob.get_val('traj.cruise.rhs_all.rc_aero_analysis.CD_gear'))
print('Lifting surface CD:', prob.get_val('traj.cruise.rhs_all.rc_aero_analysis.lifting_surface_CD'))

print('Fuselage length:', prob.get_val('aircraft:fuselage:length'))
print('Fuselage height:', prob.get_val('aircraft:fuselage:max_height'))
print('Angle of attack:', prob.get_val('traj.cruise.rhs_all.rc_aero_analysis.alpha'))
# print('Wing span:', prob.get_val(Aircraft.Wing.SPAN))

# prob=om.Problem()

# aviary_inputs = AviaryValues()

# aviary_inputs.set_val(Dynamic.Mission.ALTITUDE, 400, units='m') # wichita
# aviary_inputs.set_val(Dynamic.Mission.VELOCITY, 32, units='m/s') # M1 max velocity

# aviary_inputs.set_val(Aircraft.Wing.SPAN, 1.524, units='m')
# aviary_inputs.set_val(Aircraft.Wing.ROOT_CHORD, 0.508, units='m')
# aviary_inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.16, units='unitless')
# aviary_inputs.set_val(Aircraft.Wing.MAX_THICKNESS_LOCATION, 0.363, units='unitless')
# aviary_inputs.set_val(Aircraft.Wing.TAPER_RATIO, 0.5, units='unitless')
# aviary_inputs.set_val(Aircraft.Wing.SWEEP, 0, units='deg')
# aviary_inputs.set_val(Aircraft.Wing.INCIDENCE, 0, units='deg')

# aviary_inputs.set_val(Aircraft.HorizontalTail.SPAN, 0.674, units='m')
# aviary_inputs.set_val(Aircraft.HorizontalTail.ROOT_CHORD, 0.229, units='m')
# aviary_inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.14, units='unitless')
# aviary_inputs.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 1, units='unitless')
# aviary_inputs.set_val(Aircraft.HorizontalTail.SWEEP, 0, units='deg')

# aviary_inputs.set_val(Aircraft.VerticalTail.SPAN, 0.3048, units='m')
# aviary_inputs.set_val(Aircraft.VerticalTail.ROOT_CHORD, 0.22225, units='m')
# aviary_inputs.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.14, units='unitless')
# aviary_inputs.set_val(Aircraft.VerticalTail.TAPER_RATIO, 1, units='unitless')

# aviary_inputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 0.172, units='m')
# aviary_inputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 0.114, units='m')
# aviary_inputs.set_val(Aircraft.Wing.CENTER_DISTANCE, 0.3, units='unitless')
# aviary_inputs.set_val(Aircraft.Fuselage.WETTED_AREA, 0.61, units='m**2') # rough rough estimate
# aviary_inputs.set_val(Aircraft.Fuselage.LENGTH, 1.190244, units='m')

# aviary_inputs.set_val(Aircraft.LandingGear.DRAG_COEFFICIENT, 0.011, units='unitless') # buzzair

# prob.model.add_subsystem(
#     'rc_aero',
#     TotalAircraftAero(num_nodes=1, aviary_inputs=aviary_inputs),
#     promotes=['*']
# )

# prob.driver = om.ScipyOptimizeDriver()
# prob.driver.options['tol'] = 1e-9

# recorder = om.SqliteRecorder('aero_analysis_test.db')
# prob.driver.add_recorder(recorder)
# prob.driver.recording_options['record_derivatives'] = True
# prob.driver.recording_options['includes'] = ['*']

# prob.model.add_design_var('aircraft:wing:span', lower=0.1, upper=2.0)
# prob.model.add_constraint('lifting_surface_CL', equals=0.08)
# prob.model.add_objective('avg_CD', scaler=1e3)

# # for name, val in aviary_inputs.items():
# #     try:
# #         prob.model.set_input_defaults(name, val=val[0], units=val[1])
# #     except:
# #         print(name)
# #         continue

# prob.model.set_input_defaults('velocity', 0, units='m/s')
# prob.model.set_input_defaults('aircraft:wing:area', 0, units='m**2')

# prob.setup()
# set_aviary_initial_values(prob, aviary_inputs)
# prob.run_model()

# # # try:
# # #     prob.run_model()
# # # except:

# # with open("variables.txt", "w") as f:
# #     prob.model.list_vars(out_stream=f, print_arrays=True, units=True)

# print('Lift:', prob.get_val('lift', units='N')) # M1 TOW 8.35 =~ 37.14
# print('Drag:', prob.get_val('drag', units='N'))
# print('CL:', prob.get_val('lifting_surface_CL'))
# print('CD:', prob.get_val('CD'))

# print('CD_fus:', prob.get_val('CD_fus'))
# print('CD_vtail:', prob.get_val('CD_vtail'))
# print('CD_gear:', prob.get_val('CD_gear'))
# print('Lifting surface CD:', prob.get_val('lifting_surface_CD'))

# print('Avg CD:', prob.get_val('avg_CD'))

# print('q:', prob.get_val(Dynamic.Atmosphere.DYNAMIC_PRESSURE, units='N/m**2'))
# print('S:', prob.get_val(Aircraft.Wing.AREA, units='m**2'))