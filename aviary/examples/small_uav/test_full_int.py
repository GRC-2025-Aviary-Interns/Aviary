"""
Define an aviary mission with an NPSS defined engine and RC aerodynamics. During pre-mission the engine is designed and
an engine deck is made. During the mission the deck is used for performance with RC aero analysis. Weight is estimated
using the default Aviary method. The engine model was developed using NPSS v3.2.
"""
from copy import deepcopy
import numpy as np
import openmdao.api as om
import aviary.api as av
from aviary.examples.small_uav.dbf_example_phase import phase_info
from aviary.examples.external_subsystems.dbf_based_mass.dbf_mass_builder import DBFMassBuilder
from aviary.examples.external_subsystems.dbf_based_geometry.dbf_geometry_builder import DBFGeometryBuilder
from aviary.subsystems.propulsion.rc_electric.rc_builder import RCBuilder
from aviary.subsystems.aerodynamics.rc_aero_builder import RCAeroBuilder
from aviary.variable_info.variables import Aircraft, Dynamic

# Create builders
rc_prop = RCBuilder()
aero_builder = RCAeroBuilder()

phase_info = deepcopy(phase_info)

# Remove climb and descent phases as in original
phase_info.pop('climb')
phase_info.pop('descent')

# Set up pre-mission systems
phase_info['pre_mission']['external_subsystems'] = [DBFGeometryBuilder(), DBFMassBuilder()]

# Configure cruise phase with RC aerodynamics
phase_info['cruise']['external_subsystems'] = [aero_builder]
phase_info['cruise']['subsystem_options']['core_aerodynamics'] = {'method': 'external'}

# Add aerodynamics configuration from first script
phase_info['cruise']['user_options'].update({
    'mach_optimize': False,
    'mach_polynomial_order': 1,
    'mach_initial': (0.08, 'unitless'),
    'mach_final': (0.09, 'unitless'),
    'mach_bounds': ((0.07, 0.11), 'unitless'),
    'altitude_optimize': False,
    'altitude_polynomial_order': 1,
    'altitude_initial': (200, 'ft'),
    'altitude_final': (200, 'ft'),
    'altitude_bounds': ((100, 200), 'ft'),
    'throttle_enforcement': 'control',
    'throttle_optimize': True,
    'time_initial': (0, 's'),
    'time_duration_bounds': ((1.0, 100.0), 's'),
})

# Add initial guesses
phase_info['cruise']['initial_guesses'] = {
    'mass': (4.53, 'kg'),
    'time': ([0, 60], 's'),
    'distance': ([0, 300], 'ft')
}

# Create problem
prob = av.AviaryProblem(verbosity=2)
prob.options['group_by_pre_opt_post'] = True

# Load aircraft and options data
prob.load_inputs(
    'models/aircraft/test_aircraft/small_scale_uav.csv',
    phase_info,
    engine_builders=[rc_prop],
)

# Set aircraft geometry parameters from first script
prob.aviary_inputs.set_val(Dynamic.Mission.ALTITUDE, 200, units='ft') 
prob.aviary_inputs.set_val(Dynamic.Mission.VELOCITY, 37, units='m/s')

prob.aviary_inputs.set_val(Aircraft.Wing.THICKNESS_TO_CHORD, 0.10) 
prob.aviary_inputs.set_val(Aircraft.Wing.MAX_THICKNESS_LOCATION, 0.266) 
prob.aviary_inputs.set_val(Aircraft.Wing.TAPER_RATIO, 1, units='unitless')
prob.aviary_inputs.set_val(Aircraft.Wing.SWEEP, 0, units='deg')
prob.aviary_inputs.set_val(Aircraft.Wing.INCIDENCE, 0, units='deg')

prob.aviary_inputs.set_val(Aircraft.HorizontalTail.THICKNESS_TO_CHORD, 0.14) 
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.TAPER_RATIO, 1, units='unitless')
prob.aviary_inputs.set_val(Aircraft.HorizontalTail.SWEEP, 0, units='deg')

prob.aviary_inputs.set_val(Aircraft.VerticalTail.THICKNESS_TO_CHORD, 0.14) 
prob.aviary_inputs.set_val(Aircraft.VerticalTail.TAPER_RATIO, 1, units='unitless')

prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_HEIGHT, 0.172, units='m')
prob.aviary_inputs.set_val(Aircraft.Fuselage.MAX_WIDTH, 0.114, units='m')
prob.aviary_inputs.set_val(Aircraft.Wing.CENTER_DISTANCE, 0.511, units='unitless')
prob.aviary_inputs.set_val(Aircraft.Fuselage.LENGTH, 1.190244, units='m')

prob.aviary_inputs.set_val(Dynamic.Vehicle.MASS, 3.787, units='kg')

# Build the problem
prob.check_and_preprocess_inputs()
prob.add_pre_mission_systems()
prob.add_phases()
prob.add_post_mission_systems()
prob.link_phases()

# Add driver
optimizer = 'IPOPT'
prob.add_driver(optimizer=optimizer)

# Add design variables
prob.add_design_variables()

# Add specific design variables from first script
prob.model.add_design_var('traj.cruise.rhs_all.rc_aero_analysis.alpha', lower=-5.0, upper=15.0)
prob.model.add_constraint('traj.cruise.rhs_all.rc_aero_analysis.lifting_surface_CD', lower=0, upper=1)
prob.model.add_constraint('traj.cruise.rhs_all.rc_aero_analysis.lifting_surface_CL', lower=0, upper=1)


# Configure driver recording options

recorder = om.SqliteRecorder("opt_record.sql")

prob.driver.add_recorder(recorder)
prob.driver.recording_options["includes"] = ["*"]  # record everything
prob.driver.recording_options["record_objectives"] = True
prob.driver.recording_options["record_constraints"] = True
prob.driver.recording_options["record_desvars"] = True

prob.add_objective('time')

# Setup and run
prob.setup()

# Set initial values
prob.set_val('traj.cruise.rhs_all.rc_aero_analysis.alpha', np.array([10.0, 10.0, 10.0, 10.0]), units='deg')

prob.set_initial_guesses()

# Run the problem
# prob.run_aviary_problem(suppress_solver_print=False)
prob.run_model()

prob.check_partials(method='fd', step=1.01e-6, compact_print=True, includes=['*htail*'])

# Output results
with open("variables.txt", "w") as f:
    prob.model.list_vars(out_stream=f, print_arrays=True, units=True)

# Print aerodynamic results
print('Lift:', prob.get_val('traj.cruise.rhs_all.lift', units='lbf')) 
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
print('Wing span:', prob.get_val(Aircraft.Wing.SPAN))