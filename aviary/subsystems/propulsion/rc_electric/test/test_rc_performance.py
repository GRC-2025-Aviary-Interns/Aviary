import unittest

import numpy as np
import openmdao.api as om
from openmdao.utils.assert_utils import assert_check_partials, assert_near_equal

from aviary.subsystems.atmosphere.atmosphere import Atmosphere
from aviary.subsystems.propulsion.rc_electric.rc_performance import RCPropGroup 
from aviary.variable_info.functions import setup_model_options
from aviary.variable_info.options import get_option_defaults
from aviary.variable_info.variables import Aircraft, Dynamic, Settings

#TODO: dependant on what prediction method is chosen: change metamodel to its own class to be tested.
# class RCPerformanceTest(unittest.TestCase):
    # """Test computations for RC Performance Group."""
    
    # def test_1d_residual(self):
    #     tol = 1e-5
    #     prob = om.Problem()
    #     prob.model.add_subsystem(
    #         'group',
    #         RCPropGroup(num_nodes=1),
    #         promotes=['*'],
    #     )
        
    #     prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    #     prob.model.nonlinear_solver.options["maxiter"] = 30
    #     prob.model.nonlinear_solver.options["err_on_non_converge"] = False
    #     prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()
    #     prob.model.nonlinear_solver.linesearch.options["bound_enforcement"] = "scalar"
    #     prob.model.nonlinear_solver.linesearch.options["print_bound_enforce"] = True
    #     prob.model.linear_solver = om.DirectSolver(assemble_jac=True)#, rhs_checking =True)

    #     prob.setup()
    #     prob.set_val(Aircraft.Battery.MASS, .707, units='kg')
    #     prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
    #     prob.set_val(Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
    #     prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, 0.8, units='unitless')
    #     prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
    #     prob.set_val(Aircraft.Engine.Motor.PEAK_CURRENT, 120, units='A')
    #     prob.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.032, units='ohm')
    #     prob.set_val(Aircraft.Engine.Motor.KV, 420, units='rpm/V')
    #     prob.set_val(Aircraft.Engine.Motor.MASS, 0.288, units='kg')
    #     prob.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')
    #     prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
    #     prob.set_val(Aircraft.Engine.Propeller.PITCH, 10, units='inch')
    #     prob.set_val(Aircraft.Engine.NUM_ENGINES, 1, units='unitless')
    #     prob.set_val(Dynamic.Mission.VELOCITY, 20, units='ft/s')
    #     prob.run_model()
    #     battery_power = prob.get_val('battery.power', units='W')
    #     esc_power = prob.get_val('esc.power', units='W')
    #     motor_power = prob.get_val('motor.power', units='W')
    #     prop_power = prob.get_val(Dynamic.Vehicle.Propulsion.PROP_POWER, units='W')
    #     power_residual = battery_power + esc_power + motor_power - prop_power
    #     assert_near_equal(power_residual, 0, tolerance=tol)
    #     partial_data = prob.check_partials(
    #         out_stream=None,
    #         compact_print=True,
    #         show_only_incorrect=True,
    #         form='central',
    #         method='fd',
    #         minimum_step=1e-12,
    #         abs_err_tol=5.0e-4,
    #         rel_err_tol=5.0e-5,
    #     )
    #     assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)

    # def test_multid_residual(self):
tol = 1e-5
prob = om.Problem()
prob.model.add_subsystem(
    'group',
    RCPropGroup(num_nodes=3),
    promotes=['*'],
)

prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
prob.model.nonlinear_solver.options["maxiter"] = 30
prob.model.nonlinear_solver.options["err_on_non_converge"] = False
prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()
prob.model.nonlinear_solver.linesearch.options["bound_enforcement"] = "scalar"
prob.model.nonlinear_solver.linesearch.options["print_bound_enforce"] = True
prob.model.linear_solver = om.DirectSolver(assemble_jac=True)#, rhs_checking =True
prob.setup()
prob.set_val(Aircraft.Battery.MASS, .707, units='kg')
prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
prob.set_val(Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, 0.8, units='unitless')
prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
prob.set_val(Aircraft.Engine.Motor.PEAK_CURRENT, 120, units='A')
prob.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.032, units='ohm')
prob.set_val(Aircraft.Engine.Motor.KV, 420, units='rpm/V')
prob.set_val(Aircraft.Engine.Motor.MASS, 0.288, units='kg')
prob.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')
prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
prob.set_val(Aircraft.Engine.Propeller.PITCH, 10, units='inch')
prob.set_val(Aircraft.Engine.NUM_ENGINES, 1, units='unitless')
prob.set_val(Dynamic.Mission.VELOCITY, 20, units='ft/s')
prob.run_model()
battery_power = prob.get_val('battery.power', units='W')
esc_power = prob.get_val('esc.power', units='W')
motor_power = prob.get_val('motor.power', units='W')
prop_power = prob.get_val(Dynamic.Vehicle.Propulsion.PROP_POWER, units='W')
power_residual = battery_power + esc_power + motor_power - prop_power
assert_near_equal(power_residual, 0, tolerance=tol)
partial_data = prob.check_partials(
    out_stream=None,
    compact_print=True,
    show_only_incorrect=True,
    form='central',
    method='fd',
    minimum_step=1e-12,
    abs_err_tol=5.0e-4,
    rel_err_tol=5.0e-5,
)
assert_check_partials(partial_data, atol=5e-4, rtol=1e-4)
    # def opti_test_1d(self):
    #     tol = 1e-5
    #     prob = om.Problem()
    #     prob.model.add_subsystem(
    #         'group',
    #         RCPropGroup(num_nodes=1),
    #         promotes=['*'],
    #     )

    #     prob.driver = om.pyOptSparseDriver()
    #     prob.driver.options["optimizer"] = "IPOPT"
    #     prob.driver.options["debug_print"] = ["desvars", "nl_cons", "objs"]

    #     prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
    #     prob.model.nonlinear_solver.options["maxiter"] = 30
    #     prob.model.nonlinear_solver.options["err_on_non_converge"] = False
    #     prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS()
    #     prob.model.nonlinear_solver.linesearch.options["bound_enforcement"] = "scalar"
    #     prob.model.nonlinear_solver.linesearch.options["print_bound_enforce"] = True
    #     prob.model.linear_solver = om.DirectSolver(assemble_jac=True)#, rhs_checking =True)

    #     prob.setup()
    #     prob.set_val(Aircraft.Battery.MASS, .707, units='kg')
    #     prob.set_val(Aircraft.Battery.VOLTAGE, 22.2, units='V')
    #     prob.set_val(Aircraft.Battery.RESISTANCE, 0.05, units='ohm')
    #     prob.set_val(Dynamic.Vehicle.Propulsion.THROTTLE, 0.8, units='unitless')
    #     prob.set_val(Aircraft.Engine.Motor.IDLE_CURRENT, 0.91, units='A')
    #     prob.set_val(Aircraft.Engine.Motor.PEAK_CURRENT, 120, units='A')
    #     prob.set_val(Aircraft.Engine.Motor.RESISTANCE, 0.032, units='ohm')
    #     prob.set_val(Aircraft.Engine.Motor.KV, 420, units='rpm/V')
    #     prob.set_val(Aircraft.Engine.Motor.MASS, 0.288, units='kg')
    #     prob.set_val(Dynamic.Atmosphere.DENSITY, 1.225, units='kg/m**3')
    #     prob.set_val(Aircraft.Engine.Propeller.DIAMETER, 20, units='inch')
    #     prob.set_val(Aircraft.Engine.Propeller.PITCH, 10, units='inch')
    #     prob.set_val(Aircraft.Engine.NUM_ENGINES, 1, units='unitless')
    #     prob.set_val(Dynamic.Mission.VELOCITY, 20, units='ft/s')
    #     prob.run_model()

    #     prob.model.add_design_var(Aircraft.Battery.MASS)
    #     prob.model.add_design_var(Aircraft.Engine.Motor.IDLE_CURRENT)
    #     prob.model.add_design_var(Aircraft.Engine.Motor.MASS)
    #     prob.model.add_design_var(Aircraft.Engine.Propeller.DIAMETER)
    #     prob.model.add_design_var(Aircraft.Engine.Propeller.PITCH)
    #     prob.model.add_design_var(Dynamic.Mission.VELOCITY, units='m/s', lower=0, upper=35)

# if __name__ == '__main__':
    # unittest.main()
    # test = PropellerPerformanceTest()
    # test.setUp()
    # test.test_case_15_16_17()
