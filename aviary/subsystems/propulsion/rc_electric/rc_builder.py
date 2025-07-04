from aviary.subsystems.propulsion.rc_electric.rc_performance import RCPropGroup
from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.variable_info.variables import Aircraft, Dynamic, Mission

class RCBuilder(SubsystemBuilderBase):
    def __init__(self, name='rc_eletric'):
        """Initializes the PropellerBuilder object with a given name."""
        super().__init__(name)

    def build_pre_mission(self, aviary_inputs):
        """Builds an OpenMDAO system for the pre-mission computations of the subsystem."""
        return

    def build_mission(self, num_nodes, aviary_inputs):
        """Builds an OpenMDAO system for the mission computations of the subsystem."""
        return RCPropGroup(num_nodes=num_nodes, aviary_options=aviary_inputs)

    def get_design_vars(self):
        """
        Design vars are only tested to see if they exist in pre_mission
        Returns a dictionary of design variables for the gearbox subsystem, where the keys are the
        names of the design variables, and the values are dictionaries that contain the units for
        the design variable, the lower and upper bounds for the design variable, and any
        additional keyword arguments required by OpenMDAO for the design variable.

        Returns
        -------
        parameters : dict
        A dict of names for the propeller subsystem.
        """
        # TODO bounds are rough placeholders
        # TODO add the rest of the resign variables, also remove the feature as an external subsystem
        # TODO check with eliot about adding potentially new design variables (i.e. kv can be declared or calculated)
        DVs = {
            Aircraft.Engine.Propeller.PITCH: {
                'units': 'inch',
                'lower': 100,
                'upper': 200,
                # 'val': 100,  # initial value
            },
            Aircraft.Engine.Propeller.DIAMETER: {
                'units': 'in',
                'lower': 0.0,
                'upper': None,
                # 'val': 8,  # initial value
            },

        }
        return DVs

    def get_parameters(self, aviary_inputs=None, phase_info=None):
        """
        Parameters are only tested to see if they exist in mission.
        The value doesn't change throughout the mission.
        Returns a dictionary of fixed values for the propeller subsystem, where the keys
        are the names of the fixed values, and the values are dictionaries that contain
        the fixed value for the variable, the units for the variable, and any additional
        keyword arguments required by OpenMDAO for the variable.

        Returns
        -------
        parameters : dict
        A dict of names for the propeller subsystem.
        """

        #TODO add new variables, including dvs and optional inputs
        parameters = {
            Aircraft.Engine.Propeller.DIAMETER: {
                'val': 0.0,
                'units': 'ft',
            },
            Aircraft.Nacelle.PITCH: {
                'val': 0.0,
                'units': 'ft',
            },
        }

        return parameters

    def get_mass_names(self):
        return [Aircraft.Engine.Motor.MASS]
    
    #TODO add new outputs
    def get_outputs(self):
        return [
            Dynamic.Vehicle.Propulsion.PROP_POWER + '_out',
            Dynamic.Vehicle.Propulsion.RPM + '_out',
            Dynamic.Vehicle.Propulsion.PROP_THRUST + '_out',
        ]
