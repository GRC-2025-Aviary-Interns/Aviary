import openmdao.api as om

from aviary.subsystems.subsystem_builder_base import SubsystemBuilderBase
from aviary.examples.external_subsystems.OAS_weight.OAS_aero_analysis import OASAero
from aviary.variable_info.variables import Aircraft, Dynamic
from aviary.utils.aviary_values import AviaryValues


class OASAeroBuilder(SubsystemBuilderBase):
    def __init__(self, name='aero_analysis'):
        super().__init__(name)

    def build(self, aviary_inputs: AviaryValues, aviary_outputs: AviaryValues):
        return OASAero(aviary_inputs=aviary_inputs, num_nodes=1)

    def build_mission(self, num_nodes, aviary_inputs, **kwargs):
        return OASAero(
            aviary_inputs=aviary_inputs,
            num_nodes=num_nodes
        )

    def mission_inputs(self, **kwargs):
        return [
            Dynamic.Mission.ALTITUDE,
            Dynamic.Mission.VELOCITY,
        ]
    
    def mission_outputs(self, **kwargs):
        return [
            Dynamic.Vehicle.DRAG,
            Dynamic.Vehicle.LIFT,
            'aero_point_0.CL',
            'aero_point_0.CD'
        ]
    
    def get_parameters(self, aviary_inputs=None, **kwargs):
        params = {}

        params[Aircraft.Wing.SPAN] = {
            'units': 'm',
            'static_target': True
        }
        params[Aircraft.Wing.ROOT_CHORD] = {
            'units': 'm',
            'static_target': True
        }
        params[Aircraft.Wing.SWEEP] = {
            'units': 'deg',
            'static_target': True
        }
        params[Aircraft.Wing.INCIDENCE] = {
            'units': 'deg',
            'static_target': True
        }
        params[Aircraft.HorizontalTail.SPAN] = {
            'units': 'm',
            'static_target': True
        }
        params[Aircraft.HorizontalTail.ROOT_CHORD] = {
            'units': 'm',
            'static_target': True
        }
        params[Aircraft.HorizontalTail.SWEEP] = {
            'units': 'deg',
            'static_target': True
        }

        return params

    def needs_mission_solver(self, aviary_inputs):
        return False
