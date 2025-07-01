import numpy as np
import openmdao.api as om

from aviary.variable_info.functions import add_aviary_input, add_aviary_option, add_aviary_output
from aviary.variable_info.variables import Aircraft, Dynamic, Settings

class Battery(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
        # add_aviary_option(self, Settings.VERBOSITY)

    def setup(self):
        nn = self.options['num_nodes']

        #TODO: add Battery options
        add_aviary_input(self, Aircraft.Engine.Battery.VOLTAGE, val=np.zeros(nn), units = "V")
        add_aviary_input(self, Aircraft.Engine.Battery.MASS, val=np.zeros(nn), units='kg')
        add_aviary_input(self, Aircraft.Engine.Battery.RESISTANCE, val=np.zeros(nn), units='ohm')
        self.add_input('current', val=np.zeros(nn), units='A')

        self.add_output('voltage_out', val=np.zeros(nn), units='V')
        self.add_output('power', val=np.zeros(nn), units='W')
        self.add_output('nominal_capacity', val=np.zeros(nn), units='A*h')
        self.add_output('energy', val=np.zeros(nn), units='W*h', desc='For an individual battery')
        ar = np.arange(nn)

        self.declare_partials(
            ['voltage_out', 'power'], 
            [
                Aircraft.Engine.Battery.VOLTAGE, 
                'current', 
                Aircraft.Engine.Battery.RESISTANCE
                ],
            rows=ar, cols=ar
        )
        #TODO: check if arange is necessary

        self.declare_partials(
            'nominal_capacity',
            Aircraft.Engine.Battery.MASS,
            rows=ar, cols=ar,
        )

        self.declare_partials(
            'energy',
            [
                Aircraft.Engine.Battery.MASS,
                Aircraft.Engine.Battery.VOLTAGE,
            ],
            rows=ar, cols=ar
        )

    def compute(self, inputs, outputs):
        V = inputs[Aircraft.Engine.Battery.VOLTAGE]
        I = inputs['current']
        R = inputs[Aircraft.Engine.Battery.RESISTANCE]

        outputs['nominal_capacity'] = inputs[Aircraft.Engine.Battery.MASS] * 7.3 - 0.246 # Per Peter Sharpe TODO VERIFY
        outputs['energy'] = V * outputs['nominal_capacity']
        outputs['power'] = V * I - I**2 * R
        outputs['voltage_out'] = V - I * R

    def compute_partials(self, inputs, outputs, partials):
        V = inputs[Aircraft.Engine.Battery.VOLTAGE]
        I = inputs['current']
        R = inputs[Aircraft.Engine.Battery.RESISTANCE]
        nominal_capacity = inputs[Aircraft.Engine.Battery.MASS] * 7.3 - 0.246

        partials['voltage_out', Aircraft.Engine.Battery.VOLTAGE] = 1
        partials['voltage_out', 'current'] = -R
        partials['voltage_out', Aircraft.Engine.Battery.RESISTANCE] = -I

        partials['power', 'voltage_supply'] = I
        partials['power', 'current'] = V - 2 * I * R
        partials['power', Aircraft.Engine.Battery.RESISTANCE] = -I**2

        partials['nominal_capacity', Aircraft.Engine.Battery.MASS] = 7.3

        partials['energy', Aircraft.Engine.Battery.VOLTAGE] = nominal_capacity
        partials['energy', 'mass'] = V * 7.3


class ElectronicSpeedController(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

        self.options.declare('a', default = 1.6054, desc = 'a coefficient for efficiency(throttle) equation: efficiency = a * (1 - 1 / (1 + c*throttle^d))')
        self.options.declare('b', default = 1.6519, desc = 'b coefficient for efficiency(throttle) equation: efficiency = a * (1 - 1 / (1 + c*throttle^d))')
        self.options.declare('c', default = 0.6455, desc = 'c coefficient for efficiency(throttle) equation: efficiency = a * (1 - 1 / (1 + c*throttle^d))')
    
    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('voltage_in', val=np.zeros(nn), units = 'V')
        self.add_input('current_in', val=np.zeros(nn), units = 'A')
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.THROTTLE, val=np.zeros(nn), units='unitless')

        self.add_output('efficiency', val=np.zeros(nn), units='unitless')
        self.add_output('voltage_out', val=np.zeros(nn), units = 'V')
        self.add_output('current_out', val=np.zeros(nn), units = 'A')
        self.add_output('power', val=np.zeros(nn), units = 'W')
        ar = np.arange(nn)

        self.declare_partials('efficiency', 'throttle', rows=ar, cols=ar)
        self.declare_partials('voltage_out', ['voltage_in', 'throttle'], rows=ar, cols=ar)
        self.declare_partials('current_out', ['current_in', 'throttle'], rows=ar, cols=ar)
        self.declare_partials('power', ['voltage_in', 'current_in', 'throttle'], rows=ar, cols=ar)

    def compute(self, inputs, outputs):
        
        a = self.options['a']
        b = self.options['b']
        c = self.options['c']
        outputs['efficiency'] = a * (1 - 1 / (1 + b*inputs['throttle']**c))

        outputs['voltage_out'] = inputs['voltage_in'] * inputs['throttle'] * outputs['efficiency']
        outputs['current_out'] = inputs['current_in'] / inputs['throttle']
        outputs['power'] = (outputs['efficiency'] - 1) * inputs['current_in'] * inputs['voltage_in']

    def compute_partials(self, inputs, partials):

        a = self.options['a']
        b = self.options['b']
        c = self.options['c']
        t = inputs['throttle']
        efficiency = a * (1 - 1 / (1 + b*t**c))
        partials['efficiency', 'throttle'] = a*b*c*t**(c - 1) / (b*t**c + 1)**2

        partials['voltage_out', 'voltage_in'] = inputs['throttle'] * efficiency
        partials['voltage_out', 'throttle'] = inputs['voltage_in'] * (efficiency + inputs['throttle'] * partials['efficiency', 'throttle'])

        partials['current_out', 'current_in'] = 1 / inputs['throttle']
        partials['current_out', 'throttle'] = -inputs['current_in'] / inputs['throttle']**2

        partials['power', 'voltage_in'] = (efficiency - 1) * inputs['current_in']
        partials['power', 'current_in'] = (efficiency - 1) * inputs['voltage_in']
        partials['power', 'throttle'] = inputs['current_in'] * inputs['voltage_in'] * partials['efficiency', 'throttle']


class Motor(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']

        add_aviary_input(self, Aircraft.Engine.Motor.IDLE_CURRENT, val=np.zeros(nn), units='A')
        add_aviary_input(self, Aircraft.Engine.Motor.PEAK_CURRENT, val=np.zeros(nn), units='A')
        add_aviary_input(self, Aircraft.Engine.Motor.RESISTANCE, val=np.zeros(nn), units='ohm')
        add_aviary_input(self, Aircraft.Engine.Motor.KV, val=np.zeros(nn), units='rpm/V')
        add_aviary_input(self, Aircraft.Engine.Motor.MASS, val=np.zeros(nn), units ='kg')
        self.add_input('voltage_in', val=np.zeros(nn), units = 'V')
        self.add_input('current', val=np.zeros(nn), units = 'A')

        add_aviary_output(self, Dynamic.Vehicle.Propulsion.RPM, val=np.zeros(nn), units='rpm')
        self.add_output('power', val=np.zeros(nn), units='W')
        self.add_output('current_con', val=np.zeros(nn), units="A")
        ar=np.arange(nn)

        self.declare_partials(
            [
                Dynamic.Vehicle.Propulsion.RPM, 
                'power'
            ], 
            [
                'voltage_in', 
                'current', 
                Aircraft.Engine.Motor.RESISTANCE, 
                Aircraft.Engine.Motor.KV, 
                Aircraft.Engine.Motor.IDLE_CURRENT,
            ],
            rows=ar, cols=ar
        )

        self.declare_partials(
            'current_con',
            ['current', Aircraft.Engine.Motor.PEAK_CURRENT],
            rows=ar, cols=ar
        )

    def compute(self, inputs, outputs):
        #TODO: Add computation for no given kv, optimizing the motor
        R = inputs[Aircraft.Engine.Motor.RESISTANCE]
        kv = inputs[Aircraft.Engine.Motor.KV]
        voltage_prop = inputs['voltage_in'] - inputs['current'] * R

        outputs[Dynamic.Vehicle.Propulsion.RPM] = kv * voltage_prop
        outputs['power'] = -inputs['current']**2 * R - inputs['Aircraft.Engine.Motor.IDLE_CURRENT'] * voltage_prop
        outputs["current_con"] = inputs["current"] - inputs[Aircraft.Engine.Motor.PEAK_CURRENT] #TODO verify this is best way to do

    def compute_partials(self, inputs, partials):
        R = inputs[Aircraft.Engine.Motor.RESISTANCE]
        
        voltage_prop = inputs['voltage_in'] - inputs['current'] * R
        dvoltage_prop_dvoltage_in = 1
        dvoltage_prop_dcurrent = -R
        dvoltage_prop_dresistance = -inputs['current']

        partials[Dynamic.Vehicle.Propulsion.RPM, 'voltage_in'] = inputs[Aircraft.Engine.Motor.KV] * dvoltage_prop_dvoltage_in
        partials[Dynamic.Vehicle.Propulsion.RPM, 'current'] = inputs[Aircraft.Engine.Motor.KV] * dvoltage_prop_dcurrent
        partials[Dynamic.Vehicle.Propulsion.RPM, Aircraft.Engine.Motor.RESISTANCE] = inputs[Aircraft.Engine.Motor.KV] * dvoltage_prop_dresistance
        partials[Dynamic.Vehicle.Propulsion.RPM, Aircraft.Engine.Motor.KV] = voltage_prop
        partials[Dynamic.Vehicle.Propulsion.RPM, Aircraft.Engine.Motor.IDLE_CURRENT] = 0

        partials['power', 'voltage_in'] = -inputs[Aircraft.Engine.Motor.IDLE_CURRENT] * dvoltage_prop_dvoltage_in
        partials['power', 'current'] = -2 * inputs['current'] * R - inputs[Aircraft.Engine.Motor.IDLE_CURRENT] * dvoltage_prop_dcurrent
        partials['power', Aircraft.Engine.Motor.RESISTANCE] = -inputs['current']**2 - inputs[Aircraft.Engine.Motor.IDLE_CURRENT] * dvoltage_prop_dresistance
        partials['power', Aircraft.Engine.Motor.KV] = 0
        partials['power', Aircraft.Engine.Motor.IDLE_CURRENT] = -voltage_prop
        
        partials['current_con', Aircraft.Engine.Motor.PEAK_CURRENT] = -1
        partials['current_con', 'current'] = 1


class Propeller(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
    #TODO: ask about adding more propellers
    def setup(self): 
        nn = self.options['num_nodes']
        add_aviary_input(self, Dynamic.Atmosphere.DENSITY, val=np.zeros(nn), units = 'kg/m**3')
        add_aviary_input(self, Aircraft.Engine.Propeller.DIAMETER, units = 'm')
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.RPM, val=np.zeros(nn), units = 'rev/s')
        self.add_input("ct", val=np.zeros(nn), units='unitless')
        self.add_input("cp", val=np.zeros(nn), units='unitless')
        add_aviary_input(self, Aircraft.Engine.NUM_ENGINES, val=np.zeros(nn), units='unitless') #TODO verify no number of nodes and change eventually

        add_aviary_output(self, Dynamic.Vehicle.Propulsion.PROP_THRUST, val=np.zeros(nn), units='N')
        add_aviary_output(self, Dynamic.Vehicle.Propulsion.PROP_POWER, val=np.zeros(nn), units='W')
        ar =np.arange(nn)

        self.declare_partials(
            Dynamic.Vehicle.Propulsion.PROP_THRUST,
            [
                Dynamic.Atmosphere.DENSITY, 
                Aircraft.Engine.Propeller.DIAMETER,
                Dynamic.Vehicle.Propulsion.RPM,
                'ct',
                Aircraft.Engine.NUM_ENGINES,
            ],
            rows=ar, cols=ar,
        )

        self.declare_partials(
            Dynamic.Vehicle.Propulsion.PROP_POWER,
            [
                Dynamic.Atmosphere.DENSITY, 
                Aircraft.Engine.Propeller.DIAMETER,
                Dynamic.Vehicle.Propulsion.RPM,
                'cp',
            ],
            rows=ar, cols=ar
        )

    def compute(self, inputs, outputs):
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        D = inputs[Aircraft.Engine.Propeller.DIAMETER]
        n = inputs [Dynamic.Vehicle.Propulsion.RPM]

        outputs[Dynamic.Vehicle.Propulsion.PROP_THRUST] = (rho * n**2 * D**4 * inputs["ct"] * inputs[Aircraft.Engine.NUM_ENGINES])
        outputs[Dynamic.Vehicle.Propulsion.PROP_POWER] = (rho * n**3 * D**5 * inputs["cp"])

    def compute_partials(self, inputs, partials):
        rho = inputs[Dynamic.Atmosphere.DENSITY]
        D = inputs[Aircraft.Engine.Propeller.DIAMETER]
        n = inputs [Dynamic.Vehicle.Propulsion.RPM]

        partials[Dynamic.Vehicle.Propulsion.PROP_THRUST, Dynamic.Atmosphere.DENSITY] = n**2 * D**4 * inputs["ct"] * inputs[Aircraft.Engine.NUM_ENGINES]
        partials[Dynamic.Vehicle.Propulsion.PROP_THRUST, Aircraft.Engine.Propeller.DIAMETER] = rho * n**2 * 4 * D**3 * inputs["ct"] * inputs[Aircraft.Engine.NUM_ENGINES]
        partials[Dynamic.Vehicle.Propulsion.PROP_THRUST, Dynamic.Vehicle.Propulsion.RPM] = rho * 2 * n * D**4 * inputs["ct"] * inputs[Aircraft.Engine.NUM_ENGINES]
        partials[Dynamic.Vehicle.Propulsion.PROP_THRUST, 'ct'] = rho * n**2 * D**4 * inputs[Aircraft.Engine.NUM_ENGINES]
        partials[Dynamic.Vehicle.Propulsion.PROP_THRUST, Aircraft.Engine.NUM_ENGINES] = rho * n**2 * D**4 * inputs["ct"]

        partials[Dynamic.Vehicle.Propulsion.PROP_POWER, Dynamic.Atmosphere.DENSITY] = n**3 * D**5 * inputs["cp"]
        partials[Dynamic.Vehicle.Propulsion.PROP_POWER, Aircraft.Engine.Propeller.DIAMETER] = rho * n**3 * 5 * D**4 * inputs["cp"]
        partials[Dynamic.Vehicle.Propulsion.PROP_POWER, Dynamic.Vehicle.Propulsion.RPM] = rho * 3 * n**2 * D**5 * inputs["cp"]
        partials[Dynamic.Vehicle.Propulsion.PROP_POWER, 'cp'] = rho * n**3 * D**5 


class PowerResiduals(om.ImplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)

    def setup(self):
        nn = self.options['num_nodes']
        self.add_input('power_batt', val=np.zeros(nn), units='W')
        self.add_input('power_esc', val=np.zeros(nn), units='W')
        self.add_input('power_motor', val=np.zeros(nn), units='W')
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.PROP_POWER, val=np.zeros(nn), units='W') #TODO verify no connection needed in group
        
        #We want to make a good initial guess on the current
        self.add_output('current', val=np.ones(nn)*30, units='A') 
        self.add_residual('power_net', shape=(nn, ), units='W')

        self.declare_partials('*', '*', method='cs') #TODO try to change

    def apply_nonlinear(self, inputs, residuals):
        residuals['current'] = inputs['power_batt'] + inputs['power_esc'] + inputs['power_motor'] - inputs[Dynamic.Vehicle.Propulsion.PROP_POWER]

#TODO: reading in of data should be changed later:
from Parsing.PropDataReader import PropDataReader
from Parsing.LHS_cosine_Datasampler import SampleDataLHScosine
x, ct, cp = PropDataReader()
x, ct, cp, _, _, _ = SampleDataLHScosine(x, ct, cp)
class RCPropGroup(om.Group):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
    def setup(self):
        nn = self.options['num_nodes']
        self.add_subsystem('battery', Battery())
        self.add_subsystem('esc', ElectronicSpeedController())
        self.add_subsystem('motor', Motor(), promotes_outputs=Dynamic.Vehicle.Propulsion.RPM)
        prop_coef = om.MetaModelUnStructuredComp(vec_size = nn, default_surrogate = om.KrigingSurrogate(eval_rmse=True, training_cache='rc_surrogate'))
        #TODO: check promotions
        self.add_subsystem(
            'propco', 
            prop_coef, 
            promotes_inputs=[
                Dynamic.Vehicle.Propulsion.RPM, 
                Dynamic.Mission.VELOCITY, 
                Aircraft.Engine.Propeller.DIAMETER, 
                Aircraft.Engine.Propeller.PITCH
            ],
            promotes_outputs=['ct', 'cp']
        )
        add_aviary_input(prop_coef, Aircraft.Engine.Propeller.DIAMETER, val=np.zeros(nn), training_data=x[:, 0], units = 'm')
        add_aviary_input(prop_coef, Aircraft.Engine.Propeller.PITCH, training_data=x[:, 1], units='deg')
        add_aviary_input(prop_coef, Dynamic.Vehicle.Propulsion.RPM, val=np.zeros(nn), training_data=x[:, 2], units = 'rev/s')
        add_aviary_input(prop_coef, Dynamic.Mission.VELOCITY, val = np.zeros(nn), training_data = x[:,3], units='m/s')

        prop_coef.add_output('ct', val=np.zeros(nn), training_data = ct, units='unitless', surrogate = om.KrigingSurrogate(eval_rmse = True, lapack_driver='gesdd', training_cache = 'thrust_surrogate'))
        prop_coef.add_output('cp', val=np.zeros(nn), training_data = cp, units='unitless', surrogate = om.KrigingSurrogate(eval_rmse = True, lapack_driver='gesdd', training_cache = 'power_surrogate'))

        self.add_subsystem(
            'prop', 
            Propeller(), 
            promotes_inputs=[Aircraft.Engine.Propeller.Diameter, Aircraft.Engine.Propeller.PITCH, 'ct', 'cp', Aircraft.Engine.NUM_ENGINES, Dynamic.Atmosphere.DENSITY])
        self.add_subsystem('power_net', PowerResiduals())

        self.connect('battery.voltage_out', 'esc.voltage_in')
        self.connect('esc.voltage_out', 'motor.voltage_in')
        self.connect('esc.current_out', 'motor.current')
        # self.connect('motor.rpm', prop.rpm) TODO check if this is needed

        self.connect('battery.power', 'power_net.power_batt')
        self.connect('esc.power', 'power_net.power_esc')
        self.connect('motor.power', 'power_net.power_motor')
        self.connect('power_net.current', ['battery.current', 'esc.current_in'])
        # self.connect('prop.power', 'power_net.power_prop') TODO check if this is needed