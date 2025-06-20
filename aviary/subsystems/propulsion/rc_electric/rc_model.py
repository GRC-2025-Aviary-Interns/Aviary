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
        
        self.declare_partials(
            ['voltage_out', 'power'], 
            [
                Aircraft.Engine.Battery.VOLTAGE, 
                'current', 
                Aircraft.Engine.Battery.RESISTANCE
                ],
        )
        #TODO: check if arange is necessary

        self.declare_partials(
            'nominal_capacity',
            Aircraft.Engine.Battery.MASS,
        )

        self.declare_partials(
            'energy',
            [
                Aircraft.Engine.Battery.MASS,
                'nominal_capacity'
            ]
        )

    def compute(self, inputs, outputs):
        V = inputs[Aircraft.Engine.Battery.VOLTAGE]
        I = inputs['current']
        R = inputs['resistance']

        outputs['nominal_capacity'] = inputs[Aircraft.Engine.Battery.MASS] * 7.3 - 0.246 # Per Peter Sharpe TODO VERIFY
        outputs['energy'] = V * outputs['nominal_capacity']
        outputs['voltage_out'] = V * I - I**2 * R

    def compute_partials(self, inputs, outputs, partials):
        V = inputs[Aircraft.Engine.Battery.VOLTAGE]
        I = inputs['current']
        R = inputs['resistance']

        partials['voltage_out', Aircraft.Engine.Battery.VOLTAGE] = 1
        partials['voltage_out', 'current'] = -R
        partials['voltage_out', 'resistance'] = -I

        partials['power', 'voltage_supply'] = I
        partials['power', 'current'] = V - 2 * I * R
        partials['power', 'resistance'] = -I**2

        partials['nominal_capacity', Aircraft.Engine.Battery.MASS] = 7.3

        partials['energy', Aircraft.Engine.Battery.VOLTAGE] = outputs['nominal_capacity']
        partials['energy', 'nominal_capacity'] = V


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
        self.add_output('power', val=np.eros(nn), units = 'W')

        self.declare_partials('efficiency', 'throttle')
        self.declare_partials('voltage_out', ['voltage_in', 'throttle'])
        self.declare_partials('current_out', ['current_in', 'throttle'])
        self.declare_partials('power', ['voltage_in', 'current_in', 'throttle'])

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
        add_aviary_input(self, Aircraft.Engine.Motor.KV, val=np.zeros(nn), units='A')
        self.add_input('voltage_in', val=np.zeros(nn), units = 'V')
        self.add_input('current', val=np.zeros(nn), units = 'A')
        
        add_aviary_output(self, Dynamic.Vehicle.Propulsion.RPM, val=np.zeros(nn), units='rpm')
        self.add_output('power', val=np.zeros(nn), units='W')

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
        )

    def compute(self, inputs, outputs):
        #TODO: Add computation for no given kv, optimizing the motor
        R = inputs[Aircraft.Engine.Motor.RESISTANCE]
        kv = inputs[Aircraft.Engine.Motor.KV]
        voltage_prop = inputs['voltage_in'] - inputs['current'] * R

        outputs[Dynamic.Vehicle.Propulsion.RPM] = kv * voltage_prop
        outputs['power'] = -inputs['current']**2 * R - Aircraft.Engine.Motor.IDLE_CURRENT * voltage_prop

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
        

# import pickle
# #TODO: make sure I'm allowed to do this, otherwise going to be annoying
# # from smt.surrogate_models import KPLSK 
# class PropCoefficients(om.ExplicitComponent):
#     """Encapsulated surrogate model to compute thrust and power
#     coefficients from prop dimensions and flight conditions
#     using Surrogate Modeling Toolbox
#     https://smt.readthedocs.io/en/latest/index.html

#     """

#     def initialize(self):
#         self.options.declare("flight_conds", default = 3, desc= "Number of Flight Conditions to Analyze")
#         self.options.declare("flight_missions", default = 2, desc = "Number of Flight Missions ot Analyze")
#         self.options.declare("props", default = 1, desc="Number of Props to optimize, should be no more than fm")

#         """Initializing surrogate model, reading and sampling data"""
#         '''
#         self.thrust_sm = KPLSK(n_comp=4, eval_noise=True, print_global=False)
#         self.power_sm = KPLSK(n_comp=4, eval_noise=True, print_global=False)

#         x, ct, cp = PropDataReader()
#         x, ct, cp, xVal, ctVal, cpVal = SampleDataLHScosine(x, ct, cp)

#         xlim = np.array(
#             [
#                  [np.min(x[:, 0]), np.max(x[:, 0]) + 1],
#                  [np.min(x[:, 1]), np.max(x[:, 1]) + 1],
#                  [np.min(x[:, 2]), np.max(x[:, 2]) + 1],
#                  [np.min(x[:, 3]), np.max(x[:, 3]) + 1],
#             ]
#         )

#         self.thrust_sm.set_training_values(x, ct)
#         self.thrust_sm.train()
#         self.power_sm.set_training_values(x, cp)
#         self.power_sm.train()

#         with open("PickledSurrogateModels/power_sm.pkl", "wb") as fp:
#             pickle.dump(self.power_sm, fp)
#         with open("PickledSurrogateModels/thrust_sm.pkl", "wb") as fp:
#             pickle.dump(self.thrust_sm, fp)
#         '''

#         with open(
#             "PickledSurrogateModels/power_sm.pkl", #Now using 1.3.0
#             "rb",
#         ) as fp:
#             self.power_sm = pickle.load(fp) 
#         with open(
#             "PickledSurrogateModels/thrust_sm.pkl", #Now using 1.3.0
#             "rb",
#         ) as fp:
#             self.thrust_sm = pickle.load(fp)
#             #'''

#     def setup(self):
#         fc = self.options["flight_conds"]
#         fm = self.options["flight_missions"]
#         p=self.options["props"]
#         self.add_input("D_prop", units="m", shape = p, desc="propeller diameter")
#         self.add_input("pitch", units="deg", shape = p, desc="propeller pitch")

#         self.add_input('rpm', shape = (fc ,fm), units = "rev/s")
#         self.add_input('velocity', shape = (fc ,fm), units = "m/s")
#         self.add_output('ct', shape = (fc ,fm),desc = "thrust coefficients")
#         self.add_output('cp', shape = (fc ,fm),desc="power coefficients")

#         self.declare_partials('*', '*')

#     def compute_partials(self, inputs, partials):
#         fc = self.options["flight_conds"]
#         fm = self.options["flight_missions"]
#         p=self.options["props"]
#         D = inputs["D_prop"]
#         pitch = inputs["pitch"]
#         n = inputs['rpm']
#         V = inputs['velocity']
#         #Create empty array for each condition depending on how many are declared
#         #Fill the conditiosn with the amount of flight missions that have been declared
#         x=0
#         cond = []
#         #For the diameter and pitch, when adding multiprop, will be D[y]
#         if p > 1:
#             while x < fc:
#                 y= 0
#                 while y < fm:
#                     temps = np.ndarray((0,4))
#                     temps = [np.vstack((temps, np.array([D[y], pitch[y], n[x][y], V[x][y]])))]
#                     cond.append(temps)
#                     y+=1
#                 x+=1
#         if p == 1: 
#             while x < fc:
#                     y= 0
#                     while y < fm:
#                         temps = np.ndarray((0,4))
#                         temps = [np.vstack((temps, np.array([D[0], pitch[0], n[x][y], V[x][y]])))]
#                         cond.append(temps)
#                         y+=1
#                     x+=1

#         #Predict the partials for each condition with respect to each (First declare range of conditions)
#         ranger = int(np.size(cond)/4)

#         #Diameter
#         tdzeros = np.zeros((fc * fm , p))
#         td = []
#         for z in range(ranger):
#             appender=self.thrust_sm.predict_derivatives(cond[z][0], 0)
#             td.append(appender)
#         for m in range(p):
#             for r in range(m,fc * fm, p):
#                 tdzeros[r][m]= td[r]
#         partials["ct", "D_prop"] = tdzeros

#         pdzeros = np.zeros((fc * fm , p))
#         pd = []
#         for z in range(ranger):
#             appender=self.power_sm.predict_derivatives(cond[z][0], 0)
#             pd.append(appender)
#         for m in range(p):
#             for r in range(m,fc * fm, p):
#                 pdzeros[r][m]= pd[r]
#         partials["cp", "D_prop"] = pdzeros

#         #Pitch 
#         tpzeros = np.zeros((fc * fm , p))
#         tp = []
#         for z in range(ranger):
#             appender=self.thrust_sm.predict_derivatives(cond[z][0], 1)
#             tp.append(appender)
#         for m in range(p):
#             for r in range(m,fc * fm, p):
#                 tpzeros[r][m]= tp[r]
#         partials["ct", "pitch"] = tpzeros

#         ppzeros = np.zeros((fc * fm ,  p))
#         pp = []
#         for z in range(ranger):
#             appender=self.power_sm.predict_derivatives(cond[z][0], 1)
#             pp.append(appender)
#         for m in range(p):
#             for r in range(m,fc * fm, p):
#                 ppzeros[r][m]= pp[r]
#         partials["cp", "pitch"] = ppzeros

#         #RPM
#         tnzeros = np.zeros((fc * fm , fc * fm))
#         tn = []
#         for z in range(ranger):
#             appender=self.thrust_sm.predict_derivatives(cond[z][0], 2)
#             tn.append(appender)
#         np.fill_diagonal(
#             tnzeros,
#             tn
#         )
#         partials["ct", "rpm"] = tnzeros

#         pnzeros = np.zeros((fc * fm , fc * fm ))
#         pn = []
#         for z in range(ranger):
#             appender= self.power_sm.predict_derivatives(cond[z][0], 2)
#             pn.append(appender)
#         np.fill_diagonal(
#             pnzeros,
#             pn
#         )
#         partials["cp", "rpm"] = pnzeros

#         #Velocity
#         tvzeros = np.zeros((fc * fm , fc * fm))
#         tv = []
#         for z in range(ranger):
#             appender=self.thrust_sm.predict_derivatives(cond[z][0], 3)
#             tv.append(appender)
#         np.fill_diagonal(
#             tvzeros,
#             tv
#         )
#         partials["ct", "velocity"] = tvzeros
        
#         pvzeros = np.zeros((fc * fm , fc * fm ))
#         pv = []
#         for z in range(ranger):
#             appender= self.power_sm.predict_derivatives(cond[z][0], 3)
#             pv.append(appender)
#         np.fill_diagonal(
#             pvzeros,
#             pv
#         )
#         partials["cp", "velocity"] = pvzeros
    
#     def compute(self, inputs, outputs):
#         fc = self.options["flight_conds"]
#         fm = self.options["flight_missions"]
#         p = self.options["props"]
#         D = inputs["D_prop"]
#         pitch = inputs["pitch"]
#         n = inputs['rpm']
#         V = inputs['velocity']
#         #Create empty array for each condition depending on how many are declared
#         #Fill the conditiosn with the amount of flight missions that have been declared
#         x=0
#         cond = []
#         #For the diameter and pitch, when adding multiprop, will be D[y], &&
#         if p > 1: 
#             while x < fc:
#                 y= 0
#                 while y < fm:
#                     appender = np.array([D[y], pitch[y], n[x][y], V[x][y]])
#                     cond.append(appender)
#                     y+=1
#                 x+=1
#         if p == 1: 
#             while x < fc:
#                 y= 0
#                 while y < fm:
#                     appender = np.array([D[0], pitch[0], n[x][y], V[x][y]])
#                     cond.append(appender)
#                     y+=1
#                 x+=1
#         #Unlike in partials, here one stacks all conditions, predicting all at once. 
#         topredict = np.vstack(cond)
        
#         #test this first, if it works could be crazy
#         outputs["ct"] = self.thrust_sm.predict_values(topredict)
#         outputs["cp"] = self.power_sm.predict_values(topredict)


class PropCoefficients(om.MetaModelSemiStructuredComp):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
   
    def setup(self): 
        nn = self.options['num_nodes']
class Propeller(om.ExplicitComponent):
    def initialize(self):
        self.options.declare('num_nodes', default=1, types=int)
    #TODO: ask about adding more propellers
    def setup(self): 
        nn = self.options['num_nodes']
        add_aviary_input(self, Dynamic.Atmosphere.DENSITY, val=np.zeros(nn), units = 'kg/m**3')
        add_aviary_input(self, Aircraft.Engine.Propeller.DIAMETER, units = 'm')
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.RPM, units = 'rev/s')
        self.add_input("ct", val=np.zeros(nn), units='unitless')
        self.add_input("cp", val=np.zeros(nn), units='unitless')
        add_aviary_input(self, Aircraft.Engine.NUM_ENGINES, val=np.zeros(nn), units='unitless')

        add_aviary_output(self, Dynamic.Vehicle.Propulsion.PROP_THRUST, val=np.zeros(nn), units='N')
        add_aviary_output(self, Dynamic.Vehicle.Propulsion.PROP_POWER, val=np.zeros(nn), units='W')
        
        self.declare_partials(
            Dynamic.Vehicle.Propulsion.PROP_THRUST,
            [
                Dynamic.Atmosphere.DENSITY, 
                Aircraft.Engine.Propeller.DIAMETER,
                Dynamic.Vehicle.Propulsion.RPM,
                'ct',
                Aircraft.Engine.NUM_ENGINES,
            ],
        )

        self.declare_partials(
            Dynamic.Vehicle.Propulsion.PROP_POWER,
            [
                Dynamic.Atmosphere.DENSITY, 
                Aircraft.Engine.Propeller.DIAMETER,
                Dynamic.Vehicle.Propulsion.RPM,
                'cp',
            ],
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
        add_aviary_input(self, Dynamic.Vehicle.Propulsion.PROP_POWER, val=np.zeros(nn), units='W')

        self.add_output('current', val=np.ones(nn)*30, units='A') #we want to make a good initial guess on the current
        self.add_residual('power_net', shape=nn, units='W')

        self.declare_partials('current', ['power_batt', 'power_esc', 'power_motor',  Dynamic.Vehicle.Propulsion.PROP_POWER], val = 1)

    def apply_nonlinear(self, inputs, residuals):
        residuals['current'] = inputs['power_batt'] + inputs['power_esc'] + inputs['power_motor'] - inputs[Dynamic.Vehicle.Propulsion.PROP_POWER]


class RCPropGroup(om.Group):
    def setup(self):
        self.add_subsystem('battery', Battery())
        self.add_subsystem('esc', ElectronicSpeedController())
        self.add_subsystem('motor', Motor())
        self.add_subsystem('propco', PropCoefficients())
        self.add_subsystem('prop', Propeller())
        self.add_subsystem('net_power', PowerResiduals())

        self.connect('battery.voltage_out', 'esc.voltage_in')
        self.connect('esc.voltage_out', 'motor.voltage_in')
        self.connect('esc.current_in', 'motor.current')
        # self.connect('motor.rpm', prop.rpm) TODO check if this is needed

        self.connect('battery.power', 'power_net.power_batt')
        self.connect('esc.power', 'power_net.power_esc')
        self.connect('motor.power', 'power_net.power_motor')
        self.connect('power_net.current', ['battery.current', 'esc.current_in'])
        # self.connect('prop.power', 'power_net.power_prop') TODO check if this is needed


        
    