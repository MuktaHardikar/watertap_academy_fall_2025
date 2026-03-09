# IMPORTS FROM PYOMO

from pyomo.environ import (
    Block,
    ConcreteModel,
    Var,
    Param,
    Constraint,
    Objective,
    Expression,
    value,
    check_optimal_termination,
    assert_optimal_termination,
    TransformationFactory,
    units as pyunits,
)

from pyomo.util.check_units import assert_units_consistent
from pyomo.network import Arc


# IMPORTS FROM IDAES
from idaes.core import FlowsheetBlock, UnitModelCostingBlock
from idaes.models.unit_models import Feed, Product, StateJunction
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.scaling import calculate_scaling_factors, set_scaling_factor, constraint_scaling_transform
from idaes.core.util.initialization import propagate_state


# IMPORTS FROM WaterTAP
from watertap.property_models.NaCl_prop_pack import NaClParameterBlock
from watertap.property_models.seawater_prop_pack import SeawaterParameterBlock
from watertap.unit_models.pressure_changer import Pump
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
    PressureChangeType,
)
from watertap.unit_models.reverse_osmosis_1D import (
    ReverseOsmosis1D,
    PressureChangeType,
    MassTransferCoefficient,
    ConcentrationPolarizationType,
)

from watertap.core.solvers import get_solver

# TRANSLATOR FUNCTION
import idaes.logger as idaeslog
from idaes.core import declare_process_block_class
from idaes.core.util.exceptions import InitializationError
from idaes.models.unit_models.translator import TranslatorData

from watertap.core.util.model_diagnostics.infeasible import *
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
import idaes.core.util.scaling as iscale


def build_ro_model():
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.ro_properties = NaClParameterBlock()

    m.fs.RO = ReverseOsmosis0D(
    property_package=m.fs.ro_properties,
    has_pressure_change=True,
    pressure_change_type=PressureChangeType.calculated,
    mass_transfer_coefficient=MassTransferCoefficient.calculated,
    concentration_polarization_type=ConcentrationPolarizationType.calculated,
    module_type="spiral_wound",
    )

    return m

def set_ro_parameters(
        m,
        A_comp,
        B_comp,
        channel_height,
        spacer_porosity,
        membrane_area,
        deltaP,):
    
    # Fix (2) membrane properties
    m.fs.RO.A_comp.fix(A_comp)
    m.fs.RO.B_comp.fix(B_comp)

    # Fix (4) module specifications
    m.fs.RO.feed_side.channel_height.fix(channel_height)
    m.fs.RO.feed_side.spacer_porosity.fix(spacer_porosity)
    m.fs.RO.area.fix(membrane_area)
    m.fs.RO.deltaP.fix(deltaP)                        

    # (1) outlet state variable
    m.fs.RO.permeate.pressure[0].fix(101325*pyunits.Pa)  # atmospheric pressure

    return None

def set_ro_operating_conditions(
        m, 
        mass_flow_water, 
        mass_flow_salt,
        operating_pressure,):
    
    # Fix (2) feed state variables
    m.fs.RO.inlet.flow_mass_phase_comp[0, "Liq", "H2O"].fix(mass_flow_water)
    m.fs.RO.inlet.flow_mass_phase_comp[0, "Liq", "NaCl"].fix(mass_flow_salt)
    m.fs.RO.inlet.pressure.fix(operating_pressure)
    m.fs.RO.inlet.temperature.fix(298)

    return None


def initialize_ro_model(m):
    '''
    Initialize RO model with stable initial guess to help with convergence. 
    The initial guess is based on the following assumptions:
    '''

    set_ro_parameters(
        m,
        A_comp = 2.027e-11 * pyunits.m/(pyunits.s * pyunits.Pa),
        B_comp = 3e-8 * pyunits.m/(pyunits.s),
        channel_height = 1 * pyunits.mm,
        spacer_porosity = 0.75  * pyunits.dimensionless,
        membrane_area = 50  * pyunits.m**2,
        deltaP = -3 * pyunits.bar,
        )

    # Initialize RO unit with stable initial guess
    set_ro_operating_conditions(
        m, 
        mass_flow_water  = 0.965 * pyunits.kg / pyunits.s, 
        mass_flow_salt = 0.005 * pyunits.kg / pyunits.s,
        operating_pressure = 10 * pyunits.bar,
        )
    
    set_ro_scaling(
        m, 
        mass_flow_water= 0.965 * pyunits.kg / pyunits.s,
        mass_flow_salt= 0.01 * pyunits.kg / pyunits.s
    )

    m.fs.RO.initialize()
    
    return m

def set_ro_scaling(m, mass_flow_water, mass_flow_salt):
    # Set flow rate scaling factors
    sf_flow_mass_phase_comp_water = 1/mass_flow_water()
    sf_flow_mass_phase_comp_salt = 1/mass_flow_salt()

    m.fs.ro_properties.set_default_scaling("flow_mass_phase_comp", sf_flow_mass_phase_comp_water, index=("Liq", "H2O"))
    m.fs.ro_properties.set_default_scaling("flow_mass_phase_comp", sf_flow_mass_phase_comp_salt, index=("Liq", "NaCl"))

    set_scaling_factor(m.fs.RO.area, 1e-3)
    iscale.constraint_scaling_transform(m.fs.RO.eq_recovery_vol_phase[0.0], 1e2)

    m.fs.RO.feed_side.cp_modulus.setub(10)
    m.fs.RO.feed_side.cp_modulus.setlb(1e-5)

    m.fs.RO.flux_mass_phase_comp.setub(0.1)
    m.fs.RO.flux_mass_phase_comp.setlb(1e-8)

    m.fs.RO.feed_side.friction_factor_darcy.setub(200)
    m.fs.RO.feed_side.K.setlb(1e-5)
    calculate_scaling_factors(m)

    return None

if __name__ == "__main__":

    '''
    Steps:
    1. Build and solve stable initialize guess RO model 
    2. Initialize model with stable initial guess
    3. Set operating conditions and parameters to match the original problem -
    4. Set scaling factors
    5. Initialize model again with new operating conditions and parameters
    6. Solve model
    '''

    A_comp = 2.027e-11 * pyunits.m/(pyunits.s * pyunits.Pa)
    B_comp = 3e-8 * pyunits.m/(pyunits.s)
    channel_height = 1 * pyunits.mm
    spacer_porosity = 0.75  * pyunits.dimensionless
    membrane_area = 20000  * pyunits.m**2
    deltaP = -3 * pyunits.bar
    recovery_vol_phase = 0.9

    salt_concentration = 5 * pyunits.g / pyunits.L
    flow_vol = 500 * pyunits.m**3/pyunits.hour
    density = 995 * pyunits.kg / pyunits.m**3
    mass_flow_water = pyunits.convert(flow_vol * density, to_units=pyunits.kg / pyunits.s)
    mass_flow_salt = pyunits.convert(salt_concentration * flow_vol, to_units=pyunits.kg / pyunits.s)  

    operating_pressure = 10 * pyunits.bar

    m = build_ro_model()

    initialize_ro_model(m)

    solver = get_solver()
    results = solver.solve(m, tee=False)
    assert_optimal_termination(results)

    set_ro_parameters(
        m,
        A_comp,
        B_comp,
        channel_height,
        spacer_porosity,
        membrane_area,
        deltaP,)

    set_ro_operating_conditions(
        m, 
        mass_flow_water, 
        mass_flow_salt,
        operating_pressure,
        )
    set_ro_scaling(
        m, 
        mass_flow_water,
        mass_flow_salt,
    )
    
    m.fs.RO.initialize()

    
    print("\nAfter initialization with stable initial guess:")
    print("Estimated area: ", value(m.fs.RO.area), pyunits.get_units(m.fs.RO.area))
    print("Estimated recovery: ", value(m.fs.RO.recovery_vol_phase[0.0, 'Liq']), pyunits.get_units(m.fs.RO.recovery_vol_phase[0.0, 'Liq']))
    print("Estimated inlet pressure: ", value(m.fs.RO.inlet.pressure[0]), pyunits.get_units(m.fs.RO.inlet.pressure[0]))

    m.fs.RO.inlet.pressure.unfix()
    m.fs.RO.recovery_vol_phase[0.0, "Liq"].fix(0.9)


    solver = get_solver()
    results = solver.solve(m, tee=False)
    assert_optimal_termination(results)


    print("\nAfter solving with new operating conditions:")
    print("Estimated area: ", value(m.fs.RO.area), pyunits.get_units(m.fs.RO.area))
    print("Estimated recovery: ", value(m.fs.RO.recovery_vol_phase[0.0, 'Liq']), pyunits.get_units(m.fs.RO.recovery_vol_phase[0.0, 'Liq']))
    print("Estimated inlet pressure: ", value(m.fs.RO.inlet.pressure[0]), pyunits.get_units(m.fs.RO.inlet.pressure[0]))