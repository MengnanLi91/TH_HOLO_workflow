# 2-D axisymmetric PINSFV flow-contraction-expansion case parameterized
# for Re_43938__Dr_0p522__Lr_0p073. The middle (throat) block reads its
# Forchheimer coefficient axially from a CSV produced by
# src/cases/alpha_d/export_friction_profile.py.
#
# Material class used: ADGenericVectorFunctorMaterial
# Justification: prop_values is typed MooseFunctorName and resolved via
# getFunctor<GenericReal<is_ad>>; Function objects are registered as
# functors in MOOSE, so a function name is a valid prop_values entry.
# (framework/src/functormaterials/GenericVectorFunctorMaterial.C, line 69)
#
# CSV coordinate convention:
# The exporter (export_friction_profile.py) writes z in MOOSE mesh
# coordinates: z_csv = z_throat_local + end_length, where
# end_length = buffer_diams * D_big = 1.0 * 0.2 = 0.2 m.
# This means the CSV x column spans [0.2, 0.273] for this case,
# matching the throat block's mesh coordinates directly.
# PiecewiseLinear (axis=x) therefore evaluates correctly without any
# coordinate shift in this file.

mu = 2.376e-6
rho = 1
outer_radius = 0.1
middle_radius = 0.0522
middle_porosity = ${fparse (middle_radius / outer_radius)^2}
total_length = 0.473
middle_length = 0.073
end_length = ${fparse (total_length - middle_length) / 2}
inlet_velocity = 1
outlet_pressure = 0

[Mesh]
  [mesh]
    type = CartesianMeshGenerator
    dim = 2
    dx = '${end_length} ${middle_length} ${end_length}'
    dy = '${outer_radius}'
    ix = '21 21 21'
    iy = '11'
    subdomain_id = '1 2 3'
  []
  [interface_12]
    type = SideSetsBetweenSubdomainsGenerator
    input = mesh
    primary_block = '1'
    paired_block = '2'
    new_boundary = 'interface_12'
  []
  [interface_23]
    type = SideSetsBetweenSubdomainsGenerator
    input = interface_12
    primary_block = '2'
    paired_block = '3'
    new_boundary = 'interface_23'
  []
  [inlet]
    type = RenameBoundaryGenerator
    input = interface_23
    old_boundary = left
    new_boundary = inlet
  []
  [outlet]
    type = RenameBoundaryGenerator
    input = inlet
    old_boundary = right
    new_boundary = outlet
  []
  coord_type = RZ
  rz_coord_axis = 'X'
[]

[GlobalParams]
  advected_interp_method = 'upwind'
  velocity_interp_method = 'rc'
  rhie_chow_user_object = 'rc'
[]

[UserObjects]
  [rc]
    type = PINSFVRhieChowInterpolator
    u = superficial_vel_x
    v = superficial_vel_y
    pressure = pressure
    porosity = porosity
  []
[]

[Variables]
  [superficial_vel_x]
    type = PINSFVSuperficialVelocityVariable
    initial_condition = ${inlet_velocity}
  []
  [superficial_vel_y]
    type = PINSFVSuperficialVelocityVariable
    initial_condition = 1e-6
  []
  [pressure]
    type = BernoulliPressureVariable
    u = superficial_vel_x
    v = superficial_vel_y
    porosity = porosity
    rho = ${rho}
  []
[]

[Functions]
  # Reads the axial Forchheimer profile produced by export_friction_profile.py.
  # The CSV x column is already in MOOSE mesh coordinates (pre-shifted by the
  # exporter); see header comment.
  [forchheimer_profile_fn]
    type = PiecewiseLinear
    data_file = forchheimer_profile.csv
    format = columns
    x_index_in_file = 0
    y_index_in_file = 1
    axis = x
  []
[]

[FVKernels]
  [mass]
    type = PINSFVMassAdvection
    variable = pressure
    rho = ${rho}
  []
  [u_advection]
    type = PINSFVMomentumAdvection
    variable = superficial_vel_x
    rho = ${rho}
    porosity = porosity
    momentum_component = 'x'
  []
  [u_viscosity]
    type = PINSFVMomentumDiffusion
    variable = superficial_vel_x
    mu = ${mu}
    porosity = porosity
    momentum_component = 'x'
  []
  [u_pressure]
    type = PINSFVMomentumPressure
    variable = superficial_vel_x
    momentum_component = 'x'
    pressure = pressure
    porosity = porosity
  []
  [u_friction]
    type = PINSFVMomentumFriction
    variable = superficial_vel_x
    momentum_component = 'x'
    Forchheimer_name = 'Forchheimer_coefficient'
    rho = ${rho}
    speed = speed
    block = 2
  []
  [v_advection]
    type = PINSFVMomentumAdvection
    variable = superficial_vel_y
    rho = ${rho}
    porosity = porosity
    momentum_component = 'y'
  []
  [v_viscosity]
    type = PINSFVMomentumDiffusion
    variable = superficial_vel_y
    mu = ${mu}
    porosity = porosity
    momentum_component = 'y'
  []
  [v_pressure]
    type = PINSFVMomentumPressure
    variable = superficial_vel_y
    momentum_component = 'y'
    pressure = pressure
    porosity = porosity
  []
  [v_friction]
    type = PINSFVMomentumFriction
    variable = superficial_vel_y
    momentum_component = 'y'
    Forchheimer_name = 'Forchheimer_coefficient'
    rho = ${rho}
    speed = speed
    block = 2
  []
[]

[FVBCs]
  [inlet-u]
    type = INSFVInletVelocityBC
    boundary = inlet
    variable = superficial_vel_x
    functor = ${inlet_velocity}
  []
  [inlet-v]
    type = INSFVInletVelocityBC
    boundary = inlet
    variable = superficial_vel_y
    functor = 0
  []
  [free-slip-u]
    type = INSFVNaturalFreeSlipBC
    boundary = top
    variable = superficial_vel_x
    momentum_component = 'x'
  []
  [free-slip-v]
    type = INSFVNaturalFreeSlipBC
    boundary = top
    variable = superficial_vel_y
    momentum_component = 'y'
  []
  [symmetry-u]
    type = PINSFVSymmetryVelocityBC
    boundary = bottom
    variable = superficial_vel_x
    u = superficial_vel_x
    v = superficial_vel_y
    mu = ${mu}
    momentum_component = 'x'
  []
  [symmetry-v]
    type = PINSFVSymmetryVelocityBC
    boundary = bottom
    variable = superficial_vel_y
    u = superficial_vel_x
    v = superficial_vel_y
    mu = ${mu}
    momentum_component = 'y'
  []
  [symmetry-p]
    type = INSFVSymmetryPressureBC
    boundary = bottom
    variable = pressure
  []
  [outlet-p]
    type = INSFVOutletPressureBC
    boundary = outlet
    variable = pressure
    function = ${outlet_pressure}
  []
[]

[FunctorMaterials]
  [porosity]
    type = PiecewiseByBlockFunctorMaterial
    prop_name = porosity
    subdomain_to_prop_value = '1 1.0 2 ${middle_porosity} 3 1.0'
  []
  [forchheimer_2]
    type = ADGenericVectorFunctorMaterial
    prop_names = 'Forchheimer_coefficient'
    prop_values = 'forchheimer_profile_fn forchheimer_profile_fn forchheimer_profile_fn'
    block = 2
  []
  [speed]
    type = PINSFVSpeedFunctorMaterial
    superficial_vel_x = superficial_vel_x
    superficial_vel_y = superficial_vel_y
    porosity = porosity
  []
[]

[Executioner]
  type = Steady
  solve_type = 'NEWTON'
  petsc_options_iname = '-pc_type -ksp_gmres_restart -sub_pc_type -sub_pc_factor_shift_type'
  petsc_options_value = 'asm      300                lu           NONZERO'
  line_search = 'none'
  nl_rel_tol = 1e-11
  nl_abs_tol = 1e-14
[]

[Postprocessors]
  [inlet-p]
    type = SideIntegralVariablePostprocessor
    variable = pressure
    boundary = inlet
  []
  [outlet-u]
    type = SideIntegralVariablePostprocessor
    variable = superficial_vel_x
    boundary = outlet
  []
[]

[Outputs]
  exodus = true
  csv = true
[]
