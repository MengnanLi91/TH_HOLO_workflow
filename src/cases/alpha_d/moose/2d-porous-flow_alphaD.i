# 2-D axisymmetric PINSFV flow-contraction-expansion case parameterized
# for Re_43938__Dr_0p522__Lr_0p073. All three blocks (upstream buffer,
# throat, downstream buffer) read their Forchheimer coefficient axially
# from `forchheimer_profile.csv` (produced by
# src/cases/alpha_d/export_friction_profile.py). The CSV must sit
# next to this .i file because MOOSE resolves `data_file` relative to
# the input file's directory.
# Consumed by: [Functions/forchheimer_profile_fn] (PiecewiseLinear).
#
# ─── How to run (full pipeline) ─────────────────────────────────────────
#
# Two apptainer SIFs are needed:
#   PY_SIF=/data/lim2/projects/multifid-th/worktrees/refactor/multifid-th-cpu.sif
#   MOOSE_SIF=/data/lim2/containers/moose-dev-openmpi-x86_64_latest.sif
#   REPO=/data/lim2/projects/multifid-th/worktrees/integration
#   BIND="--bind /data/lim2/projects/multifid-th:/data/lim2/projects/multifid-th"
#
# 1) Run the exporter to generate forchheimer_profile.csv + sidecar:
#   apptainer exec $BIND $PY_SIF bash -lc "
#     cd $REPO/src && PYTHONPATH=. python -m cases.alpha_d.export_friction_profile \
#       --zarr      ../data/flow_contraction_expansion/parametric_study/processed/Re_43938__Dr_0p522__Lr_0p073.zarr \
#       --checkpoint ../data/cases/train_conv1d/model.mdlus \
#       --run-meta   ../data/cases/train_conv1d/run_meta.json \
#       --output-csv ../data/cases/train_conv1d/Re_43938__Dr_0p522__Lr_0p073/forchheimer_profile.csv"
#
# 2) Stage the CSV next to this .i file (MOOSE resolves data_file relatively):
#   cp $REPO/data/cases/train_conv1d/Re_43938__Dr_0p522__Lr_0p073/forchheimer_profile.csv \
#      $REPO/src/cases/alpha_d/moose/forchheimer_profile.csv
#
# 3) Run MOOSE:
#   apptainer exec $BIND $MOOSE_SIF bash -lc "
#     cd $REPO/src/cases/alpha_d/moose && \
#     $REPO/moose/modules/navier_stokes/navier_stokes-opt \
#       -i 2d-porous-flow_alphaD.i Outputs/file_base=2d-porous-flow_alphaD_out"
#
# 4) Verify (three-ΔP comparison: truth vs surrogate integral vs MOOSE):
#   apptainer exec $BIND $PY_SIF bash -lc "
#     cd $REPO && PYTHONPATH=src python -m cases.alpha_d.verify_delta_p \
#       --sidecar   data/cases/train_conv1d/Re_43938__Dr_0p522__Lr_0p073/forchheimer_profile.meta.json \
#       --moose-csv src/cases/alpha_d/moose/2d-porous-flow_alphaD_out.csv"
#
# The end-to-end version of steps 1-4 is automated in
# tests/case_pressure_drop/test_alpha_d_delta_p.py (skips silently when
# apptainer isn't on PATH; run from a host shell, not from inside a SIF).
#
# ─── Forchheimer mapping ────────────────────────────────────────────────
#
# Derived 2026-05-28, full derivation in
# docs/dev/alpha_d_coupling_physics.md §5):
#
#   F(z) = α_D(z) · ε(z)² / D_h(z)
#
# Specializing per block:
#   Block 1/3 (buffer, ε=1,   D_h=D_outer):    F = α_D / D_outer
#   Block 2   (throat, ε=Dr², D_h=Dr·D_outer): F = α_D · Dr³ / D_outer
#
# Derivation: PINSFV's combined momentum balance for 1D steady plug flow
# in a porous block is -ε∇p = (ρ/2)·F·|v_inter|·v_super, where the ε on
# the pressure gradient comes from PINSFVMomentumPressure.C:41-44 and the
# speed = |v_super|/ε = |v_inter| comes from PINSFVSpeedFunctorMaterial.C
# lines 69-82. Solving for the effective ∇p and using v_super = V_bulk
# (mass conservation in incompressible PINSFV) gives
#   -dP/dz_MOOSE = ρ·F·V_bulk² / (2·ε²).
# Equating to the training-data Darcy-Weisbach definition
#   -dP/dz = α_D · ρ·V_bulk² / (2·D_h)
# yields F = α_D·ε²/D_h. Verified empirically: constant-F=1 throat run
# gives 0.484 Pa vs analytical 0.493 Pa (within finite-volume error).
#
# Full-ROI application (not throat-only) captures the upstream
# vena-contracta spikes the surrogate predicts at z ≈ 0.185-0.194 m,
# which account for ~64% of the surrogate's predicted ΔP.
#
# Material class used: ADGenericVectorFunctorMaterial
# Justification: prop_values is typed MooseFunctorName and resolved via
# getFunctor<GenericReal<is_ad>>; Function objects are registered as
# functors in MOOSE, so a function name is a valid prop_values entry.
# (framework/src/functormaterials/GenericVectorFunctorMaterial.C, line 69)
#
# CSV coordinate convention:
# The exporter writes z_phys = z_hat * roi_length directly (no pre-shift).
# The CSV x column spans [~0, roi_length=0.473] for this case,
# covering the full ROI in MOOSE mesh coordinates (x=0 at inlet).
# PiecewiseLinear (axis=x) evaluates correctly without any coordinate shift.

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
  [forchheimer_all]
    type = ADGenericVectorFunctorMaterial
    prop_names = 'Forchheimer_coefficient'
    prop_values = 'forchheimer_profile_fn forchheimer_profile_fn forchheimer_profile_fn'
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
