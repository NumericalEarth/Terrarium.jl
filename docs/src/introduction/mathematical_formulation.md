# Mathematical formulation

Unlike many other land models, Terrarium.jl strictly follows a philosophy of dynamical modeling via discretized differential equations. This means that all physical processes in the model must be able to be written in terms of a set of continuous-time equations of the form:
```math
\frac{\partial u(x,t)}{\partial t} = G(u(x,t)) + F(x,t)
```
where $u(x,t)$ is a field describing the prognostic state of the system, $t$ is time, $F$ is a focing term, and $G$ is a (differentiable) function which computes the tendencies, i.e. the change in the prognostic state at the current time, $t$.

In many areas of science and engineering, this approach to modeling is fairly standard. It can be contrasted, however, with discrete-time dynamical modeling where state changes are computed according to a series of discrete update rules, $\mathbf{D}$, with some fixed time resolution,
```math
\mathbf{u}_{t+1} = \mathbf{D}(\mathbf{u}_{t})
```

The continuous-time formulation has three key advantages over the discrete approach:

1. **Faithfulness to realistic physical processes.** Within the context of Earth system modeling, we are primarily interested in processes that operate continuously in time, not in discrete steps. Although the physics of land models may lack a well-defined fluid “dynamical core” like those found in atmosphere and ocean models, they are still ultimately centered around hydrological, thermodynamic, and biological processes that have continuous-time dynamics. As such, we believe that it is generally preferable to build models that follow this structure.

2. **Flexibility in timestepping and error control.** Discrete-time formulations of dynamical systems necessarily require strong assumptions about the time-discretization of the system which often amount to a form of forward Euler with a fixed timestep size. Re-adjustment of the timestep in the discrete dynamical operator $\mathbf{D}$ is sometimes achieved by simply rescaling rate parameters. However, this rescaling makes the strong assumption that the dynamics of the system scale linearly with the timescale and ignores potential feedback mechanisms and scale-dependent interactions in the dynamics. Continuous-time formulations relax this assumption and permit a much broader range of timestepping strategies that can account for nonlinearity and are thus, in our view, better suited to the simulation of coupled physical processes.

3. **Heterogeneous temporal resolution.** Data assimilation and parameter estimation problems often necessitate the comparison of model outputs with observational time series data. These data often come at differing temporal resolutions and have irregular sample spacing due to, e.g. missing values. This can pose a problem for discrete-time models which must either (i) rescale the timestep to match the data, which has significant physical implications, or (ii) resort to temporal interpolation of the model outputs which is likely to violate the underlying physical conservation laws. Continuous-time systems do not have this problem since the timestep can always be adapted to obtain output at any set of time points.

Note that one common problem in continuous-time dynamical modeling is the representation of discontinuous or instantaneous events/disruptions to the system. Such events can, however,  be handled through the use of callback functions based on the (discretized) state $\mathbf{u}$. These cases should nonetheless be considered the exception rather than the rule.

## Practical implications

While the above modeling philosophy has many advantages, it also places some practical restrictions on how we code the model physics. The most important constraints are:
- Prognostic (i.e. time-integrated) state variables must be clearly and coherently distinguished from all other auxiliary variables derived from the prognostic state.
- Prognostic variables of the system should only be updated by a timestepper, callback, or closure opreation, and should not be otherwise modified by the physical processes within a timestep. Note that this precludes the implementation of some traditional land model process formulations, such as the so-called “bucket schemes” for soil hydrology which rely on non-physical, instantaneous routing of water between soil layers.
- For physical coherence, tendencies must be computed only based on the current state of the system. Similarly, all non-prognostic (auxiliary) variables should be derived only from the prognostic state and/or forcings within each timestep and are not allowed to depend on non-prognostic values from previous timesteps, except in special cases where previous values are used only for computational efficiency (e.g. iterative solvers).

These restrictions can potentially be relaxed in some cases through the use of callbacks and/or nested time-stepping schemes, but the goal should always be to adhere to them as much as possible to maximize efficiency and minimize potential sources of error.
