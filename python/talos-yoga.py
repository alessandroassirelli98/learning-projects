"""
OCP with constrained dynamics

Simple example with regularization cost and terminal com+velocity constraint, known initial config.

min ||q0 + Dq - q*||**2 
Dq    
s.t
        com(q_t)[2] = com(x_T[:nq])[2] = 0.8

So the robot should just bend to reach altitude COM 10cm while stoping at the end of the movement.
"""

import pinocchio as pin
from pinocchio import casadi as cpin
import casadi
import numpy as np
import example_robot_data as robex
from pinocchio.visualize import GepettoVisualizer

### Load and display Talos
robot = robex.load("talos")
try:
    viz = GepettoVisualizer(robot.model, robot.collision_model, robot.visual_model)
    viz.initViewer()
    viz.loadViewerModel()
except:
    print("No viewer")

# The pinocchio model is what we are really interested by.
model = robot.model
cmodel = cpin.Model(robot.model)
data = model.createData()
nq = cmodel.nq
nx = cmodel.nv

### PROBLEM
opti = casadi.Opti()
Dq = opti.variable(nx)

qnext = cpin.integrate(cmodel, robot.q0, Dq)

com = casadi.Function("com", [qnext], [cpin.centerOfMass(cmodel, data, qnext)])
opti.subject_to(com(qnext)[2] == 0.8)

cost = casadi.sumsqr(Dq)

### SOLVE
opti.minimize(cost)
opti.set_initial(Dq, np.zeros(nx))

opti.solver("ipopt")
try:
    sol = opti.solve_limited()
    dq = opti.value(Dq)
    q = pin.integrate(model, robot.q0, dq)
except:
    print("ERROR in convergence, plotting debug info.")
    dq = opti.debug.value(Dq)

viz.display(q)

