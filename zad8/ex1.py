#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from qiskit.circuit.library import UnitaryGate
from qiskit import (ClassicalRegister,
 QuantumRegister,
QuantumCircuit,
transpile)
from qiskit_aer import Aer


import helpers as oq
#%%
#task 1
def oracle(n,winner_index):
    Uw = np.eye(2**n)
    Uw[winner_index, winner_index] = -1
    return UnitaryGate(Uw)

#%%
n = 3
win = 1
assert win < n
qreg = QuantumRegister(3, name="qubit")
creg = ClassicalRegister(3, name="class_bit")

t1_circuit = QuantumCircuit(qreg,creg)

t1_circuit.initialize(1, qreg[0])
t1_circuit.initialize(0, qreg[1])
t1_circuit.initialize(0, qreg[2])

for i in range(n):
#   t1_circuit.h(qreg[i])
  if i != win:
        t1_circuit.x(qreg[i])

t1_circuit.ccz(*qreg)

for i in range(n):
  if i != win:
        t1_circuit.x(qreg[i])

# t1_circuit.append(oracle(n, win), qreg)
print(oq.Wavefunction(t1_circuit))

t1_circuit.measure(qreg,creg)
print(oq.Measurement(t1_circuit, shots=100))

print(t1_circuit)

#%%
#task 2
def gen_oracle(circ,n,win,q):
    for i in range(n):
        if i != win:
            circ.x(q[i])

    circ.ccz(*q)

    for i in range(n):
        if i != win:
            circ.x(q[i])
    circ.barrier(q)

def gen_diff(circ,n,q):
    for i in range(n):
        circ.h(q[i])
        circ.x(q[i])

    circ.h(q[n-1])
    circ.ccx(*q)
    circ.h(q[n-1])

    for i in range(n):
        circ.x(q[i])
        circ.h(q[i])

    circ.barrier(q)
#PARAMS
q2 = QuantumRegister(3, name="qubit")
c2 = ClassicalRegister(3, name="class_bit")

t2_circuit = QuantumCircuit(q2,c2)

n = 3
winner = 2

#STATE INIT
t1_circuit.initialize(0, qreg[0])
t1_circuit.initialize(1, qreg[1])
t1_circuit.initialize(0, qreg[2])

for i in range(n):
    t2_circuit.h(qreg[i])

t2_circuit.barrier(qreg)

#REPETITIONS FOR GROVER
num_reps = 2
for j in range(num_reps):
    gen_oracle(t2_circuit,n,winner,q2)
    gen_diff(t2_circuit,n,q2)

#MEASUREMENT
print(t2_circuit)
t2_circuit.measure(q2,c2)
print(oq.Measurement(t2_circuit, shots=1000))
#%%
'''
#IBM
#CIRCUIT
# from token_qibm import ibm_token
q2 = QuantumRegister(3, name="qubit")
c2 = ClassicalRegister(3, name="class_bit")

ibm_circ = QuantumCircuit(q2,c2)

n = 3
winner = 2

#STATE INIT
# ibm_circ.initialize(0, qreg[0])
# ibm_circ.initialize(1, qreg[1])
# ibm_circ.initialize(0, qreg[2])

for i in range(n):
    ibm_circ.h(qreg[i])

ibm_circ.barrier(qreg)

#REPETITIONS FOR GROVER
num_reps = 2
for j in range(num_reps):
    gen_oracle(ibm_circ,n,winner,q2)
    gen_diff(ibm_circ,n,q2)

#%%
#RUN
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
service = QiskitRuntimeService()
backend = service.backend("ibmq_qasm_simulator")
# backend = service.least_busy(operational=True, simulator=True, min_num_qubits=3)
#%%
sampler = SamplerV2(backend)
job = sampler.run([ibm_circ])
print(f"job id: {job.job_id()}")
result = job.result()
print(result)
'''



#%%
'''
#EXAMPLES

from qiskit import (ClassicalRegister,
 QuantumRegister,
QuantumCircuit,
transpile)
q = QuantumRegister(2, name="qubit")
c = ClassicalRegister(2, name="bit")
circuit = QuantumCircuit(q, c)
S_simulator = (
 Aer.backends(name='statevector_simulator'))[0]
# circuit simulation
job = S_simulator.run(
 transpile(
 circuit, S_simulator
 )
)
print(job.result().get_statevector())
# Statevector([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
# dims=(2, 2))
#%%
q = QuantumRegister(2, name="qubit")
c = ClassicalRegister(2, name="bit")
circuit = QuantumCircuit(q, c)
circuit.h(q[0])
circuit.cx(q[0], q[1])
# adding the instruction to measure
circuit.measure(q, c)
# actual measurement simulation
M_simulator = (
 Aer.backends(name='qasm_simulator'))[0]
job = M_simulator.run(
 transpile(circuit, M_simulator)
)
print(job.result().get_counts())
# {'11': 496, '00': 528}
# visualising the circuit
print(circuit)
# or
circuit.draw('mpl')
plt.show()

# wraps measurement simulation and get_counts()
print(oq.Measurement(circuit, shots=100))
# 52|11> 48|00>
# let’s delete the final two measurement instructions
circuit.data = circuit.data[:-2]
# wraps state simulation and get_statevector()
print(oq.Wavefunction(circuit))
# 0.70711 |00> 0.70711 |11>

#%%
D = np.array([[0,1],[-1,0]])
def J_gate(gamma, adj=False):
 J_circuit = QuantumCircuit(QuantumRegister(2),
 name=('J' if not adj
 else 'J+'))
 J_circuit.cx(0, 1)
 J_circuit.cz(0, 1)
 J_circuit.rx((gamma if not adj else -gamma), 0)
 J_circuit.cz(0, 1)
 J_circuit.cx(0, 1)
 return J_circuit.to_gate()
# or simply
from qiskit.circuit.library import UnitaryGate
def J_gate(gamma, adj=False):
 return UnitaryGate(
 sp.linalg.expm(
 (-1)**adj * (-1j) * gamma / 2 * np.kron(D, D)
 ), label=('J' if not adj else 'J+')
 )
gamma = np.pi/3
circuit.append(J_gate(gamma, adj=False), q)
circuit.y(q[0]) # U_A = D = 1j * Y
circuit.z(q[1]) # U_B = Q = 1j * Z
circuit.append(J_gate(gamma, adj=True), q)
print(circuit)
print(oq.Wavefunction(circuit))
#END OF EXAMPLES
'''