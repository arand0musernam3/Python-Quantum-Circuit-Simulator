# Quantum Circuit Simulator

This is a simple Python-based quantum circuit simulator that supports standard single-, two-, and three-qubit gates, measurements, and reading circuits from `.circuit` files. It uses a **state vector representation** and allows simulating up to a few qubits efficiently.

It is based on the premises established by this website: [Quantum Computing Simulator](https://courses.physics.illinois.edu/phys446/sp2023/QC/1a-QuantumComputingSimulator.html).

## Features

- **Single-qubit gates**: `H`, `X`, `Y`, `Z`, `S`, `T`, `Rphi`, `Rtheta`, `SQRTX`
- **Two-qubit gates**: `CNOT`, `CZ`, `SWAP`
- **Three-qubit gates**: `CCNOT` (Toffoli), `CSWAP` (Fredkin)
- **Measurement**: single-shot or repeated (`measure_shots`) with statistics
- **Circuit file support**: `.circuit` files with commands like `H 0`, `CNOT 0 1`, `MEASURE`, etc.
- **Custom initial states**: from basis state `|001>` or from a file containing amplitudes

## Usage

### Run a circuit file:
```bash
python simulator.py my_circuit.circuit
```

### Run default test examples (EPR, Toffoli, reference example):
```bash
python simulator.py
```

## Circuit file format
Example `.circuit` file:
```
3
H 0
H 1
P 2 0.3
CNOT 2 1
H 1
H 2
CNOT 2 0
MEASURE
```
- The first line specifies the number of qubits.
- Subsequent lines are gate commands.
- `INITSTATE` can set the initial state from a file or basis.
- Lines starting with `#` are treated as comments.
- `MEASURE` triggers measurement statistics.

## Code example
```py
s = QState(2)
s.H(0)
s.CNOT(0, 1)
s.measure_shots(1000)
```