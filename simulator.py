import numpy as np
import sys
from collections import Counter

isq2 = 1.0 / (2.0 ** 0.5)

class Qstate:
    def __init__(self, n):
        self.n = n
        self.state = np.zeros(2**n, dtype=complex)
        self.state[0] = 1

    def op(self, t, i):
        """Apply a gate matrix t starting at qubit i (MSB-first)."""
        eyeL = np.eye(2**i, dtype=complex)
        k = int(t.shape[0]**0.5)
        eyeR = np.eye(2**(self.n - i - k), dtype=complex)
        t_all = np.kron(np.kron(eyeL, t), eyeR)
        self.state = np.matmul(t_all, self.state)

    def _bitpos(self, qubit_index):
        """Convert MSB-first index to bit position for state vector indexing."""
        return self.n - qubit_index - 1

    # --- Single-qubit gates ---
    def H(self, i):
        """Hadamard gate on qubit i."""
        h = isq2 * np.array([[1, 1], [1, -1]], dtype=complex)
        self.op(h, i)

    def X(self, i):
        """Pauli-X (NOT) gate on qubit i."""
        x = np.array([[0, 1], [1, 0]], dtype=complex)
        self.op(x, i)

    def Y(self, i):
        """Pauli-Y gate on qubit i."""
        y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        self.op(y, i)

    def Z(self, i):
        """Pauli-Z gate on qubit i."""
        z = np.array([[1, 0], [0, -1]], dtype=complex)
        self.op(z, i)

    def S(self, i):
        """S (phase) gate on qubit i."""
        s = np.array([[1, 0], [0, 1j]], dtype=complex)
        self.op(s, i)

    def T(self, i):
        """T (π/8) gate on qubit i."""
        t = np.array([[1, 0], [0, np.exp(1j*np.pi/4)]], dtype=complex)
        self.op(t, i)

    def sqrtX(self, i):
        """Square root of X gate on qubit i."""
        sx = 0.5 * np.array([[1+1j, 1-1j], [1-1j, 1+1j]], dtype=complex)
        self.op(sx, i)

    def Rtheta(self, i, theta):
        """Rotation around Y axis by θ on qubit i."""
        r = np.array([
            [np.cos(theta/2), -np.sin(theta/2)],
            [np.sin(theta/2),  np.cos(theta/2)]
        ], dtype=complex)
        self.op(r, i)

    def Rphi(self, i, phi):
        """Phase-shift by φ on qubit i."""
        r = np.array([[1, 0], [0, np.exp(1j*phi)]], dtype=complex)
        self.op(r, i)

    # --- Two-qubit gates ---
    def CNOT(self, control, target):
        """Controlled-NOT with given control and target."""
        new = np.zeros_like(self.state)
        # convert MSB-first qubit indices to bit positions
        p_c, p_t = self._bitpos(control), self._bitpos(target)
        for idx in range(2**self.n):
            amp = self.state[idx]
            # check if control qubit is 1
            if (idx >> p_c) & 1:
                # flip target qubit using XOR mask
                j = idx ^ (1 << p_t)
                new[j] += amp
            else:
                new[idx] += amp
        self.state = new

    def CZ(self, control, target):
        """Controlled-Z with given control and target."""
        new = np.zeros_like(self.state)
        p_c, p_t = self._bitpos(control), self._bitpos(target)
        for idx in range(2**self.n):
            amp = self.state[idx]
            # if both control and target qubits are 1, apply phase flip
            if ((idx >> p_c) & 1) and ((idx >> p_t) & 1):
                new[idx] += -amp
            else:
                new[idx] += amp
        self.state = new

    def SWAP(self, a, b):
        """Swap qubits a and b."""
        new = np.zeros_like(self.state)
        p_a, p_b = self._bitpos(a), self._bitpos(b)
        for idx in range(2**self.n):
            bit_a = (idx >> p_a) & 1  # extract qubit a
            bit_b = (idx >> p_b) & 1  # extract qubit b
            if bit_a == bit_b:
                # same bits, no change
                new[idx] += self.state[idx]
            else:
                # flip both bits using XOR mask
                j = idx ^ ((1 << p_a) | (1 << p_b))
                new[j] += self.state[idx]
        self.state = new

    # --- Three-qubit gates ---
    def CCNOT(self, control1, control2, target):
        """Toffoli gate with two controls and one target."""
        new = np.zeros_like(self.state)
        p_c1, p_c2, p_t = self._bitpos(control1), self._bitpos(control2), self._bitpos(target)
        for idx in range(2**self.n):
            amp = self.state[idx]
            # if both control qubits are 1, flip the target
            if ((idx >> p_c1) & 1) and ((idx >> p_c2) & 1):
                j = idx ^ (1 << p_t)
                new[j] += amp
            else:
                new[idx] += amp
        self.state = new

    def CSWAP(self, control, a, b):
        """Fredkin gate: controlled swap of qubits a and b."""
        new = np.zeros_like(self.state)
        p_c, p_a, p_b = self._bitpos(control), self._bitpos(a), self._bitpos(b)
        for idx in range(2**self.n):
            amp = self.state[idx]
            # if control qubit is 1, swap a and b
            if (idx >> p_c) & 1:
                bit_a = (idx >> p_a) & 1
                bit_b = (idx >> p_b) & 1
                if bit_a == bit_b:
                    new[idx] += amp
                else:
                    # flip both bits using XOR mask
                    j = idx ^ ((1 << p_a) | (1 << p_b))
                    new[j] += amp
            else:
                new[idx] += amp
        self.state = new


    # --- Measurement ---
    def measure(self):
        """Single projective measurement, collapses the state."""
        probs = np.abs(self.state)**2
        outcome = np.random.choice(2**self.n, p=probs)
        new = np.zeros_like(self.state)
        new[outcome] = 1.0
        self.state = new
        return f"|{format(outcome, f'0{self.n}b')}⟩"

    def measure_shots(self, shots=100000):
        """Repeat measurement many times, return statistics."""
        probs = np.abs(self.state)**2
        outcomes = np.random.choice(2**self.n, size=shots, p=probs)
        counts = Counter(outcomes)
        print(f"Results after {shots} shots:")
        for outcome, count in sorted(counts.items()):
            perc = count/shots*100
            print(f"  |{format(outcome, f'0{self.n}b')}⟩ : {count} ({perc:.2f}%)")
        return counts

    

# ------------------ Circuit Parser ------------------
def run_circuit_file(filename):
    with open(filename, "r") as f:
        lines = []
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            lines.append(stripped)

    n_qubits = int(lines[0])
    q = Qstate(n_qubits)

    print(f"\n--- Running circuit from {filename} with {n_qubits} qubits ---")

    saw_measure = False
    for line in lines[1:]:
        parts = line.split()
        cmd = parts[0].upper()

        if cmd == "H":
            q.H(int(parts[1]))
        elif cmd in ("X", "NOT"):
            q.X(int(parts[1]))
        elif cmd == "Y":
            q.Y(int(parts[1]))
        elif cmd == "Z":
            q.Z(int(parts[1]))
        elif cmd == "S":
            q.S(int(parts[1]))
        elif cmd == "T":
            q.T(int(parts[1]))
        elif cmd in ("SQRTX", "SQRTX"):
            q.sqrtX(int(parts[1]))
        elif cmd in ("R", "RTHETA"):
            q.Rtheta(int(parts[1]), float(parts[2]))
        elif cmd in ("P", "RPHI"):
            q.Rphi(int(parts[1]), float(parts[2]))
        elif cmd == "CNOT":
            q.CNOT(int(parts[1]), int(parts[2]))
        elif cmd in ("CZ", "CPHASE"):
            q.CZ(int(parts[1]), int(parts[2]))
        elif cmd == "SWAP":
            q.SWAP(int(parts[1]), int(parts[2]))
        elif cmd in ("CCNOT", "TOFFOLI"):
            q.CCNOT(int(parts[1]), int(parts[2]), int(parts[3]))
        elif cmd == "CSWAP":
            q.CSWAP(int(parts[1]), int(parts[2]), int(parts[3]))
        elif cmd == "INITSTATE":
            mode = parts[1].upper()
            if mode == "FILE":
                fname = parts[2]
                vec = []
                with open(fname, "r") as fin:
                    for line in fin:
                        re, im = map(float, line.split())
                        vec.append(complex(re, im))
                q.state = np.array(vec, dtype=complex)
                q.state /= np.linalg.norm(q.state)
            elif mode == "BASIS":
                basis = parts[2].strip()
                assert basis.startswith("|") and basis.endswith(">")
                bitstring = basis[1:-1]
                idx = int(bitstring, 2)
                q.state = np.zeros(2**q.n, dtype=complex)
                q.state[idx] = 1.0
        elif cmd == "MEASURE":
            q.measure_shots()
            saw_measure = True
        else:
            raise ValueError(f"Unknown command: {line}")

    if not saw_measure:
        print("Final state (no measurement command found):")
        print(q.state)
    return q

# ------------------ Main / Testing ------------------
if __name__ == "__main__":
    if len(sys.argv) > 1:
        # run the circuit file passed as argument
        filename = sys.argv[1]
        run_circuit_file(filename)
    else:
        # --- Default tests if no argument is given ---

        # Constructing an EPR pair and measuring
        s = Qstate(2)
        s.H(0)
        s.CNOT(0, 1)
        print("EPR pair state vector:")
        print(s.state)
        s.measure_shots(1000)
        print("State after measurement:")
        print(s.state)
        print("Single measurement outcome:", s.measure())
        print("State after single measurement:")
        print(s.state)

        # Toffoli example
        q = Qstate(3)
        q.H(0)
        q.H(1)
        q.X(2)  # set target to |1>
        q.CCNOT(0, 1, 2)
        print("Toffoli gate example:")
        q.measure_shots(200000)

        # Website example
        s = Qstate(3)
        s.H(1)
        s.H(2)
        s.Rphi(2, 0.3)
        s.CNOT(2, 1)
        s.H(1)
        s.H(2)
        s.CNOT(2, 0)
        print("Website example:")
        s.measure_shots(100000)

        # Example circuit files
        print("Running example circuit files:")
        run_circuit_file("rand.circuit")
        run_circuit_file("measure.circuit")
        run_circuit_file("input.circuit")