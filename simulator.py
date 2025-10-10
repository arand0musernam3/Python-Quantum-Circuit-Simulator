import numpy as np
import sys
from collections import Counter

isq2 = 1.0 / (2.0 ** 0.5)

class QState:
    def __init__(self, n):
        self.n = n
        self.state = np.zeros(2**n, dtype=complex)
        self.state[0] = 1

    def op(self, t, i):
        """Apply a gate matrix t starting at qubit i (MSB-first)."""
        # Not used anymore, but kept in case matrix-based gates are needed
        eyeL = np.eye(2**i, dtype=complex)
        k = int(np.log2(t.shape[0]))
        eyeR = np.eye(2**(self.n - i - k), dtype=complex)
        t_all = np.kron(np.kron(eyeL, t), eyeR)
        self.state = np.matmul(t_all, self.state)

    def _bitpos(self, qubit_index):
        """Convert MSB-first index to bit position for state vector indexing."""
        return self.n - qubit_index - 1

    # --- Single-qubit gates ---
    def H(self, i):
        """Hadamard gate on qubit i."""
        new = np.zeros_like(self.state)
        p_i = self._bitpos(i)
        for idx in range(2**self.n):
            amp = self.state[idx]
            j = idx ^ (1 << p_i)
            if (idx >> p_i) & 1: # Qubit is 1
                new[j] += amp * isq2
                new[idx] += amp * -isq2 # Minus sign for the |1> state
            else: # Qubit is 0
                new[idx] += amp * isq2
                new[j] += amp * isq2
        self.state = new

    def X(self, i):
        """Pauli-X (NOT) gate on qubit i."""
        new = np.zeros_like(self.state)
        p_i = self._bitpos(i)
        for idx in range(2**self.n):
            amp = self.state[idx]
            # flip qubit i using XOR mask
            j = idx ^ (1 << p_i)
            new[j] += amp
        self.state = new

    def Y(self, i):
        """Pauli-Y gate on qubit i."""
        new = np.zeros_like(self.state)
        p_i = self._bitpos(i)
        for idx in range(2**self.n):
            amp = self.state[idx]
            j = idx ^ (1 << p_i)
            if (idx >> p_i) & 1: # Qubit is 1, so the new amplitude goes to the '0' state
                new[j] += amp * 1j
            else: # Qubit is 0, so the new amplitude goes to the '1' state
                new[j] += amp * -1j
        self.state = new

    def Z(self, i):
        """Pauli-Z gate on qubit i."""
        new = np.zeros_like(self.state)
        p_i = self._bitpos(i)
        for idx in range(2**self.n):
            amp = self.state[idx]
            if (idx >> p_i) & 1:
                new[idx] += -amp
            else:
                new[idx] += amp
        self.state = new

    def S(self, i):
        """S (phase) gate on qubit i."""
        new = np.zeros_like(self.state)
        p_i = self._bitpos(i)
        for idx in range(2**self.n):
            amp = self.state[idx]
            if (idx >> p_i) & 1:
                # qubit i is 1, apply phase i
                new[idx] += 1j * amp
            else:
                new[idx] += amp
        self.state = new

    def T(self, i):
        """T (π/8) gate on qubit i."""
        new = np.zeros_like(self.state)
        p_i = self._bitpos(i)
        phase = np.exp(1j * np.pi / 4)
        for idx in range(2**self.n):
            amp = self.state[idx]
            if (idx >> p_i) & 1:
                # qubit i is 1, apply phase exp(iπ/4)
                new[idx] += phase * amp
            else:
                new[idx] += amp
        self.state = new

    def sqrtX(self, i):
        """Square root of X gate on qubit i."""
        new = np.zeros_like(self.state)
        p_i = self._bitpos(i)
        
        c00 = 0.5 * (1 + 1j) # Top-left element
        c01 = 0.5 * (1 - 1j) # Top-right element
        c10 = 0.5 * (1 - 1j) # Bottom-left element
        c11 = 0.5 * (1 + 1j) # Bottom-right element

        for idx in range(2**self.n):
            amp = self.state[idx]
            j = idx ^ (1 << p_i) # The flipped index
            
            # Distribute amplitude based on the original qubit's value
            if (idx >> p_i) & 1: # Qubit is 1
                # Adds to the '0' state with a factor (c10)
                new[j] += amp * c10
                # Adds to the '1' state with a factor (c11)
                new[idx] += amp * c11
            else: # Qubit is 0
                # Adds to the '0' state with a factor (c00)
                new[idx] += amp * c00
                # Adds to the '1' state with a factor (c01)
                new[j] += amp * c01
                
        self.state = new

    def Rtheta(self, i, theta):
        """Rotation around Y axis by θ on qubit i."""
        new = np.zeros_like(self.state)
        p_i = self._bitpos(i)

        c = np.cos(theta/2)
        s = np.sin(theta/2)

        for idx in range(2**self.n):
            amp = self.state[idx]
            j = idx ^ (1 << p_i)
            
            if (idx >> p_i) & 1: # Qubit is 1
                new[j] += amp * s
                new[idx] += amp * c
            else: # Qubit is 0
                new[idx] += amp * c
                new[j] += amp * -s
                
        self.state = new

    def Rphi(self, i, phi):
        """Phase-shift by φ on qubit i."""
        new = np.zeros_like(self.state)
        p_i = self._bitpos(i)
        phase = np.exp(1j * phi)

        for idx in range(2**self.n):
            amp = self.state[idx]
            if (idx >> p_i) & 1:
                new[idx] += phase * amp
            else:
                new[idx] += amp
        self.state = new

    def CRphi(self, control, target, phi):
        """
        Controlled-Rphi gate, applies a phase shift of phi to the target qubit
        if the control qubit is in the |1> state.
        """
        new = np.zeros_like(self.state)
        p_c, p_t = self._bitpos(control), self._bitpos(target)
        phase = np.exp(1j * phi)

        for idx in range(2**self.n):
            amp = self.state[idx]
            # If the control qubit is 1 and target is also 1, apply the phase shift
            if (idx >> p_c) & 1 and (idx >> p_t) & 1:
                new[idx] += amp * phase
            # Else do nothing
            else:
                new[idx] += amp
        
        self.state = new

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

    # --- Multi-qubit gates ---
    def QFT(self, qubits=None):
        """Full Quantum Fourier Transform on all qubits or a subset."""

        # qubits variable allows to specify a subset of qubits for QFT, 
        # we assume input is within range
        if qubits is None:
            qubits = list(range(self.n))
        
        num_qubits = len(qubits)
        for i in range(num_qubits):
            # target qubit is qubits[i]
            self.H(qubits[i]) # Hadamard on qubit i

            for j in range(i+1, num_qubits):
                # control qubit is qubits[j]
                # Controlled phase rotation between qubits j and i
                angle = np.pi / (2 ** (j - i))
                self.CRphi(qubits[j], qubits[i], angle)
        
        # Apply SWAPs to reverse the order of qubits
        for i in range(num_qubits // 2):
            q1 = qubits[i]
            q2 = qubits[num_qubits - i - 1]
            self.SWAP(q1, q2)


    # --- Measurement ---
    def measure(self):
        """Single projective measurement, collapses the state."""
        probs = np.abs(self.state)**2
        outcome = np.random.choice(2**self.n, p=probs)
        new = np.zeros_like(self.state)
        new[outcome] = 1.0
        self.state = new
        return f"|{format(outcome, f'0{self.n}b')}⟩"

    def measure_shots(self, shots=1000000):
        """Repeat measurement many times, return statistics."""
        probs = np.abs(self.state)**2
        outcomes = np.random.choice(2**self.n, size=shots, p=probs)
        counts = Counter(outcomes)
        print(f"Results after {shots} shots:")
        for outcome, count in sorted(counts.items()):
            perc = count/shots*100
            print(f"  |{format(outcome, f'0{self.n}b')}⟩ : {count} ({perc:.2f}%)")
        return counts
    
    # --- Printing ---
    def print_state(self, decimals=4):
        """Prints the state vector in a more readable format."""
        for idx, amp in enumerate(self.state):
            # Use a tolerance for floating-point comparison
            if not np.isclose(amp, 0, atol=1e-12):
                bitstring = format(idx, f'0{self.n}b')
                
                # Format the real and imaginary parts
                real_part = f"{amp.real:.{decimals}f}"
                imag_part = f"{abs(amp.imag):.{decimals}f}"
                
                # Determine the sign for the imaginary part
                sign = '+' if amp.imag >= 0 else '-'
                
                # Print the state
                print(f"  |{bitstring}⟩: {real_part} {sign} {imag_part}j")

        print() # Add an extra newline for better readability

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
    q = QState(n_qubits)

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
        elif cmd == "CRPHI":
            q.CRphi(int(parts[1]), int(parts[2]), float(parts[3]))
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
        q.print_state()
    return q

# ------------------ Main / Testing ------------------
def test_qft_matrix(n):
    """Tests the full QFT against the theoretical matrix."""
    N = 2**n
    print(f"\n--- Testing: Full QFT matrix for n={n} (N={N}) ---\n")

    # simulate QFT on each basis state
    simulated_matrix = np.zeros((N, N), dtype=complex)
    for i in range(N):
        q = QState(n)
        q.state = np.zeros(N, dtype=complex); q.state[i] = 1.0
        q.QFT()
        simulated_matrix[:, i] = q.state

    # by definition of QFT
    omega = np.exp(2 * np.pi * 1j / N)
    theoretical_matrix = np.array([[omega**(j*k) for k in range(N)] for j in range(N)]) / np.sqrt(N)
    
    # comparison
    if np.allclose(simulated_matrix, theoretical_matrix):
        print("SUCCESS.\n")
    else:
        print("FAILURE.\n")

def test_qft_periodic_state(n=6, period=8):
    """Tests the full QFT on a periodic state."""
    N = 2**n
    print(f"\n--- Testing: Full QFT on a periodic state (n={n}, period={period}) ---\n")
    q = QState(n)
    initial_state = np.zeros(N, dtype=complex)
    # Create a periodic superposition state (what we would obtain after modular exponentiation)
    for i in range(N // period): 
        initial_state[i * period] = 1.0
    
    q.state = initial_state / np.linalg.norm(initial_state)
    
    print(f"Applying QFT... expecting peaks at multiples of N/period = {N}/{period} = {N/period}.")
    q.QFT()
    # try to visualize the result, if N/period is not exact peaks should still be visible
    q.measure_shots()


def problem2_week8():
    test_qft_matrix(n=3)
    test_qft_periodic_state(n=5, period=4)
    test_qft_periodic_state(n=5, period=6)

    

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # run the circuit file passed as argument
        filename = sys.argv[1]
        run_circuit_file(filename)
    else:
        print("\nRunning problem 2 from week 8:\n")
        problem2_week8()
        