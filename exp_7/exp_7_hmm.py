"""
21CSC305P — Machine Learning Lab
Experiment 7: Hidden Markov Model (Viterbi Algorithm)

AIM: To implement HMM to predict the most likely sequence of hidden states.

ALGORITHM:
1. Define states, observations, and model parameters:
   - Start probabilities
   - Transition probabilities
   - Emission probabilities
2. For each observation, calculate probability of being in each state.
3. Use Viterbi algorithm to find the most likely state sequence.
4. Backtrack to reconstruct the full path.
"""

import numpy as np
import matplotlib.pyplot as plt


class SimpleHMM:
    """
    Hidden Markov Model with Viterbi decoding.

    States: Hidden states (e.g., Rainy, Sunny)
    Observations: Visible outputs (e.g., Walk, Shop, Clean)
    """

    def __init__(self, states, observations, start_prob, trans_prob, emit_prob):
        self.states = states
        self.observations = observations
        self.start_prob = start_prob      # P(initial state)
        self.trans_prob = trans_prob      # P(state_j | state_i)
        self.emit_prob = emit_prob        # P(observation | state)

    def viterbi(self, obs_sequence):
        """
        Viterbi Algorithm:
        Find the most likely sequence of hidden states given observations.

        Returns: (best_path, best_probability)
        """
        n_states = len(self.states)
        n_obs = len(obs_sequence)

        # Initialize dynamic programming tables
        viterbi_table = np.zeros((n_states, n_obs))
        backpointer_table = np.zeros((n_states, n_obs), dtype=int)

        # ── Initialization step ────────────────────────────────────
        first_obs = obs_sequence[0]
        for s in range(n_states):
            viterbi_table[s, 0] = self.start_prob[s] * self.emit_prob[s, first_obs]
            backpointer_table[s, 0] = 0

        # ── Recursion step ─────────────────────────────────────────
        for t in range(1, n_obs):
            for s in range(n_states):
                # For each state, find the max probability path leading here
                probabilities = (
                    viterbi_table[:, t-1] *
                    self.trans_prob[:, s] *
                    self.emit_prob[s, obs_sequence[t]]
                )
                viterbi_table[s, t] = np.max(probabilities)
                backpointer_table[s, t] = np.argmax(probabilities)

        # ── Termination step ───────────────────────────────────────
        best_path_prob = np.max(viterbi_table[:, n_obs-1])
        best_last_state = np.argmax(viterbi_table[:, n_obs-1])

        # ── Path backtracking ──────────────────────────────────────
        best_path = np.zeros(n_obs, dtype=int)
        best_path[-1] = best_last_state
        for t in range(n_obs - 2, -1, -1):
            best_path[t] = backpointer_table[best_path[t+1], t+1]

        return best_path, best_path_prob


# ─── Define the HMM (Weather example from lab manual) ────────────────
states = ['Rainy', 'Sunny']
observations = ['Walk', 'Shop', 'Clean']

# P(initial state)
start_probability = np.array([0.6, 0.4])

# P(next_state | current_state)
transition_probability = np.array([
    [0.7, 0.3],   # From Rainy → Rainy/Sunny
    [0.4, 0.6],   # From Sunny → Rainy/Sunny
])

# P(observation | state)
emission_probability = np.array([
    [0.1, 0.4, 0.5],   # From Rainy: Walk/Shop/Clean
    [0.6, 0.3, 0.1],   # From Sunny: Walk/Shop/Clean
])

# ─── Create HMM and run Viterbi ──────────────────────────────────────
hmm = SimpleHMM(states, observations, start_probability,
                transition_probability, emission_probability)

# Encode observations as integers
obs_map = {obs: i for i, obs in enumerate(observations)}
obs_sequence = np.array([obs_map['Walk'], obs_map['Shop'],
                         obs_map['Clean'], obs_map['Walk']])

print("Observations:", [observations[i] for i in obs_sequence])
state_sequence, probability = hmm.viterbi(obs_sequence)

# Decode state sequence
state_names = [states[state] for state in state_sequence]
print(f"Most likely states: {state_names}")
print(f"Probability: {probability:.7f}")

# ─── Visualize the Viterbi table ─────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: State sequence
ax1 = axes[0]
state_colors = {'Rainy': '#3498db', 'Sunny': '#f1c40f'}
for t, (obs, state) in enumerate(zip([observations[i] for i in obs_sequence],
                                      state_names)):
    ax1.barh(0, 1, left=t, height=0.5, color=state_colors[state], alpha=0.8)
    ax1.text(t + 0.5, 0.15, state, ha='center', va='center', fontsize=12,
             fontweight='bold')
    ax1.text(t + 0.5, -0.15, f'Obs: {obs}', ha='center', va='center', fontsize=10)
ax1.set_xlim(0, len(obs_sequence))
ax1.set_ylim(-0.5, 0.5)
ax1.set_title(f'Viterbi Path (P={probability:.6f})')
ax1.set_xlabel('Time Step')
ax1.set_yticks([])

# Right: Transition diagram
ax2 = axes[1]
ax2.set_xlim(-1, 3)
ax2.set_ylim(-1, 3)
ax2.set_aspect('equal')

# Draw states
circle_rainy = plt.Circle((0.5, 1.5), 0.4, color='#3498db', alpha=0.3)
circle_sunny = plt.Circle((2.0, 1.5), 0.4, color='#f1c40f', alpha=0.3)
ax2.add_patch(circle_rainy)
ax2.add_patch(circle_sunny)
ax2.text(0.5, 1.5, 'Rainy\n0.6', ha='center', va='center', fontsize=11, fontweight='bold')
ax2.text(2.0, 1.5, 'Sunny\n0.4', ha='center', va='center', fontsize=11, fontweight='bold')

# Draw transitions
ax2.annotate('', xy=(1.6, 1.5), xytext=(0.9, 1.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
ax2.annotate('', xy=(0.9, 1.7), xytext=(1.6, 1.7),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
ax2.text(1.25, 1.85, '0.3', ha='center', fontsize=9)
ax2.text(1.25, 1.35, '0.4', ha='center', fontsize=9)
ax2.set_title('Transition Probabilities')
ax2.axis('off')

plt.tight_layout()
plt.savefig('hmm_result.png', dpi=150)
plt.show()

print("\nResult: HMM with Viterbi algorithm executed successfully.")
