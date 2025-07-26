import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from qyphos.core.simulator import QyphosSimulator
from qyphos.core.circuit import QuantumCircuit
from qyphos.utils.logger import log

st.set_page_config(layout="wide", page_title="Qyphos Simulator")

st.title("🧠 Qyphos: Next-Gen Quantum Search Simulator")
st.markdown("An interactive, matrix-free quantum search simulator with CPU/GPU backends.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("⚙️ Simulation Parameters")
    n_qubits = st.slider("Number of Qubits (n)", 2, 24, 8)
    n_items = 2 ** n_qubits
    
    st.markdown(f"**Search Space (N = 2ⁿ)**: `{n_items}` items")

    solutions_str = st.text_input("Solution Indices (comma-separated)", "42")
    try:
        solutions = sorted(list(set([int(s.strip()) for s in solutions_str.split(',') if s.strip()])))
        if any(s >= n_items for s in solutions):
            st.error(f"All solutions must be less than N={n_items}.")
            st.stop()
        st.success(f"Found {len(solutions)} unique solutions: {solutions}")
    except ValueError:
        st.error("Invalid input. Please enter comma-separated integers.")
        st.stop()

    n_sols = len(solutions)
    optimal_iters = 0
    if n_sols > 0:
        optimal_iters = int(np.round(np.pi / 4 * np.sqrt(n_items / n_sols)))
    
    st.markdown(f"**Optimal Iterations (R)**: `{optimal_iters}`")
    
    iterations_to_run = st.slider(
        "Iterations to Run", 0, int(optimal_iters * 2.5) + 1, optimal_iters
    )

    backend = st.selectbox("Select Backend", ['auto', 'numpy', 'cupy'], help="Auto will pick CuPy if available.")
    
    run_button = st.button("🚀 Run Simulation", type="primary", use_container_width=True)


# --- Main Panel ---
if run_button:
    # Build Circuit
    circuit = QuantumCircuit(n_qubits)
    circuit.gates.extend([('h', [i], {}) for i in range(n_qubits)]) # Superposition
    for i in range(iterations_to_run):
        circuit.index_oracle(solutions)
        circuit.diffusion()

    try:
        with st.spinner(f"Simulating on '{backend}' backend... This may take a moment for many qubits."):
            sim = QyphosSimulator(backend_name=backend)
            results = sim.run(circuit, store_history=True)
    
        st.success(f"Simulation Complete! ({results['simulation_time_sec']:.4f}s on {results['backend']} backend)")

        # --- Data Analysis ---
        final_probs = results['final_probabilities']
        history = results['state_history']
        
        # --- Create Visualizations ---
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Final Probability Distribution")
            
            # Prepare data for plotting
            prob_df = pd.DataFrame({
                'Index': np.arange(n_items),
                'Probability': final_probs
            })
            prob_df['Type'] = np.where(prob_df['Index'].isin(solutions), 'Solution', 'Non-Solution')
            
            # Downsample for large N to keep plot responsive
            if n_items > 2048:
                non_sol_sample = prob_df[prob_df['Type'] == 'Non-Solution'].sample(n=2048 - n_sols)
                sol_sample = prob_df[prob_df['Type'] == 'Solution']
                plot_df = pd.concat([non_sol_sample, sol_sample]).sort_values('Index')
            else:
                plot_df = prob_df

            fig_prob = go.Figure()
            fig_prob.add_trace(go.Bar(
                x=plot_df[plot_df['Type'] == 'Non-Solution']['Index'],
                y=plot_df[plot_df['Type'] == 'Non-Solution']['Probability'],
                name='Non-Solution',
                marker_color='cornflowerblue'
            ))
            fig_prob.add_trace(go.Bar(
                x=plot_df[plot_df['Type'] == 'Solution']['Index'],
                y=plot_df[plot_df['Type'] == 'Solution']['Probability'],
                name='Solution',
                marker_color='tomato'
            ))
            fig_prob.update_layout(
                title_text=f"Probabilities after {iterations_to_run} Iterations",
                xaxis_title="Item Index",
                yaxis_title="Probability",
                bargap=0.0
            )
            st.plotly_chart(fig_prob, use_container_width=True)

        with col2:
            st.subheader("📈 Success Probability Evolution")
            
            success_probs = []
            # Calculate success probability for each step
            initial_superposition_steps = n_qubits
            grover_steps = 2 # Oracle + Diffusion
            
            # Get success probability after initial superposition
            prob_after_H = np.sum(np.abs(history[initial_superposition_steps])**2)
            success_probs.append(np.sum(prob_after_H[solutions]))

            # Get success probability after each Grover op
            for i in range(iterations_to_run):
                step_index = initial_superposition_steps + (i * grover_steps) + 1 # After oracle
                prob_state = np.abs(history[step_index])**2
                success_probs.append(np.sum(prob_state[solutions]))
            
            iter_axis = np.arange(len(success_probs))

            fig_evol = go.Figure()
            fig_evol.add_trace(go.Scatter(
                x=iter_axis, y=success_probs, mode='lines+markers', name='Success Probability'
            ))
            fig_evol.add_vline(x=optimal_iters, line_dash="dash", line_color="red",
                              annotation_text=f"Optimal: {optimal_iters}", annotation_position="top left")
            fig_evol.update_layout(
                title_text="Total Success Probability vs. Iteration",
                xaxis_title="Grover Iteration",
                yaxis_title="Probability(Solution)",
                yaxis_range=[0, 1.05]
            )
            st.plotly_chart(fig_evol, use_container_width=True)
            
        # --- Summary Stats ---
        st.subheader("📝 Run Summary")
        theoretical_prob = np.sum(final_probs[solutions])
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Theoretical Success", f"{theoretical_prob:.3%}")
        c2.metric("Most Likely Index", f"{np.argmax(final_probs)}")
        c3.metric("Probability of M.L.", f"{np.max(final_probs):.3%}")
        if np.argmax(final_probs) in solutions:
            c4.metric("Outcome", "✅ Correct")
        else:
            c4.metric("Outcome", "❌ Incorrect")

    except MemoryError as e:
        st.error(f"Memory Error: {e}. Please reduce the number of qubits.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
else:
    st.info("Configure your simulation in the sidebar and click 'Run Simulation'.")
