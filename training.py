import tensorflow as tf
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from environment import ElevatorEnv
from state_abstraction import StateExtractor
from bayesian_network import BayesianInference
from model import QNetwork

# Hyperparameters
NUM_HOURS = 10
SIMULATION_TIME = NUM_HOURS * 3600
GAMMA = 0.99 # Decay representing the continuous discount factor roughly
BASE_TEMP = 10.0
MIN_TEMP = 0.1
TEMP_DECAY = 3600 * 2 # Halves approximately every 2 hours

def choose_action(qs, temp):
    """
    Boltzmann action selection for COST minimization:
    P(a) = exp(-Q(s, a) / T) / sum_b exp(-Q(s, b) / T)
    """
    # qs is a list or array of shape (4,)
    # We want to minimize cost, so smaller Q means higher probability.
    # We shift Q for numerical stability
    qs = np.array(qs)
    shifted_qs = qs - np.min(qs)
    exp_q = np.exp(-shifted_qs / temp)
    probs = exp_q / np.sum(exp_q)
    
    # Select index
    return np.random.choice(len(qs), p=probs)

import sys

def train():
    # Keep higher arrival rate (0.8) to ensure enough samples
    env = ElevatorEnv(traffic_type='down-peak', arrival_rate=0.8)
    # Use smaller learning rate for stability
    q_net = QNetwork(lr=0.0005)
    
    t_prev = 0.0
    s_prev_features = None
    
    hourly_wait_times = []
    hourly_wait_sq = []
    hourly_long_waits = []
    times_to_log = []
    cumulative_finished = []
    
    # Log every 15 minutes instead of 1 hour
    LOG_INTERVAL = 900.0 
    next_log_time = LOG_INTERVAL
    last_ui_update = 0.0
    total_loss = 0.0
    update_count = 0
    
    state, p, done = env._run_sim_until_next_decision()
    
    print(f"Starting 10-hour simulation (Arrival Rate: {env.arrival_rate}, LR: {q_net.optimizer.learning_rate.numpy()})...")
    sys.stdout.flush()
    
    while env.time < SIMULATION_TIME:
        # Status update
        if env.time - last_ui_update > 500:
            avg_l = total_loss / max(1, update_count)
            # Find how many passengers are actually finishing
            total_fin = len(env.finished_passengers)
            
            # Additional clarity metrics
            waiting_passengers = sum(len(w) for w in env.waiting_passengers.values())
            in_car_passengers = sum(len(el.passengers) for el in env.elevators)
            
            sys.stdout.write(f"\rProgress: {env.time/SIMULATION_TIME*100:4.1f}% | Time: {env.time/3600:4.2f}h | AvgLoss: {avg_l:7.4f} | Fin: {total_fin} | Waiting: {waiting_passengers} | In-Car: {in_car_passengers}   ")
            sys.stdout.flush()
            last_ui_update = env.time

        # Check for logging binned metrics
        while env.time >= next_log_time:
            # Look at passengers who finished in this 15-min bin
            bin_start = next_log_time - LOG_INTERVAL
            bin_end = next_log_time
            passengers_in_bin = [psg for psg in env.finished_passengers if bin_start < psg.alight_time <= bin_end]
            
            if passengers_in_bin:
                waits = [psg.wait_time for psg in passengers_in_bin]
                avg_wait = np.mean(waits)
                max_wait = np.max(waits)
                avg_sq_wait = np.mean([w**2 for w in waits])
                long_waits_pct = np.mean([1 if w > 60 else 0 for w in waits]) * 100
                
                # Report bin statistics
                sys.stdout.write(f"\n[METRIC {bin_end / 3600:.2f} Hr] Avg Wait: {avg_wait:.2f}s | Max Wait: {max_wait:.2f}s | >60s: {long_waits_pct:.1f}% | Fin: {len(passengers_in_bin)} | Total: {len(env.finished_passengers)}\n")
                sys.stdout.flush()
                
                hourly_wait_times.append(avg_wait)
                hourly_wait_sq.append(avg_sq_wait)
                hourly_long_waits.append(long_waits_pct)
                times_to_log.append(bin_end / 3600)
                cumulative_finished.append(len(env.finished_passengers))
                
            next_log_time += LOG_INTERVAL
        
        if state is None:
             break
            
        current_time = env.time
        call_floor = p.origin
        call_dir = 1 if p.destination > p.origin else -1
        
        # Calculate reward
        raw_r = env.calculate_reward()
        # Scale the reward using log to prevent explosion: -log(1 + sqrt(sum squares) / 100)
        # This keeps reward in a manageable [-10, 0] range typically.
        r = -np.log1p(abs(raw_r) / 100.0)
        
        qs = []
        features_list = []
        for el in env.elevators:
            s_abs = StateExtractor.get_abstract_state(el, call_floor, call_dir)
            n_floor, n_people, n_stop = BayesianInference.infer(s_abs)
            
            # Feature scaling to [0, 1] range
            f = [n_floor/10.0, n_people/10.0, n_stop/10.0, float(el.id)/4.0]
            features_list.append(f)
            
            f_tensor = tf.convert_to_tensor([f], dtype=tf.float32)
            q_val = float(q_net(f_tensor)[0][0])
            qs.append(q_val)
            
        temp = max(MIN_TEMP, BASE_TEMP * math.exp(-current_time / TEMP_DECAY))
        action_idx = choose_action(qs, temp)
        
        if s_prev_features is not None:
            tau = current_time - t_prev
            # Discount factor for SMDP: gamma ^ tau
            df = math.pow(GAMMA, tau / 10.0)
            
            min_q_next = min(qs)
            target_q = r + df * min_q_next 
            
            target_tensor = tf.convert_to_tensor([[target_q]], dtype=tf.float32)
            s_prev_tensor = tf.convert_to_tensor([s_prev_features], dtype=tf.float32)
            
            loss = q_net.update(s_prev_tensor, target_tensor)
            total_loss += loss
            update_count += 1
            
        s_prev_features = features_list[action_idx]
        t_prev = current_time
        
        state, p, done = env.step_assign(action_idx, call_floor, call_dir)

    print("\nTraining Complete. Computing summary statistics...")
    all_waits = [psg.wait_time for psg in env.finished_passengers]
    total_fin = len(all_waits)
    if total_fin > 0:
        overall_avg_wait = np.mean(all_waits)
        overall_max_wait = np.max(all_waits)
        overall_long_pct = np.mean([1 if w > 60 else 0 for w in all_waits]) * 100
    else:
        overall_avg_wait = 0.0
        overall_max_wait = 0.0
        overall_long_pct = 0.0
    
    summary_text = (
        "\n==================================================\n"
        "               SIMULATION SUMMARY\n"
        "==================================================\n"
        f"Total Passengers Served: {total_fin}\n"
        f"Overall Average Wait Time: {overall_avg_wait:.2f} s\n"
        f"Overall Maximum Wait Time: {overall_max_wait:.2f} s\n"
        f"Percentage of Wait > 60s: {overall_long_pct:.2f} %\n"
        f"Total Training Steps: {update_count}\n"
        f"Final Exploration Temp: {temp:.4f}\n"
    )
    if update_count > 0:
        summary_text += f"Average Loss: {(total_loss / update_count):.4f}\n"
    summary_text += "==================================================\n"
    
    print(summary_text)
    
    # Save the text output to a file
    with open("output.txt", "w", encoding="utf-8") as f:
        f.write(summary_text)

    print("Plotting metrics...")
    
    # Plotting
    plt.figure(figsize=(12, 10))

    
    plt.subplot(2, 2, 1)
    plt.plot(times_to_log, hourly_wait_times, marker='o', color='b')
    plt.title("Average Wait Time over Simulation")
    plt.ylabel("Wait Time (s)")
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(times_to_log, hourly_wait_sq, marker='o', color='r')
    plt.title("Average Squared Wait Time")
    plt.ylabel("Squared Wait Time (s^2)")
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(times_to_log, hourly_long_waits, marker='o', color='g')
    plt.title("% Passengers Waiting > 60s")
    plt.xlabel("Simulation Time (Hours)")
    plt.ylabel("Percentage (%)")
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    if total_fin > 0:
        plt.hist(all_waits, bins=min(50, total_fin), color='purple', alpha=0.7)
    plt.title("Distribution of Wait Times")
    plt.xlabel("Wait Time (s)")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("metrics_plot.png")
    print("Saved metrics plot to metrics_plot.png")

    print("Plotting extra metrics...")
    all_travels = [psg.travel_time for psg in env.finished_passengers]
    all_total_times = [psg.wait_time + psg.travel_time for psg in env.finished_passengers]

    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 2, 1)
    if len(times_to_log) == len(cumulative_finished):
        plt.plot(times_to_log, cumulative_finished, marker='s', color='orange')
    plt.title("Cumulative Passengers Served")
    plt.xlabel("Simulation Time (Hours)")
    plt.ylabel("Total Passengers")
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    if total_fin > 0:
        plt.hist(all_travels, bins=min(50, total_fin), color='teal', alpha=0.7)
    plt.title("Distribution of Travel Times")
    plt.xlabel("Travel Time (s)")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    if total_fin > 0:
        plt.scatter(all_waits, all_travels, color='darkred', alpha=0.3, s=10)
    plt.title("Wait Time vs Travel Time")
    plt.xlabel("Wait Time (s)")
    plt.ylabel("Travel Time (s)")
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    if total_fin > 0:
        plt.hist(all_total_times, bins=min(50, total_fin), color='navy', alpha=0.7)
    plt.title("Distribution of Total Journey Time")
    plt.xlabel("Total Time (s)")
    plt.ylabel("Frequency")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("extra_metrics.png")
    print("Saved extra metrics plot to extra_metrics.png")

if __name__ == "__main__":
    train()
