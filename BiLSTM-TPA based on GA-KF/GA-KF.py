import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
from scipy import signal as scipy_signal
import warnings
import time

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

try:
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning)

    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')

    def setup_gpu():
        print("Detecting GPU...")
        tf.autograph.set_verbosity(0)

        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                tf.config.set_visible_devices(gpus, 'GPU')

                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f"Detected {len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs")
                print("GPU configuration successful, using GPU for calculations")
                return True
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
                return False
        else:
            print("No GPU detected, using CPU for calculations")
            return False

    HAS_GPU = setup_gpu()

    if HAS_GPU:
        print("Using TensorFlow to accelerate matrix operations")

        def matrix_multiply(a, b):
            a_np = np.asarray(a)
            b_np = np.asarray(b)

            a_shape = a_np.shape
            b_shape = b_np.shape

            if len(a_shape) == 2 and len(b_shape) == 1:
                b_reshaped = b_np.reshape(-1, 1)
                result = np.matmul(a_np, b_reshaped)
                return result.flatten()
            elif len(a_shape) == 1 and len(b_shape) == 2:
                a_reshaped = a_np.reshape(1, -1)
                result = np.matmul(a_reshaped, b_np)
                return result.flatten()
            elif len(a_shape) == 1 and len(b_shape) == 1:
                return np.dot(a_np, b_np)
            else:
                return np.matmul(a_np, b_np)

        def matrix_inverse(a):
            if len(np.shape(a)) == 2:
                return np.linalg.inv(a)
            else:
                a_shape = np.shape(a)
                a_reshaped = np.reshape(a, (a_shape[0], a_shape[0]))
                result = np.linalg.inv(a_reshaped)
                return result
    else:
        def matrix_multiply(a, b):
            a_np = np.asarray(a)
            b_np = np.asarray(b)

            a_shape = a_np.shape
            b_shape = b_np.shape

            if len(a_shape) == 2 and len(b_shape) == 1:
                b_reshaped = b_np.reshape(-1, 1)
                result = np.matmul(a_np, b_reshaped)
                return result.flatten()
            elif len(a_shape) == 1 and len(b_shape) == 2:
                a_reshaped = a_np.reshape(1, -1)
                result = np.matmul(a_reshaped, b_np)
                return result.flatten()
            elif len(a_shape) == 1 and len(b_shape) == 1:
                return np.dot(a_np, b_np)
            else:
                return np.matmul(a_np, b_np)

        def matrix_inverse(a):
            if len(np.shape(a)) == 2:
                return np.linalg.inv(a)
            else:
                a_shape = np.shape(a)
                a_reshaped = np.reshape(a, (a_shape[0], a_shape[0]))
                result = np.linalg.inv(a_reshaped)
                return result
except ImportError:
    print("TensorFlow not installed, cannot use GPU acceleration")
    HAS_GPU = False

    def matrix_multiply(a, b):
        a_np = np.asarray(a)
        b_np = np.asarray(b)

        a_shape = a_np.shape
        b_shape = b_np.shape

        if len(a_shape) == 2 and len(b_shape) == 1:
            b_reshaped = b_np.reshape(-1, 1)
            result = np.matmul(a_np, b_reshaped)
            return result.flatten()
        elif len(a_shape) == 1 and len(b_shape) == 2:
            a_reshaped = a_np.reshape(1, -1)
            result = np.matmul(a_reshaped, b_np)
            return result.flatten()
        elif len(a_shape) == 1 and len(b_shape) == 1:
            return np.dot(a_np, b_np)
        else:
            return np.matmul(a_np, b_np)

    def matrix_inverse(a):
        if len(np.shape(a)) == 2:
            return np.linalg.inv(a)
        else:
            a_shape = np.shape(a)
            a_reshaped = np.reshape(a, (a_shape[0], a_shape[0]))
            result = np.linalg.inv(a_reshaped)
            return result

def enhanced_kalman_filter(data, Q, R, alpha=1.0, beta=0.0, model_type='constant'):
    n = len(data)
    if n == 0:
        return np.array([])

    Q = max(1e-10, min(1.0, Q))
    R = max(1e-10, min(1.0, R))
    alpha = max(0.1, min(1.0, alpha))
    beta = max(0.0, min(0.5, beta))

    filtered = np.zeros(n)

    if np.any(~np.isfinite(data)):
        print("Warning: Input data contains NaN or infinity, replacing with 0")
        data_clean = np.copy(data)
        data_clean[~np.isfinite(data_clean)] = 0
    else:
        data_clean = data

    if model_type == 'constant':
        state_dim = 1
        F = np.array([[1.0]])
    elif model_type == 'linear':
        state_dim = 2
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
    else:
        state_dim = 1
        F = np.array([[1.0]])

    if state_dim == 1:
        x = np.array([data_clean[0]])
        P = np.array([[1.0]])
    else:
        x = np.array([data_clean[0], 0.0])
        P = np.array([[1.0, 0.0], [0.0, 1.0]])

    if state_dim == 1:
        H = np.array([[1.0]])
    else:
        H = np.array([[1.0, 0.0]])

    if state_dim == 1:
        Q_mat = np.array([[Q]])
    else:
        Q_mat = np.array([[Q, 0.0], [0.0, Q/10.0]])

    R_mat = np.array([[R]])

    filtered[0] = x[0]

    try:
        for i in range(1, n):
            if model_type == 'adaptive':
                window = min(5, i)
                if window > 0:
                    recent_trend = (data_clean[i] - data_clean[i-window]) / window
                    recent_trend = max(-0.5, min(0.5, recent_trend))
                    F[0, 0] = 1.0 + beta * np.sign(recent_trend) * min(0.1, abs(recent_trend))

            try:
                x_pred = matrix_multiply(F, x)
                P_pred = matrix_multiply(F, matrix_multiply(P, F.T)) + Q_mat

                if state_dim == 1:
                    P_pred = np.array([[max(1e-10, P_pred[0, 0])]])
                else:
                    P_pred[0, 0] = max(1e-10, P_pred[0, 0])
                    P_pred[1, 1] = max(1e-10, P_pred[1, 1])
                    det = P_pred[0, 0] * P_pred[1, 1] - P_pred[0, 1] * P_pred[1, 0]
                    if det <= 0:
                        P_pred = np.array([[P_pred[0, 0], 0.0], [0.0, P_pred[1, 1]]])

                S = matrix_multiply(H, matrix_multiply(P_pred, H.T)) + R_mat

                if state_dim == 1:
                    S = np.array([[max(1e-10, S[0, 0])]])
                    S_inv = np.array([[1.0 / S[0, 0]]])
                else:
                    S[0, 0] = max(1e-10, S[0, 0])
                    S_inv = matrix_inverse(S)

                K = matrix_multiply(P_pred, matrix_multiply(H.T, S_inv))

                z = np.array([data_clean[i]])
                y = z - matrix_multiply(H, x_pred)
                x_update = x_pred + matrix_multiply(K, y)

                if np.all(np.isfinite(x_update)):
                    x = x_update
                else:
                    x = x_pred

                I_KH = np.eye(state_dim) - matrix_multiply(K, H)
                P_update = matrix_multiply(I_KH, P_pred)

                if state_dim > 1:
                    P_update = (P_update + P_update.T) / 2

                if np.all(np.isfinite(P_update)):
                    P = P_update
                else:
                    P = np.eye(state_dim)

                filtered[i] = x[0]
            except Exception as e:
                print(f"Filtering calculation error (i={i}): {str(e)}")
                filtered[i] = filtered[i-1]
    except Exception as e:
        print(f"Filtering process error: {str(e)}")
        return np.array(data_clean)

    if alpha < 1.0:
        try:
            smoothed = np.copy(filtered)
            for i in range(1, n):
                smoothed[i] = alpha * filtered[i] + (1 - alpha) * smoothed[i-1]

            backward_smoothed = np.copy(smoothed)
            for i in range(n-2, -1, -1):
                backward_smoothed[i] = alpha * smoothed[i] + (1 - alpha) * backward_smoothed[i+1]

            final_smoothed = (smoothed + backward_smoothed) / 2

            if np.all(np.isfinite(final_smoothed)):
                return final_smoothed
            else:
                print("Smoothing result contains NaN or infinity, returning unsmoothed result")
                return filtered
        except Exception as e:
            print(f"Smoothing process error: {str(e)}")
            return filtered

    return filtered

class KalmanFilter:
    def __init__(self, transition_matrices=None, observation_matrices=None, initial_state_mean=None,
                 initial_state_covariance=None, transition_covariance=None, observation_covariance=None,
                 alpha=0.8, beta=0.1, model_type='adaptive'):
        self.Q = float(transition_covariance) if transition_covariance is not None else 1e-4
        self.R = float(observation_covariance) if observation_covariance is not None else 1e-2
        self.alpha = alpha
        self.beta = beta
        self.model_type = model_type
        self.residue = None

    def filter(self, measurements):
        filtered = enhanced_kalman_filter(
            measurements,
            self.Q,
            self.R,
            alpha=self.alpha,
            beta=self.beta,
            model_type=self.model_type
        )

        self.residue = measurements - filtered

        filtered_state = filtered.reshape(-1, 1)
        return filtered_state, None

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = True
plt.style.use('seaborn-v0_8-whitegrid')

try:
    import matplotlib.font_manager as fm
    system_fonts = [f.name for f in fm.fontManager.ttflist]

    linux_fonts = ['DejaVu Sans', 'Liberation Sans', 'Ubuntu', 'FreeSans', 'Arial']

    for font in linux_fonts:
        if font in system_fonts:
            plt.rcParams['font.family'] = font
            print(f"Using system font: {font}")
            break
    else:
        print("No suitable system font found, using default font")
except:
    print("Font setting failed, using default font")

def load_data(file_path='********'):
    try:
        print(f"Reading data: {file_path}")
        df = pd.read_excel(file_path)

        print(f"File contains columns: {df.columns.tolist()}")

        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        else:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]) or 'date' in col.lower() or 'time' in col.lower():
                    print(f"Using column '{col}' as date column")
                    df['date'] = pd.to_datetime(df[col])
                    df.set_index('date', inplace=True)
                    df = df.drop(col, axis=1, errors='ignore')
                    break
            else:
                print("Date column not found, creating default date index")
                df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='D')
                df.set_index('date', inplace=True)

        if 'displacement_1' in df.columns:
            displacement_col = 'displacement_1'
        elif 'G4' in df.columns:
            displacement_col = 'G4'
            print(f"Using 'G4' column as displacement data")
        else:
            displacement_col = df.columns[0]
            print(f"Using first column '{displacement_col}' as displacement data")

        new_df = pd.DataFrame({'displacement_1': df[displacement_col]}, index=df.index)

        full_date_range = pd.date_range(start=new_df.index.min(), end=new_df.index.max(), freq='D')
        new_df = new_df.reindex(full_date_range)

        new_df['displacement_1'] = new_df['displacement_1'].interpolate(method='linear')

        data = new_df['displacement_1']

        print(f"Data loading completed, total {len(data)} data points")
        return data

    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        print("Creating sample data for testing...")

        dates = pd.date_range(start='2020-01-01', periods=365, freq='D')

        trend = np.linspace(0, 10, len(dates))
        seasonal = 2 * np.sin(np.linspace(0, 12*np.pi, len(dates)))
        noise = np.random.normal(0, 0.5, len(dates))

        values = trend + seasonal + noise

        df = pd.DataFrame({
            'date': dates,
            'displacement_1': values
        })

        df.set_index('date', inplace=True)

        data = df['displacement_1']

        print(f"Sample data created, total {len(data)} data points")
        return data

def calculate_snr(signal, noise):
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

class AdvancedGeneticKalmanFilter:
    def __init__(self, data, pop_size=120, generations=200, tournament_size=7):
        self.data = data
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size

        self.param_ranges = {
            'Q': (1e-12, 0.5),
            'R': (1e-12, 0.5),
            'alpha': (0.3, 0.9999),
            'beta': (0.0, 0.3),
            'model_type_idx': (0, 2)
        }

        self.special_regions = [
            {'Q': (1e-15, 1e-10), 'R': (1e-15, 1e-10), 'alpha': (0.95, 0.9999), 'beta': (0.0, 0.05)},
            {'Q': (1e-10, 1e-7), 'R': (1e-10, 1e-7), 'alpha': (0.9, 0.99), 'beta': (0.0, 0.1)},
            {'Q': (1e-7, 1e-4), 'R': (1e-7, 1e-4), 'alpha': (0.8, 0.95), 'beta': (0.05, 0.2)},
            {'Q': (1e-4, 1e-1), 'R': (1e-4, 1e-1), 'alpha': (0.5, 0.8), 'beta': (0.1, 0.3)},
            {'Q': (0.001, 0.01), 'R': (1e-6, 1e-5), 'alpha': (0.9, 0.95), 'beta': (0.2, 0.3)}
        ]

        self.model_types = ['constant', 'linear', 'adaptive']

        self.best_fitness_history = []
        self.best_params_history = []
        self.best_snr_history = []
        self.avg_fitness_history = []

        self.fitness_cache = {}

        self.best_filtered = None
        self.best_snr = 0

    def kalman_filter(self, params):
        Q, R, alpha, beta, model_type_idx = params

        model_type = self.model_types[int(np.clip(model_type_idx, 0, 2))]

        kf = KalmanFilter(
            transition_covariance=Q,
            observation_covariance=R,
            alpha=alpha,
            beta=beta,
            model_type=model_type
        )

        filtered_state, _ = kf.filter(self.data.values)
        filtered_series = pd.Series(filtered_state.flatten(), index=self.data.index)

        self.residue = kf.residue

        return filtered_series

    def fitness(self, params):
        for i, param_name in enumerate(['Q', 'R', 'alpha', 'beta', 'model_type_idx']):
            if not np.isfinite(params[i]):
                print(f"Invalid parameter {param_name}: {params[i]}")
                return 0

            low, high = self.param_ranges[param_name]
            if params[i] < low or params[i] > high:
                print(f"Parameter {param_name} out of range: {params[i]}, should be in [{low}, {high}]")
                return 0

        params_tuple = tuple(params)
        if params_tuple in self.fitness_cache:
            return self.fitness_cache[params_tuple]

        try:
            filtered = self.kalman_filter(params)

            if np.any(~np.isfinite(filtered.values)):
                print(f"Filtering result contains NaN or infinity, Params: Q={params[0]:.2e}, R={params[1]:.2e}, alpha={params[2]:.2f}")
                return 0

            noise = self.data.values - filtered.values

            snr = calculate_snr(filtered.values, noise)

            if not np.isfinite(snr):
                print(f"SNR calculation invalid: {snr}")
                return 0

            mse = mean_squared_error(self.data, filtered)

            if not np.isfinite(mse):
                print(f"MSE calculation invalid: {mse}")
                return 0

            diff_var = np.var(np.diff(filtered.values))
            if not np.isfinite(diff_var) or diff_var <= 0:
                smoothness = 0
            else:
                smoothness = 1 / (1 + diff_var)

            try:
                correlation = np.corrcoef(self.data.values, filtered.values)[0, 1]
                if not np.isfinite(correlation):
                    correlation = 0
            except:
                correlation = 0

            try:
                from scipy import signal as scipy_signal

                peaks_orig = scipy_signal.find_peaks(self.data.values, prominence=np.std(self.data.values)*0.5)[0]
                valleys_orig = scipy_signal.find_peaks(-self.data.values, prominence=np.std(self.data.values)*0.5)[0]
                extrema_orig = np.sort(np.concatenate([peaks_orig, valleys_orig]))

                if len(extrema_orig) > 0:
                    diffs = np.abs(self.data.values[extrema_orig] - filtered.values[extrema_orig])
                    denoms = np.abs(self.data.values[extrema_orig]) + 1e-10
                    ratios = diffs / denoms
                    valid_ratios = ratios[np.isfinite(ratios)]
                    if len(valid_ratios) > 0:
                        peak_preservation = 1 - np.mean(valid_ratios)
                    else:
                        peak_preservation = 0.5
                else:
                    peak_preservation = 1.0
            except Exception as e:
                print(f"Peak preservation calculation error: {str(e)}")
                peak_preservation = 0.5

            try:
                from scipy import signal as scipy_signal

                f_orig, Pxx_orig = scipy_signal.welch(self.data.values, fs=1.0, nperseg=min(256, len(self.data)//2))
                f_filt, Pxx_filt = scipy_signal.welch(filtered.values, fs=1.0, nperseg=min(256, len(self.data)//2))

                if np.any(~np.isfinite(Pxx_orig)) or np.any(~np.isfinite(Pxx_filt)):
                    low_freq_preservation = 0.5
                    high_freq_suppression = 0.5
                else:
                    low_freq_idx = f_orig <= 0.1
                    if np.sum(Pxx_orig[low_freq_idx]) > 0:
                        low_freq_preservation = np.sum(Pxx_filt[low_freq_idx]) / np.sum(Pxx_orig[low_freq_idx])
                        low_freq_preservation = min(1.0, max(0.0, low_freq_preservation))
                    else:
                        low_freq_preservation = 1.0

                    high_freq_idx = f_orig > 0.3
                    if np.sum(Pxx_orig[high_freq_idx]) > 0:
                        high_freq_suppression = 1 - np.sum(Pxx_filt[high_freq_idx]) / np.sum(Pxx_orig[high_freq_idx])
                        high_freq_suppression = min(1.0, max(0.0, high_freq_suppression))
                    else:
                        high_freq_suppression = 1.0
            except Exception as e:
                print(f"Spectrum characteristic calculation error: {str(e)}")
                low_freq_preservation = 0.5
                high_freq_suppression = 0.5

            try:
                from scipy import signal
                f_orig, Pxx_orig = signal.welch(self.data.values, fs=1.0, nperseg=min(256, len(self.data)//2))

                high_freq_idx = f_orig > 0.3
                noise_power_orig = np.mean(Pxx_orig[high_freq_idx]) if np.any(high_freq_idx) else 0.1
                signal_power_orig = np.mean(Pxx_orig) - noise_power_orig

                original_snr = 10 * np.log10(signal_power_orig / noise_power_orig) if noise_power_orig > 0 else 10
            except:
                original_snr = 10

            snr_improvement = snr - original_snr

            score = ((0.85 * min(snr_improvement, 40)) + (0.05 * correlation) + (0.03 * peak_preservation)
                    + (0.03 * low_freq_preservation) + (0.02 * high_freq_suppression) + (0.02 * smoothness))

            if snr_improvement < 15:
                penalty = 0.9 * (1 - snr_improvement / 15)
                score = score * (1 - penalty)

            if snr_improvement > 20:
                bonus = 0.5 * (snr_improvement - 20) / 10
                score = score * (1 + min(bonus, 1.0))

            if snr_improvement > 35:
                score = score * 1.5

            if not np.isfinite(score):
                print(f"Score calculation invalid: {score}")
                return 0

            if np.isfinite(snr) and snr > self.best_snr:
                self.best_snr = snr
                self.best_filtered = filtered

            self.fitness_cache[params_tuple] = max(score, 0)

            return max(score, 0)
        except Exception as e:
            print(f"Fitness calculation error: {str(e)}")
            return 0

    def tournament_selection(self, population, scores):
        selected = []
        for _ in range(self.pop_size // 2):
            candidates_idx = np.random.choice(len(population), self.tournament_size, replace=False)
            best_idx = candidates_idx[np.argmax(scores[candidates_idx])]
            selected.append(population[best_idx])
        return np.array(selected)

    def evolve(self):
        print("Initializing population using super-optimized hybrid sampling strategy...")

        population = np.zeros((self.pop_size, 5))

        try:
            from scipy.stats import qmc

            general_ratio = 0.4
            special_ratio = 0.5
            random_ratio = 0.1

            general_count = int(self.pop_size * general_ratio)
            special_count = int(self.pop_size * special_ratio)
            random_count = self.pop_size - general_count - special_count

            if general_count > 0:
                sampler = qmc.LatinHypercube(d=5)
                samples = sampler.random(n=general_count)

                for i, param in enumerate(['Q', 'R', 'alpha', 'beta', 'model_type_idx']):
                    low, high = self.param_ranges[param]
                    if param in ['Q', 'R']:
                        log_low, log_high = np.log10(low), np.log10(high)
                        population[:general_count, i] = 10 ** (log_low + samples[:, i] * (log_high - log_low))
                    elif param == 'model_type_idx':
                        population[:general_count, i] = np.floor(low + samples[:, i] * (high - low + 1))
                        population[:general_count, i] = np.clip(population[:general_count, i], low, high)
                    else:
                        population[:general_count, i] = low + samples[:, i] * (high - low)

            if special_count > 0:
                region_counts = []
                remaining = special_count
                for i in range(len(self.special_regions) - 1):
                    count = remaining // (len(self.special_regions) - i)
                    region_counts.append(count)
                    remaining -= count
                region_counts.append(remaining)

                start_idx = general_count
                for r, region in enumerate(self.special_regions):
                    count = region_counts[r]
                    if count <= 0:
                        continue

                    region_sampler = qmc.LatinHypercube(d=5)
                    region_samples = region_sampler.random(n=count)

                    for i, param in enumerate(['Q', 'R', 'alpha', 'beta']):
                        if param in region:
                            low, high = region[param]
                        else:
                            low, high = self.param_ranges[param]

                        if param in ['Q', 'R']:
                            log_low, log_high = np.log10(low), np.log10(high)
                            population[start_idx:start_idx+count, i] = 10 ** (log_low + region_samples[:, i] * (log_high - log_low))
                        else:
                            population[start_idx:start_idx+count, i] = low + region_samples[:, i] * (high - low)

                    if r == 0:
                        population[start_idx:start_idx+count, 4] = 0
                    elif r == 1:
                        population[start_idx:start_idx+count, 4] = 1
                    else:
                        population[start_idx:start_idx+count, 4] = 2

                    start_idx += count

            if random_count > 0:
                for i, param in enumerate(['Q', 'R', 'alpha', 'beta', 'model_type_idx']):
                    low, high = self.param_ranges[param]
                    if param in ['Q', 'R']:
                        log_low, log_high = np.log10(low), np.log10(high)
                        population[-random_count:, i] = 10 ** (np.random.uniform(log_low, log_high, random_count))
                    elif param == 'model_type_idx':
                        population[-random_count:, i] = np.random.randint(low, high + 1, random_count)
                    else:
                        population[-random_count:, i] = np.random.uniform(low, high, random_count)

            print(f"Population initialized successfully: {general_count} general space, {special_count} special regions, {random_count} random individuals")

        except Exception as e:
            print(f"Advanced hybrid sampling initialization failed: {str(e)}")
            print("Using backup random initialization...")

            for i, param in enumerate(['Q', 'R', 'alpha', 'beta', 'model_type_idx']):
                low, high = self.param_ranges[param]
                if param in ['Q', 'R']:
                    log_low, log_high = np.log10(low), np.log10(high)
                    population[:, i] = 10 ** (np.random.uniform(log_low, log_high, self.pop_size))
                elif param == 'model_type_idx':
                    population[:, i] = np.random.randint(low, high + 1, self.pop_size)
                else:
                    population[:, i] = np.random.uniform(low, high, self.pop_size)

        print("Starting advanced genetic algorithm optimization...")
        start_time = time.time()

        for gen in range(self.generations):
            gen_start_time = time.time()

            scores = np.array([self.fitness(ind) for ind in population])

            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            best_params = population[best_idx].copy()

            self.best_fitness_history.append(best_score)
            self.best_params_history.append(best_params)
            self.avg_fitness_history.append(np.mean(scores))

            best_filtered = self.kalman_filter(best_params)
            noise = self.data.values - best_filtered.values
            best_snr = calculate_snr(best_filtered.values, noise)
            self.best_snr_history.append(best_snr)

            if gen % 10 == 0 or gen == self.generations - 1:
                elapsed = time.time() - start_time
                gen_time = time.time() - gen_start_time
                remaining = (self.generations - gen - 1) * gen_time

                print(f"Gen: {gen+1}/{self.generations} | "
                      f"Best Fitness: {best_score:.4f} | "
                      f"Best SNR: {best_snr:.2f} dB | "
                      f"Time: {elapsed:.1f}s | "
                      f"Remaining: {remaining:.1f}s")

            if gen == self.generations - 1:
                break

            parents = self.tournament_selection(population, scores)

            children = []
            while len(children) < self.pop_size - len(parents):
                p1, p2 = parents[np.random.choice(len(parents), 2, replace=False)]

                for i, param_name in enumerate(['Q', 'R', 'alpha', 'beta', 'model_type_idx']):
                    low, high = self.param_ranges[param_name]
                    p1[i] = np.clip(p1[i], low, high)
                    p2[i] = np.clip(p2[i], low, high)

                eta = 15
                u = np.random.random(5)

                beta = np.zeros(5)
                for i in range(5):
                    if u[i] <= 0.5:
                        beta[i] = (2 * u[i]) ** (1 / (eta + 1))
                    else:
                        beta[i] = (1 / (2 * (1 - u[i]))) ** (1 / (eta + 1))

                c1 = np.zeros(5)
                c2 = np.zeros(5)

                for i, param_name in enumerate(['Q', 'R', 'alpha', 'beta', 'model_type_idx']):
                    low, high = self.param_ranges[param_name]

                    c1[i] = 0.5 * ((1 + beta[i]) * p1[i] + (1 - beta[i]) * p2[i])
                    c2[i] = 0.5 * ((1 - beta[i]) * p1[i] + (1 + beta[i]) * p2[i])

                    c1[i] = np.clip(c1[i], low, high)
                    c2[i] = np.clip(c2[i], low, high)

                for child in [c1, c2]:
                    if np.random.rand() < 0.3:
                        for i in range(5):
                            if np.random.rand() < 0.5:
                                param_name = ['Q', 'R', 'alpha', 'beta', 'model_type_idx'][i]
                                low, high = self.param_ranges[param_name]

                                decay_rate = 1 - gen / self.generations

                                if param_name in ['Q', 'R']:
                                    log_val = np.log10(max(child[i], low))
                                    log_low, log_high = np.log10(low), np.log10(high)
                                    sigma = decay_rate * 0.1 * (log_high - log_low)
                                    log_val += np.random.normal(0, sigma)
                                    child[i] = 10 ** np.clip(log_val, log_low, log_high)
                                else:
                                    sigma = decay_rate * 0.1 * (high - low)
                                    child[i] += np.random.normal(0, sigma)
                                    child[i] = np.clip(child[i], low, high)

                for child in [c1, c2]:
                    for i, param_name in enumerate(['Q', 'R', 'alpha', 'beta', 'model_type_idx']):
                        low, high = self.param_ranges[param_name]
                        child[i] = np.clip(child[i], low, high)

                children.extend([c1, c2])

            children = children[:self.pop_size - len(parents)]

            elite_size = max(1, int(self.pop_size * 0.1))
            elite_indices = np.argsort(scores)[-elite_size:]
            elites = population[elite_indices]

            population = np.vstack([elites, parents[:self.pop_size-elite_size-len(children)], children])

        top_n = 5

        snr_indices = np.argsort(self.best_snr_history)[::-1]
        top_snr_indices = snr_indices[:top_n]

        print("\nTop 5 Best SNR Results:")
        for i, idx in enumerate(top_snr_indices):
            params = self.best_params_history[idx]
            snr = self.best_snr_history[idx]
            fitness = self.best_fitness_history[idx]

            model_type_idx = int(np.clip(params[4], 0, 2))
            model_type = self.model_types[model_type_idx]

            print(f"\nResult #{i+1} - SNR: {snr:.2f} dB, Fitness: {fitness:.6f}")
            print(f"Process Noise Covariance (Q): {params[0]:.8f}")
            print(f"Observation Noise Covariance (R): {params[1]:.8f}")
            print(f"Smoothing Factor (alpha): {params[2]:.4f}")
            print(f"Trend Weight (beta): {params[3]:.4f}")
            print(f"Model Type: {model_type}")

        best_idx = top_snr_indices[0]
        best_params = self.best_params_history[best_idx]

        model_type_idx = int(np.clip(best_params[4], 0, 2))
        model_type = self.model_types[model_type_idx]

        print(f"\nSelecting result with highest SNR (SNR: {self.best_snr_history[best_idx]:.2f} dB)")

        return [best_params[0], best_params[1], best_params[2], best_params[3], model_type]

def analyze_spectrum(data, title="Spectrum Analysis", fs=1.0):
    from scipy import signal as scipy_signal

    f, Pxx = scipy_signal.welch(data, fs=fs, nperseg=min(256, len(data)//2))

    plt.figure(figsize=(10, 6))
    plt.semilogy(f, Pxx)
    plt.title(title)
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    plt.grid(True)
    plt.tight_layout()
    return f, Pxx

def calculate_energy_distribution(psd, freq):
    total_energy = np.sum(psd)

    low_freq_idx = freq <= 0.1
    mid_freq_idx = (freq > 0.1) & (freq <= 0.3)
    high_freq_idx = freq > 0.3

    low_energy_pct = np.sum(psd[low_freq_idx]) / total_energy * 100
    mid_energy_pct = np.sum(psd[mid_freq_idx]) / total_energy * 100
    high_energy_pct = np.sum(psd[high_freq_idx]) / total_energy * 100

    return {
        "Low Freq Ratio (%)": low_energy_pct,
        "Mid Freq Ratio (%)": mid_energy_pct,
        "High Freq Ratio (%)": high_energy_pct
    }

if __name__ == "__main__":
    output_dir = '*****'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    try:
        data_file = 'data_6.xlsx'
        if os.path.exists(data_file):
            print(f"Loading raw data: {data_file}")
            original_data = load_data(data_file)
        else:
            print(f"Data file {data_file} not found, using default data...")
            original_data = load_data()

        print(f"Data loaded successfully, total {len(original_data)} data points")
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        exit()

    print("Starting optimization of Kalman filter parameters...")
    optimizer = AdvancedGeneticKalmanFilter(
        original_data,
        pop_size=300,
        generations=50,
        tournament_size=10
    )

    best_params = optimizer.evolve()
    best_Q, best_R, best_alpha, best_beta, best_model_type = best_params

    print("\nOptimization completed! Best parameters:")
    print(f"Process Noise Covariance (Q): {best_Q:.8f}")
    print(f"Observation Noise Covariance (R): {best_R:.8f}")
    print(f"Smoothing Factor (alpha): {best_alpha:.4f}")
    print(f"Trend Weight (beta): {best_beta:.4f}")
    print(f"Model Type: {best_model_type}")

    kf = KalmanFilter(
        transition_covariance=best_Q,
        observation_covariance=best_R,
        alpha=best_alpha,
        beta=best_beta,
        model_type=best_model_type
    )
    filtered_state, _ = kf.filter(original_data.values)
    filtered_data = pd.Series(filtered_state.flatten(), index=original_data.index)

    noise = original_data.values - filtered_data.values

    from scipy import signal as scipy_signal
    f_orig, Pxx_orig = scipy_signal.welch(original_data.values, fs=1.0, nperseg=min(256, len(original_data)//2))

    high_freq_idx = f_orig > 0.3
    noise_power = np.mean(Pxx_orig[high_freq_idx]) if np.any(high_freq_idx) else 0.1
    signal_power = np.mean(Pxx_orig) - noise_power

    snr_before = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 10

    snr_after = calculate_snr(filtered_data.values, noise)

    snr_improvement = snr_after - snr_before

    print(f"\nSNR Analysis:")
    print(f"Estimated SNR of Raw Signal: {snr_before:.2f} dB")
    print(f"SNR after Filtering: {snr_after:.2f} dB")
    print(f"SNR Improvement: {snr_improvement:.2f} dB")

    print("\nPerforming spectrum analysis...")
    f_orig, Pxx_orig = analyze_spectrum(original_data.values, title="Raw Signal Spectrum")
    f_filt, Pxx_filt = analyze_spectrum(filtered_data.values, title="Filtered Signal Spectrum")

    orig_energy = calculate_energy_distribution(Pxx_orig, f_orig)
    filt_energy = calculate_energy_distribution(Pxx_filt, f_filt)

    print("\nEnergy Distribution Analysis:")
    print("Raw Signal:")
    for k, v in orig_energy.items():
        print(f"  {k}: {v:.2f}%")

    print("Filtered Signal:")
    for k, v in filt_energy.items():
        print(f"  {k}: {v:.2f}%")

    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(original_data.index, original_data,
             label='Raw Data', linewidth=1, alpha=0.7, color='#1f77b4')
    plt.plot(filtered_data.index, filtered_data,
             label='Denoised Data', linewidth=1.5, color='#ff7f0e')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Displacement', fontsize=12)
    plt.title(f'Comparison of GA-Optimized Kalman Filter Denoising', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(loc='best')

    plt.subplot(2, 2, 2)
    bars = plt.bar(['Before Denoising', 'After Denoising'], [snr_before, snr_after],
                  color=['#1f77b4', '#ff7f0e'], alpha=0.8)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.2f} dB', ha='center', va='bottom', fontsize=12)
    plt.annotate(f'Improvement: {snr_improvement:.2f} dB',
                xy=(1, snr_before + (snr_improvement)/2),
                xytext=(1.5, snr_before + (snr_improvement)/2),
                arrowprops=dict(facecolor='green', shrink=0.05, width=2),
                fontsize=12, color='green', weight='bold')
    plt.ylabel('SNR (dB)', fontsize=12)
    plt.title('Kalman Filter SNR Improvement', fontsize=14, color='darkgreen')
    plt.grid(axis='y', alpha=0.3)

    plt.ylim(0, max(snr_before, snr_after) * 1.3)

    plt.subplot(2, 2, 3)
    plt.plot(original_data.index, noise, label='Removed Noise', color='#d62728', alpha=0.7)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Noise Amplitude', fontsize=12)
    plt.title('Analysis of Removed Noise', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(loc='best')

    noise_std = np.std(noise)
    noise_mean = np.mean(noise)
    textstr = f'Noise Std: {noise_std:.4f}\nNoise Mean: {noise_mean:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    plt.subplot(2, 2, 4)
    mid_point = len(original_data) // 2
    window = len(original_data) // 8
    start_idx = mid_point - window
    end_idx = mid_point + window

    plt.plot(original_data.index[start_idx:end_idx],
             original_data.values[start_idx:end_idx],
             label='Raw Data', linewidth=1.5, alpha=0.7, color='#1f77b4')
    plt.plot(filtered_data.index[start_idx:end_idx],
             filtered_data.values[start_idx:end_idx],
             label='Denoised Data', linewidth=2, color='#ff7f0e')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Displacement', fontsize=12)
    plt.title('Local Zoom - Detail Comparison', fontsize=14)
    plt.grid(alpha=0.3)
    plt.legend(loc='best')

    textstr = (f'Optimized Params:\nQ={best_Q:.2e}, R={best_R:.2e}\n'
              f'α={best_alpha:.2f}, β={best_beta:.2f}\n'
              f'Model: {best_model_type}')
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.5)
    plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Kalman_Filtering_Results_{timestamp}.png'),
                dpi=300, bbox_inches='tight')

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    plt.semilogy(f_orig, Pxx_orig, color='#1f77b4', label='Raw Signal')
    plt.semilogy(f_filt, Pxx_filt, color='#ff7f0e', label='Filtered Signal')
    plt.title('Spectrum Comparison Analysis', fontsize=14)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('PSD (Log Scale)', fontsize=12)
    plt.grid(True, alpha=0.5)
    plt.legend()

    plt.subplot(2, 2, 2)
    high_freq_idx = f_orig > 0.3
    if np.any(high_freq_idx):
        orig_high_energy = np.sum(Pxx_orig[high_freq_idx])
        filt_high_energy = np.sum(Pxx_filt[high_freq_idx])
        high_freq_reduction = (orig_high_energy - filt_high_energy) / orig_high_energy * 100

        bars = plt.bar(['Raw Signal', 'Filtered Signal'],
                      [orig_high_energy, filt_high_energy],
                      color=['#1f77b4', '#ff7f0e'], alpha=0.8)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height * 1.05,
                    f'{height:.2e}', ha='center', va='bottom', fontsize=10)

        plt.annotate(f'Reduced: {high_freq_reduction:.1f}%',
                    xy=(1, filt_high_energy + (orig_high_energy - filt_high_energy)/2),
                    xytext=(1.5, filt_high_energy + (orig_high_energy - filt_high_energy)/2),
                    arrowprops=dict(facecolor='green', shrink=0.05, width=1.5),
                    fontsize=12, color='green', weight='bold')

        plt.title('High Freq Noise Suppression', fontsize=14)
        plt.ylabel('High Freq Energy', fontsize=12)
        plt.grid(axis='y', alpha=0.5)
    else:
        plt.text(0.5, 0.5, 'No High Freq Components', ha='center', va='center', fontsize=14)
        plt.title('High Freq Noise Suppression', fontsize=14)

    plt.subplot(2, 2, 3)
    plt.semilogy(f_orig, Pxx_orig, color='#1f77b4')
    low_freq_idx = f_orig <= 0.1
    mid_freq_idx = (f_orig > 0.1) & (f_orig <= 0.3)
    high_freq_idx = f_orig > 0.3

    if np.any(low_freq_idx):
        plt.fill_between(f_orig[low_freq_idx], Pxx_orig[low_freq_idx],
                        alpha=0.3, color='green', label='Low Freq Region')
    if np.any(mid_freq_idx):
        plt.fill_between(f_orig[mid_freq_idx], Pxx_orig[mid_freq_idx],
                        alpha=0.3, color='yellow', label='Mid Freq Region')
    if np.any(high_freq_idx):
        plt.fill_between(f_orig[high_freq_idx], Pxx_orig[high_freq_idx],
                        alpha=0.3, color='red', label='High Freq Region')

    plt.title('Raw Signal Spectrum', fontsize=14)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Power Spectral Density', fontsize=12)
    plt.grid(True, alpha=0.5)
    plt.legend(loc='best')

    plt.subplot(2, 2, 4)
    plt.semilogy(f_filt, Pxx_filt, color='#ff7f0e')
    if np.any(low_freq_idx):
        plt.fill_between(f_filt[low_freq_idx], Pxx_filt[low_freq_idx],
                        alpha=0.3, color='green', label='Low Freq Region')
    if np.any(mid_freq_idx):
        plt.fill_between(f_filt[mid_freq_idx], Pxx_filt[mid_freq_idx],
                        alpha=0.3, color='yellow', label='Mid Freq Region')
    if np.any(high_freq_idx):
        plt.fill_between(f_filt[high_freq_idx], Pxx_filt[high_freq_idx],
                        alpha=0.3, color='red', label='High Freq Region')

    plt.title('Filtered Signal Spectrum', fontsize=14)
    plt.xlabel('Frequency', fontsize=12)
    plt.ylabel('Power Spectral Density', fontsize=12)
    plt.grid(True, alpha=0.5)
    plt.legend(loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Spectrum_Analysis_{timestamp}.png'),
                dpi=300, bbox_inches='tight')

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    generations = range(1, len(optimizer.best_fitness_history) + 1)
    plt.plot(generations, optimizer.best_fitness_history, 'b-', label='Best Fitness', linewidth=2)
    plt.plot(generations, optimizer.avg_fitness_history, 'g--', label='Avg Fitness', alpha=0.7)
    plt.title('GA Optimization Process - Fitness History', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 2, 2)
    plt.plot(generations, optimizer.best_snr_history, 'r-', label='Best SNR', linewidth=2)
    plt.axhline(y=snr_before, color='k', linestyle='--', label=f'Raw SNR: {snr_before:.2f} dB', alpha=0.7)
    plt.title('GA Optimization Process - SNR History', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('SNR (dB)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 2, 3)
    q_history = [params[0] for params in optimizer.best_params_history]
    r_history = [params[1] for params in optimizer.best_params_history]

    plt.semilogy(generations, q_history, 'b-', label='Process Noise (Q)', linewidth=1.5)
    plt.semilogy(generations, r_history, 'r-', label='Observation Noise (R)', linewidth=1.5)
    plt.title('Parameter Convergence History - Q and R', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Parameter Value (Log Scale)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.subplot(2, 2, 4)
    alpha_history = [params[2] for params in optimizer.best_params_history]
    beta_history = [params[3] for params in optimizer.best_params_history]

    plt.plot(generations, alpha_history, 'g-', label='Smoothing Factor (alpha)', linewidth=1.5)
    plt.plot(generations, beta_history, 'm-', label='Trend Weight (beta)', linewidth=1.5)
    plt.title('Parameter Convergence History - alpha and beta', fontsize=14)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Parameter Value', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Optimization_Process_{timestamp}.png'),
                dpi=300, bbox_inches='tight')

    result_df = pd.DataFrame({
        'Date': original_data.index,
        'Raw Data': original_data.values,
        'Denoised Data': filtered_data.values,
        'Noise': noise
    })

    info_df = pd.DataFrame({
        'Parameter': ['Process Noise Covariance (Q)', 'Observation Noise Covariance (R)', 'Smoothing Factor (alpha)', 'Trend Weight (beta)',
                'Model Type', 'Raw Signal SNR (dB)', 'Filtered SNR (dB)', 'SNR Improvement (dB)'],
        'Value': [best_Q, best_R, best_alpha, best_beta, best_model_type,
              snr_before, snr_after, snr_improvement]
    })

    energy_df = pd.DataFrame({
        'Freq Band': list(orig_energy.keys()),
        'Raw Signal (%)': list(orig_energy.values()),
        'Filtered Signal (%)': list(filt_energy.values())
    })

    excel_path = os.path.join(output_dir, f'Denoising_Results_Data_{timestamp}.xlsx')
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        result_df.to_excel(writer, sheet_name='Denoising Results', index=False)
        info_df.to_excel(writer, sheet_name='Parameter Info', index=False)
        energy_df.to_excel(writer, sheet_name='Spectrum Analysis', index=False)

    mse = mean_squared_error(original_data, filtered_data)
    mae = mean_absolute_error(original_data, filtered_data)
    r2 = r2_score(original_data, filtered_data)

    print("\nError Analysis Results:")
    print(f"MSE: {mse:.10f}")
    print(f"MAE: {mae:.10f}")
    print(f"R2: {r2:.10f}")
    print(f"SNR: {snr_after:.2f} dB")

    print(f"\nResults saved to: {output_dir}")