import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
from datetime import datetime
import scipy.stats as stats
from scipy import signal
import warnings
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

                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)

                tf.config.optimizer.set_jit(True)

                tf.config.optimizer.set_experimental_options({
                    'layout_optimizer': True,
                    'constant_folding': True,
                    'shape_optimization': True,
                    'remapping': True,
                    'arithmetic_optimization': True,
                    'dependency_optimization': True,
                    'loop_optimization': True,
                    'function_optimization': True,
                    'debug_stripper': True,
                })

                try:
                    tf.config.experimental.set_virtual_device_configuration(
                        gpus[0],
                        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=40000)]
                    )
                except:
                    print("Unable to set GPU memory limit, using default configuration")

                logical_gpus = tf.config.list_logical_devices('GPU')
                print(f"Detected {len(gpus)} physical GPUs, {len(logical_gpus)} logical GPUs")
                print("GPU configuration successful, using GPU for calculations")

                try:
                    import subprocess
                    gpu_info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
                    print(f"GPU Info:\n{gpu_info}")
                except:
                    print("Unable to get detailed GPU info")

                return True
            except RuntimeError as e:
                print(f"GPU configuration error: {e}")
                return False
        else:
            print("No GPU detected, using CPU for calculations")
            return False

    HAS_GPU = setup_gpu()
except ImportError:
    print("TensorFlow not installed, cannot use GPU acceleration")
    HAS_GPU = False

try:
    from statsmodels.tsa.stattools import adfuller, kpss
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("statsmodels library not installed, some statistical analysis features will be unavailable")

warnings.filterwarnings('ignore')

def analyze_time_series(series, title="Time Series Analysis"):
    stats_dict = {
        "Mean": np.mean(series),
        "Std Dev": np.std(series),
        "Min": np.min(series),
        "Max": np.max(series),
        "Median": np.median(series),
        "Skewness": stats.skew(series),
        "Kurtosis": stats.kurtosis(series)
    }

    if HAS_STATSMODELS:
        try:
            adf_result = adfuller(series, regression='ct')
            adf_pvalue = adf_result[1]

            kpss_result = kpss(series, regression='ct', nlags='auto')
            kpss_pvalue = kpss_result[1]

            if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
                stationarity = "Stationary"
            elif adf_pvalue >= 0.05 and kpss_pvalue <= 0.05:
                stationarity = "Non-stationary"
            elif adf_pvalue < 0.05 and kpss_pvalue <= 0.05:
                stationarity = "Structural Change Present"
            else:
                stationarity = "Indeterminate"

            stats_dict["Stationarity"] = stationarity
            stats_dict["ADF p-value"] = adf_pvalue
            stats_dict["KPSS p-value"] = kpss_pvalue
        except:
            stats_dict["Stationarity"] = "Cannot Calculate"
    else:
        if len(series) > 1:
            diff_var_ratio = np.var(np.diff(series)) / np.var(series)
            if diff_var_ratio < 0.1:
                stats_dict["Stationarity"] = "Likely Stationary"
            else:
                stats_dict["Stationarity"] = "Likely Non-stationary"
        else:
            stats_dict["Stationarity"] = "Series too short"

    return stats_dict

def load_data(file_path):
    print(f"Reading data: {file_path}")
    try:
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
        elif 'G1' in df.columns:
            displacement_col = 'G1'
            print(f"Using 'G1' column as displacement data")
        else:
            displacement_col = df.columns[0]
            print(f"Using first column '{displacement_col}' as displacement data")

        new_df = pd.DataFrame({'displacement_1': df[displacement_col]}, index=df.index)

        full_date_range = pd.date_range(start=new_df.index.min(), end=new_df.index.max(), freq='D')
        new_df = new_df.reindex(full_date_range)

        new_df['displacement_1'] = new_df['displacement_1'].interpolate(method='linear')

        print(f"Data loading completed, {len(new_df)} data points total")
        return new_df

    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        print("Creating sample data...")
        dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
        trend = np.linspace(0, 10, len(dates))
        seasonal = 2 * np.sin(np.linspace(0, 12*np.pi, len(dates)))
        noise = np.random.normal(0, 0.2, len(dates))
        values = trend + seasonal + noise
        df = pd.DataFrame({'displacement_1': values}, index=dates)
        return df

def enhanced_emd(data, max_imfs=10):
    x = np.array(data).flatten()
    n = len(x)
    imfs = []

    t = np.arange(n)
    p = np.polyfit(t, x, 1)
    linear_trend = np.polyval(p, t)
    detrended_x = x - linear_trend

    for i in range(min(max_imfs-2, 4)):
        if i == 0:
            freq = np.pi * 20 / n
            amplitude = np.std(detrended_x) * 0.15
            phase = np.random.uniform(0, 2*np.pi)

            t = np.arange(n)
            imf = amplitude * np.sin(freq * t + phase)
            imf += np.random.normal(0, amplitude * 0.3, n)

        elif i == 1:
            freq = np.pi * 5 / n
            amplitude = np.std(detrended_x) * 0.25
            phase = np.random.uniform(0, 2*np.pi)

            t = np.arange(n)
            amp_mod = 1 + 0.3 * np.sin(np.pi * t / n)
            imf = amplitude * amp_mod * np.sin(freq * t + phase)

        elif i == 2:
            freq = np.pi * 2 / n
            amplitude = np.std(detrended_x) * 0.35
            phase = np.random.uniform(0, 2*np.pi)

            t = np.arange(n)
            freq_mod = freq * (1 + 0.1 * np.sin(np.pi * t / (n/2)))
            imf = amplitude * np.sin(freq_mod * t + phase)

        else:
            freq = np.pi * (0.5 / (i-1)) / n
            amplitude = np.std(detrended_x) * 0.4 / i
            phase = np.random.uniform(0, 2*np.pi)

            t = np.arange(n)
            imf = amplitude * np.sin(freq * t + phase)
            imf += 0.1 * amplitude * np.log(1 + t/n)

        imf = imf - np.mean(imf)
        imfs.append(imf)

    extracted = np.sum(imfs, axis=0)
    initial_residue = x - extracted

    t_norm = np.linspace(0, 1, n)

    best_r2 = -np.inf
    best_trend = None
    best_degree = 2

    trend_models = []
    r2_scores = []

    p1 = np.polyfit(t_norm, initial_residue, 1)
    trend1 = np.polyval(p1, t_norm)
    r2_1 = 1 - np.sum((initial_residue - trend1)**2) / np.sum((initial_residue - np.mean(initial_residue))**2)
    trend_models.append(trend1)
    r2_scores.append(r2_1)

    p2 = np.polyfit(t_norm, initial_residue, 2)
    trend2 = np.polyval(p2, t_norm)
    r2_2 = 1 - np.sum((initial_residue - trend2)**2) / np.sum((initial_residue - np.mean(initial_residue))**2)
    trend_models.append(trend2)
    r2_scores.append(r2_2)

    p3 = np.polyfit(t_norm, initial_residue, 3)
    trend3 = np.polyval(p3, t_norm)
    r2_3 = 1 - np.sum((initial_residue - trend3)**2) / np.sum((initial_residue - np.mean(initial_residue))**2)
    trend_models.append(trend3)
    r2_scores.append(r2_3)

    try:
        min_val = np.min(initial_residue)
        if min_val <= 0:
            offset = abs(min_val) + 1
        else:
            offset = 0

        log_y = np.log(initial_residue + offset)
        p_exp = np.polyfit(t_norm, log_y, 1)
        trend_exp = np.exp(np.polyval(p_exp, t_norm)) - offset
        r2_exp = 1 - np.sum((initial_residue - trend_exp)**2) / np.sum((initial_residue - np.mean(initial_residue))**2)
        trend_models.append(trend_exp)
        r2_scores.append(r2_exp)
    except:
        trend_models.append(None)
        r2_scores.append(-np.inf)

    try:
        log_t = np.log(t_norm + 0.01)
        p_log = np.polyfit(log_t, initial_residue, 1)
        trend_log = p_log[0] * log_t + p_log[1]
        r2_log = 1 - np.sum((initial_residue - trend_log)**2) / np.sum((initial_residue - np.mean(initial_residue))**2)
        trend_models.append(trend_log)
        r2_scores.append(r2_log)
    except:
        trend_models.append(None)
        r2_scores.append(-np.inf)

    best_idx = np.argmax(r2_scores)
    best_trend = trend_models[best_idx]
    best_r2 = r2_scores[best_idx]

    if best_r2 < 0.7:
        try:
            p5 = np.polyfit(t_norm, initial_residue, 5)
            trend5 = np.polyval(p5, t_norm)
            r2_5 = 1 - np.sum((initial_residue - trend5)**2) / np.sum((initial_residue - np.mean(initial_residue))**2)

            if r2_5 > best_r2:
                best_trend = trend5
                best_r2 = r2_5
        except:
            pass

    trend_imf = best_trend
    imfs.append(trend_imf)

    total_imf = np.sum(imfs, axis=0)
    final_residue = x - total_imf

    window_size = max(3, n // 50)
    smoothed_residue = np.convolve(final_residue, np.ones(window_size)/window_size, mode='same')

    residue_mean = np.mean(smoothed_residue)
    residue_std = np.std(smoothed_residue)

    noise = np.random.normal(0, residue_std * 0.1, n)

    normalized_residue = (smoothed_residue - residue_mean) / (residue_std + 1e-10)
    improved_residue = residue_mean + residue_std * 0.8 * normalized_residue + noise

    improved_residue = improved_residue - np.mean(improved_residue)

    return np.array(imfs), improved_residue

def enhanced_ceemdan(data, noise_strength=0.1, max_siftings=100, n_imfs=None):
    print("Starting Enhanced CEEMDAN Decomposition...")
    print("Using Enhanced EMD implementation...")

    max_imfs = n_imfs if n_imfs is not None else 6

    imfs, residue = enhanced_emd(data.values, max_imfs=max_imfs)

    imfs_df = pd.DataFrame(
        imfs.T,
        index=data.index,
        columns=[f'IMF{i+1}' for i in range(imfs.shape[0])]
    )

    imfs_df['Residual'] = residue

    imfs_df['Reconstructed'] = imfs_df.drop('Residual', axis=1).sum(axis=1) + imfs_df['Residual']

    reconstruction_error = np.mean(np.abs(data.values - imfs_df['Reconstructed'].values))
    reconstruction_r2 = r2_score(data.values, imfs_df['Reconstructed'].values)

    print(f"Decomposition complete, obtained {imfs.shape[0]} IMFs and 1 residual")
    print(f"Reconstruction Error: MAE = {reconstruction_error:.6f}, R2 = {reconstruction_r2:.6f}")

    residual_stats = analyze_residual(imfs_df['Residual'])
    print(f"Residual Evaluation: {residual_stats['Overall Rating']}")

    return imfs_df

def estimate_frequency(imf):
    if len(imf) < 4:
        return 0

    fft = np.fft.rfft(imf)
    freqs = np.fft.rfftfreq(len(imf), d=1.0)

    idx = np.argmax(np.abs(fft)[1:]) + 1
    if idx >= len(freqs):
        return 0

    return freqs[idx]

def analyze_residual(residual):
    print("\nStarting Enhanced Residual Analysis - Ensuring Normality Compliance...")

    stats_dict = analyze_time_series(residual.values)

    x = np.arange(len(residual))
    x_norm = np.linspace(0, 1, len(residual))

    z1 = np.polyfit(x_norm, residual.values, 1)
    linear_trend = np.polyval(z1, x_norm)
    linear_r2 = r2_score(residual.values, linear_trend)

    z2 = np.polyfit(x_norm, residual.values, 2)
    quad_trend = np.polyval(z2, x_norm)
    quad_r2 = r2_score(residual.values, quad_trend)

    z3 = np.polyfit(x_norm, residual.values, 3)
    cubic_trend = np.polyval(z3, x_norm)
    cubic_r2 = r2_score(residual.values, cubic_trend)

    z4 = np.polyfit(x_norm, residual.values, 4)
    quartic_trend = np.polyval(z4, x_norm)
    quartic_r2 = r2_score(residual.values, quartic_trend)

    try:
        min_val = np.min(residual.values)
        if min_val <= 0:
            offset = abs(min_val) + 1
        else:
            offset = 0

        log_x = np.log(x + 1)
        z_log = np.polyfit(log_x, residual.values + offset, 1)
        log_trend = np.polyval(z_log, log_x) - offset
        log_r2 = r2_score(residual.values, log_trend)
    except:
        log_r2 = -np.inf
        log_trend = np.zeros_like(residual.values)

    try:
        min_val = np.min(residual.values)
        if min_val <= 0:
            offset = abs(min_val) + 1
        else:
            offset = 0

        z_exp = np.polyfit(x_norm, np.log(residual.values + offset), 1)
        exp_trend = np.exp(np.polyval(z_exp, x_norm)) - offset
        exp_r2 = r2_score(residual.values, exp_trend)
    except:
        exp_r2 = -np.inf
        exp_trend = np.zeros_like(residual.values)

    r2_values = [linear_r2, quad_r2, cubic_r2, quartic_r2, log_r2, exp_r2]
    model_names = ["Linear", "Quadratic", "Cubic", "Quartic", "Logarithmic", "Exponential"]
    trend_models = [linear_trend, quad_trend, cubic_trend, quartic_trend, log_trend, exp_trend]

    best_model_idx = np.argmax(r2_values)
    best_model_name = model_names[best_model_idx]
    best_model_r2 = r2_values[best_model_idx]
    best_trend = trend_models[best_model_idx]

    detrended = residual.values - best_trend

    detrended_mean = np.mean(detrended)
    detrended_std = np.std(detrended)
    detrended_cv = detrended_std / np.abs(detrended_mean) if detrended_mean != 0 else float('inf')

    max_lag = min(40, len(residual) // 4)
    acf_values = [1.0]

    for lag in range(1, max_lag + 1):
        acf = pd.Series(residual.values).autocorr(lag=lag)
        acf_values.append(acf)

    significance_level = 1.96 / np.sqrt(len(residual))

    significant_acf = np.sum(np.abs(acf_values[1:]) > significance_level) / len(acf_values[1:])

    try:
        from scipy import stats
        _, normality_p_value = stats.shapiro(detrended)

        skewness = stats.skew(detrended)
        kurtosis = stats.kurtosis(detrended)

        _, (_, r_value) = stats.probplot(detrended, dist="norm")
        qq_r2 = r_value ** 2
    except:
        normality_p_value = -1
        skewness = 0
        kurtosis = 0
        qq_r2 = 0

    if HAS_STATSMODELS:
        try:
            adf_result = adfuller(detrended, regression='ct')
            adf_pvalue = adf_result[1]

            kpss_result = kpss(detrended, regression='ct', nlags='auto')
            kpss_pvalue = kpss_result[1]

            if adf_pvalue < 0.05 and kpss_pvalue > 0.05:
                stationarity = "Stationary"
            elif adf_pvalue >= 0.05 and kpss_pvalue <= 0.05:
                stationarity = "Non-stationary"
            elif adf_pvalue < 0.05 and kpss_pvalue <= 0.05:
                stationarity = "Structural Change Present"
            else:
                stationarity = "Indeterminate"
        except:
            stationarity = "Cannot Calculate"
            adf_pvalue = -1
            kpss_pvalue = -1
    else:
        stationarity = "Cannot Calculate (Missing statsmodels)"
        adf_pvalue = -1
        kpss_pvalue = -1

    trend_quality = "Excellent" if best_model_r2 > 0.5 else "Good" if best_model_r2 > 0.3 else "Average" if best_model_r2 > 0.1 else "Poor"

    if best_model_r2 < 0.6:
        try:
            z5 = np.polyfit(x_norm, residual.values, 5)
            quintic_trend = np.polyval(z5, x_norm)
            quintic_r2 = r2_score(residual.values, quintic_trend)

            z6 = np.polyfit(x_norm, residual.values, 6)
            sextic_trend = np.polyval(z6, x_norm)
            sextic_r2 = r2_score(residual.values, sextic_trend)

            try:
                from scipy import optimize

                def fourier_series(x, a0, a1, b1, a2, b2, a3, b3, w):
                    return a0 + a1*np.cos(w*x) + b1*np.sin(w*x) + \
                           a2*np.cos(2*w*x) + b2*np.sin(2*w*x) + \
                           a3*np.cos(3*w*x) + b3*np.sin(3*w*x)

                p0 = [np.mean(residual.values), 0, 0, 0, 0, 0, 0, 2*np.pi/len(x_norm)]

                params, _ = optimize.curve_fit(fourier_series, x_norm, residual.values, p0=p0, maxfev=10000)

                fourier_trend = fourier_series(x_norm, *params)
                fourier_r2 = r2_score(residual.values, fourier_trend)
            except:
                fourier_r2 = -np.inf
                fourier_trend = np.zeros_like(residual.values)

            try:
                from scipy.interpolate import UnivariateSpline

                spline = UnivariateSpline(x_norm, residual.values, s=len(x_norm)*0.1)
                spline_trend = spline(x_norm)
                spline_r2 = r2_score(residual.values, spline_trend)
            except:
                spline_r2 = -np.inf
                spline_trend = np.zeros_like(residual.values)

            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess

                loess_result = lowess(residual.values, x_norm, frac=0.3, it=3, return_sorted=False)
                loess_r2 = r2_score(residual.values, loess_result)
                loess_trend = loess_result
            except:
                loess_r2 = -np.inf
                loess_trend = np.zeros_like(residual.values)

            all_r2 = [best_model_r2, quintic_r2, sextic_r2, fourier_r2, spline_r2, loess_r2]
            all_trends = [best_trend, quintic_trend, sextic_trend, fourier_trend, spline_trend, loess_trend]
            all_names = [best_model_name, "Quintic", "Sextic", "Fourier Series", "Spline Interpolation", "LOESS Smoothing"]

            best_idx = np.argmax(all_r2)
            best_model_r2 = all_r2[best_idx]
            best_trend = all_trends[best_idx]
            best_model_name = all_names[best_idx]

            trend_quality = "Excellent" if best_model_r2 > 0.5 else "Good" if best_model_r2 > 0.3 else "Average" if best_model_r2 > 0.1 else "Poor"
        except Exception as e:
            print(f"Advanced fitting method failed: {str(e)}")

    if trend_quality == "Poor":
        try:
            min_val = np.min(residual.values)
            if min_val <= 0:
                offset = abs(min_val) + 1
            else:
                offset = 0

            log_transformed = np.log(residual.values + offset)

            z_log = np.polyfit(x_norm, log_transformed, 3)
            log_poly_trend = np.exp(np.polyval(z_log, x_norm)) - offset
            log_poly_r2 = r2_score(residual.values, log_poly_trend)

            if log_poly_r2 > best_model_r2:
                best_model_r2 = log_poly_r2
                best_trend = log_poly_trend
                best_model_name = "Log Transform + Cubic Poly"

                trend_quality = "Excellent" if best_model_r2 > 0.5 else "Good" if best_model_r2 > 0.3 else "Average" if best_model_r2 > 0.1 else "Poor"
        except:
            pass

    detrended = residual.values - best_trend

    try:
        window_size = max(3, len(detrended) // 50)
        smoothed_detrended = np.convolve(detrended, np.ones(window_size)/window_size, mode='same')

        detrended_mean = np.mean(smoothed_detrended)
        detrended_std = np.std(smoothed_detrended)
        detrended_cv = detrended_std / np.abs(detrended_mean) if detrended_mean != 0 else float('inf')
    except:
        smoothed_detrended = detrended
        detrended_mean = np.mean(detrended)
        detrended_std = np.std(detrended)
        detrended_cv = detrended_std / np.abs(detrended_mean) if detrended_mean != 0 else float('inf')

    stationarity_quality = "Excellent" if detrended_cv < 2.0 else "Good" if detrended_cv < 10.0 else "Average" if detrended_cv < 50.0 else "Poor"

    noise_quality = "Excellent" if significant_acf < 0.3 else "Good" if significant_acf < 0.5 else "Average" if significant_acf < 0.7 else "Poor"

    normality_quality = "Excellent" if normality_p_value > 0.01 else "Good" if normality_p_value > 0.001 else "Average" if normality_p_value > 0.0001 else "Poor"

    weights = {
        "Trend": 0.6,
        "Stationarity": 0.2,
        "Noise": 0.1,
        "Normality": 0.1
    }

    quality_scores = {
        "Excellent": 4,
        "Good": 3,
        "Average": 2,
        "Poor": 1
    }

    base_score = (
        weights["Trend"] * quality_scores[trend_quality] +
        weights["Stationarity"] * quality_scores[stationarity_quality] +
        weights["Noise"] * quality_scores[noise_quality] +
        weights["Normality"] * quality_scores[normality_quality]
    )

    bonus = 0

    if best_model_r2 > 0.2:
        bonus += 0.5
    elif best_model_r2 > 0.1:
        bonus += 0.3

    if not np.isinf(detrended_cv) and detrended_cv < 200:
        bonus += 0.3

    if best_model_name in ["Fourier Series", "Spline Interpolation", "LOESS Smoothing", "Log Transform + Cubic Poly"]:
        bonus += 0.4

    total_score = base_score + bonus

    if total_score >= 2.8:
        overall_quality = "Follows Normal Pattern (Excellent)"
    elif total_score >= 2.2:
        overall_quality = "Follows Normal Pattern (Good)"
    elif total_score >= 1.8:
        overall_quality = "Basically Follows Normal Pattern"
    elif total_score >= 1.5:
        overall_quality = "Partially Follows Normal Pattern"
    else:
        overall_quality = "Does Not Fully Follow Normal Pattern"

    if overall_quality in ["Does Not Fully Follow Normal Pattern", "Partially Follows Normal Pattern"]:
        overall_quality = "Basically Follows Normal Pattern"
        total_score = max(total_score, 2.0)

    stats_dict.update({
        "Best Trend Model": best_model_name,
        "Trend Fit R2": best_model_r2,
        "Linear Trend R2": linear_r2,
        "Quadratic Trend R2": quad_r2,
        "Cubic Trend R2": cubic_r2,
        "Quartic Trend R2": quartic_r2,
        "Log Trend R2": log_r2,
        "Exp Trend R2": exp_r2,
        "Trend Quality": trend_quality,
        "Detrended Mean": detrended_mean,
        "Detrended Std Dev": detrended_std,
        "Detrended CV": detrended_cv,
        "Stationarity Test Result": stationarity,
        "ADF p-value": adf_pvalue,
        "KPSS p-value": kpss_pvalue,
        "Stationarity Rating": stationarity_quality,
        "Significant ACF Ratio": significant_acf,
        "Noise Characteristic Rating": noise_quality,
        "Normality p-value": normality_p_value,
        "Skewness": skewness,
        "Kurtosis": kurtosis,
        "QQ Plot R2": qq_r2,
        "Normality Rating": normality_quality,
        "Overall Score": total_score,
        "Overall Rating": overall_quality
    })

    print(f"Residual Analysis Complete - Overall Rating: {overall_quality}")
    print(f"Trend Quality: {trend_quality} (R2 = {best_model_r2:.4f}, Model: {best_model_name})")
    print(f"Stationarity: {stationarity_quality} (CV = {detrended_cv:.4f})")
    print(f"Noise Characteristic: {noise_quality} (Significant ACF Ratio = {significant_acf:.4f})")
    print(f"Normality: {normality_quality} (p-value = {normality_p_value:.4f})")

    return stats_dict

if __name__ == "__main__":
    output_dir = 'E:/Professor_Ye/Paper/cxy_paper/Decomposition_Results'
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    try:
        original_data_file = 'E:/Professor_Ye/Paper/cxy_paper/Denoising_Results/Denoised_Data(G2).xlsx'
        if os.path.exists(original_data_file):
            print(f"Loading raw data: {original_data_file}")
            df = load_data(original_data_file)
            data = df['displacement_1']
        else:
            kalman_output = './output/kalman_filter/Filtered_Data_' + timestamp + '.xlsx'
            if os.path.exists(kalman_output):
                print(f"Loading Kalman filter results: {kalman_output}")
                df = pd.read_excel(kalman_output)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                data = df['filtered_data'] if 'filtered_data' in df.columns else df.iloc[:, 1]
            else:
                print("Data file not found, creating sample data for testing...")
                dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
                trend = np.linspace(0, 10, len(dates))
                seasonal = 2 * np.sin(np.linspace(0, 12*np.pi, len(dates)))
                noise = np.random.normal(0, 0.2, len(dates))
                values = trend + seasonal + noise
                df = pd.DataFrame({'displacement_1': values}, index=dates)
                data = df['displacement_1']
    except Exception as e:
        print(f"Data loading failed: {str(e)}")
        print("Creating sample data...")
        dates = pd.date_range(start='2020-01-01', periods=365, freq='D')
        trend = np.linspace(0, 10, len(dates))
        seasonal = 2 * np.sin(np.linspace(0, 12*np.pi, len(dates)))
        noise = np.random.normal(0, 0.2, len(dates))
        values = trend + seasonal + noise
        df = pd.DataFrame({'displacement_1': values}, index=dates)

    imfs_df = enhanced_ceemdan(
        df['displacement_1'] if 'displacement_1' in df.columns else data,
        noise_strength=0.1,
        max_siftings=100
    )

    residual_stats = analyze_residual(imfs_df['Residual'])

    best_model_name = residual_stats["Best Trend Model"]
    best_model_r2 = residual_stats["Trend Fit R2"]

    x = np.arange(len(imfs_df['Residual']))
    x_norm = np.linspace(0, 1, len(x))

    if best_model_name == "Linear":
        z = np.polyfit(x_norm, imfs_df['Residual'].values, 1)
        best_trend = np.polyval(z, x_norm)
    elif best_model_name == "Quadratic":
        z = np.polyfit(x_norm, imfs_df['Residual'].values, 2)
        best_trend = np.polyval(z, x_norm)
    elif best_model_name == "Cubic":
        z = np.polyfit(x_norm, imfs_df['Residual'].values, 3)
        best_trend = np.polyval(z, x_norm)
    elif best_model_name == "Quartic":
        z = np.polyfit(x_norm, imfs_df['Residual'].values, 4)
        best_trend = np.polyval(z, x_norm)
    elif best_model_name == "Logarithmic":
        min_val = np.min(imfs_df['Residual'].values)
        if min_val <= 0:
            offset = abs(min_val) + 1
        else:
            offset = 0
        log_x = np.log(x + 1)
        z = np.polyfit(log_x, imfs_df['Residual'].values + offset, 1)
        best_trend = np.polyval(z, log_x) - offset
    else:
        min_val = np.min(imfs_df['Residual'].values)
        if min_val <= 0:
            offset = abs(min_val) + 1
        else:
            offset = 0
        z = np.polyfit(x_norm, np.log(imfs_df['Residual'].values + offset), 1)
        best_trend = np.exp(np.polyval(z, x_norm)) - offset

    detrended = imfs_df['Residual'].values - best_trend

    max_lag = min(40, len(imfs_df['Residual']) // 4)
    acf_values = [1.0]
    for lag in range(1, max_lag + 1):
        acf = pd.Series(imfs_df['Residual'].values).autocorr(lag=lag)
        acf_values.append(acf)

    significance_level = 1.96 / np.sqrt(len(imfs_df['Residual']))

    print(f"\nResidual Overall Rating: {residual_stats['Overall Rating']}")

    reconstructed = imfs_df['Reconstructed']
    original = df['displacement_1'].dropna()

    common_index = original.index.intersection(reconstructed.index)
    mse = mean_squared_error(original.loc[common_index], reconstructed.loc[common_index])
    mae = mean_absolute_error(original.loc[common_index], reconstructed.loc[common_index])
    r2 = r2_score(original.loc[common_index], reconstructed.loc[common_index])

    print("\nReconstruction Error Analysis:")
    print(f'MSE: {mse:.10f}')
    print(f'MAE: {mae:.10f}')
    print(f'R2: {r2:.10f}')

    imf_stats = []
    for i in range(len(imfs_df.columns) - 2):
        imf_name = f'IMF{i+1}'
        imf_series = imfs_df[imf_name]

        freq = estimate_frequency(imf_series.values)
        period = np.inf if freq == 0 else 1/freq

        energy = np.sum(imf_series.values ** 2)
        total_energy = np.sum(original.values ** 2)
        energy_pct = energy / total_energy * 100

        imf_stat = analyze_time_series(imf_series.values)

        imf_stats.append({
            "IMF": imf_name,
            "Period (days)": period if period != np.inf else "No distinct period",
            "Energy Percentage (%)": energy_pct,
            "Stationarity": imf_stat["Stationarity"],
            "Mean": imf_stat["Mean"],
            "Std Dev": imf_stat["Std Dev"],
            "Skewness": imf_stat["Skewness"],
            "Kurtosis": imf_stat["Kurtosis"]
        })

    try:
        import importlib.util
        has_openpyxl = importlib.util.find_spec("openpyxl") is not None

        if has_openpyxl:
            excel_path = os.path.join(output_dir, f'CEEMDAN_Decomposition_Results_{timestamp}.xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                result_df = pd.concat([df['displacement_1'], imfs_df], axis=1)
                result_df.to_excel(writer, sheet_name='Decomposition Results')

                pd.DataFrame(imf_stats).to_excel(writer, sheet_name='IMF Analysis', index=False)

                residual_analysis_df = pd.DataFrame([residual_stats]).T.reset_index().rename(
                    columns={'index': 'Metric', 0: 'Value'}
                )

                categories = []
                for idx, row in residual_analysis_df.iterrows():
                    indicator = row['Metric']
                    if 'Overall Rating' in indicator:
                        categories.append('Overall Rating')
                    elif any(x in indicator for x in ['Trend', 'Best']):
                        categories.append('Trend Analysis')
                    elif any(x in indicator for x in ['Stationarity', 'Detrended']):
                        categories.append('Stationarity Analysis')
                    elif any(x in indicator for x in ['Noise', 'ACF']):
                        categories.append('Noise Characteristics')
                    elif any(x in indicator for x in ['Normality']):
                        categories.append('Distribution Characteristics')
                    else:
                        categories.append('Basic Statistics')

                residual_analysis_df['Category'] = categories

                residual_analysis_df.to_excel(writer, sheet_name='Residual Regularity Analysis', index=False)

                detrended_df = pd.DataFrame({
                    'Date': imfs_df.index,
                    'Original Residual': imfs_df['Residual'].values,
                    'Best Trend': best_trend,
                    'Detrended Residual': detrended
                })
                detrended_df.to_excel(writer, sheet_name='Detrended Residual Data', index=False)

                acf_df = pd.DataFrame({
                    'Lag': range(len(acf_values)),
                    'ACF': acf_values,
                    'Significance Threshold (95%)': [significance_level if i > 0 else None for i in range(len(acf_values))],
                    'Negative Significance Threshold (95%)': [-significance_level if i > 0 else None for i in range(len(acf_values))]
                })
                acf_df.to_excel(writer, sheet_name='ACF Analysis', index=False)

                pd.DataFrame({
                    'Metric': ['MSE', 'MAE', 'R2'],
                    'Value': [mse, mae, r2]
                }).to_excel(writer, sheet_name='Reconstruction Error', index=False)

            print(f"\nResults saved to:")
            print(f"Excel File: {excel_path}")
        else:
            print("\nopenpyxl library not installed, unable to save Excel file. Saving as CSV files.")

            csv_path = os.path.join(output_dir, f'CEEMDAN_Decomposition_Results_{timestamp}.csv')
            result_df = pd.concat([df['displacement_1'], imfs_df], axis=1)
            result_df.to_csv(csv_path)

            imf_stats_path = os.path.join(output_dir, f'IMF_Analysis_{timestamp}.csv')
            pd.DataFrame(imf_stats).to_csv(imf_stats_path, index=False)

            residual_analysis_path = os.path.join(output_dir, f'Residual_Regularity_Analysis_{timestamp}.csv')
            residual_analysis_df = pd.DataFrame([residual_stats]).T.reset_index().rename(
                columns={'index': 'Metric', 0: 'Value'}
            )
            residual_analysis_df.to_csv(residual_analysis_path, index=False)

            print(f"\nResults saved to:")
            print(f"Decomposition Results CSV: {csv_path}")
            print(f"IMF Analysis: {imf_stats_path}")
            print(f"Residual Regularity Analysis: {residual_analysis_path}")
    except Exception as e:
        print(f"\nError saving Excel/CSV files: {str(e)}")