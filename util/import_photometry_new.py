'''New version of import photometry from Marta, use a double exponential fit to correct
signals for photobleaching'''

# Function for opening pyPhotometry data files in Python.
from datetime import datetime
import numpy as np
from scipy.signal import medfilt, butter, filtfilt, detrend
from scipy.stats import linregress
from scipy.ndimage.filters import percentile_filter
from numba import jit
from scipy.signal._arraytools import axis_slice, axis_reverse, odd_ext, even_ext, const_ext
from parse import *
from scipy.optimize import curve_fit
import scipy.optimize as opt


from scipy import sparse
from scipy.sparse.linalg import spsolve

def als(y, lam=1e4, p=0.05, niter=10):
  L = len(y)
  # D = sparse.csc_matrix(np.diff(np.eye(L), 2))
  diag = np.ones(L - 2)
  D = sparse.spdiags([diag, -2 * diag, diag], [0, -1, -2], L, L - 2).tocsc()
  w = np.ones(L)
  for i in range(niter):
    W = sparse.spdiags(w, 0, L, L)
    Z = W + lam * D.dot(D.transpose())
    z = spsolve(Z, w*y)
    w = p * (y > z) + (1-p) * (y < z)
    print(w)
  return z

def import_data(file_path, subject_format, low_pass=2, high_pass=0.001, order=3, old_format=False,
                signal_name='df'):
    with open(file_path, 'rb') as f:
        header_size = int.from_bytes(f.read(2), 'little')
        data_header = f.read(header_size)
        data = np.frombuffer(f.read(), dtype=np.dtype('<u2'))
    # Extract header information
    if old_format:
      subject_ID = data_header[:12].decode().strip()
      date_time = datetime.strptime(data_header[12:31].decode(), '%Y-%m-%dT%H:%M:%S')
      mode = {1: 'GCaMP/RFP', 2: 'GCaMP/iso', 3: 'GCaMP/RFP_dif'}[data_header[31]]
      sampling_rate = int.from_bytes(data_header[32:34], 'little')
      volts_per_division = np.frombuffer(data_header[34:42], dtype='<u4') * 1e-9
    else:
      subject_ID = parse(subject_format, data_header.decode().split('",')[0].split()[1][1:])['id']
      if 'hemisphere' in parse(subject_format, data_header.decode().split('",')[0].split()[1][1:]).named.keys():
        hemisphere =parse(subject_format, data_header.decode().split('",')[0].split()[1][1:])['hemisphere']
      else:
        hemisphere = 'NaN'
      if 'region' in parse(subject_format, data_header.decode().split('",')[0].split()[1][1:]).named.keys():
        region =parse(subject_format, data_header.decode().split('",')[0].split()[1][1:])['region']
      else:
        region = 'NaN'
      # subject_ID = data_header.decode().split('",')[0].split()[1][1:4]
      # if no_region_format == True:
      #   region = 'NaN'
      #   hemisphere = data_header.decode().split('",')[0].split()[1][5:8]
      # else:
      #   region = data_header.decode().split('",')[0].split()[1][5:8]
      #   if no_hemisphere_format == True:
      #     hemisphere = 'NaN'
      #   else:
      #     hemisphere = data_header.decode().split('",')[0].split()[1][9]
      date_time = datetime.strptime(data_header.decode().split('",')[1].split()[1][1:], '%Y-%m-%dT%H:%M:%S')
      mode = data_header.decode().split('",')[2].split('"')[-1]
      sampling_rate = int(data_header.decode().split('",')[3].split(', "')[0].split()[-1])
      volts_per_division = np.array([float(data_header.decode().split('",')[3].split(', "')[1].split(': ')[-1].
                                           replace(']','').replace('[','').split(',')[0]),
                                     float(data_header.decode().split('",')[3].split(', "')[1].split(': ')[-1].
                                           replace(']','').replace('[','').split(',')[1])])
    # Extract signals.
    signal  = data >> 1       # Analog signal is most significant 15 bits.
    digital = (data % 2) == 1 # Digital signal is least significant bit.
    # Alternating samples are signals 1 and 2.
    ADC1 = signal[ ::2] * volts_per_division[0] #convert the analog signals into volts
    ADC2 = signal[1::2] * volts_per_division[1]
    DI1 = digital[ ::2]
    DI2 = digital[1::2]
    t = np.arange(ADC1.shape[0]) / sampling_rate # Time relative to start of recording (seconds).
    #Median filtering to remove electrical artifact
    ADC1_denoised = medfilt(ADC1, kernel_size=5)
    ADC2_denoised = medfilt(ADC2, kernel_size=5)
    if signal_name in ['df', 'det']:
    #Detrend the signal with a nth order polynomial
    # ADC1_p = percentile_filter(ADC1, 20, sampling_rate * 20)
      model = np.polyfit(range(len(ADC1_denoised)), ADC1_denoised, order)
      predicted = np.polyval(model, range(len(ADC1_denoised)))
      ADC1_f = ADC1_denoised - predicted
      #detrend ADC2
      model = np.polyfit(range(len(ADC2_denoised)), ADC2_denoised, order)
      predicted = np.polyval(model, range(len(ADC2_denoised)))
      ADC2_f = ADC2_denoised - predicted
    #detrend by substracting a linear least-square fit to the data
    elif signal_name in ['det_sq']:
      ADC1_f = detrend(ADC1_denoised)
      ADC2_f = detrend(ADC2_denoised)
    # Filter signals.
    if low_pass and high_pass:
        b, a = butter(2, np.array([high_pass, low_pass])/(0.5*sampling_rate), 'bandpass')
    elif low_pass:
        b, a = butter(2, low_pass/(0.5*sampling_rate), 'low')
    elif high_pass:
        # b, a = butter(2, high_pass/(0.5*sampling_rate), 'high')
        b, a = butter(2, high_pass, btype='high', fs=sampling_rate)
    if signal_name in ['filt']:
      if low_pass or high_pass:
          ADC1_f = filtfilt(b, a, ADC1_denoised, padtype='even')
          ADC2_f = filtfilt(b, a, ADC2_denoised, padtype='even')
      else:
          ADC1_f = ADC2_f = None

    if signal_name in ['als']:
      # Remove bleaching using asymmetrical least squares smoothing
      ADC1_f = ADC1_denoised - als(ADC1_denoised)
      ADC2_f = ADC2_denoised - als(ADC2_denoised)

    if signal_name is 'denoised':
      ADC1_f = ADC1_denoised
      ADC2_f = ADC2_denoised

    #lowpass detrended signal
    b, a = butter(2, low_pass / (0.5 * sampling_rate), 'low')
    ADC1_f = filtfilt(b, a, ADC1_f, padtype='even')
    ADC2_f = filtfilt(b, a, ADC2_f, padtype='even')


    return {'subject_ID'   : subject_ID,
            'region'       : region,
            'hemisphere'   : hemisphere,
            'datetime'     : date_time,
            'datetime_str' : date_time.strftime('%Y-%m-%d %H:%M:%S'),
            'mode'         : mode,
            'sampling_rate': sampling_rate,
            'volts_per_div': volts_per_division,
            'ADC1'         : ADC1,
            'ADC2'         : ADC2,
            # 'ADC1_denoised': ADC1_denoised,
            # 'ADC2_denoised': ADC2_denoised,
            'ADC1_f'       : ADC1_f,
            'ADC2_f'       : ADC2_f,
            'DI1'          : DI1,
            'DI2'          : DI2,
            't'            : t}

def double_exp(x, a, b, c, d):
  return a * np.exp(b * x) + c * np.exp(d * x)

def linear_func(x, a, b):
  return a + b * x

def fit_motion(x, TdTom_motionband, GCaMP_motionband):
  return TdTom_motionband - x[0] * GCaMP_motionband - x[1]

def signal_correction(ADC1_denoised, ADC2_denoised, sampling_rate, low_pass=20, high_pass=0.001):
  # Motion correction using a bandpass filter
  b, a = butter(2, [high_pass, low_pass], btype='bandpass', fs=sampling_rate)
  GCaMP_motionband = filtfilt(b, a, ADC1_denoised, padtype='even')
  TdTom_motionband = filtfilt(b, a, ADC2_denoised, padtype='even')
  slope, intercept, r_value, p_value, std_err = linregress(x=TdTom_motionband, y=GCaMP_motionband)
  GCaMP_est_motion = intercept + slope * ADC2_denoised
  GCaMP_corrected = ADC1_denoised - GCaMP_est_motion

  #Use a double exponential fit to correct for bleaching
  try:
    popt, pcov = curve_fit(double_exp, np.arange(len(GCaMP_corrected)), GCaMP_corrected, p0=(1, 1e-6, 1, 1e-6))
    fit_line = double_exp(np.arange(len(GCaMP_corrected)), *popt)
  except RuntimeError:
    try:
      popt, pcov = curve_fit(double_exp, np.arange(len(GCaMP_corrected)), GCaMP_corrected, p0=(1, 1e-6, 0, 1e-6))
      fit_line = double_exp(np.arange(len(GCaMP_corrected)), *popt)
    except RuntimeError:
      try:
        popt, pcov = curve_fit(double_exp, np.arange(len(GCaMP_corrected)), GCaMP_corrected, p0=(-1, 1e-6, 0, 1e-6))
        fit_line = double_exp(np.arange(len(GCaMP_corrected)), *popt)
      except RuntimeError:
        popt, pcov = curve_fit(linear_func, np.arange(len(GCaMP_corrected)), GCaMP_corrected, p0=(1e-6, 0))
        fit_line = linear_func(np.arange(len(GCaMP_corrected)), *popt)
        print('LINEAR FIT')
  GCaMP_corrected = GCaMP_corrected - fit_line

  return GCaMP_corrected

@jit
def percentile_filter(x, percentile=50, a=0.001, i=None):
    '''
    Online percentile filter implemented by incrementing or decrementing the filtered
    signal by a fixed amount at each timestep depending on whether the raw signal is
    larger or smaller than the current value of the filtered signal.  Positive and
    negative increments are aysmmetric and the extent of this asymmetry determines
    the percentile that is filtered for.
    Arguments:
    x          : The data to be filtered, 1D numpy array.
    percentile : The percentile to be filtered for.
    a          : Size of increments of the filtered signal relative to the raw signal's
                 standard deviation.  Determines the effective time window over which
                 the percentile is calculated, with low values of 'a' producing
                 a longer time window.
    i          : initial value for filter.
    '''
    if not (0 < percentile < 100):
        raise ValueError("'percentile' must be between 0 and 100.")

    if not (0 < a < 1):
        raise ValueError("'a' must be between 0 and 1.")

    x_sd = np.std(x)
    pos = x_sd*a*percentile/100        # postitive increment
    neg = x_sd*a*(100-percentile)/100  # negative increment
    y = np.zeros(x.shape)
    y[0] = i if i is not None else x[0]
    for i,xs in enumerate(x[:-1]):
        if xs > y[i]:
            y[i+1] = y[i] + pos
        else:
            y[i+1] = y[i] - neg
    return y

def percentile_filtfilt(x, percentile=50, a=0.001, padtype='even', padlen=None):
    '''
    Applies percentile filtering in the forward and reverse directions to the raw
    signal and averages the results to avoid inducing phase shifts.  By default even
    padding is applied to the edges of the signal to avoid edge effects. The type
    of padding used can be specified using the same arguments as scipy.signal.filtfilt.
    Arguments:
    x          : The data to be filtered, 1D numpy array.
    percentile : The percentile to be filtered for.
    a          : Size of increments of the filtered signal relative to the raw signal's
                 standard deviation.  Determines the effective time window over which
                 the percentile is calculated, with low values of 'a' producing
                 a longer time window.
    '''
    if padtype not in ['even', 'odd', 'constant', None]:
        raise ValueError(("Unknown value '%s' given to padtype.  padtype must "
                         "be 'even', 'odd', 'constant', or None.") %
                         padtype)

    x = np.asarray(x)

    if padtype is None:
        padlen = 0

    if padlen is None:
        edge = int(1/(np.std(x)*a))
    else:
        edge = padlen

    # x's 'axis' dimension must be bigger than edge.
    if x.shape[0] <= edge:
        raise ValueError("The length of the input vector x must be at least "
                         "padlen, which is %d." % edge)

    if padtype is not None and edge > 0:
        # Make an extension of length `edge` at each
        # end of the input array.
        if padtype == 'even':
            ext = even_ext(x, edge, axis=0)
        elif padtype == 'odd':
            ext = odd_ext(x, edge, axis=0)
        else:
            ext = const_ext(x, edge, axis=0)
        # Filter initial conditions.
        i_f = np.percentile(x[:edge ], percentile)
        i_b = np.percentile(x[-edge:], percentile)
    else:
        ext = x
        i_f = x[0]
        i_b = x[-1]

    # Forward filter.
    y = percentile_filter(ext, percentile, a, i_f)

    # Backward filter.
    y += axis_reverse(percentile_filter(axis_reverse(ext, axis=0), percentile, a, i_b), axis=0)

    # Average forward and backwards filtered signals.
    y = y/2

    if edge > 0:
        # Slice the actual signal from the extended signal.
        y = axis_slice(y, start=edge, stop=-edge, axis=0)

    return y