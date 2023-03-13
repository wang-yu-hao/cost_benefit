import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, filtfilt
import util.import_photometry_new as pi
import util.rsync as rs

def sync_photometry_data(photometry_filepath, subject_format_photometry, session, low_pass=5,
                         high_pass=0.001, old_format=False, plot=False):
  # Import photometry data.
  photo_data = pi.import_data(photometry_filepath,
                              subject_format=subject_format_photometry,
                              low_pass=low_pass,
                              high_pass=high_pass,  # Set high_pass to False to see bleaching otherwise 0.01
                              old_format=old_format,
                              signal_name='denoised')
  # Correct for motion artifacts and photobleaching ('ADC1_f': low_pass denoised signal)
  print('{} {}'.format(photo_data['subject_ID'], photo_data['datetime_str']))
  GCaMP_corrected = pi.signal_correction(photo_data['ADC1_f'], photo_data['ADC2_f'],
                                         photo_data['sampling_rate'], low_pass=low_pass, high_pass=high_pass)
  
  # Setup synchronisation
  sync_signal = photo_data['DI2'].astype(int)  # Signal from digital input 2 which has sync pulses.
  pulse_times_pho = (1 + np.where(np.diff(sync_signal) == 1)[0]  # Photometry sync pulse times (ms).
                     * 1000 / photo_data['sampling_rate'])
  if old_format:
    pulse_times_pyc = session.times['Rsync']  # pyControl sync pulse times (ms).
  else:
    pulse_times_pyc = session.times['rsync']  # pyControl sync pulse times (ms).
  aligner = rs.Rsync_aligner(pulse_times_A=pulse_times_pyc, pulse_times_B=pulse_times_pho, plot=plot)

  # Convert photometry sample times into pyControl reference frame.
  sample_times_pho = photo_data['t'] * 1000  # Time of photometry samples in photomery reference frame (ms).
  sample_times_pyc = aligner.B_to_A(sample_times_pho)  # Time of photometry samples in pyControl reference frame (ms).

  # remove the denoised signal before returning the dictionary
  del photo_data['ADC1_f']
  del photo_data['ADC2_f']
  return photo_data, sample_times_pho, sample_times_pyc, GCaMP_corrected

def sync_photometry_data_old(photometry_filepath, subject_format_photometry, session, signal_name, low_pass=20, high_pass=0.01, order=3, old_format=False, plot=False):
  '''
  signal name: [ADC1_filt, ADC2_filt] or [ADC1_perc, ADC2_perc], [ADC1_df, ADC2_df] #old version
  signal name: 'df', 'det', 'det_sq', 'filt', 'als'
  '''
  photo_data = pi.import_data(photometry_filepath,  # Import photometry data.
                              subject_format=subject_format_photometry,
                              low_pass=low_pass,
                              high_pass=high_pass, # Set high_pass to False to see bleaching otherwise 0.01
                              order=order,
                              old_format=old_format,
                              signal_name=signal_name)
  # Setup synchronisation.
  sync_signal = photo_data['DI2'].astype(int)  # Signal from digital input 2 which has sync pulses.
  pulse_times_pho = (1 + np.where(np.diff(sync_signal) == 1)[0]  # Photometry sync pulse times (ms).
                     * 1000 / photo_data['sampling_rate'])
  if old_format:
    pulse_times_pyc = session.times['Rsync']  # pyControl sync pulse times (ms).
  else:
    pulse_times_pyc = session.times['rsync']  # pyControl sync pulse times (ms).
  aligner = rs.Rsync_aligner(pulse_times_A=pulse_times_pyc, pulse_times_B=pulse_times_pho, plot=plot)

  # Convert photometry sample times into pyControl reference frame.
  sample_times_pho = photo_data['t'] * 1000  # Time of photometry samples in photomery reference frame (ms).
  sample_times_pyc = aligner.B_to_A(sample_times_pho)  # Time of photometry samples in pyControl reference frame (ms).

  # Motion correction
  # OLS = LinearRegression()  # Instantiate linear regression object.
  # OLS.fit(photo_data['ADC2_filt'][:, None],
  #         photo_data['ADC1_filt'][:, None])  # Fit model with TdTomato predicting GCaMP signals.
  # estimated_motion = OLS.predict(photo_data['ADC2_filt'][:, None]).squeeze()  # Prediction of motion in GCaMP signal.
  # corrected_signal = photo_data['ADC1_filt'] - estimated_motion  # Motion corrected GCaMP signal.

  # First bandpass between 0.2 and 5Hz - eliminates most of the calcium signals
  b, a = butter(2, [0.2,5], btype='bandpass', fs=photo_data['sampling_rate'])
  GCaMP_motionband = filtfilt(b, a, photo_data['ADC1_f'], padtype='even')
  TdTom_motionband = filtfilt(b, a, photo_data['ADC2_f'], padtype='even')

  # Motion correction by finding the best linear fit of the TdTomato signal to the GCaMP signal and subtracting this estimated motion component from the GCaMP signal
  slope, intercept, r_value, p_value, std_err = linregress(x=TdTom_motionband, y=GCaMP_motionband)

  # GCaMP_est_motion = intercept + slope * photo_data[signal_name[1]] #photo_data['ADC2_filt']
  GCaMP_est_motion = intercept + slope * photo_data['ADC2_f'] #photo_data['ADC2_filt']
  # GCaMP_corrected = GCaMP_highpass - GCaMP_est_motion
  # GCaMP_corrected = photo_data[signal_name[0]] - GCaMP_est_motion
  GCaMP_corrected = photo_data['ADC1_f'] - GCaMP_est_motion
  # GCaMP_corrected = photo_data['ADC1_df'] - GCaMP_est_motion

  return photo_data, sample_times_pho, sample_times_pyc, GCaMP_corrected

def session_information_from_photo_data(photo_data, keys='long'):
  # reduced all_photo_data variable, removed keys with photometry signals, only kept information about the session
  if keys == 'long':
    keys = ['subject_ID', 'region', 'hemisphere', 'datetime', 'datetime_str', 'sampling_rate', 'ADC1', 'ADC2']
  else:
    keys = ['subject_ID', 'region', 'hemisphere', 'datetime', 'datetime_str', 'sampling_rate']
  all_photo_data_info = [{x:photo_data[i][x] for x in keys} for i in range(len(photo_data))]
  return all_photo_data_info

def plot_scaling_GCaMP_scaling(photometry_filepaths, low_pass=20, high_pass=0.01):
  X = []
  y = []
  y_pred = []
  coef = []
  for pf in photometry_filepaths:
    photo_data = pi.import_data(pf,  # Import photometry data.
                                low_pass=low_pass,
                                high_pass=high_pass)
    OLS = LinearRegression()  # Instantiate linear regression object.
    OLS.fit(photo_data['ADC2_filt'][:, None],
            photo_data['ADC1_filt'][:, None]) # Fit model with TdTomato predicting GCaMP signals.
    X.append(photo_data['ADC2_filt'][:, None])
    y.append(photo_data['ADC1_filt'][:, None])
    y_pred.append(OLS.predict(photo_data['ADC2_filt'][:, None]))
    coef.append(OLS.coef_)
  plt.figure()
  [plt.plot(xi, y_predi) for xi, y_predi in zip(X, y_pred)]
  plt.xlabel('TdTomato')
  plt.ylabel('GCaMP')
  plt.figure()
  plt.hist(np.vstack(coef))
