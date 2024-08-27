import pandas as pd
from scipy.signal import savgol_filter, firwin, lfilter


class AccPreprocessing:
    def __init__(
        self,
        df,
        window_size=31,
        poly_order=5,
        fir_length=64,
        fir_cut_off=0.4,
        wrist=False,
        fs = 32
    ):
        self.df = df
        self.wrist = wrist
        self.window_size = window_size
        self.poly_order = poly_order
        self.fir_length = fir_length
        self.fir_cut_off = fir_cut_off

    def process(self):
        if self.wrist:
            # If wrist is True, process wrist acceleration data
            for acc in ["w_acc_x", "w_acc_y", "w_acc_z"]:
                acc_signal = self.df[acc].values
                filtered_signal = self.wrist_filter(acc_signal)
                self.df[acc] = filtered_signal
        else:
            # Check if the acceleration columns ["acc1", "acc2", "acc3"] exist
            acc_columns = ["acc1", "acc2", "acc3"]
            default_columns = ['x', 'y', 'z']

            if set(acc_columns).issubset(self.df.columns):
                # Use acc_columns if they are all present in the dataframe
                columns_to_use = acc_columns
            else:
                # Fall back to default columns ['x', 'y', 'z'] if acc_columns are not present
                columns_to_use = default_columns

            # Process the selected acceleration data
            for acc in columns_to_use:
                acc_signal = self.df[acc].values
                filtered_signal = self.acc_filter(acc_signal)
                self.df[acc] = filtered_signal

        return self.df

    def acc_filter(self, signal):
        return savgol_filter(signal, self.window_size, self.poly_order)

    def wrist_filter(self, signal):
        normalized_cutoff = self.fir_cut_off / (self.fs / 2) 
        fir_coeff = firwin(self.fir_length, normalized_cutoff)
        return lfilter(fir_coeff, 1.0, signal)

