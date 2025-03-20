import os
import sys

import numpy as np
import pandas as pd
import random
import seaborn as sns

class TraceDataGeneration():
    def __init__(self, n, global_range, seed=1, jitter_bool=False):
        self.n = n
        self.global_range = global_range
        self.jitter_bool = jitter_bool
        self.seed_everything(seed=seed)

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)


    def gaussian_noise(self, signal_length, mean=0, std=None):
        if std == None:
            std = self.global_range * 0.01

        return np.random.randn(signal_length) * std + mean

    def jitter(self, length, *time):
        # assert max(time) < length, "length must be larger than time value"
        new_length = int(np.random.randn(1) * length * 0.05 + length)
        if new_length > length:
            new_length += 1
        else:
            new_length = length
        if time:
            new_time = []
            for key, t in enumerate(time):
                time_difference = max(0, int(np.random.randn(1) * t * 0.4 + t))

                new_time.append(time_difference)

            new_time.sort()

            if max(new_time) >= new_length:
                return length, time

            # check duplicated time
            if len(set(new_time)) != len(time):
                return length, time

            return new_length, new_time

        else:
            return new_length

    def spike(self, maximum, minimum):
        return np.random.uniform(minimum, maximum)

    def combine_traces(self, *args):
        data = []
        for n in range(self.n):
            temp = []
            for wafer in args:
                temp.append(wafer[n])
            temp = np.concatenate(temp, axis=0)
            data.append(temp)

        return data

    def constant(self, start_value, end_value, length, jitter=False, sampling=True):
        if not sampling:
            noise = self.gaussian_noise(signal_length=length, mean=0)
            return np.ones(length) * (start_value + noise)

        constant = []
        for n in range(self.n):
            new_length = length
            if jitter or self.jitter_bool:
                new_length = self.jitter(length)
            noise = self.gaussian_noise(signal_length=new_length, mean=0)
            constant.append(np.ones(new_length) * (start_value + noise))
        return constant

    def step_like(self, start_value, end_value, t1, length, jitter=False, sampling=True):

        if not sampling:
            noise = self.gaussian_noise(signal_length=t1 + 1, mean=0)
            start = np.ones(t1 + 1) * (start_value + noise)
            noise = self.gaussian_noise(signal_length=length - t1 - 1, mean=0)
            end = np.ones(length - t1 - 1) * (end_value + noise)
            signal = np.concatenate([start, end])

            return signal

        step_like = []
        for n in range(self.n):
            new_length, new_t1 = length, t1
            if jitter or self.jitter_bool:
                new_length, new_time = self.jitter(length, t1)
                new_t1 = new_time[0]
            noise = self.gaussian_noise(signal_length=new_t1 + 1, mean=0)
            start = np.ones(new_t1 + 1) * (start_value + noise)
            noise = self.gaussian_noise(signal_length=new_length - new_t1 - 1, mean=0)
            end = np.ones(new_length - new_t1 - 1) * (end_value + noise)
            signal = np.concatenate([start, end])
            step_like.append(signal)

        return step_like

    def linear_transition(self, start_value, end_value, t1, t2, length, jitter=False, sampling=True):
        # print(f"transition_length = t2 - t1 + 1: {t2 - t1 + 1}\nt1, t2, length = {t1}, {t2}, {length}\n\n")
        if not sampling:
            transition_length = t2 - t1 + 1
            noise = self.gaussian_noise(signal_length=t1, mean=0)
            start = np.ones(t1) * (start_value + noise)
            t = np.arange(0, transition_length)
            m = (end_value - start_value) / (transition_length - 1)
            noise = self.gaussian_noise(signal_length=len(t), mean=0)
            transition = start_value + m * t + noise
            noise = self.gaussian_noise(signal_length=length - t2 - 1, mean=0)
            end = np.ones(length - t2 - 1) * (end_value + noise)
            return np.concatenate([start, transition, end])

        else:
            transition_list = []
            for n in range(self.n):
                new_t1, new_t2, new_length = t1, t2, length
                if jitter or self.jitter_bool:
                    new_length, new_time = self.jitter(length, t1, t2)
                    new_t1, new_t2 = new_time

                transition_length = new_t2 - new_t1 + 1
                noise = self.gaussian_noise(signal_length=new_t1, mean=0)
                start = np.ones(new_t1) * (start_value + noise)
                t = np.arange(0, transition_length)
                m = (end_value - start_value) / (transition_length - 1)
                noise = self.gaussian_noise(signal_length=len(t), mean=0)
                transition = start_value + m * t + noise
                # print(f"new_length: {new_length}\nnew_t2 {new_t2}\nnew_length-new_t2-1:{new_length-new_t2-1}\nnew_t1 {new_t1}")
                noise = self.gaussian_noise(signal_length=new_length - new_t2 - 1, mean=0)
                end = np.ones(new_length - new_t2 - 1) * (end_value + noise)
                signal = np.concatenate([start, transition, end])
                transition_list.append(signal)
            return transition_list

    def exponential_form(self, start_value, end_value, t1, b, length, jitter=False, sampling=True):
        if not sampling:
            t1 += 1
            noise = self.gaussian_noise(signal_length=t1, mean=0)
            start = np.ones(t1) * (start_value + noise)
            t = np.arange(t1, length)
            noise = self.gaussian_noise(signal_length=len(t), mean=0)
            exponen = start_value + (end_value - start_value) * (1 - np.exp(-t / b)) + noise
            return np.append(start, exponen)

        exponential = []
        for n in range(self.n):
            new_t1, new_length = t1, length
            if jitter or self.jitter_bool:
                new_length, new_time = self.jitter(length, t1)
                new_t1 = new_time[0]
            noise = self.gaussian_noise(signal_length=new_t1 + 1, mean=0)
            start = np.ones(new_t1 + 1) * (start_value + noise)
            t = np.arange(new_t1 + 1, new_length)
            noise = self.gaussian_noise(signal_length=len(t), mean=0)
            exponen = start_value + (end_value - start_value) * (1 - np.exp(-t / b)) + noise
            signal = np.append(start, exponen)
            exponential.append(signal)
        return exponential

    def impulse_like(self, start_value, peak_value, end_value, t1, length, jitter=False):
        impulse = []
        for n in range(self.n):
            new_t1, new_length = t1, length
            if jitter or self.jitter_bool:
                new_length, new_time = self.jitter(length, t1)
                new_t1 = new_time[0]

            noise = self.gaussian_noise(signal_length=new_t1, mean=0)
            start = np.ones(new_t1) * (start_value + noise)
            new_peak_value = self.spike(maximum=peak_value * 1.5, minimum=(0.8 * peak_value + 0.2 * end_value))
            peak = np.ones(1) * new_peak_value
            noise = self.gaussian_noise(signal_length=(new_length - new_t1 - 1), mean=0)
            end = np.ones(new_length - new_t1 - 1) * (end_value + noise)
            signal = np.concatenate([start, peak, end])
            impulse.append(signal)

        return impulse

    def rectangular_pulse(self, start_value, high_value, end_value, t1, t2, length, jitter=False):
        rectangular = []
        for n in range(self.n):
            new_t1, new_t2, new_length = t1, t2, length
            if jitter or self.jitter_bool:
                new_length, new_time = self.jitter(length, t1, t2)
                new_t1, new_t2 = new_time

            noise = self.gaussian_noise(signal_length=new_t1, mean=0)
            start = np.ones(new_t1) * start_value + noise
            noise = self.gaussian_noise(signal_length=(new_t2 - new_t1 + 1), mean=0)
            high = np.ones(new_t2 - new_t1 + 1) * high_value + noise
            noise = self.gaussian_noise(signal_length=new_length - new_t2 - 1, mean=0)
            end = np.ones(new_length - new_t2 - 1) * end_value + noise
            signal = np.concatenate([start, high, end])
            rectangular.append(signal)
        return rectangular

    def trapezoidal(self, start_value, high_value, end_value, t1, t2, t3, t4, length, jitter=False):
        trapezoidal = []
        for n in range(self.n):
            new_t1, new_t2, new_t3, new_t4, new_length = t1, t2, t3, t4, length
            if jitter or self.jitter_bool:
                new_length, new_time = self.jitter(length, t1, t2, t3, t4)
                new_t1, new_t2, new_t3, new_t4 = new_time
            # print(f"t1, t2, t3, t4 = {new_t1}, {new_t2}, {new_t3}, {new_t4}")
            noise = self.gaussian_noise(signal_length=new_t1, mean=0)
            start = np.ones(new_t1) * (start_value + noise)
            # print(f"new_t2-new_t1: {new_t2-new_t1}")
            up_trend = self.linear_transition(start_value=start_value, end_value=high_value, t1=0, t2=new_t2 - new_t1,
                                              length=(new_t2 - new_t1 + 1), sampling=False)
            noise = self.gaussian_noise(signal_length=new_t3 - new_t2, mean=0)
            up_stable = np.ones(new_t3 - new_t2) * (high_value + noise)
            # print(f"new_t4-new_t3: {new_t4 - new_t3}")
            down_trend = self.linear_transition(start_value=high_value, end_value=end_value, t1=0, t2=new_t4 - new_t3,
                                                length=(new_t4 - new_t3 + 1), sampling=False)
            noise = self.gaussian_noise(signal_length=new_length - new_t4 - 1, mean=0)
            end = np.ones(new_length - new_t4 - 1) * (end_value + noise)
            signal = np.concatenate([start, up_trend, up_stable, down_trend, end])
            trapezoidal.append(signal)
        return trapezoidal

    def triangular(self, start_value, peak_value, end_value, t1, t2, t3, length, jitter=False):
        triangular = []
        for n in range(self.n):
            new_t1, new_t2, new_t3, new_length = t1, t2, t3, length
            if jitter or self.jitter_bool:
                new_length, new_time = self.jitter(length, t1, t2, t3)
                new_t1, new_t2, new_t3 = new_time

            noise = self.gaussian_noise(signal_length=new_t1, mean=0)
            start = np.ones(new_t1) * (start_value + noise)
            noise = self.gaussian_noise(signal_length=new_t2 - new_t1, mean=0)

            # print(peak_value)
            new_peak_value = self.spike(maximum=peak_value * 1., minimum=(0.8 * peak_value + 0.2 * end_value))
            # peak_value = new_peak_value

            up_trend = start_value + np.arange(new_t2 - new_t1) * (new_peak_value - start_value) / (
                        new_t2 - new_t1) + noise
            peak = np.array([new_peak_value])
            noise = self.gaussian_noise(signal_length=new_t3 - new_t2, mean=0)
            down_trend = new_peak_value - np.arange(1, new_t3 - new_t2 + 1) * (new_peak_value - end_value) / (
                    new_t3 - new_t2) + noise
            noise = self.gaussian_noise(signal_length=new_length - new_t3 - 1, mean=0)
            end = np.ones(new_length - new_t3 - 1) * (end_value + noise)
            signal = np.concatenate([start, up_trend, peak, down_trend, end])
            triangular.append(signal)

        return triangular

    def exponential_pulse(self, start_value, high_value, end_value, t1, t2, b, c, length, jitter=False):
        exponential = []
        for n in range(self.n):
            new_t1, new_t2, new_length = t1, t2, length
            if jitter or self.jitter_bool:
                new_length, new_time = self.jitter(length, t1, t2)
                new_t1, new_t2 = new_time

            noise = self.gaussian_noise(signal_length=(new_t2 + 1), mean=0)
            form1 = self.exponential_form(start_value, high_value, new_t1, b, new_t2 + 1, sampling=False)
            form1 += noise
            t = np.arange(1, new_length - new_t2)
            noise = self.gaussian_noise(signal_length=len(t), mean=0)
            form2 = end_value + (high_value - end_value) * (1 - np.exp(-(t - new_t1 + new_t2) / b)) * np.exp(-t / c)
            form2 += noise
            signal = np.concatenate([form1, form2])
            exponential.append(signal)
        return exponential