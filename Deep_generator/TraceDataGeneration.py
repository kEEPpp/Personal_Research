import os
import sys

import numpy as np
import pandas as pd
import random
import seaborn as sns

FUNCTION_LIST = ['constant',
                 'exponential_form',
                 'exponential_pulse',
                 'impulse_like',
                 'linear_transition',
                 'rectangular_pulse',
                 'step_like',
                 'trapezoidal',
                 'triangular']

DEFECT_LIST = ['add_spike',
               'add_stability',
               'add_drifting',
               'add_different']


class TraceDataGenerationCommon:
    def sample_function_list(self, num_func):
        function = []
        for i in range(num_func):
            function.append(random.choice(FUNCTION_LIST))
        return function


class TraceDataGeneration(TraceDataGenerationCommon):
    def __init__(self, n, global_range, seed=1, jitter_bool=False):
        self.n = n
        self.global_range = global_range
        self.jitter_bool = jitter_bool
        self.seed_everything(seed=seed)

    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

    def random_mean(self, num=10, mean_max=100, mean_min=10):
        mean_range = np.random.rand(num)
        mean_range = (mean_range - np.min(mean_range)) / (np.max(mean_range) - np.min(mean_range))
        mean_range = mean_range * (mean_max - mean_min) + mean_min

        return mean_range

    def get_param_normal(self, fnc_name, mean_val, length_mean=50, length_std=20, length_min=20):
        step_length = int(np.max([length_min, np.random.normal(length_mean, length_std)]))
        step_length = 40

        if fnc_name == 'constant':
            return {'start_value': mean_val, 'end_value': mean_val, 'length': step_length}

        elif fnc_name == 'step_like':
            start_val = mean_val - np.max([np.random.normal(10, 5), 5])
            t1 = int(np.random.uniform(2, step_length - 2))

            return {'start_value': start_val, 'end_value': mean_val, 't1': t1, 'length': step_length}

        elif fnc_name == 'linear_transition':
            start_val = mean_val - np.max([np.random.normal(10, 5), 5])
            ts = np.random.uniform(2, step_length - 2, 6)
            ts.sort()
            t1 = int(ts[2])
            t2 = int(ts[4])
            param = {'start_value': start_val, 'end_value': mean_val, 't1': t1, 't2': t2, 'length': step_length}
            # return {'start_value': start_val, 'end_value': mean_val, 't1':t1, 't2':t2, 'length': step_length}
            return param

        elif fnc_name == 'exponential_form':
            start_val = mean_val - np.max([np.random.normal(10, 5), 5])
            t1 = int(np.random.uniform(3, step_length - 3))
            b = np.random.poisson(3) + 1

            return {'start_value': start_val, 'end_value': mean_val, 't1': t1, 'b': b, 'length': step_length}

        elif fnc_name == 'impulse_like':
            start_val = mean_val - np.max([np.random.normal(10, 5), 5])
            peak_val = mean_val + np.max([np.random.normal(10, 5), 5])
            t1 = int(np.random.uniform(3, step_length - 3))

            return {'start_value': start_val, 'peak_value': peak_val, 'end_value': mean_val, 't1': t1,
                    'length': step_length}

        elif fnc_name == 'rectangular_pulse':
            start_val = mean_val - np.max([np.random.normal(10, 5), 5])
            high_val = mean_val + np.max([np.random.normal(10, 5), 5])
            ts = np.random.uniform(3, step_length - 3, 4)
            t1 = int(np.min(ts))
            t2 = int(np.max(ts))

            return {'start_value': start_val, 'high_value': high_val, 'end_value': mean_val, 't1': t1, 't2': t2,
                    'length': step_length}

        elif fnc_name == 'trapezoidal':
            start_val = mean_val - np.max([np.random.normal(10, 4), 3])
            high_val = mean_val + np.max([np.random.normal(10, 4), 3])
            ts = np.random.uniform(5, step_length - 5, 10)
            ts.sort()
            t1 = int(ts[1])
            t2 = int(ts[4])
            t3 = int(ts[7])
            t4 = int(ts[9])

            return {'start_value': start_val, 'high_value': high_val, 'end_value': mean_val, 't1': t1, 't2': t2,
                    't3': t3, 't4': t4, 'length': step_length}

        elif fnc_name == 'triangular':
            start_val = mean_val - np.max([np.random.normal(10, 4), 3])
            peak_val = mean_val + np.max([np.random.normal(10, 4), 3])
            ts = np.random.uniform(5, step_length - 5, 9)
            ts.sort()
            t1 = int(ts[2])
            t2 = int(ts[5])
            t3 = int(ts[7])

            return {'start_value': start_val, 'peak_value': peak_val, 'end_value': mean_val, 't1': t1, 't2': t2,
                    't3': t3, 'length': step_length}

        elif fnc_name == 'exponential_pulse':
            start_val = mean_val - np.max([np.random.normal(10, 4), 3])
            high_val = mean_val + np.max([np.random.normal(10, 4), 3])
            ts = np.random.uniform(5, step_length - 5, 9)
            ts.sort()
            t1 = int(ts[2])
            t2 = int(ts[6])

            b = np.random.poisson(6) + 1
            c = np.random.poisson(6) + 1

            return {'start_value': start_val,
                    'high_value': high_val,
                    'end_value': mean_val,
                    't1': t1,
                    't2': t2,
                    'b': b,
                    'c': c,
                    'length': step_length}

        else:
            return None

    def automated_generation_random_para(self, function, jitter=False, variation=False):

        # function = super().sample_function_list(step_num)
        step = []
        params = []
        start_value = -1

        check_start_value = -1
        check_end_value = -1

        mean_profiles = self.random_mean(len(function))
        #print(mean_profiles)
        for key, f in enumerate(function):
            if f == 'cloud_pulse':
                param = [30, 4, 20 + 20]
                # func = getattr(self.cloud_pulse, f)
                func = self.cloud_pulse
                # all_value.append(func(*param))

            elif f == 'constant':
                # param = [10, 20 + 20]
                # param = {'start_value': 10,
                #          'end_value': 10,
                #          'length': 40}

                param = self.get_param_normal('constant', mean_profiles[key])
                func = self.constant

            elif f == 'step_like':
                # param = [12, 34, 6, 10 + 20]
                # param = {'start_value': 12,
                #          'end_value': 34,
                #          't1': 6,
                #          'length': 30}
                # func = getattr(self.step_like, f)

                param = self.get_param_normal('step_like', mean_profiles[key])
                func = self.step_like
                # all_value.append(func(*param))

            elif f == 'linear_transition':
                # param = [12, 42, 5, 7, 20 + 20]
                # param = {'start_value': 12,
                #          'end_value': 42,
                #          't1': 5,
                #          't2': 7,
                #          'length': 40}
                # func = getattr(self.linear_transition, f)
                param = self.get_param_normal('linear_transition', mean_profiles[key])
                func = self.linear_transition
                # all_value.append(func(*param))

            elif f == 'exponential_form':
                # param = [12, 42, 5, 10, 20 + 20]
                # param = {'start_value': 12,
                #          'end_value': 42,
                #          't1': 5,
                #          'b': 10,
                #          'length': 40}
                # func = getattr(self.exponential_form, f)

                param = self.get_param_normal('exponential_form', mean_profiles[key])
                func = self.exponential_form
                # all_value.append(func(*param))

            elif f == 'impulse_like':
                # param = [12, 24, 11, 12, 30 + 20]
                # param = {'start_value': 12,
                #          'peak_value': 24,
                #          'end_value': 11,
                #          't1': 12,
                #          'length': 50}
                # func = getattr(self.impulse_like, f)

                param = self.get_param_normal('impulse_like', mean_profiles[key])
                func = self.impulse_like
                # all_value.append(func(*param))

            elif f == 'rectangular_pulse':
                # param = [23, 14, 53, 8, 23, 30 + 20]
                # param = {'start_value': 23,
                #          'high_value': 14,
                #          'end_value': 53,
                #          't1': 7,
                #          't2': 23,
                #          'length': 50}
                # func = getattr(self.rectangular_pulse, f)

                param = self.get_param_normal('rectangular_pulse', mean_profiles[key])
                func = self.rectangular_pulse
                # all_value.append(func(*param))

            elif f == 'trapezoidal':
                # param = [42, 52, 12, 7, 10, 14, 16, 20 + 20]
                # param = {'start_value': 42,
                #          'high_value': 52,
                #          'end_value': 12,
                #          't1': 7,
                #          't2': 10,
                #          't3': 14,
                #          't4': 16,
                #          'length': 40}
                # func = getattr(self.trapezoidal, f)

                param = self.get_param_normal('trapezoidal', mean_profiles[key])
                func = self.trapezoidal
                # all_value.append(func(*param))

            elif f == 'triangular':
                # param = [23, 12, 32, 1, 5, 9, 10 + 20]
                # param = {'start_value': 23,
                #          'peak_value': 12,
                #          'end_value': 32,
                #          't1': 1,
                #          't2': 5,
                #          't3': 9,
                #          'length': 30}
                # func = getattr(self.triangular, f)

                param = self.get_param_normal('triangular', mean_profiles[key])
                func = self.triangular
                # all_value.append(func(*param))

            elif f == 'exponential_pulse':
                # param = [0, 23, 9, 3, 6, 10, 10, 20 + 20]
                # param = {'start_value': 0,
                #          'high_value': 23,
                #          'end_value': 9,
                #          't1': 3,
                #          't2': 6,
                #          'b': 10,
                #          'c': 10,
                #          'length': 40}
                # func = getattr(self.exponential_pulse, f)

                param = self.get_param_normal('exponential_pulse', mean_profiles[key])
                func = self.exponential_pulse
                # all_value.append(func(*param))

        #     # start & end value check logic
        #     if key == 0:
        #         start_value = param['start_value']
        #     if key == len(function) - 1:
        #         if 'end_value' in param.keys():
        #             param['end_value'] = start_value
        #         else:
        #             param['start_value'] = start_value
        #
        #     # add jitter
        #     if jitter:
        #         param['jitter'] = True
        #
            all_value = func(**param)
        #
        #     # start & end value check logic
        #     if key > 0:
        #         check_start_value = param['start_value']
        #         if int(check_end_value) != int(check_start_value):
        #             all_value = self.fill_step_to_step(all_value, check_end_value, check_start_value)
        #         else:
        #             pass
        #
        #     check_end_value = param['end_value']
        #     # print(f"key = {key}\nparam[start]: {param['start_value']}\nparam[end]: {param['end_value']}\ncheck_end_value: {check_end_value}\ncheck_start_value: {check_start_value}")
        #     #print(param)
            step.append(all_value)
            params.append(param)
        #
        # if variation:
        #     # select defect step index
        #     step_num = np.random.poisson(2) + 1
        #     if len(step) <= step_num:
        #         step_num = 2
        #     selected_defect_step_idx = np.random.choice(len(step), step_num, replace=False)
        #
        #     # select defect
        #     defect_list = np.random.choice(DEFECT_LIST, len(selected_defect_step_idx))
        #
        #     defect_pair = {}
        #     for key, idx in enumerate(selected_defect_step_idx):
        #         defect_pair.update({idx: defect_list[key]})
        #     # print(defect_pair)
        #
        #     for step_idx, defect in defect_pair.items():
        #         func = getattr(self, defect)
        #         step = func(step, step_idx)

        #return self.make_trace_format(*step)
        return self.combine_traces(*step), params

    def defined_function_call(self, function):
        # for key, f in enumerate(function):
        f = function

        if f == 'cloud_pulse':
            param = [30, 4, 20 + 20]
            # func = getattr(self.cloud_pulse, f)
            func = self.cloud_pulse
            # all_value.append(func(*param))

        elif f == 'constant':
            param = [10, 20 + 20]
            param = {'start_value': 10,
                     'end_value': 10,
                     'length': 40}
            # func = getattr(self.constant, f)
            func = self.constant
            # all_value.append(func(*param))

        elif f == 'step_like':
            param = [12, 34, 6, 10 + 20]
            param = {'start_value': 12,
                     'end_value': 34,
                     't1': 6,
                     'length': 30}
            # func = getattr(self.step_like, f)
            func = self.step_like
            # all_value.append(func(*param))

        elif f == 'linear_transition':
            param = [12, 42, 5, 7, 20 + 20]
            param = {'start_value': 12,
                     'end_value': 42,
                     't1': 5,
                     't2': 7,
                     'length': 40}
            # func = getattr(self.linear_transition, f)
            func = self.linear_transition
            # all_value.append(func(*param))

        elif f == 'exponential_form':
            param = [12, 42, 5, 10, 20 + 20]
            param = {'start_value': 12,
                     'end_value': 42,
                     't1': 5,
                     'b': 10,
                     'length': 40}
            # func = getattr(self.exponential_form, f)
            func = self.exponential_form
            # all_value.append(func(*param))

        elif f == 'impulse_like':
            param = [12, 24, 11, 12, 30 + 20]
            param = {'start_value': 12,
                     'peak_value': 24,
                     'end_value': 11,
                     't1': 12,
                     'length': 50}
            # func = getattr(self.impulse_like, f)
            func = self.impulse_like
            # all_value.append(func(*param))

        elif f == 'rectangular_pulse':
            param = [23, 14, 53, 8, 23, 30 + 20]
            param = {'start_value': 23,
                     'high_value': 14,
                     'end_value': 53,
                     't1': 7,
                     't2': 23,
                     'length': 50}
            # func = getattr(self.rectangular_pulse, f)
            func = self.rectangular_pulse
            # all_value.append(func(*param))

        elif f == 'trapezoidal':
            param = [42, 52, 12, 7, 10, 14, 16, 20 + 20]
            param = {'start_value': 42,
                     'high_value': 52,
                     'end_value': 12,
                     't1': 7,
                     't2': 10,
                     't3': 14,
                     't4': 16,
                     'length': 40}
            # func = getattr(self.trapezoidal, f)
            func = self.trapezoidal
            # all_value.append(func(*param))

        elif f == 'triangular':
            param = [23, 12, 32, 1, 5, 9, 10 + 20]
            param = {'start_value': 23,
                     'peak_value': 12,
                     'end_value': 32,
                     't1': 1,
                     't2': 5,
                     't3': 9,
                     'length': 30}
            # func = getattr(self.triangular, f)
            func = self.triangular
            # all_value.append(func(*param))

        elif f == 'exponential_pulse':
            param = [0, 23, 9, 3, 6, 10, 10, 20 + 20]
            param = {'start_value': 0,
                     'high_value': 23,
                     'end_value': 9,
                     't1': 3,
                     't2': 6,
                     'b': 10,
                     'c': 10,
                     'length': 40}
            # func = getattr(self.exponential_pulse, f)
            func = self.exponential_pulse
            # all_value.append(func(*param)).25
        return func, param

    def automated_generation(self, function, jitter=False, variation=False):

        # function = super().sample_function_list(step_num)
        step = []
        start_value = -1

        check_start_value = -1
        check_end_value = -1
        for key, f in enumerate(function):
            func, param = self.defined_function_call(f)

            # start & end value check logic
            if key == 0:
                start_value = param['start_value']
            if key == len(function) - 1:
                if 'end_value' in param.keys():
                    param['end_value'] = start_value
                else:
                    param['start_value'] = start_value

            # add jitter
            if jitter:
                param['jitter'] = True

            all_value = func(**param)

            # start & end value check logic
            if key > 0:
                check_start_value = param['start_value']
                if int(check_end_value) != int(check_start_value):
                    all_value = self.fill_step_to_step(all_value, check_end_value, check_start_value)
                else:
                    pass

            check_end_value = param['end_value']
            # print(f"key = {key}\nparam[start]: {param['start_value']}\nparam[end]: {param['end_value']}\ncheck_end_value: {check_end_value}\ncheck_start_value: {check_start_value}")
            print(param)
            step.append(all_value)

        if variation:
            # select defect step index
            step_num = np.random.poisson(2) + 1
            if len(step) <= step_num:
                step_num = 2
            selected_defect_step_idx = np.random.choice(len(step), step_num, replace=False)

            # select defect
            defect_list = np.random.choice(DEFECT_LIST, len(selected_defect_step_idx))

            defect_pair = {}
            for key, idx in enumerate(selected_defect_step_idx):
                defect_pair.update({idx: defect_list[key]})
            # print(defect_pair)

            for step_idx, defect in defect_pair.items():
                func = getattr(self, defect)
                step = func(step, step_idx)

        return self.make_trace_format(*step)

    def add_drifting(self, steps, step_idx):

        for num in range(self.n):
            # function_type = np.random.choice(['constant', 'step_like'])
            # growth
            function_type = 'exponential_form'
            new_shape = getattr(self, function_type)
            param = {'start_value': 12,
                     'end_value': 20,
                     't1': 5,
                     'b': 10,
                     'length': 40,
                     'jitter': True,
                     'sampling': False}
            steps[step_idx][num] = new_shape(**param)
        return steps

    def add_different(self, steps, step_idx):

        for num in range(self.n):
            function_type = np.random.choice(['constant', 'step_like'])
            new_shape = getattr(self, function_type)
            if function_type == 'constant':
                param = {'start_value': 10,
                         'end_value': 10,
                         'length': 40,
                         'jitter': True,
                         'sampling': False}
            else:
                param = {'start_value': 12,
                         'end_value': 34,
                         't1': 6,
                         'length': 30,
                         'jitter': True,
                         'sampling': False}

            steps[step_idx][num] = new_shape(**param)

        return steps

    def add_stability(self, steps, step_idx):

        return steps
        # for num in range(self.n):
        #     target = steps[step_idx][num]
        #     target_len = steps[step_idx][num].shape[0]
        #     stability_idx = np.random.choice(target_len - 2) + 1  # position of spike
        #
        #     t = np.linspace(-10, 10, 100)
        #     f = 10 * np.sin(3 * t) + (3 * t + 10)
        #     stability_value = f
        #     target = stability_value
        #     steps[step_idx][num] = target
        #
        # return steps

    def add_spike(self, steps, step_idx, noise_min=50, noise_max=80, wafers_lambda=1, steps_lambda=1):

        for num in range(self.n):
            target = steps[step_idx][num]
            target_len = steps[step_idx][num].shape[0]
            spike_idx = np.random.choice(target_len - 2) + 1  # position of spike

            spike_value = self.spike(noise_max, noise_min)
            target[spike_idx] = spike_value
            steps[step_idx][num] = target

        return steps

    def fill_step_to_step(self, steps, end_value, start_value):
        for key, wafer in enumerate(steps):
            fill_point_num = np.random.randint(low=1,
                                               high=5 + 1)  # bridge the gap between step[n] and step[n+1] using 1 ~ 5 points
            values = np.sort([end_value, start_value])

            try:
                filled_point_values = np.random.randint(values[0], values[1], size=fill_point_num)
            except:
                print(values[0], values[1])
                print(int(values[0]) != int(values[1]))
                filled_point_values = np.random.randint(values[0], values[1], size=fill_point_num)

            if end_value > start_value:
                filled_point_values = np.sort(filled_point_values)[::-1]
            else:  # end_value < start_value:
                filled_point_values = np.sort(filled_point_values)

            steps[key] = np.append(wafer, filled_point_values)
            # print(wafer)
            # print('\n')
        return steps

    def gaussian_noise(self, signal_length, mean=0, std=None):
        if std == None:
            std = self.global_range * 0.01

        return np.random.randn(signal_length) * std + mean

    """jiter mean time difference between recorded signal"""

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
                # if key == 0:
                # print(t, time_difference)
                # time_difference = t
                new_time.append(time_difference)
            # check if t1 > t2 ==> change two values
            new_time.sort()

            # check: length must larger than time value
            # print(f"jitter logic\nnew_time: {new_time}\nnew_length: {new_length}\nlength: {length}")
            if max(new_time) >= new_length:
                return length, time

            # check duplicated time
            if len(set(new_time)) != len(time):
                return length, time

            return new_length, new_time

        # if time parameter don't exist then only return jitter length value
        else:
            return new_length

    def spike(self, maximum, minimum):
        return np.random.uniform(minimum, maximum)

    def data_to_list_format(self, *args):
        data_list = []
        for n in range(self.n):
            data = []
            for key, step in enumerate(args):
                data.append(step[n])
            data_list.append(data)

        return data_list

    def make_trace_format(self, *args):
        # data = self.combine_traces(args)

        lot_id = 'LOT_ID'
        wafer_id = 'WAFER_ID'
        process = 'PROCESS'
        process_step = 'PROCESS_STEP'
        recipe = 'RECIPE'
        recipe_step = 'RECIPE_STEP'
        parameter_name = 'PARAMETER_NAME'
        parameter_value = 'PARAMETER_VALUE'
        time = 'TIME'
        col = ['LOT_ID', 'WAFER_ID', 'PROCESS', 'PROCESS_STEP', 'RECIPE', 'RECIPE_STEP', 'PARAMETER_NAME',
               'PARAMETER_VALUE', 'TIME']

        # data = []
        # for n in range(self.n):
        #     df = []
        #     for key, step in enumerate(args):

        data = []
        for n in range(self.n):
            df = []
            for key, step in enumerate(args):
                step_temp = pd.DataFrame([], columns=col)
                step_temp['PARAMETER_VALUE'] = step[n]
                step_temp['RECIPE_STEP'] = key + 1
                df.append(step_temp)

            df = pd.concat(df)
            df = df.reset_index(drop=True)
            df['LOT_ID'] = 'lot_a'
            df['WAFER_ID'] = f'wafer{n + 1}'
            df['PROCESS'] = 'process'
            df['PROCESS_STEP'] = 'process_step'
            df['RECIPE'] = 'recipe'
            df['PARAMETER_NAME'] = 'parameter_name'
            data.append(df)

        data = pd.concat(data)
        data['TIME'] = pd.date_range("2018-01-01", periods=data.shape[0], freq="S")
        temp_list = []
        for wafer in data['WAFER_ID'].unique():
            temp_list.append(data[data['WAFER_ID'] == wafer])

        return temp_list

    def substep_aggregation(self, *args):
        step = []
        for n in range(self.n):
            temp = []
            for sub_step in args:
                temp.append(sub_step[n])
            temp = np.concatenate(temp, axis=0)
            step.append(temp)
        return step

    def combine_traces(self, *args):
        data = []
        for n in range(self.n):
            temp = []
            for wafer in args:
                temp.append(wafer[n])
            temp = np.concatenate(temp, axis=0)
            data.append(temp)

        return data

    def cloud_pulse(self, start_value, std, length, jitter=False):
        cloud = []
        for n in range(self.n):
            new_length = length
            if jitter or self.jitter_bool:
                new_length = self.jitter(length)
            cloud.append(self.gaussian_noise(new_length, mean=start_value, std=std))
        return cloud

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
            new_peak_value = self.spike(maximum=peak_value * 1, minimum=(0.99 * peak_value + 0.01 * end_value))
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

    # steps: [step1, step2, step3 ...]. each step contain N number of different wafer's individual step
    def wafers_add_spike(self, *steps, step_idx, noise_min=50, noise_max=80, wafers_lambda=1, steps_lambda=1):
        # wafer_num = np.random.poisson(wafers_lambda) + 1

        # if self.n < wafer_num:
        #     wafer_num = self.n
        # wafer_idx = np.random.choice(self.n, wafer_num, replace=False)

        # this is for random step.
        # step_num = np.random.poisson(steps_lambda) + 1
        # if len(steps) <= step_num:
        #     step_num = 2
        # step_idx = np.random.choice(len(steps), step_num, replace=False)

        for s_idx in step_idx:
            for num in range(self.n):
                target = steps[s_idx][num]
                target_len = steps[s_idx][num].shape[0]
                spike_idx = np.random.choice(target_len - 2) + 1  # position of spike

                spike_value = self.spike(noise_max, noise_min)
                target[spike_idx] = spike_value
                steps[s_idx][num] = target
        # for s_idx in step_idx:
        #     for w_idx in wafer_idx:
        #         target = steps[s_idx][w_idx]
        #         target_len = steps[s_idx][w_idx].shape[0]
        #         spike_idx = np.random.choice(target_len - 2) + 1
        #
        #         # target_max = np.max(target)
        #         # target_min = np.min(target)
        #         spike_value = self.spike(noise_max, noise_min)
        #         target[spike_idx] = spike_value
        #
        #         steps[s_idx][w_idx] = target

        return steps

    def wafers_add_shape_noise(self, steps, step_idx=[1], wafer_idx=[2, 5], noise_max=1.5, noise_duration=0.3):
        # wafer_num = np.random.poisson(wafers_lambda) + 1
        # wafer_idx = np.random.choice(self.n, wafer_num, replace=False)

        for s_idx in step_idx:
            for w_idx in wafer_idx:
                target = steps[s_idx][w_idx]
                target_len = steps[s_idx][w_idx].shape[0]

                start_idx = int(target_len * (1 - noise_duration - 0.05))
                noise_len = int(target_len * noise_duration)

                xs = np.linspace(0, np.pi / 2, noise_len)
                ys = np.cos(xs) * (np.random.rand(noise_len) * 2 - 1) * noise_max

                target[start_idx: start_idx + noise_len] += ys
                steps[s_idx][w_idx] = target

        return steps


class TraceDataGenerationCheck(TraceDataGenerationCommon):
    def gaussian_noise(self, signal_length, mean=0, std=1):
        return np.random.randn(signal_length) * std + mean

    """jiter mean time difference between recorded signal"""

    def jitter(self, length, *time):
        pass

    def spike(self, maximum, minimum):
        return np.random.uniform(minimum, maximum)

    def combine_traces(self, *args):
        all_recipe_step = np.concatenate(args, axis=0)
        # return pd.DataFrame(all_recipe_step)
        return all_recipe_step

    def constant(self, start_value, end_value, length):
        return np.ones(length) * start_value

    def step_like(self, start_value, end_value, t1, length):
        t1 += 1
        start = np.ones(t1) * start_value
        end = np.ones(length - t1) * end_value

        signal = np.concatenate([start, end])

        return signal

    def linear_transition(self, start_value, end_value, t1, t2, length):
        # t1 += 1
        # t2 += 1
        transition_length = t2 - t1 + 1
        start = np.ones(t1) * start_value
        t = np.arange(0, transition_length)
        m = (end_value - start_value) / (transition_length - 1)
        transition = start_value + m * t
        end = np.ones(length - t2 - 1) * end_value

        return np.concatenate([start, transition, end])

    def exponential_form(self, start_value, end_value, t1, b, length):
        t1 += 1
        start = np.ones(t1) * start_value
        t = np.arange(t1, length)
        return np.append(start, start_value + (end_value - start_value) * (1 - np.exp(-t / b)))

    def impulse_like(self, start_value, peak_value, end_value, t1, length):
        start = np.ones(t1) * start_value
        peak = np.ones(1) * peak_value
        end = np.ones(length - t1 - 1) * end_value

        return np.concatenate([start, peak, end])

    def rectangular_pulse(self, start_value, high_value, end_value, t1, t2, length):
        start = np.ones(t1) * start_value
        high = np.ones(t2 - t1 + 1) * high_value
        end = np.ones(length - t2 - 1) * end_value

        return np.concatenate([start, high, end])

    def trapezoidal(self, start_value, high_value, end_value, t1, t2, t3, t4, length):
        start = np.ones(t1) * start_value
        up_trend = self.linear_transition(start_value=start_value, end_value=high_value, t1=0, t2=t2 - t1,
                                          length=(t2 - t1 + 1))
        up_stable = np.ones(t3 - t2 - 1) * high_value
        down_trend = self.linear_transition(start_value=high_value, end_value=end_value, t1=0, t2=t4 - t3,
                                            length=(t4 - t3 + 1))
        end = np.ones(length - t4 - 1) * end_value

        return np.concatenate([start, up_trend, up_stable, down_trend, end])

    def triangular(self, start_value, peak_value, end_value, t1, t2, t3, length):
        start = np.ones(t1)
        up_trend = start_value + np.arange(t2 - t1) * (peak_value - start_value) / (t2 - t1)
        peak = np.array([peak_value])
        down_trend = peak_value - np.arange(1, t3 - t2 + 1) * (peak_value - end_value) / (t3 - t2)
        end = np.ones(length - t3 - 1) * end_value

        return np.concatenate([start, up_trend, peak, down_trend, end])

    def exponential_pulse(self, start_value, high_value, end_value, t1, t2, b, c, length):
        form1 = self.exponential_form(start_value, high_value, t1, b, t2 + 1)
        t = np.arange(1, length - t2)
        form2 = end_value + (high_value - end_value) * (1 - np.exp(-(t - t1 + t2) / b)) * np.exp(-t / c)

        return np.concatenate([form1, form2])

    # def exponential_down(self, start_value, end_value, signal_length, factor):
    #     t = np.arange(signal_length)
    #     #form2 = end_value + (start_value - end_value) * (1 - np.exp(-(t2 - t1[0]) / factor1)) * np.exp(-(t2 - t1[-1]) / factor2)
    #     form2 = end_value + (start_value - end_value) * np.exp(-t / factor)
    #
    #     return form2

    def cloud_pulse(self, start_value, std, length):
        return self.gaussian_noise(length, mean=start_value, std=std)

    def automated_generation(self, step_num):
        all_value = []
        function = super().sample_function_list(step_num)
        for f in function:
            if f == 'cloud_pulse':
                param = [30, 4, 20 + 20]
                # func = getattr(generator, f)
                func = self.cloud_pulse
                all_value.append(func(*param))

            elif f == 'constant':
                param = [10, 20 + 20]
                # func = getattr(generator, f)
                func = self.constant
                all_value.append(func(*param))

            elif f == 'step_like':
                param = [12, 34, 6, 10 + 20]
                # func = getattr(generator, f)
                func = self.step_like
                all_value.append(func(*param))

            elif f == 'linear_transition':
                param = [12, 42, 5, 7, 20 + 20]
                # func = getattr(generator, f)
                func = self.linear_transition
                all_value.append(func(*param))

            elif f == 'exponential_form':
                param = [12, 42, 5, 10, 20 + 20]
                # func = getattr(generator, f)
                func = self.exponential_form
                all_value.append(func(*param))

            elif f == 'impulse_like':
                param = [12, 24, 11, 12, 30 + 20]
                # func = getattr(generator, f)
                func = self.impulse_like
                all_value.append(func(*param))

            elif f == 'rectangular_pulse':
                param = [23, 14, 53, 8, 23, 30 + 20]
                # func = getattr(generator, f)
                func = self.rectangular_pulse
                all_value.append(func(*param))

            elif f == 'trapezoidal':
                param = [42, 52, 12, 7, 10, 14, 16, 20 + 20]
                # func = getattr(generator, f)
                func = self.trapezoidal
                all_value.append(func(*param))

            elif f == 'triangular':
                param = [23, 12, 32, 1, 5, 9, 10 + 20]
                # func = getattr(generator, f)
                func = self.triangular
                all_value.append(func(*param))

            elif f == 'exponential_pulse':
                param = [0, 23, 9, 3, 6, 10, 10, 20 + 20]
                # func = getattr(generator, f)
                func = self.exponential_pulse
                all_value.append(func(*param))

        return self.combine_traces(*all_value)