import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.linear_model import LinearRegression


class Analysis:

    def __init__(self, data=None, feature='feature', target='target', lot_col='Carrier ID', wafer_col='Wafer ID'):
        self.data = data
        self.feature = feature
        self.target = target
        self.lot_col = lot_col
        self.wafer_col = wafer_col

        if data is None:
            self.data = self.make_toy_dataset()
        
    def make_toy_dataset(self):
        # Generate toy example dataset
        tool_chambers = [
            "EP00018/PM-1", "EP00018/PM-1", "EP00018/PM-2",
            "EP00019/PM-3", "EP00019/PM-3", "EP00019/PM-4",
            "EP00020/PM-5", "EP00020/PM-5", "EP00020/PM-5"
        ]

        # Generate random features and targets
        #features = [round(random.uniform(5.0, 15.0), 2) for _ in range(len(tool_chambers))]
        #targets = [round(random.uniform(0.1, 1.0), 2) for _ in range(len(tool_chambers))]
        features = np.arange(9)
        targets = np.arange(10, 19)

        # Generate Carrier IDs and Wafer IDs
        #carrier_ids = [f"PFP0073" for _ in range(len(tool_chambers))]
        carrier_ids = ['PFP0073', 'PFP0073', 'PFP0074',
                       'PFP0075', 'PFP0075', 'PFP0076',
                       'PFP0077', 'PFP0077', 'PFP0077']
        wafer_ids = [f"{carrier_ids[i]}.{i+1}" for i in range(len(carrier_ids)) for j in range(1)]

        # Create the DataFrame
        data = {
            "Tool/Chamber": tool_chambers,
            "feature": features,
            "target": targets,
            "Carrier ID": carrier_ids,
            "Wafer ID": wafer_ids
        }

        df = pd.DataFrame(data)
        df['Start time'] = pd.date_range("2024-01-01", periods=df.shape[0], freq="0.1S")
        return df

    def create_lot_col(self, data):
        data['lot_identifier'] = data['Tool/Chamber'] + '/' + data['Carrier ID']

        data_lot = data.groupby('lot_identifier').agg({
            self.target: 'mean',
            self.feature: 'mean'
            }).reset_index().rename(columns={self.target:'lot_y', self.feature:'lot_x'})
        
        return data_lot

    def create_within_lot_col(self, data, data_lot):
        data = data.merge(data_lot, on='lot_identifier')
        data['wlot_x'] = data[self.feature] - data['lot_x']
        data['wlot_y'] = data[self.target] - data['lot_y']

        return data
    
    def correlation(self, data, feature, target):
        x = data[feature].values.reshape(-1,1)
        y = data[target].values.reshape(-1,1)

        coeff, pval = stats.pearsonr(data[feature], data[target])
        
        model = LinearRegression()
        model.fit(x, y)
        
        gradient = model.coef_[0]
        intercept = model.intercept_
        r2 = model.score(x,y)
        result = {'corr': coeff,
                  'pval': pval,
                  'slope': gradient[0],
                  'intercept': intercept,
                  'r2': r2}
    
        return result

    def calculate_score(self, data, data_lot, corr_wf, corr_lot, corr_wlot):
        
        pval_lot = corr_lot['pval']
        slope_lot = corr_lot['slope']
        r2_lot = corr_lot['r2']

        pval_wlot = corr_wlot['pval']
        slope_wlot = corr_wlot['slope']
        r2_wlot = corr_wlot['r2']

        pval_wf = corr_wf['pval']
        slope_wf = corr_wf['slope']        
        r2_wf = corr_wf['r2']
        
        sigma_lot = np.sqrt(data_lot['lot_x'].var())
        sigma_wlot = np.sqrt(data['wlot_x'].var())

        pval_thres = 0.2
        wratio = 2 * sigma_wlot / (sigma_lot + 2*sigma_wlot)
        
        cond1 = slope_wlot * slope_lot > 0
        cond2 = pval_lot < pval_thres
        cond3 = pval_wlot < pval_thres
        if cond1 and cond2 and cond3:
            apval_lot = min(1, pval_lot / pval_thres)
            apval_wlot = min(1, pval_wlot / pval_thres)
            awratio = min(2/3, max(1/3, wratio))
            sig_score = 1 - (apval_lot * (1-awratio) + apval_wlot * awratio)
            orig_score = 1 - max(pval_lot, pval_wlot)

        else:
            sig_score = 0
            apval_lot = min(1, pval_lot / 0.1)
            apval_wlot = min(1, pval_wlot / 0.1)
            
            if wratio < 0.2:
                apval_lot = min(1, (1-apval_lot) * 2 * r2_lot)
            if wratio > 0.8:
                sig_score = min(1, (1-apval_wlot) * 5 * r2_lot)
            if pval_lot < 0.02 or pval_wlot < 0.02:
                orig_score = 1 - max(pval_lot, pval_wlot)

        res = {
            'parameter': [self.feature],
            'pval': [pval_wf],
            'r2': [r2_wf],
            'slope': [slope_wf],
            'pval_l': [pval_lot],
            'r2_l': [r2_lot],
            'slope_l': [slope_lot],
            'pval_wl': [pval_wlot],
            'r2_wl': [r2_wlot],
            'slope_wl': [slope_wlot],
            'class': [''],
            'score': [sig_score]
        }
    
        return res

               
    def create_data_norm_format(self, data, norm_col, statistics='mean'):
        data_norm = data.groupby(norm_col).agg({
            self.target : statistics,
            self.feature : statistics
        }).reset_index().rename(columns={self.feature:'norm_x', self.target:'norm_y'})
        
        data = data.merge(data_norm, on=norm_col)

        data[self.feature] -= data['norm_x']
        data[self.target] -= data['norm_y']

        return data
        
    def make_analysis_form(self, norm_col=None):
        data = self.data.copy()
        
        if norm_col is not None:
            data = self.create_data_norm_format(data, norm_col)
            
        data_lot = self.create_lot_col(data)
        data = self.create_within_lot_col(data, data_lot)
        
        return data, data_lot
        
    def get_score(self):
        data, data_lot = self.make_analysis_form()

        corr_wf = self.correlation(data, self.feature, self.target)
        corr_lot = self.correlation(data_lot, 'lot_x', 'lot_y')
        corr_wl = self.correlation(data, 'wlot_x', 'wlot_y')

        score = self.calculate_score(data, data_lot, corr_wf, corr_lot, corr_wl)

        return pd.DataFrame(score)
        
    def get_color_map(self, color_by, color_type='gist_rainbow'):
        unique_group = self.data[color_by].unique()
        cmap = plt.colormaps[color_type]
        colors = cmap(np.linspace(0, 1, len(unique_group)))
        color_map = {group: color[i] for i, group in enumerate(unique_group)}
        
        return color_map
        
    def draw_trend_chart(self, para, time_col, color_by, ax):
        color_map = self.get_color_map(color_by)
        for group, group_data in self.data.groupby(color_by):
            ax.plot(group_data[time_col], group_data[para],
            label=f'{group}', color=color_map[group], marker='o')
        
        ax.set_label(time_col)
        ax.set_ylabel(para)
        ax.legend()
        ax.grid()
        
        return ax
        
    def draw_chart(self, color_by, trend=False):
        data, data_lot - self.make_analysis_form()
        if trend:
            fig, ax1 = plt.subplots(1, 2, figsize=(18,3))
            self.draw_trend_chart(self.feature, 'start time', color_by, ax1[0])
            self.draw_trend_chart(self.target, 'start time', color_by, ax1[1])
        
        fig, ax2 = plt.subplots(1, 3, figsize=(21,5))
        self.draw_wafer_level_chart(data, color_by, ax=ax2[0])
        self.draw_lot_level_chart(data, color_by, ax=ax2[1])
        self.draw_within_lot_level_chart(data, color_by, ax=ax2[2])
        
        plt.show()
        
    def draw_corr_chart(self, data, feature, target, corr, color_by, level, ax):
        color_map = self.get_color_map(color_by)
        for group, group_data in data.groupby(color_by):
            ax.scatter(group_data[feature], group_data[target],
            s=40, alpha=0.7,
            color=color_map[group], marker='o')
            
        rsq = corr[rsq]
        pval = corr[pval]
        b = corr[slope]
        a = corr[intercept]
        
        xax, yax = data[feature] data[target]
        xseq = np.linspace(xax.min()*0.9, xax.max()*1.1)
        
        ax.plot(xseq, a+b*xseq, alpha=0.8, color='black', lw=3, linestyle='--')
        ax.grid()
        ax.set_title(f'{level}, level correlatuin')
        ax.set_xlabel(feature)
        ax.set_ylabel(target)
        ax.legend([f'p-value:{pval:.3f}\br2:{}]bslopeL{};'])
        
        return ax
        
    def draw_wafer_level_chart(self, data, color_by, ax):
        
        corr = self.corr(data, self.feature, self.target)
        ax = self.draw_corr_chart(data, self.feature, self.target, corr, color_by, 'wafer', ax)
        
        return ax
        
    def draw_lot_level_chart(data, color_by, ax):
        data = data.drop_duplicates(subset='lot_identifier', keep='first', ignore_index=True, inplace-True)
        corr=self.corr(data, lot_y, lot_x))
        ax = self.draw_corr_chart(data, lot_x_x, lot_y, corr, color_by, 'lot', ax)
        
        return ax
        
    def draw_within_lot_level_chart(self, data, color_by, ax):
        corr = (wlot_x, wot_y)
        ax = 
