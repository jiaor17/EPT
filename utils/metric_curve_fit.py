#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
from scipy.optimize import curve_fit


def log_curve(x, a, b, c):
    return a * np.log((x + c)) + b


def pred_metric(x_past, y_past, x_pred):
    x_past, y_past = np.array(x_past), np.array(y_past)
    start_value = np.min(x_past) - 1
    x_past -= start_value

    func = log_curve
    param, _ = curve_fit(func, x_past, y_past)

    y_past_pred = func(x_past, *param)
    rel_error = np.mean(np.abs((y_past_pred - y_past) / y_past))

    x_pred = np.array(x_pred)
    x_pred -= start_value

    return func(x_pred, *param), rel_error, param


if __name__ == '__main__':
    y_pred, rel_error, param = pred_metric(
        [0, 1, 2, 3, 4, 10, 11],
        [0.1017, 0.1357, 0.1426, 0.1716, 0.1822, 0.2929, 0.3055],
        [12, 20, 30, 50]
    )
    y_pred, rel_error, param = pred_metric(
        [5, 11, 15, 20, 90, 154, 371, 446],
        [0.1615, 0.1247, 0.1186, 0.1065, 0.05551, 0.04795, 0.04152, 0.04273],
        # [56, 154, 446, 1000],
        1000
    )
    print(y_pred)
    print(rel_error)
    print(param)