#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# TODO: Définissez vos fonctions ici (il en manque quelques unes)


def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return np.array([(np.sqrt(c[0] ** 2 + c[1] ** 2), np.arctan2(c[1], c[0])) for c in cartesian_coordinates])


def find_closest_index(values: np.ndarray, number: float) -> int:
    return np.abs(values - number).argmin()


def create_plot():
    x = np.linspace(-1, 1, num=250)
    y = x ** 2 * np.sin(1 / x ** 2) + x

    plt.scatter(x, y)
    plt.xlim((-1, 1))
    plt.scatter(x, y, label="nuage de points")
    plt.plot(x, y, label="ligne", color="b")
    plt.title("Titre")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()


def estimate_pi(iteration: int = 5000) -> float:
    x_inside_dots = []
    y_inside_dots = []
    x_outside_dots = []
    y_outside_dots = []
    for i in range(iteration):
        x = np.random.random()
        y = np.random.random()
        if np.sqrt(x ** 2 + y ** 2) < 1:
            x_inside_dots.append(x)
            y_inside_dots.append(y)
        else:
            x_outside_dots.append(x)
            y_outside_dots.append(y)

    plt.scatter(x_inside_dots, y_inside_dots, label="inside dot")
    plt.scatter(x_outside_dots, y_outside_dots, label="outside dot")
    plt.xlabel("X")
    plt.xlabel("Y")
    plt.title("Calcul Monte Carlo")
    plt.show()

    return float(len(x_inside_dots)) / iteration * 4


def integrate_and_plt() -> tuple:
    result_inf = integrate.quad(lambda x: np.exp(-x ** 2), -np.inf, np.inf)
    x = np.arange(-4, 4, 0.1)
    y = [integrate.quad(lambda x: np.exp(-x ** 2)[0], 0, value) for value in x]

    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    # print(linear_values())
    # print(coordinate_conversion([(0, 0), (1, 1), (5, 8)]))
    # print(find_closest_index(np.array([0, 5, 10, 12, 8]), 10.5))
    # create_plot()
    # print(estimate_pi())
    integrate_and_plt()
