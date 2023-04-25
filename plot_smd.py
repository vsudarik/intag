import numpy as np
import matplotlib.pyplot as plt
import os 
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pdb
from typing import Callable

class Measurement():

    def __init__(self, file_name, path = ''):
        self.load_file(file_name, path)
        self.name = file_name
        self.reverse_check()

    def load_file(self, file_name, path):
        data = np.loadtxt(path + '/' + file_name, skiprows=1) 

        self.step_array = data[:, 0]  
        self.pmf_array = data[:, 3] 
        self.z_array = data[:, 2] 

    def reverse_check(self):
        self.calculate_mnk()
        if self.coefficients[0] > 0:
            self.isreverse = False
        else:
            self.isreverse = True

    def calculate_mnk(self):
        self.coefficients = np.polyfit(self.step_array,self.z_array , 1)

    def reverse(self):
        #self.step_array = np.flip(self.step_array)
        self.pmf_array = np.flip(self.pmf_array)
        self.z_array = np.flip(self.z_array) 
        self.reverse_check()

    def form(self, reference_coefficients):
        delta = (self.coefficients[1] -  reference_coefficients[1])/self.coefficients[0]
        self.step_array = self.step_array + delta   
        self.calculate_mnk()

class Comparator:
    def __init__(self, func: Callable[[Measurement, Measurement], Measurement]):
        self.func = func

class MeasurementComparator:
    MIN_BOTTOM = 'MIN_BOTTOM'

COMPARATORS = {
    MeasurementComparator.MIN_BOTTOM: Comparator(
        lambda current, measure: current if current.z_array[0] <  measure.z_array[0] else measure
    )
}

class Measurements_Storage():
    def __init__(self, path):
        self.load_all_data(path)
        self.isCheckReverse = False
        self.isCheckForward = True
        
    def load_all_data(self, path):
        with os.scandir(path) as entries:
            self.measurement_list =[]
            for entry in entries:
                print(entry.name)  
                measure = Measurement(entry.name, path)
                self.measurement_list.append(measure)

    def find(self, comparator: Comparator):
        current_value = next(iter(self)) 
        for measure in self:
            current_value = comparator.func(current_value, measure)
        return current_value


    def normalize(self):
        measure_reference = self.find(COMPARATORS[MeasurementComparator.MIN_BOTTOM])
        for measure in self:
            measure.form(measure_reference.coefficients)
        

    def __iter__(self):
        for i in self.measurement_list:
            if (i.isreverse and self.isCheckReverse) or  (not i.isreverse and self.isCheckForward):
                yield i 


storage = Measurements_Storage('data/')
storage.normalize()


# Создание графиков
fig, ax = plt.subplots(1, 3, figsize=(12, 6))

def plot_measures():
    for measure in storage:
            ax[0].plot(measure.step_array, measure.pmf_array , label=measure.name)
            ax[1].plot(measure.step_array, measure.z_array , label=measure.name)
            ax[2].plot(measure.z_array, measure.pmf_array , label=measure.name)
        
def plot_one_measure(number):
    measure =  next(iter(storage))
    ax[0].plot(measure.step_array, measure.pmf_array , label=measure.name)
    ax[1].plot(measure.step_array, measure.z_array , label=measure.name)

#plot_one_measure(0)
plot_measures()
ax[0].set_xlabel("step")
ax[0].set_ylabel("PMF")
ax[0].legend()
ax[1].set_xlabel("step")
ax[1].set_ylabel("Z")
ax[1].legend()
ax[2].set_xlabel("Z")
ax[2].set_ylabel("PMF")
ax[2].legend()
# Отображение графиков
plt.show()


