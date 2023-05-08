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
        self.name = file_name.replace('.txt', '')
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

    def reverse_dict(self):
        if self.isreverse:
            self.pmf_array_straight = np.flip(self.pmf_array)
            self.z_array_straight = np.flip(self.z_array_straight) 
        else:
            self.pmf_array_straight = self.pmf_array

    def reverse(self):
        #self.step_array = np.flip(self.step_array)
        self.pmf_array = np.flip(self.pmf_array)
        self.z_array = np.flip(self.z_array) 
        self.reverse_check()

    def form(self, reference_coefficients):
        delta = (self.coefficients[1] -  reference_coefficients[1])/self.coefficients[0]
        self.step_array = self.step_array + delta   
        self.calculate_mnk()

    def calculate_straight_z(self):
        self.z_array_straight = self.step_array * self.coefficients[0] + self.coefficients[1]

    def fit(self, bottom_border, top_border):
        start_index = np.searchsorted(self.z_array_straight, bottom_border, side='left')
        end_index = np.searchsorted(self.z_array_straight, top_border, side='right')
        self.z_array_straight = self.z_array_straight[start_index:end_index]
        self.pmf_array_straight = self.pmf_array_straight[start_index:end_index]

    def __len__(self):
        return len(self.pmf_array_straight)

class AveragedMeasurement(Measurement):
    def __init__(self, z_array_straight, pmf_array_straight,  name ='average' ):
        self.z_array_straight = z_array_straight 
        self.pmf_array_straight = pmf_array_straight
        self.name = name


class Comparator:
    def __init__(self, func: Callable[[Measurement, Measurement], Measurement]):
        self.func = func

class MeasurementComparator:
    MIN_BOTTOM = 'MIN_BOTTOM'
    MAX_BOTTOM = 'MAX_BOTTOM'
    MAX_TOP = 'MAX_TOP'
    MAX_BOTTOM_STRAIGHT = 'MAX_BOTTOM_STRAIGHT'
    MIN_TOP_STRAIGHT = 'MIN_TOP_STRAIGHT'


COMPARATORS = {
    MeasurementComparator.MIN_BOTTOM: Comparator(
        lambda current, measure: current if current.z_array[0] <  measure.z_array[0] else measure
    ),
    MeasurementComparator.MAX_BOTTOM: Comparator(
        lambda current, measure: current if current.z_array[0] >  measure.z_array[0] else measure
    ),
    MeasurementComparator.MAX_TOP: Comparator(
        lambda current, measure: current if current.z_array[-1] >  measure.z_array[-1] else measure
    ),
    MeasurementComparator.MAX_BOTTOM_STRAIGHT: Comparator(
        lambda current, measure: current if current.z_array_straight[0] >  measure.z_array_straight[0] else measure 
        ),
    MeasurementComparator.MIN_TOP_STRAIGHT: Comparator(
        lambda current, measure: current if current.z_array_straight[-1] <  measure.z_array_straight[-1] else measure 
        )
}

class Measurements_Storage():
    def __init__(self, path):
        self.load_all_data(path)
        self.isCheckReverse = True
        self.isCheckForward = True
        
    def load_all_data(self, path):
        with os.scandir(path) as entries:
            self.measurement_dict ={}
            for entry in entries:
                print(entry.name)  
                measure = Measurement(entry.name, path)
                self.measurement_dict[measure.name] = measure 

    def find(self, comparator: Comparator):
        current_value = next(iter(self)) 
        for measure in self:
            current_value = comparator.func(current_value, measure)
        return current_value

    def fit(self):
        self.bottom_border =  self.find(COMPARATORS[MeasurementComparator.MAX_BOTTOM_STRAIGHT]).z_array_straight[0]
        self.top_border =  self.find(COMPARATORS[MeasurementComparator.MIN_TOP_STRAIGHT]).z_array_straight[-1]
        print(self.bottom_border, self.top_border) 
        for measure in self:
            measure.fit(self.bottom_border, self.top_border)

        

    def reverse(self):
        for measure in self:
            if measure.isreverse:
                measure.reverse()


    def normalize(self):
        for measure in self:
            measure.calculate_straight_z()
            measure.reverse_dict()

    def average(self, measure_list):
        number_of_steps = max([len(self[measure]) for measure in measure_list])
        new_z = np.linspace(self.top_border, self.bottom_border,number_of_steps ) 
        interp_mesures_list = [np.interp(new_z, self[measure].z_array_straight, self[measure].pmf_array_straight) for measure in measure_list]
        mean_array = np.add.reduce(interp_mesures_list)/len(interp_mesures_list) 
        new_avearage_measure =  AveragedMeasurement(new_z, mean_array) 
        self.measurement_dict[measure_list.name + '_average'] = new_avearage_measure
        measure_list.append(measure_list.name + '_average')


    

    def __iter__(self):
        for i in self.measurement_dict.values():
            if (i.isreverse and self.isCheckReverse) or  (not i.isreverse and self.isCheckForward):
                yield i 

    def __getitem__(self, index):
        return self.measurement_dict[index]



class Visualisator():
    def __init__(self, storage):
        self.storage = storage
        self.fig, self.ax = plt.subplots(1, 3, figsize=(12, 6))

    def set_components(self):
        self.ax[0].set_xlabel("step")
        self.ax[0].set_ylabel("PMF")
        self.ax[0].legend()
        self.ax[1].set_xlabel("step")
        self.ax[1].set_ylabel("Z")
        self.ax[1].legend()
        self.ax[2].set_xlabel("Z")
        self.ax[2].set_ylabel("PMF")
        self.ax[2].legend()

    def plot_one_measure(self, measure_name):
        measure =  self.storage[measure_name]
        try:
            self.ax[0].plot(measure.step_array, measure.pmf_array , label=measure.name)
            self.ax[1].plot(measure.step_array, measure.z_array , label=measure.name)
        except:
            pass
        self.ax[2].plot(measure.z_array_straight, measure.pmf_array_straight , label=measure.name)

    def plot_all(self):
        for measure in storage:
            self.plot_one_measure(measure.name)
        self.set_components()
        plt.show()
    
    def plot_from_list(self, measure_list):
        for measure_name in measure_list:
            self.plot_one_measure(measure_name)
        self.set_components()
        plt.show()

class IntagGroup():
    def __init__(self, name, info):
        self.storage = info
        self.name = name
    def __iter__(self):
        for tag_name in self.storage:
            yield tag_name
    def append(self, new):
        self.storage.append(new)

tags_160 = IntagGroup('160', ['160_2_reverse', '160_2_straight', '160_3_reverse', '160_3_straight', '160_reverse', '160_straight'] )       
tags_160_2 = IntagGroup('160_2', ['160_2_reverse', '160_2_straight'] )       
tags_160_3 = IntagGroup('160_3', ['160_3_reverse', '160_3_straight'] )       
    



def main():
    storage = Measurements_Storage('data/')
    storage.normalize()
    storage.fit()
    storage.average(tags_160_3)

    visualisator = Visualisator(storage)
    visualisator.plot_from_list(tags_160_3)

if __name__ == '__main__':
    main()



