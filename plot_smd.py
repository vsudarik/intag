import numpy as np
import matplotlib.pyplot as plt
import os 
import time
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pdb
from typing import Callable
from scipy import stats

import sys
sys.setrecursionlimit(10000) # установить максимальную глубину рекурсии в 10000

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
    def move_z_to_zero(self, incr):
        
        print(self.z_array_straight)
        print('kek')
        self.z_array_straight -= incr 


        print(self.z_array_straight)
        print('============')

        pass



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

    def move_z_to_zero(self):
        for measure in self:
            measure.move_z_to_zero(self.bottom_border)


        print('border_do')
        print(self.bottom_border)
        print(self.top_border)

        incr = self.bottom_border 
        self.bottom_border -= incr
        self.top_border -= incr
        print('border')
        print(self.bottom_border)
        print(self.top_border)

            

    def normalize(self):
        for measure in self:
            measure.calculate_straight_z()
            measure.reverse_dict()

    def single_zero_start(self, measure_list, position):
        if position == 'left':
            for measure in measure_list:
                
                print(self[measure].pmf_array_straight)
                print(self[measure].pmf_array_straight[0])
                self[measure].pmf_array_straight -= self[measure].pmf_array_straight[0] 
                print(self[measure].pmf_array_straight)

        elif position == 'right': 
            for measure in measure_list:
                print(self[measure].pmf_array_straight)
                print(self[measure].pmf_array_straight[-1])
                self[measure].pmf_array_straight -= self[measure].pmf_array_straight[-1] 
                print(self[measure].pmf_array_straight)

        else:
            raise Exception
                

    def average(self, measure_list):
        number_of_steps = max([len(self[measure]) for measure in measure_list])
        new_z = np.linspace(self.top_border, self.bottom_border,number_of_steps ) 
        new_z = np.linspace(self.bottom_border, self.top_border,number_of_steps ) 
        print('new_z')
        print(new_z)

        interp_mesures_list = [np.interp(new_z, self[measure].z_array_straight, self[measure].pmf_array_straight) for measure in measure_list]
        mean_array = np.add.reduce(interp_mesures_list)/len(interp_mesures_list) 
        

        squared_diffs = [(arr - mean_array) ** 2 for arr in interp_mesures_list]
        variance_array =   np.add.reduce(squared_diffs)/ (len(interp_mesures_list) - 1)

        deviatin_array = np.sqrt(variance_array)

    
        new_avearage_measure =  AveragedMeasurement(new_z, mean_array, measure_list.name + '_average') 
        self.measurement_dict[measure_list.name + '_average'] = new_avearage_measure
        measure_list.append(measure_list.name + '_average')

        '''
        new_variance_measure =  AveragedMeasurement(new_z, variance_array, measure_list.name + '_variance')
        self.measurement_dict[measure_list.name + '_variance'] = new_variance_measure
        measure_list.append(measure_list.name + '_variance')
        '''

        new_deviation_measure =  AveragedMeasurement(new_z, deviatin_array, measure_list.name + '_deviation')
        self.measurement_dict[measure_list.name + '_deviation'] = new_deviation_measure
        measure_list.append(measure_list.name + '_deviation')


        SE_array = deviatin_array / np.sqrt(len(interp_mesures_list))
        print(len(interp_mesures_list))

        t = stats.t.ppf(0.975, len(interp_mesures_list)-1) #для 95% интервала, тут коэффициент 0.975, чтобы учесть обе стороны хвоста
        print(t)

        CI_b_array = mean_array - SE_array * t   
        CI_t_array = mean_array + SE_array * t 

        new_CI_b_measure =  AveragedMeasurement(new_z, CI_b_array , measure_list.name + '_CI_b')
        self.measurement_dict[measure_list.name + '_CI_b'] = new_CI_b_measure
        measure_list.append(measure_list.name + '_CI_b')

        new_CI_t_measure =  AveragedMeasurement(new_z, CI_t_array, measure_list.name + '_CI_t')
        self.measurement_dict[measure_list.name + '_CI_t'] = new_CI_t_measure
        measure_list.append(measure_list.name + '_CI_t')


    def __iter__(self):
        for i in self.measurement_dict.values():
            #if (i.isreverse and self.isCheckReverse) or  (not i.isreverse and self.isCheckForward):
                yield i 

    def __getitem__(self, index):
        return self.measurement_dict[index]



class Visualisator():
    def __init__(self, storage):
        self.storage = storage
       # self.fig, self.ax = plt.subplots(1, 3, figsize=(12, 6))
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 6))

    def set_components(self):
        '''
        self.ax[0].set_xlabel("step")
        self.ax[0].set_ylabel("PMF")
        self.ax[0].legend()

        self.ax[1].set_xlabel("step")
        self.ax[1].set_ylabel("Z")
        self.ax[1].legend()

        self.ax[2].set_xlabel("Z")
        self.ax[2].set_ylabel("PMF")
        self.ax[2].legend()
        '''
        
        self.ax[0].set_xlabel("ΔZ")
        self.ax[0].set_ylabel("PMF")
        self.ax[0].legend()

        self.ax[1].set_xlabel("ΔZ")
        self.ax[1].set_ylabel("PMF")
        self.ax[1].legend()


    def plot_one_measure(self, measure_name, reverse=False):
        measure =  self.storage[measure_name]
        if not reverse:
            z = measure.z_array_straight
        else:
            z = np.flip(measure.z_array_straight)
        '''
        try:
            self.ax[0].plot(measure.step_array, measure.pmf_array , label=measure.name)
        except:
            pass
        

        try:
            self.ax[1].plot(measure.step_array, measure.z_array , label=measure.name)
        except:
            pass
            '''

        try:
            self.ax[0].plot(z, measure.pmf_array_straight , label=measure.name)
        except:
            pass

    def plot_one_measure1(self, measure_name, reverse=False):
        measure =  self.storage[measure_name]
        
        if not reverse:
            z = measure.z_array_straight
        else:
            z = np.flip(measure.z_array_straight)

        try:
            self.ax[1].plot(z, measure.pmf_array_straight , label=measure.name)
        except:
            pass


    def plot_all(self):
        for measure in self.storage:
            self.plot_one_measure(measure.name)

    def plot_CI(self, bottom, top, reverse=False):
        if not reverse:
            z = self.storage[bottom].z_array_straight 
        else:
            z = np.flip(self.storage[bottom].z_array_straight)

        self.ax[0].fill_between(z, self.storage[bottom].pmf_array_straight, self.storage[top].pmf_array_straight, alpha=0.5)
    
    def plot_from_list(self, measure_list, reverse=False):
        for measure_name in measure_list:
            
            self.plot_one_measure(measure_name, reverse)

    def show(self):
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

tags_160_reverse = IntagGroup('160_reverse', ['160_2_reverse', '160_3_reverse', '160_reverse' ] )       
tags_160_straight = IntagGroup('160_straight', [ '160_3_straight', '160_2_straight', '160_straight'] )       
tags_160_stat = IntagGroup('160_stat', [ '160_reverse_average', '160_straight_average'  ] )
    

tags_220_reverse = IntagGroup('220_reverse', ['220_pium4_reverse', '220_qiP6R_reverse' ] )       
tags_220_straight = IntagGroup('220_straight', [ '220_pium4_straight', '220_qiP6R_straight'] )       
tags_220_stat = IntagGroup('220_stat', [ '220_reverse_average', '220_straight_average'  ] )


tags_225_reverse = IntagGroup('225_reverse', ['225_4_reverse', '225_4b_reverse' ] )       
tags_225_straight = IntagGroup('225_straight', [ '225_4_straight', '225_4b_straight'] )       
tags_225_stat = IntagGroup('225_stat', [ '225_reverse_average', '225_straight_average'  ] )



def main(tags_group):
    tags_160 = IntagGroup('160', ['160_2_reverse', '160_2_straight', '160_3_reverse', '160_3_straight', '160_reverse', '160_straight'] )       

    tags_160_reverse = IntagGroup('160_reverse', ['160_2_reverse', '160_3_reverse', '160_reverse' ] )       
    tags_160_straight = IntagGroup('160_straight', [ '160_3_straight', '160_2_straight', '160_straight'] )       
    tags_160_stat = IntagGroup('160_stat', [ '160_reverse_average', '160_straight_average'  ] )
        

    tags_220_reverse = IntagGroup('220_reverse', ['220_pium4_reverse', '220_qiP6R_reverse' ] )       
    tags_220_straight = IntagGroup('220_straight', [ '220_pium4_straight', '220_qiP6R_straight'] )       
    tags_220_stat = IntagGroup('220_stat', [ '220_reverse_average', '220_straight_average'  ] )


    tags_225_reverse = IntagGroup('225_reverse', ['225_4_reverse', '225_4b_reverse' ] )       
    tags_225_straight = IntagGroup('225_straight', [ '225_4_straight', '225_4b_straight'] )       
    tags_225_stat = IntagGroup('225_stat', [ '225_reverse_average', '225_straight_average'  ] )


    storage = Measurements_Storage('data/')
    storage.normalize()
    storage.fit()
    storage.move_z_to_zero()

    storage.single_zero_start(tags_group, position = 'right')
    storage.average(tags_225_reverse)
    storage.single_zero_start(tags_225_straight, position = 'left')
    storage.average(tags_225_straight)

    visualisator = Visualisator(storage)
   # visualisator.plot_all()
    
    z = np.flip(storage['225_reverse_average'].z_array_straight)

    '''
    visualisator.ax[0].plot(z, storage['225_reverse_average'].pmf_array_straight , label=storage['225_reverse_average'].name)
    #visualisator.plot_from_list(tags_225_stat)
    visualisator.ax[0].plot(storage['225_straight_average'].z_array_straight, storage['225_straight_average'].pmf_array_straight , label=storage['225_straight_average'].name)


    visualisator.plot_CI('225_reverse_CI_b','225_reverse_CI_t', reverse=True)
    visualisator.plot_CI('225_straight_CI_b','225_straight_CI_t' )
    visualisator.plot_one_measure1('225_reverse_deviation', reverse=True)
    visualisator.plot_one_measure1('225_straight_deviation')
    '''



    tags_225_reverse = IntagGroup('225_reverse', ['225_4_reverse', '225_4b_reverse' ] )       
    tags_225_straight = IntagGroup('225_straight', [ '225_4_straight', '225_4b_straight'] )       

   # tags_225_straight = IntagGroup('225_straight', [ '225_pium4_straight', '225_qiP6R_straight'] )       

    visualisator.plot_from_list(tags_225_reverse, reverse = True)
   # visualisator.plot_from_list(tags_225_straight)

    visualisator.show()
    

if __name__ == '__main__':
    main(tags_225_reverse )



