import numpy as np
import matplotlib.pyplot as plt
import os 

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
            self.reverse = False
        else:
            self.reverse = True

    def calculate_mnk(self):
        self.coefficients = np.polyfit(self.step_array,self.z_array , 1)
        

class Measurements_storage():
    def __init__(self, path):
        self.load_all_data(path)
        
    def load_all_data(self, path):
        with os.scandir(path) as entries:
            self.measurement_list =[]
            for entry in entries:
                print(entry.name)  
                measure = Measurement(entry.name, path)
                self.measurement_list.append(measure)

    def __iter__(self):
        for i in self.measurement_list:
            yield i 

    def old_code():
        #нормирование данных
        max_range = min(z1[0], z2[-1], z3[-1])  
        min_range = max(z1[-1], z2[0], z3[0])  
        print(z1)
        mask = np.logical_and(z1 >= min_range, z1 <= max_range)
        print(mask)


measurements = Measurements_storage('data/')



# Создание графиков
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

def plot_measures():
    for measure in measurements:
            print(measure)
        #if not measure.reverse:
            ax[0].plot(measure.step_array, measure.pmf_array , label=measure.name)
            ax[1].plot(measure.step_array, measure.z_array , label=measure.name)
        
def plot_one_measure(number):
    measure =  next(measurements)
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

# Отображение графиков
plt.show()
