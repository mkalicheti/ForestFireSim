import sys
import math
import random
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib import animation as animation
fig = plt.figure()

class ForestFire:

    def __init__(self, nrows, ncols, generation):
        self.nrows = nrows
        self.ncols = ncols
        self.generation = generation

        # Initialize the environment 

        self.vegetation_matrix = self.init_vegetation()
        self.density_matrix = self.init_density()
        self.altitude_matrix = self.init_altitude()
        self.forest = self.init_forest()
        self.slope_matrix = self.get_slope(self.altitude_matrix)
        self.wind_matrix = self.get_wind()

        #Initialze the animation

        self.fig = plt.figure()
        self.im = None

    def colormap(self, i, array):

        np_array = np.array(array)
        plt.imshow(np_array, interpolation = "None", cmap = cm.plasma)
        plt.title(i)
        plt.show()

    # Define the initial vegetation matrix 

    def init_vegetation(self):
        veg_matrix = [[0 for col in range(self.ncols)] for row in range(self.nrows)]
        for i in range(self.nrows):
            for j in range(self.ncols):
                veg_matrix[i][j] = 1
        return veg_matrix
    
    # Define the initial density matrix
    
    def init_density(self):
        
        den_matrix = [[0 for col in range(self.ncols)] for row in range(self.nrows)]
        for i in range(self.nrows):
            for j in range(self.ncols):
                den_matrix[i][j] = 1
        return den_matrix
    
    # Define the initial altitude matrix

    def init_altitude(self):

        alt_matrix = [[0 for col in range(self.ncols)] for row in range(self.nrows)]
        for i in range(self.nrows):
            for j in range(self.ncols):
                alt_matrix[i][j] = 1
        return alt_matrix
    
    # Define the initial forest matrix

    def init_forest(self):

        forest = [[0 for col in range(self.ncols)] for row in range(self.nrows)]
        for i in range(self.nrows):
            for j in range(self.ncols):
                forest[i][j] = 2
        ignite_col = int(self.ncols//2)
        ignite_row = int(self.nrows//2)
        for row in range(ignite_row-1, ignite_row+1):
            for col in range(ignite_col-1,ignite_col+1):
                forest[row][col] = 3
        return forest

    # Define the slope matrix

    def tg(self, x):
        return math.degrees(math.atan(x))

    def get_slope(self, alt_matrix):

        slope_matrix = [[0 for col in range(self.ncols)] for row in range(self.nrows)]
        for row in range(self.nrows):
            for col in range(self.ncols):
                sub_slope_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                if row == 0 or row == self.nrows - 1 or col == 0 or col == self.ncols - 1: 
                    slope_matrix[row][col] = sub_slope_matrix
                    continue

            
                current_altitude = alt_matrix[row][col]
                sub_slope_matrix[0][0] = self.tg((current_altitude - alt_matrix[row - 1][col - 1]) / 1.414)
                sub_slope_matrix[0][1] = self.tg(current_altitude - alt_matrix[row - 1][col])
                sub_slope_matrix[0][2] = self.tg((current_altitude - alt_matrix[row - 1][col + 1]) / 1.414)
                sub_slope_matrix[1][0] = self.tg(current_altitude - alt_matrix[row][col - 1])
                sub_slope_matrix[1][1] = 0

                sub_slope_matrix[1][2] = self.tg(current_altitude - alt_matrix[row][col + 1])
                sub_slope_matrix[2][0] = self.tg((current_altitude - alt_matrix[row + 1][col - 1]) / 1.414)
                sub_slope_matrix[2][1] = self.tg(current_altitude - alt_matrix[row + 1][col])
                sub_slope_matrix[2][2] = self.tg((current_altitude - alt_matrix[row + 1][col + 1]) / 1.414)
                slope_matrix[row][col] = sub_slope_matrix

        return slope_matrix
    

    def calc_pw(self, theta):
        c_1 = 0.045
        c_2 = 0.131
        V = 10
        t = math.radians(theta)
        ft = math.exp(V*c_2*(math.cos(t)-1))
        return math.exp(c_1*V)*ft
    
    def get_wind(self):

        wind_matrix = [[0 for col in [0,1,2]] for row in [0,1,2]]
        thetas = [[45,0,45],
              [90,0,90],
              [135,180,135]]
        for row in [0,1,2]:
            for col in [0,1,2]:
                wind_matrix[row][col] = self.calc_pw(thetas[row][col])
        wind_matrix[1][1] = 0
        return wind_matrix


    # burn or not burn function

    

    def get_vegetation_value(self, matrix, row, col, dictval):
        try:
            value = matrix[row][col]
            return dictval[value]
        except KeyError:
            return None

        

    def burn_or_not(self, abs_row, abs_col, neighbour_matrix):

        p_veg = self.get_vegetation_value(self.vegetation_matrix, abs_row, abs_col, {1: -0.3, 2: 0, 3: 0.4})
        p_den = self.get_vegetation_value(self.density_matrix, abs_row, abs_col,  {1:-0.4,2:0,3:0.3})
        p_h = 0.58 
        a = 0.078

        for row in [0,1,2]:
            for col in [0,1,2]:
                if neighbour_matrix[row][col] == 3:
                    slope = self.slope_matrix[abs_row][abs_col][row][col]
                    p_slope = math.exp(a * slope)
                    p_wind = self.wind_matrix[row][col]
                    p_burn = p_h * (1 + p_veg) * (1 + p_den) * p_wind * p_slope
                    if p_burn > random.random():
                        return 3 

        return 2 


    def update(self, old_forest):

        result_forest = [[1 for i in range(self.ncols)] for j in range(self.nrows)]
        for row in range(1, self.nrows - 1):
            for col in range(1, self.ncols - 1):

                if old_forest[row][col] == 1 or old_forest[row][col] == 4:
                    result_forest[row][col] = old_forest[row][col] 
                if old_forest[row][col] == 3:
                    if random.random() < 0.4:
                        result_forest[row][col] = 3 
                    else:
                        result_forest[row][col] = 4
                if old_forest[row][col] == 2:
                    neighbours = [[row_vec[col_vec] for col_vec in range(col-1, col+2)]
                                for row_vec in old_forest[row-1:row+2]]
        
                    result_forest[row][col] = self.burn_or_not(row, col, neighbours)
        return result_forest
    

    # Simulate the forest fire by initializing the forest and updating it for 100 times

    def simulate(self):

        forest = self.init_forest()
        for i in range(self.generation):
            forest = self.update(forest)
            plt.imshow(forest)
            
            # Make the title based on i

            plt.title("Forest Fire Simulation: Generation {}".format(i))

            plt.show()
            if i % 5 == 0:
                plt.savefig("forest_fire_{}.png".format(i))
    # Run the simulation

    def run(self):

        self.simulate()

if __name__ == "__main__":

    forest_fire = ForestFire(50,50,100)
    forest_fire.run()



        
        



    





