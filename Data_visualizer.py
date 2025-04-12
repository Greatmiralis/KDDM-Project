# visualize the data as different graphs. ie histograph, scatterplot, ...

import matplotlib.pyplot as plt
from values_filter_table import max_filtertable, low_filtertable
from scipy import stats



def continousNumericAvg(character_list, feat1, feat2):
    character_list = filter_characters(character_list, feat1)
    character_list = filter_characters(character_list, feat2)
    x = [getattr(x, feat1) for x in character_list]
    y = [getattr(y, feat2) for y in character_list]

    slope, intercept, r, p, std_err = stats.linregress(x, y)
    
    
    def myfunc(x):
        return slope * x + intercept

    mymodel = list(map(myfunc, x))
    plt.plot(x, mymodel, color="red")
       
    return mymodel

def filter_characters(character_list, feat):
     
    filtered_list = character_list.copy()
    # removes all characters where the specified attribute is -1 or not in the
    # range defined in values_filter_table.py
    for character in character_list:
        if feat == "win_prob" or feat == "training_time" or feat == "speed":
            if getattr(character, feat) >= max_filtertable[feat] or getattr(character, feat) <= low_filtertable[feat]:
                filtered_list.remove(character)
        elif getattr(character, feat) == -1 or getattr(character, feat) == "-1":
            filtered_list.remove(character)
            
    return filtered_list

def visualize_dependancy(character_list, feat1, feat2):
    character_list = filter_characters(character_list, feat1)
    character_list = filter_characters(character_list, feat2)
    x_axis = [getattr(x, feat1) for x in character_list]
    y_axis = [getattr(y, feat2) for y in character_list]
    
    plt.scatter(x_axis, y_axis)
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.title("Dependancy plot")
    
    try:
        continousNumericAvg(character_list, feat1, feat2)
    except:
        pass
    
    plt.show()
    