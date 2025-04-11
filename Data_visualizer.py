# visualize the data as different graphs. ie histograph, scatterplot, ...

import matplotlib.pyplot as plt
from values_filter_table import max_filtertable, low_filtertable

def filter_characters(character_list, feat):
     
    for character in character_list:
        if feat == "win_prob" or feat == "training_time":
            if getattr(character, feat) > max_filtertable[feat] or getattr(character, feat) < low_filtertable[feat]:
                character_list.remove(character)
        elif getattr(character, feat) == -1 or getattr(character, feat) == "-1":
            character_list.remove(character)
            
    return character_list

def visualize_dependancy(character_list, feat1, feat2):
    character_list = filter_characters(character_list, feat1)
    character_list = filter_characters(character_list, feat2)
    x_axis = [getattr(x, feat1) for x in character_list]
    y_axis = [getattr(y, feat2) for y in character_list]
    
    plt.scatter(x_axis, y_axis)
    plt.xlabel(feat1)
    plt.ylabel(feat2)
    plt.title("Dependancy plot")
    plt.show()
    