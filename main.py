####################
#   Order of operations  
# 1. Load Data, loads the csv file into 2 seperate list for heroes and villains
# 1.0.1 Split Data into Evaluation and training Data
# 1.1 Preprocess loaded Data, processes the data of these list.
# 2. Visualize Data, visualizes the processed data
# 3. Build prediction Method, predict the winning probabilites
# 4. Evaluate
# 5. Use Method and visualization to complete Task 2 and 3
####################

from Dataloader import load_characters, split_eval

if __name__ == "__main__":
    # load data
    heroes, villains = load_characters("Data-20250331/data.csv")
    eval_heroes, heroes = split_eval(heroes)
    eval_villains, villains = split_eval(villains)
 

# next steps
# check if plots show categories in categories like intelligence. i.e stupid, smart, genius usw