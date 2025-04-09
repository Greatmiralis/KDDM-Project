# simple data loader that loads all the data into lists
# loads the data when starting the programm, returns list of superheroes, list of villains
# and an additional dataloader that only loads evaluation data
# the evaluation data must not be the same as the training data ie. 80% of rows is training, 20% is eval
# evaluation should be free from holes or blank spaces possibly

import pandas as pd

class Character:
    def __init__(self, name, role, skin_type, power_level, weight, height, age,
                 eye_color, gender, hair_color, speed, universe, body_type,
                 job, battle_iq, species, ranking, intelligence, abilities,
                 training_time, special_attack, secret_code, win_prob):
        self.name = name
        self.role = role
        self.skin_type = skin_type
        self.power_level = power_level
        self.weight = weight
        self.height = height
        self.age = age
        self.eye_color = eye_color
        self.gender = gender
        self.hair_color = hair_color
        self.speed = speed
        self.universe = universe
        self.body_type = body_type
        self.job = job
        self.battle_iq = battle_iq
        self.species = species
        self.ranking = ranking
        self.intelligence = intelligence
        self.abilities = abilities
        self.training_time = training_time
        self.special_attack = special_attack
        self.secret_code = secret_code
        self.win_prob = win_prob

def load_characters(filepath):
    df = pd.read_csv(filepath)

    heroes = []
    villains = []

    for index, row in df.iterrows():
        data = row.to_dict()

        clean_data = clean_features(data)

        if clean_data['role'] == 'Hero':
            heroes.append(Character(**clean_data))
        elif clean_data['role'] == 'Villain':
            villains.append(Character(**clean_data))


    print(f"number all entries = {len(df)}")
    # Note: 5 entries are missing as there are 5 completely empty entries in the csv file
    print(f"number of loaded characters = {len(heroes) + len(villains)}")

    return heroes, villains

def clean_features(data):
    data['role'] = clean_role_feature(data.get('role'))

    # Todo: clean other features

    return data

def clean_role_feature(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ['hero', 'h3ro', 'her0']:
            return 'Hero'
        elif value == 'villain':
            return 'Villain'
    return None




