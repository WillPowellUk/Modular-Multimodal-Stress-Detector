import pickle


# read pkl file
pkl_path = 'data_collection/recordings/S6_W/S6.pkl'

with open(pkl_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')
    print(data.keys())
