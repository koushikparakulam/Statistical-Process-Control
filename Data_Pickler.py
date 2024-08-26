import pickle
from Data_Module import pickle_path


def state_saver(var_name, curr_var):

    with open(pickle_path + '\\' + var_name + '.pkl', 'wb') as file:
        pickle.dump(curr_var, file)


def state_retriever(var_name):

    with open(pickle_path + '\\' + var_name + '.pkl', 'rb') as file:
        variable_data = pickle.load(file)

    return variable_data