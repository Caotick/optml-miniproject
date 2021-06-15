from postprocess import generate_plots
from train import  train
from postprocess import *
from helpers import *

# Setting seeds
torch.manual_seed(404)

# Creates the complete folder architecture
create_folders_structure()

#problems = ['PIMA', 'CaliforniaHousing', 'FashionMNIST']
#optimizers = ['SGD', 'Adagrad', 'Adam']

problems = ['PIMA', 'CaliforniaHousing', 'FashionMNIST']
optimizers = ['SGD', 'Adagrad', 'Adam']

# Decide number of epochs!
nb_epochs = 50
nb_fold = 10

for prob in problems:
    print(f'Problem {prob}')
    print('--------------------------------')
    for opt in optimizers:
        dataset, folds = load_data(prob, k_folds=nb_fold) ### Have to shift this here because folds don't run otherwise
        print(f'Optimizer {opt}')
        print('--------------------------------')
        train(dataset, folds, prob, opt, nb_epochs)

generate_plots(nb_epochs, nb_fold)