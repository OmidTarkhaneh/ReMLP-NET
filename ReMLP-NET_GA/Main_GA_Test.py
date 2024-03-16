'''
        In implementing ReMLP-NET we  borrowed different parts from torchani package (https://aiqm.github.io/torchani/) especially the 
        sections related to the AEV generation and dataloading and subtracting the self-energies.
'''



import numpy as np
import matplotlib.pyplot as plt
from ypstruct import structure
import GA
from Model import ModelTest
from Modelutils import EnergyShifter, load
import os
import pandas as pd
import sys
import argparse
import torch




if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path_train',
                        help='Path of the dataset, can be an hdf5 file or a directory containing hdf5 files')
    parser.add_argument('--dataset_path_test',
                        help='Path of the dataset, can be an hdf5 file or a directory containing hdf5 files')    
    parser.add_argument('-d', '--device',
                        help='Device of modules and tensors',
                        default=('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('-b', '--batch_size',
                        help='Number of conformations of each batch',
                        default=256, type=int)
    parser.add_argument('-n', '--num_epochs',
                        help='Number of epochs',
                        default=1000, type=int)
    parser.add_argument('--maxit', type=int, default=100, help='Maximum number of iterations')
    parser.add_argument('--npop', type=int, default=50, help='Population size')
    parser.add_argument('--beta', type=float, default=1, help='Beta parameter')
    parser.add_argument('--pc', type=float, default=1, help='Crossover probability')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma parameter')
    parser.add_argument('--mu', type=float, default=0.01, help='Mutation probability')
    parser.add_argument('--sigma', type=float, default=0.1, help='Sigma parameter')

    args = parser.parse_args()



if args.dataset_path_train is None:
    args.dataset_path_train = input("Enter the path of the train dataset: ")

if args.dataset_path_test is None:
    args.dataset_path_test = input("Enter the path of the test dataset: ")


#################################  loading Data ############################################
"""Tools for loading, shuffling, and batching  datasets


You can transform this iterable by using transformations.
To do a transformation, call `it.transformation_name()`. This
will return an iterable that may be cached depending on the specific
transformation.

Available transformations are listed below:

- `species_to_indices` accepts two different kinds of arguments. It converts
    species from elements (e. g. "H", "C", "Cl", etc) into internal torchani
    indices (as returned by :class:`torchani.utils.ChemicalSymbolsToInts` or
    the ``species_to_tensor`` method of a :class:`torchani.models.BuiltinModel`
    and :class:`torchani.neurochem.Constants`), if its argument is an iterable
    of species. By default species_to_indices behaves this way, with an
    argument of ``('H', 'C', 'N', 'O', 'F', 'S', 'Cl')``  However, if its
    argument is the string "periodic_table", then elements are converted into
    atomic numbers ("periodic table indices") instead. This last option is
    meant to be used when training networks that already perform a forward pass
    of :class:`torchani.nn.SpeciesConverter` on their inputs in order to
    convert elements to internal indices, before processing the coordinates.

- `subtract_self_energies` subtracts self energies from all molecules of the
    dataset. It accepts two different kinds of arguments: You can pass a dict
    of self energies, in which case self energies are directly subtracted
    according to the key-value pairs.
   

- `remove_outliers` removes some outlier energies from the dataset if present.

- `shuffle` shuffles the provided dataset. Note that if the dataset is
    not cached (i.e. it lives in the disk and not in memory) then this method
    will cache it before shuffling. This may take time and memory depending on
    the dataset size. This method may be used before splitting into validation/training
    shuffle all molecules in the dataset, and ensure a uniform sampling from
    the initial dataset, and it can also be used during training on a cached
    dataset of batches to shuffle the batches.

- `cache` cache the result of previous transformations.
    If the input is already cached this does nothing.

- `collate` creates batches and pads the atoms of all molecules in each batch
    with dummy atoms, then converts each batch to tensor. `collate` uses a
    default padding dictionary:
    ``{'species': -1, 'coordinates': 0.0, 'forces': 0.0, 'energies': 0.0}`` for
    padding, but a custom padding dictionary can be passed as an optional
    parameter, which overrides this default padding. Note that this function
    returns a generator, it doesn't cache the result in memory.

- `pin_memory` copies the tensor to pinned (page-locked) memory so that later transfer
    to cuda devices can be done faster.

you can also use `split` to split the iterable to pieces. use `split` as:

.. code-block:: python

    it.split(ratio1, ratio2, None)

where None in the end indicate that we want to use all of the rest.

Note that orderings used in :class:`torchani.utils.ChemicalSymbolsToInts` and
:class:`torchani.nn.SpeciesConverter` should be consistent with orderings used
in `species_to_indices` and `subtract_self_energies`. To prevent confusion it
is recommended that arguments to intialize converters and arguments to these
functions all order elements *by their atomic number* (e. g. if you are working
with hydrogen, nitrogen and bromine always use ['H', 'N', 'Br'] and never ['N',
'H', 'Br'] or other variations). It is possible to specify a different custom
ordering, mainly due to backwards compatibility and to fully custom atom types,
but doing so is NOT recommended, since it is very error prone.


"""

species_order = ['C', 'H', 'O', 'N', 'S', 'Cl']

num_species = len(species_order)

energy_shifter = EnergyShifter(None)

try:
        path = os.path.dirname(os.path.realpath(__file__))
except NameError:
        path = os.getcwd()
dspath = os.path.join(args.dataset_path_train)
batch_size = args.batch_size


training, validation = load(dspath)\
                                .subtract_self_energies(energy_shifter, species_order)\
                                .remove_outliers()\
                                .species_to_indices(species_order)\
                                .shuffle()\
                                .split(0.9, None)


training = training.collate(batch_size).cache()
validation = validation.collate(batch_size).cache()
print('Self atomic energies: ', energy_shifter.self_energies)

###################################################

try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
dspath_test = os.path.join(args.dataset_path_test)
batch_size = args.batch_size


_, test = load(dspath_test)\
                                    .subtract_self_energies(energy_shifter, species_order)\
                                    .remove_outliers()\
                                    .species_to_indices(species_order)\
                                    .shuffle()\
                                    .split(0.0000000001, None)

# training_test = training_test.collate(batch_size).cache()
data_test = test.collate(batch_size).cache()
print('Self atomic energies: ', energy_shifter.self_energies)





##############################################################


# fitness function of GA 
def fitness(X):
   
    # print(population[1])
    Rcr, Rca, EtaR, EtaA, zeta = X

    temp=Rcr
    if Rca > Rcr:
          Rcr=Rca
          Rca=temp

    # result=[Rcr, Rca, EtaR, EtaA, zeta]
    rmse, rmse_test, mae_test = ModelTest(Rcr, Rca, EtaR, EtaA, zeta, training, validation, data_test,species_order,energy_shifter, args.num_epochs, args.device) 
    print('rmse', rmse, 'rmse_test', rmse_test)
    print('Rcr=',Rcr, 'Rca=',Rca, 'EtaR=', EtaR, 'EtaA=', EtaA, 'Zeta=', zeta)
    
    return  rmse_test


# Problem Definition
problem = structure()
problem.costfunc = fitness
problem.nvar = 5
problem.varmin = 1.5
problem.varmax = 15

# GA Parameters
params = structure()
params.maxit = args.maxit
params.npop = args.npop
params.beta = args.beta
params.pc = args.pc
params.gamma = args.gamma
params.mu = args.mu
params.sigma = args.sigma



df_result=pd.DataFrame()
df_result['solution']=''

# Run GA
out = GA.run(problem, params)


df = pd.DataFrame({'position': out.bestsol.position})
df['cost']=out.bestsol.cost

df.to_csv('GA_ANIResults.csv')


