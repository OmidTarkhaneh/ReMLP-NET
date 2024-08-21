"""This version of ANI-2x is adjusted and fine-Tuned to work
   with the Retrieivum Dataset
"""


import torch

import math
import torch.utils.tensorboard
import tqdm
import numpy as np


from sklearn.metrics import mean_squared_error
import pandas as pd
from aev import *
from units import *
from utils import *
from nn import *
from utils import EnergyShifter, load



##################################################################################



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


Rcr = 4.30
Rca = 3.05

count=0.18750e+00
values_ShfA=[]
while count< Rca:
        count+=0.3374999999999999
        if count<Rca:
            values_ShfA.append(count)

count=0.18750e+00
values_ShfR=[]
while count< Rcr:
        count+=0.26879999999999993
        if count<Rcr:
            values_ShfR.append(count)


count=3.927000e-01
values_ShfZ=[]
while count< Rca:
          count+=0.7853999999999999
          if count<Rca:
              values_ShfZ.append(count)

EtaR = torch.tensor([10.87], device=device)
ShfR = torch.tensor([values_ShfR], device=device)
Zeta = torch.tensor([9.62], device=device)
ShfZ = torch.tensor([values_ShfZ], device=device)
EtaA = torch.tensor([10.29], device=device)
ShfA = torch.tensor([values_ShfA], device=device)


species_order = ['H','C', 'N', 'O', 'S', 'Cl']

num_species = len(species_order)

energy_shifter = EnergyShifter(None)

try:
        path = os.path.dirname(os.path.realpath(__file__))
except NameError:
        path = os.getcwd()
dspath = os.path.join('GDB28ANI_TrainSorted.h5')
batch_size = 256


training, validation = load(dspath)\
                                .subtract_self_energies(energy_shifter, species_order)\
                                .remove_outliers()\
                                .species_to_indices(species_order)\
                                .shuffle()\
                                .split(0.9, None)


training = training.collate(batch_size).cache()
validation = validation.collate(batch_size).cache()
print('Self atomic energies: ', energy_shifter.self_energies)


aev_computer = AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfZ, ShfA, num_species)


"""# Model Definition"""
import torch
import torch.nn as nn
import torch.nn.functional as F





aev_dim = aev_computer.aev_length

H_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 256),
    torch.nn.CELU(0.1),
    torch.nn.Linear(256, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 1)
)

C_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 224),
    torch.nn.CELU(0.1),
    torch.nn.Linear(224, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 1)
)

N_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 1)
)

O_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 192),
    torch.nn.CELU(0.1),
    torch.nn.Linear(192, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 1)
)

# F_network = torch.nn.Sequential(
#     torch.nn.Linear(aev_dim, 160),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(160, 128),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(128, 96),
#     torch.nn.CELU(0.1),
#     torch.nn.Linear(96, 1)
# )

Cl_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)

S_network = torch.nn.Sequential(
    torch.nn.Linear(aev_dim, 160),
    torch.nn.CELU(0.1),
    torch.nn.Linear(160, 128),
    torch.nn.CELU(0.1),
    torch.nn.Linear(128, 96),
    torch.nn.CELU(0.1),
    torch.nn.Linear(96, 1)
)


nn = ANIModel([H_network,C_network, O_network, N_network,S_network,Cl_network])

print(nn)

###############################################################################
# Initialize the weights and biases.
#
def init_params(m):
      if isinstance(m, torch.nn.Linear):
              torch.nn.init.kaiming_normal_(m.weight, a=1.0)
              torch.nn.init.zeros_(m.bias)


nn.apply(init_params)

###############################################################################
# Let's now create a pipeline of AEV Computer --> Neural Networks.
# model = torchani.nn.Sequential(aev_computer, nn).to(device)
model = Sequential(aev_computer, nn).to(device)


AdamW = torch.optim.AdamW(model.parameters())
SGD = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
###############################################################################
# Setting up a learning rate scheduler to do learning rate decay
AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=100, threshold=0)
SGD_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(SGD, factor=0.5, patience=100, threshold=0)

###############################################################################

# latest_checkpoint = 'latest.pt'


def validate():
      # run validation
      mse_sum = torch.nn.MSELoss(reduction='sum')
      total_mse = 0.0
      count = 0
      true_energies_1=[]
      predicted_energies_1=[]

      # true_dftmain_energy=[]
      # predicted_dftmain_energies=[]

      model.train(False)
      with torch.no_grad():
              for properties in validation:
                      species = properties['species'].to(device)
                      coordinates = properties['coordinates'].to(device).float()
                      true_energies = properties['energies'].to(device)
                      _, predicted_energies = model((species, coordinates))
                      total_mse += mse_sum(predicted_energies, true_energies).item()
                      count += predicted_energies.shape[0]

                      # save predicted and true energy in list
                      predicted_energies_1.append(predicted_energies.detach().cpu().numpy())
                      true_energies_1.append(true_energies.detach().cpu().numpy())

      model.train(True)
      return hartree2kcalmol(math.sqrt(total_mse / count)), predicted_energies_1, true_energies_1



"""# Model Training"""
##################################################################################

###############################################################################
# Finally, we come to the training loop.
#
# In this tutorial, we are setting the maximum epoch to a very small number,
# only to make this demo terminate fast. For serious training, this should be
# set to a much larger value
mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 1000
early_stopping_learning_rate = 1.0E-6
best_model_checkpoint = 'best.pt'

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
      rmse, predicted_energies_1, true_energies_1 = validate()
      print('RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)

      learning_rate = AdamW.param_groups[0]['lr']

      if learning_rate < early_stopping_learning_rate:
              break

      # checkpoint
      if AdamW_scheduler.is_better(rmse, AdamW_scheduler.best):
              torch.save(nn.state_dict(), best_model_checkpoint)

      AdamW_scheduler.step(rmse)
      SGD_scheduler.step(rmse)


      for i, properties in tqdm.tqdm(
              enumerate(training),
              total=len(training),
              desc="epoch {}".format(AdamW_scheduler.last_epoch)
      ):
              species = properties['species'].to(device)
              coordinates = properties['coordinates'].to(device).float()
              true_energies = properties['energies'].to(device).float()
              num_atoms = (species >= 0).sum(dim=1, dtype=true_energies.dtype)
              _, predicted_energies = model((species, coordinates))

              loss = (mse(predicted_energies, true_energies) / num_atoms.sqrt()).mean()

              AdamW.zero_grad()
              SGD.zero_grad()
              loss.backward()
              AdamW.step()
              SGD.step()



import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

true_energies_11= np.hstack( true_energies_1)
pred_energies_11= np.hstack( predicted_energies_1)

mae=np.sum(np.abs(true_energies_11-pred_energies_11))

mae=mae/(len(true_energies_11))

print('overall MAE(kcal/mol)=',(mae))


mse=mean_squared_error(true_energies_11,pred_energies_11)

rmse=np.sqrt(mse)

print('overall RMSE(kcal/mol)=',(rmse))


NonZero_NewModeldf=pd.DataFrame()
NonZero_NewModeldf['truelabel']=true_energies_11
NonZero_NewModeldf['pred']=pred_energies_11
NonZero_NewModeldf['val']=np.arange(0,len(true_energies_11))
NonZero_NewModeldf.to_csv('NonZeroData003_Predicted.csv')




device='cpu'
model=model.to(device)

"""# Test the Model"""
##################################################################################


try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()
dspath = os.path.join('GDB28ANI_TestSorted.h5')
batch_size = 256


training2, validation2 = load(dspath)\
                                    .subtract_self_energies(energy_shifter, species_order)\
                                    .remove_outliers()\
                                    .species_to_indices(species_order)\
                                    .shuffle()\
                                    .split(0.000000001, None)

training2 = training2.collate(batch_size).cache()
validation2 = validation2.collate(batch_size).cache()
print('Self atomic energies: ', energy_shifter.self_energies)

def validate2():
    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    count = 0
    true_energies_1=[]
    predicted_energies_1=[]
    true_dft1main_energy=[]
    predicted_dft1main_energies=[]

    model.train(False)
    with torch.no_grad():
        for properties in validation2:
            species = properties['species'].to(device)
            coordinates = properties['coordinates'].to(device).float()
            true_energies = properties['energies'].to(device)
            _, predicted_energies = model((species, coordinates))
            total_mse += mse_sum(predicted_energies, true_energies).item()
            count += predicted_energies.shape[0]

            energy_shift = energy_shifter.sae(species)
            true_dft1_energy = true_energies + energy_shift.to(device)
            predicted_dft1_energies= predicted_energies + energy_shift.to(device)

            # save predicted and true energy in list
            predicted_energies_1.append(predicted_energies.detach().cpu().numpy())
            true_energies_1.append(true_energies.detach().cpu().numpy())

            true_dft1main_energy.append(true_dft1_energy.detach().cpu().numpy())
            predicted_dft1main_energies.append(predicted_dft1_energies.detach().cpu().numpy())

            # print('true_dft1main_energy=',true_dft1main_energy)
            # print('true_energies=',true_energies)

    model.train(True)
    return hartree2kcalmol(math.sqrt(total_mse / count)), predicted_energies_1, true_energies_1, true_dft1main_energy, predicted_dft1main_energies



import numpy as np
import pandas as pd


rmse_1, predicted_energies_111,true_energies_111, true_dft1_energy, predicted_dft1_energies = validate2()

from sklearn.metrics import mean_squared_error

true_energies_22= np.hstack(true_energies_111)
pred_energies_22= np.hstack(predicted_energies_111)

mae=np.sum(np.abs(true_energies_22-pred_energies_22))

mae=mae/(len(true_energies_22))

print('overall MAE(kcal/mol)=',hartree2kcalmol(mae))


mse=mean_squared_error(true_energies_22,pred_energies_22)

rmse=np.sqrt(mse)

print('overall RMSE(kcal/mol)=',hartree2kcalmol(rmse))



import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, r2_score


true_dft1_energies_11= np.hstack( true_dft1_energy)
pred_dft1_energies_11= np.hstack( predicted_dft1_energies)


print( r2_score(true_dft1_energies_11,pred_dft1_energies_11))
mae=np.sum(np.abs(true_dft1_energies_11-pred_dft1_energies_11))

mae=mae/(len(true_dft1_energies_11))

print('overall MAE(kcal/mol)=',(mae))


mse=mean_squared_error(true_dft1_energies_11,pred_dft1_energies_11)

rmse=np.sqrt(mse)

print('overall RMSE(kcal/mol)=',(rmse))


NonZero_NewModeldf=pd.DataFrame()
NonZero_NewModeldf['truelabel']=true_dft1_energies_11
NonZero_NewModeldf['pred']=pred_dft1_energies_11
NonZero_NewModeldf['val']=np.arange(0,len(true_dft1_energies_11))
NonZero_NewModeldf.to_csv('TestAni2x_predicted_C3H3NS_Main.csv')

