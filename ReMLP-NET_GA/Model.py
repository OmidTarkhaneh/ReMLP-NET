def ModelTest(Rcr, Rca, EtaR, EtaA, Zeta,training, validation, data_test,species_order,energy_shifter, epochs, device):
        

        
        import torch
 
        import math
        import torch.utils.tensorboard
        import tqdm
        import numpy as np

      
        from aev import AEVComputer
        from units import hartree2kcalmol
        from nn import ANIModel, Sequential
   

        from sklearn.metrics import mean_squared_error
        import pandas as pd
    
        # helper function to convert energy unit from Hartree to kcal/mol


        # device to run the training
        device = device

        ##################################################################################

        """
        Train Your Own Neural Network Potential
        =======================================
        """

        ###############################################################################
        # To begin with, let's first import the modules and setup devices we will use:



        # helper function to convert energy unit from Hartree to kcal/mol
        # device to run the training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        ###############################################################################

        Rcr =torch.tensor([Rcr], device=device, dtype=torch.float32)
        Rca = torch.tensor([Rca], device=device, dtype=torch.float32)

        count=8.0000000e-01
        values_ShfA=[]
        while count< Rca:
                count+=0.3374999999999999
                if count<Rca:
                    values_ShfA.append(count)

        count=8.0000000e-01
        values_ShfR=[]
        while count< Rcr:
                count+=0.26879999999999993
                if count<Rcr:
                    values_ShfR.append(count)            


        count=3.927000e-01
        values_ShfZ=[]
        while count< Rca:
                count+=0.7853999999999999
                if count<Rcr:
                    values_ShfZ.append(count)         

        EtaR = torch.tensor([EtaR], device=device, dtype=torch.float32)
        ShfR = torch.tensor([values_ShfR], device=device, dtype=torch.float32)
        Zeta = torch.tensor([Zeta], device=device, dtype=torch.float32)
        ShfZ = torch.tensor([values_ShfZ], device=device, dtype=torch.float32)
        EtaA = torch.tensor([EtaA], device=device, dtype=torch.float32)
        ShfA = torch.tensor([values_ShfA], device=device, dtype=torch.float32)
        
        # species_order = ['C', 'H', 'O', 'N', 'Br', 'S', 'Cl', 'P', 'I', 'F']
    
        # ['C', 'H', 'O', 'N', 'Br', 'S', 'Cl', 'P', 'I', 'F'],
        num_species = len(species_order)
        aev_computer = AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfZ, ShfA, num_species)
        # energy_shifter = EnergyShifter(None)
        print(energy_shifter.self_energies)


        """# Model Definition"""


        aev_dim = aev_computer.aev_length


        class TinyModel(torch.nn.Module):

                def __init__(self):
                        super(TinyModel, self).__init__()

                        self.lin1 = torch.nn.Linear(aev_dim, 256)
                        self.ELU1 = torch.nn.ELU(0.1)
                        self.lin2 = torch.nn.Linear(256, 256)
                        self.ELU2 = torch.nn.ELU(0.1)
                        self.lin3 = torch.nn.Linear(256, 128)
                        self.ELU3 = torch.nn.ELU(0.1)
                        self.lin4 = torch.nn.Linear(128, 96)
                        self.ELU4 = torch.nn.ELU(0.1)
                        self.lin5 = torch.nn.Linear(224, 224)
                        self.ELU5 = torch.nn.ELU(0.1)
                        self.lin6 = torch.nn.Linear(224, 1)


                def __getitem__(self, key):
                        return self.__dict__[key]

                def forward(self, x):

                        x = self.lin1(x)
                        x = self.ELU1(x)
                        x = self.lin2(x)
                        x = self.ELU2(x)
                        x = self.lin3(x)
                        x4 = self.ELU3(x)
                        x = self.lin4(x4)
                        x5 = self.ELU4(x)
                        x6=torch.cat((x4,x5),dim=-1)
                        x = self.lin5(x6)
                        x=self.ELU5(x)
                        xx=torch.multiply(x6,x)
                        x=self.lin6(xx)


                        return x




        C_network = TinyModel()
        H_network =TinyModel()
        O_network =TinyModel()
        N_network =TinyModel()
        S_network =TinyModel()
        Cl_network =TinyModel() 


        nn = ANIModel([C_network, H_network, O_network, N_network,S_network,Cl_network])

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
                        for properties in data_test:
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


                model.train(True)
                return hartree2kcalmol(math.sqrt(total_mse / count)), predicted_energies_1, true_energies_1,true_dft1main_energy,predicted_dft1main_energies

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
        max_epochs = epochs
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



        rmse_1, predicted_energies_111,true_energies_111,true_dft1main_energy,predicted_dft1main_energies = validate2()


        true_energies_22= np.hstack(true_energies_111)
        pred_energies_22= np.hstack(predicted_energies_111)

        mae=np.sum(np.abs(true_energies_22-pred_energies_22))

        mae_test=mae/(len(true_energies_22))

        print('overall MAE(kcal/mol)=',hartree2kcalmol(mae_test))
        print('overall RMSE(kcal/mol)=',(rmse_1))

        mse=mean_squared_error(true_energies_22,pred_energies_22)

        rmse_test=np.sqrt(mse)
        # rmse_test=hartree2kcalmol(rmse_test)
        print('overall rmse_test(kcal/mol)=',hartree2kcalmol(rmse_test))
######################################################################################

        return rmse, hartree2kcalmol(rmse_test), hartree2kcalmol(mae_test)


