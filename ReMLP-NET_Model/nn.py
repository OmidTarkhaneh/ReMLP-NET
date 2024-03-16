
import torch

import torch.utils.tensorboard
from torch import Tensor
from typing import Tuple, Optional, NamedTuple
import torch.nn.functional as F
from aev import *
from units import *
from collections import OrderedDict


# helper function to convert energy unit from Hartree to kcal/mol


# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SpeciesEnergies(NamedTuple):
    species: Tensor
    energies: Tensor


class SpeciesCoordinates(NamedTuple):
    species: Tensor
    coordinates: Tensor



class RetrieviumModel(torch.nn.Module):
    """model that compute energies from species and AEVs.

    Different atom types might have different modules, when computing
    energies, for each atom, the module for its corresponding atom type will
    be applied to its AEV, after that, outputs of modules will be reduced along
    different atoms to obtain molecular energies.

    .. warning::

        The species must be indexed in 0, 1, 2, 3, ..., not the element
        index in periodic table. Check :class:`torchani.SpeciesConverter`
        if you want periodic table indexing.

    .. note:: The resulting energies are in Hartree.

    Arguments:
        modules (:class:`collections.abc.Sequence`): Modules for each atom
            types. Atom types are distinguished by their order in
            :attr:`modules`, which means, for example ``modules[i]`` must be
            the module for atom type ``i``. Different atom types can share a
            module by putting the same reference in :attr:`modules`.
    """


    def __init__(self, modules):
        super(RetrieviumModel, self).__init__()
        self.mymodule=modules

    def forward(self, species_aev: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None) -> SpeciesEnergies:
        species, aev = species_aev
        assert species.shape == aev.shape[:-1]


        atomic_energies = self._atomic_energies((species, aev))


        return SpeciesEnergies(species, torch.sum(atomic_energies, dim=1))

    @torch.jit.export
    def _atomic_energies(self, species_aev: Tuple[Tensor, Tensor]) -> Tensor:
        # Obtain the atomic energies associated with a given tensor of AEV's
        species, aev = species_aev

        aev_temp=aev
        aev=aev.view(aev.size(0), -1)
        aev = F.normalize(aev, p=2, dim=1)
        aev=aev.view_as(aev_temp)


        # print('species', species)
        assert species.shape == aev.shape[:-1]
        species_ = species.flatten()
        aev = aev.flatten(0, 1)

        output = aev.new_zeros(species_.shape)
        #  Here we defined a sigle model in ReMLP-NET Model, we see that the model only accept the whole AEV vector
        m = self.mymodule
        output=m(aev).flatten()

        output = output.view_as(species)
        return output




class Sequential(torch.nn.ModuleList):
    """Modified Sequential module that accept Tuple type as input"""

    def __init__(self, *modules):
        super().__init__(modules)

    def forward(self, input_: Tuple[Tensor, Tensor],  # type: ignore
                cell: Optional[Tensor] = None,
                pbc: Optional[Tensor] = None):
        for module in self:
            input_ = module(input_, cell=cell, pbc=pbc)
        return input_





