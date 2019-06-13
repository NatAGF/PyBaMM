#
# Lithium ion bulter-volmer class
#

import pybamm
from .base_bulter_volmer import BaseModel


class LithiumIon(BaseModel):
    """
    Lithium ion Butler-Volmer class

    *Extends:* :class:`BaseButlerVolmer`
    """

    def __init__(self, param, domain):
        super().__init__(param, domain)

    def _get_exchange_current_density(self, variables):

        c_s_surf = variables[self._domain + " particle surface concentration"]
        c_e = variables[self._domain + " electrolyte concentration"]

        if self._domain == "Negative":
            prefactor = 1 / self.param.C_r_n
        elif self._domain == "Positive":
            prefactor = self.param.gamma_p / self.param.C_r_p
        else:
            raise pybamm.DomainError

        j0 = prefactor * (
            c_e ** (1 / 2) * c_s_surf ** (1 / 2) * (1 - c_s_surf) ** (1 / 2)
        )

        return j0

