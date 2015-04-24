##########################################################################
#
#  This file is part of Lilith
#  made by J. Bernon and B. Dumont
#
#  Web page: http://lpsc.in2p3.fr/projects-th/lilith/
#
#  In case of questions email bernon@lpsc.in2p3.fr dum33@ibs.re.kr
#
#
#    Lilith is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    Lilith is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with Lilith.  If not, see <http://www.gnu.org/licenses/>.
#
##########################################################################

from ..errors import EffectiveMuError
from warnings import warn


def compute_effective_mu(user_mu, mu, alpha):
    """Compute mu_eff(mu,alpha) based on the general template"""

    effective_mu = {}
    mutemplate = mu["mutemplate"]
    prod_modes = ["ggH", "VBF", "WH", "ZH", "ttH"]
    alpha_names = alpha.keys()
    template_names = []
    for template in mutemplate:
        template_names.append(template["extra"]["syst"])
        try:
            alpha_names.index(template["extra"]["syst"])
        except ValueError:
            raise EffectiveMuError(
            'systematic uncertainty '+template["extra"]["syst"]+' not provided in user input file')
#    for name in [name for name in alpha_names if name not in template_names]:
#            warn('systematic uncertainty '+name+
#            ' in user input has no experimental equivalent: will be ignored',Warning,stacklevel=3)

    for prod, decay in user_mu:
        mu_eff = user_mu[prod,decay]
        for template in mutemplate:
            alpha_i = alpha[template["extra"]["syst"]]["val"]
            alpha_0 = template["alpha0"]
            mu_eff += template["phi"]*(alpha_i-alpha_0)
            for pprime in prod_modes:
                mu_eff += user_mu[pprime,decay]*template[prod,pprime]*(alpha_i-alpha_0)

        effective_mu[prod,decay] = mu_eff

    return effective_mu




