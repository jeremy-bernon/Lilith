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

from ..errors import PriorError
import numpy as np
from warnings import warn


def evaluate_prior(mu, alpha):
    """Evaluate priors"""

    l_prior = 0.
    mutemplate = mu["mutemplate"]
    template_names = []
    for template in mutemplate:
        name = template["extra"]["syst"]
        if "prior" in alpha[name]:
            if alpha[name]["prior"] == "normal":
                val = alpha[name]["val"]
                sigma = alpha[name]["sigma"]
                mean = alpha[name]["mean"]
                l_prior += (val-mean)**2/float(sigma)**2
            if alpha[name]["prior"] == "lognormal":
                val = alpha[name]["val"]
                sigma = alpha[name]["sigma"]
                mean = alpha[name]["mean"]
                l_prior += 2*np.log(val)+((np.log(val)-mean)/float(sigma))**2
    return l_prior






