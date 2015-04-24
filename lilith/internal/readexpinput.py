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

import sys
try:
    from lxml import etree
except:
    import xml.etree.ElementTree as etree
from ..errors import ExpInputError, ExpInputIOError
from scipy import interpolate
import numpy as np
import math
from . import brsm as BR_SM
from warnings import warn

class ReadExpInput:
    """Read the experimental input in XML and extracts all information."""

    def __init__(self):
        """Initialize the VBF, WH and ZH cross section ratios."""

        self.eff_VVH = BR_SM.geteffVVHfunctions()
        self.mu = []
        self.filepath = ""

    def warning(self, message):
        """Customized warnings."""

        warn("in the file " + self.filepath + ": " + message, Warning,
                      stacklevel=3)

    def get_filelist(self, filepath):
        """get list of files from .list experimental input"""

        expfiles = [] # list of all paths to XML input files

        filepath_split = filepath.split("/")
        expdata_dir = "/".join(filepath_split[:-1])

        try:
            with open(filepath) as linput:
                expfiles = [] # list of all paths to XML input files
                for line in linput:
                    # remove comment, space and new lines from the line
                    line = line.split("#")[0].rstrip("\n").strip()
                    if line == "": # empty line or comment
                        continue
                    if line[0] == "/": # absolute filepath
                        expfiles.append(line)
                    else: # relative filepath
                        expfiles.append(expdata_dir+"/"+line)
        except IOError as e:
            raise ExpInputIOError(
                'I/O error({0}): {1}'.format(e.errno, e.strerror) + '; cannot' +
                ' open the experimental list file "' + filepath + '".')
        
        return expfiles
    

    def read_file(self, filepath):
        """read individual xml files"""

        self.filepath = filepath

        root = self.produce_tree()
        if root.tag != "expmu":
            raise ExpInputError(self.filepath, "root tag is not <expmu>")

        (dim, decay, type) = self.get_mode(root)
        
        self.get_mass(root)
        (experiment, source, sqrts) = self.get_metadata(root)
        
        eff = self.read_eff(root, dim, decay)
        (bestfit, param, grid, Lxy, LChi2min) = self.read_mus(root, dim, type)
        
        self.mutemplate = []
        if type == "eff":
            self.mutemplate = self.get_mutemplate(root)

        self.mu.append({"filepath": self.filepath,
                        "dim": dim, "type": type,
                        "eff": eff,
                        "bestfit": bestfit, "param": param, "grid": grid,
                        "Lxy": Lxy, "LChi2min": LChi2min,
                        "experiment": experiment, "source": source,
                        "sqrts": sqrts, "eff": eff, "mutemplate": self.mutemplate})

    def produce_tree(self):
        """Produce the XML tree with ElementTree."""

        try:
            with open(self.filepath) as f:
                tree = etree.parse(f)
        except IOError as e:
            raise ExpInputIOError(
                'I/O error({0}): {1}'.format(e.errno, e.strerror) + '; cannot' +
                'open the experimental input file "' + self.filepath + '".')

        return tree.getroot()

    def get_mode(self, root):
        """Get the dimension, decay and type of the experimental mu."""
        
        allowed_decays = ["gammagamma", "ZZ", "WW", "Zgamma",
                          "tautau", "bb", "cc", "mumu", "invisible"]

        mandatory_attribs = {"dim":["1", "2"],
                             "type":["n", "f", "eff"]}
        optional_attribs = {"decay": allowed_decays}

        for mandatory_attrib, allowed_values in mandatory_attribs.items():
            if mandatory_attrib not in root.attrib:
                # if "dim" or "type" not in attribute
                raise ExpInputError(self.filepath,
                                    'mandatory attribute of root tag "' +
                                    mandatory_attrib + '" is not present.')
            
            if root.attrib[mandatory_attrib] not in allowed_values:
                # if dim="3" or type="z" for instance
                raise ExpInputError(self.filepath,
                                    'mandatory attribute of root tag "' +
                                    mandatory_attrib + '" has value "' +
                                   root.attrib[mandatory_attrib] +
                                   '" which is unknown. Allowed values are : ' +
                                   str(allowed_values))

        dim = int(root.attrib["dim"])
        type = root.attrib["type"]

        decay = "mixture"

        for optional_attrib, allowed_values in optional_attribs.items():
            if optional_attrib in root.attrib:
                # if "decay" in attribute
                if root.attrib[optional_attrib] not in allowed_values:
                    # if decay="yy" for instance
                    raise ExpInputError(self.filepath,
                                        'optional attribute of root tag "' +
                                        optional_attrib + '" has value "' +
                                        root.attrib[optional_attrib] +
                                        '" which is unknown. Allowed values ' +
                                        + 'are: ' + str(allowed_values))
                else:
                    decay = root.attrib["decay"]

        return (dim, decay, type)
    
    def get_mass(self, root):
        def_mass = 125. # default value
        mass = def_mass
    
        for child in root:
            if child.tag == "mass":
                try:
                    mass = float(child.text)
                except TypeError: # empty tag is of type NULL
                    self.warning('<mass> tag is empty; ' +
                                 'setting the mass to ' + def_mass + ' GeV')
                    mass = def_mass
                except ValueError:
                    raise ExpInputError(self.filepath,
                                        "value of <mass> tag is not a number")
        self.mass = mass

    def get_metadata(self, root):
        experiment = ""
        source = ""
        sqrts = ""

        for child in root:
            if child.tag == "experiment":
                experiment = child.text
            if child.tag == "source":
                source = child.text
            if child.tag == "sqrts":
                sqrts = child.text
    
        return (experiment, source, sqrts)

    def read_eff(self, root, dim, decay):
        allowed_decays = ["gammagamma", "ZZ", "WW", "Zgamma",
                          "tautau", "bb", "cc", "mumu", "invisible"]

        # read the efficiencies
        if dim == 1: # 1D signal strength
            eff = {"x": {}}
            axis_label = "x"

            mandatory_attribs = {"prod": ["ggH", "VVH", "VBF", "VH", "WH", "ZH", "ttH"]}
            if decay == "mixture":
                mandatory_attribs["decay"] = allowed_decays

            for child in root:
                if child.tag == "eff":
                    for mandatory_attrib, allowed_values in mandatory_attribs.items():
                        if mandatory_attrib not in child.attrib:
                            # if "axis" or "prod" not in attribute
                            raise ExpInputError(self.filepath,
                                                'mandatory attribute of <eff> tag "' +
                                                mandatory_attrib + '" is not present.')
                        if child.attrib[mandatory_attrib] not in allowed_values:
                            # if axis="z" or prod="yy"
                            raise ExpInputError(self.filepath,
                                                'mandatory attribute of <eff> tag "' +
                                                mandatory_attrib + '" has value "' +
                                                child.attrib[mandatory_attrib] + '" which is unknown. Allowed values are : ' + str(allowed_values))

                    prod_label = child.attrib["prod"]
                    if decay == "mixture":
                        decay_label = child.attrib["decay"]
                    else:
                        decay_label = decay

                    if (prod_label,decay_label) in eff["x"]:
                        self.warning('<eff> tag with prod="' + prod_label +
                                     '" and decay="' + decay_label +
                                     '" is being redefined.')
                        
                    try:
                        eff[axis_label][prod_label,decay_label] = float(child.text)
                    except TypeError: # empty tag is of type NULL
                        self.warning('<eff> tag for axis="' + axis_label +
                                     '", prod="' + prod_label + '" and decay="' +
                                     decay_label + '" is empty; setting to ' +
                                     'default value of 0')
                        eff[axis_label][prod_label,decay_label] = 0.
                    except ValueError:
                        raise ExpInputError(self.filepath,
                                            'value of <eff> tag with axis="' + axis_label +
                                            '" and prod="' + prod_label + '" and decay="' + decay_label + '" is not a number')

        else: # 2D signal strength
            eff = {"x": {}, "y": {}}

            mandatory_attribs = {"axis": ["x", "y"],
                             "prod": ["ggH", "VVH", "VBF", "VH", "WH", "ZH", "ttH"]}
            if decay == "mixture":
                mandatory_attribs["decay"] = allowed_decays
            
            for child in root:
                if child.tag == "eff":
                    for mandatory_attrib, allowed_values in mandatory_attribs.items():
                        if mandatory_attrib not in child.attrib:
                            # if "axis" or "prod" not in attribute
                            raise ExpInputError(self.filepath,
                                                'mandatory attribute of <eff> tag "' +
                                                mandatory_attrib + '" is not present.')
                        if child.attrib[mandatory_attrib] not in allowed_values:
                            # if axis="z" or prod="yy"
                            raise ExpInputError(self.filepath,
                                                'mandatory attribute of <eff> tag "' +
                                                mandatory_attrib + '" has value "' +
                                                child.attrib[mandatory_attrib] + '" which is unknown. Allowed values are : ' + str(allowed_values))

                        axis_label = child.attrib["axis"]
                        prod_label = child.attrib["prod"]
                        if decay == "mixture":
                            decay_label = child.attrib["decay"]
                        else:
                            decay_label = decay


                    if (prod_label,decay_label) in eff[axis_label]:
                        self.warning('<eff> tag with axis="' + axis_label +
                                     '", prod="' + prod_label +
                                     '" and decay="' +decay_label +
                                     '" is being redefined.')
                        
                    try:
                        eff[axis_label][prod_label,decay_label] = float(child.text)
                    except TypeError: # empty tag is of type NULL
                        self.warning('<eff> tag for axis="' + axis_label +
                                     '", prod="' + prod_label + '" and decay="' +
                                     decay_label + '" is empty; setting to ' +
                                     'default value of 0')
                        eff[axis_label][prod_label,decay_label] = 0.
                    except ValueError:
                        raise ExpInputError(self.filepath,
                                            'value of <eff> tag with axis="' + axis_label +
                                            '" and prod="' + prod_label + '" and decay="' + decay_label + '" is not a number')

        effWH = self.eff_VVH["eff_WH"](self.mass)
        effZH = self.eff_VVH["eff_ZH"](self.mass)
        
        effVBF = self.eff_VVH["eff_VBF"](self.mass)
        effVH = self.eff_VVH["eff_VH"](self.mass)
        effVWH = effVH * effWH
        effVZH = effVH * effZH

        multiprod = {"VH": {"WH": effWH, "ZH": effZH}, "VVH": {"VBF": effVBF, "WH": effVWH, "ZH": effVZH}}

        self.check_multiprod(eff["x"], multiprod)
        if dim == 2:
            self.check_multiprod(eff["y"], multiprod)

        # now all reduced couplings have been properly defined, one can
        # delete all multiparticle labels
        effCleanX = eff["x"].copy()
        for (p,decay) in eff["x"]:
            if p in multiprod:
                del effCleanX[p,decay]

        if dim == 2:
            effCleanY = eff["y"].copy()

            for (p,decay) in eff["y"]:
                if p in multiprod:
                    del effCleanY[p,decay]

        eff["x"] = effCleanX
        if dim == 2:
            eff["y"] = effCleanY

        # check that efficiencies add up to 1, otherwise issue a warning
        # or an error
        for axis in eff:
            sumeff = 0
            for prod in eff[axis]:
                sumeff += eff[axis][prod]
            
            if sumeff == 0:
                raise ExpInputError(self.filepath,
                                    "no <eff> tag found for " + axis + " axis")

            if sumeff < 0.99:
                self.warning('the sum of efficiencies for axis="' +
                             axis + '" is less than 1 (value: ' +
                             str(sumeff) + ')')
            elif sumeff > 1.01:
                raise ExpInputError(self.filepath,
                                    'the sum of efficiencies for axis="' +
                                    axis + '" is greater than 1 (value: ' +
                                    str(sumeff) + ')')

        return eff


    def read_mus(self, root, dim, type):
        # first, read the bestfit
        bestfit = {}
        LChi2min = 0
                
        for child in root:
            if child.tag == "bestfit":
                if type == "f" or type == "eff":
                    self.warning('block <bestfit> in experimental mu of ' +
                                 'type "full"... skipping.')
                    continue

                if dim == 1:
                    # read directly the number
                    
                    if "x" in bestfit:
                        self.warning("redefinition of the bestfit...")
                    
                    try:
                        bestfit["x"] = float(child.text)
                    except TypeError: # empty tag is of type NULL
                        self.warning('<x> tag in <bestfit> block is empty; ' +
                                     'setting to 0')
                        bestfit["x"] = 0.
                    except ValueError:
                        raise ExpInputError(self.filepath,
                                            "value of <besfit> tag is not a number")
            
                elif dim == 2:
                    bestfit_allowedsubtags = ["x", "y"]
                
                    for bfit in child:
                        if bfit.tag in bestfit_allowedsubtags:
                            
                            if bfit.tag in bestfit:
                                self.warning("redefinition of the bestfit...")
                        
                            try:
                                bestfit[bfit.tag] = float(bfit.text)
                            except TypeError: # empty tag is of type NULL
                                self.warning('<' + bfit.tag + '> tag in ' +
                                             '<bestfit> block is empty; ' +
                                             'setting to 0')
                                bestfit[bfit.tag] = 0.
                            except ValueError:
                                raise ExpInputError(self.filepath,
                                                    "value of <besfit> tag is not a number")
                        else:
                            raise ExpInputError(self.filepath,
                                                "subtag in bestfit not known")
                    
                if dim == 1 and "x" not in bestfit:
                    raise ExpInputError(self.filepath,
                                        "best fit point should be specified.")
                if dim == 2 and ("x" not in bestfit or "y" not in bestfit):
                    raise ExpInputError(self.filepath,
                                        "best fit point should be specified for x and y.")
        
        # then, read the param...
        param = {}
        
        for child in root:
            if child == "param":
                break
        param_tag = child

        param["uncertainty"] = {}
        
        for child in param_tag:
            if child.tag is etree.Comment:
                # ignore all comments
                continue

            if dim == 1:
                if child.tag == "uncertainty":
                    if "side" not in child.attrib:
                        try:
                            unc_value = float(child.text)
                        except TypeError: # empty tag is of type NULL
                            self.warning('<uncertainty> tag is empty; ' +
                                         'setting to 0')
                            unc_value = 0.
                        except ValueError:
                            raise ExpInputError(self.filepath,
                                                "value of <uncertainty> tag is not a number")

                        param["uncertainty"]["left"] = unc_value
                        param["uncertainty"]["right"] = unc_value
                    else:
                        if child.attrib["side"] not in ["left", "right"]:
                            raise ExpInputError(self.filepath,
                                                "attribute of uncertainty is not left nor right")
                        else:
                            try:
                                unc_value = float(child.text)
                            except TypeError: # empty tag is of type NULL
                                self.warning('<uncertainty> tag is empty; ' +
                                             'setting to 0')
                                unc_value = 0.
                            except ValueError:
                                raise ExpInputError(self.filepath,
                                                    "value of <uncertainty> tag is " +
                                                    "not a number")
                                
                        param["uncertainty"][child.attrib["side"]] = unc_value
                else:
                    raise ExpInputError(self.filepath,
                                        "subtag or param should be uncertainty")
                    
            elif dim == 2:
                allowed_tags = ["a", "b", "c"]
                if child.tag not in allowed_tags:
                    raise ExpInputError(self.filepath,
                                        "only allowed tags are <a>, <b> and <c> in " +
                                        "block param in 2D normal mode")

                if child.tag in param:
                    self.warning("redefinition of tag <" + child.tag + ">")
                    
                try:
                    param_value = float(child.text)
                except TypeError: # empty tag is of type NULL
                    self.warning('<' + child.tag + '> tag is empty; ' +
                                 'setting to 0')
                    param_value = 0.
                except ValueError:
                    raise ExpInputError(self.filepath,
                                        "value of <" + child.tag + "> tag is not a number")
                
                param[child.tag] = param_value
        
        # check that everything is there
        if type == "n" and dim == 1:
            if ("uncertainty" not in param or
                "left" not in param["uncertainty"] or
                "right" not in param["uncertainty"]):
                raise ExpInputError(self.filepath,
                                    "uncertainties are not given consistently in block param")
        elif type == "n" and dim == 2:
            if "a" not in param or "b" not in param or "c" not in param:
                raise ExpInputError(self.filepath,
                                    "a, b, c tags are not given in block param")
        
        # or the grid
        grid = {}
        Lxy = None
        
        for child in root:
            if child == "grid":
                break
        grid_raw = child.text

        if type == "f" and dim == 1:
            x = []
            L = []
        
            grid_raw = grid_raw.strip("\n").strip().split("\n")
        
            i = -1
        
            for line in grid_raw:
                tab = line.split()
                if len(tab) != 2:
                    raise ExpInputError(self.filepath,
                                        'incorrect <grid> entry on line "' + line + '"')

                cur_x = float(tab[0])
                cur_L = float(tab[1])

                if cur_x not in x:
                    x.append(cur_x)
                    L.append(cur_L)
                    i += 1
                else:
                    i = x.index(cur_x)

            grid["x"] = x
            grid["L"] = L
            LChi2min = min(grid["L"])

            Lxy = interpolate.UnivariateSpline(grid["x"], grid["L"], k = 3, s = 0)
        
        elif type == "f" and dim == 2 or type == "eff" and dim == 2:
            x = []
            y = []
            L = []
            
            grid_raw = grid_raw.strip("\n").strip().split("\n")
            
            i = -1
            
            for line in grid_raw:
                tab = line.split()
                if len(tab) != 3:
                    raise ExpInputError(self.filepath,
                                        'incorrect <grid> entry on line "' + line + '"')

                cur_x = float(tab[0])
                cur_y = float(tab[1])
                cur_L = float(tab[2])

                if cur_x not in x:
                    x.append(cur_x)
                    L.append([])
                    i += 1
                else:
                    i = x.index(cur_x)

                if cur_y not in y:
                    y.append(cur_y)

                L[i].append(cur_L)

            grid["x"] = np.array(x)
            grid["y"] = np.array(y)
            grid["L"] = np.array(L)
            
            LChi2min = min(min(p[1:]) for p in grid["L"])

            Lxy = interpolate.RectBivariateSpline(grid["x"],
                    grid["y"], grid["L"])
            
        return (bestfit, param, grid, Lxy, LChi2min)

    def check_multiprod(self, eff_dict, multiprod):
        """..."""

        # check consistency in the definition of multi-particle labels
        for (prod,decay) in eff_dict:
            if prod in multiprod:
                # there is a multi-particle label
                # in that case, check if individual particle are also defined
                for label in multiprod[prod]:
                    if (label,decay) in eff_dict:
                        raise ExpInputError(self.filepath,
                                            '<eff> tags for "' + label + '" and "' +
                                            prod + '" cannot both be defined')
                # also, only one multi-particle label can be used (VH or VVH),
                # not both
                for label in multiprod:
                    if label != prod and (label,decay) in eff_dict:
                        raise ExpInputError(self.filepath,
                                            '<eff> tags for "' + label + '" and "' +
                                             prod + '" cannot both be defined')

        # it is consistent, resolve multi-particle labels
        new_eff = {}
        for (prod,decay) in eff_dict:
            if prod in multiprod:
                for label in multiprod[prod]:
                    new_eff[label,decay] = eff_dict[prod,decay]*multiprod[prod][label]

        for elem in new_eff:
            eff_dict[elem] = new_eff[elem]


    def get_mutemplate(self, root):
    
        mutemplate=[]
        accepted_prod = ["ggH", "VVH", "ttH", "VBF", "VH", "WH", "ZH"]
        
        for child in root:
            if child.tag == "mutemplate":
                mutemplate_cur = {"extra":{}}
                mutemplate_block = child
                if "syst" not in child.attrib:
                    raise ExpInputError(self.filepath, 'syst attribute of tag <mutemplate>' +
                                                        ' not defined')
                else:
                    mutemplate_cur["extra"]["syst"] = child.attrib["syst"]
                
                for child in mutemplate_block:
                    if child.tag == "alpha0":
                        if ("alpha0") in mutemplate_cur:
                            self.warning('<alpha0> tag is being ' + 'redefined')
                        
                        try:
                            mutemplate_cur["alpha0"] = float(child.text)
                        except TypeError: # empty tag is of type NULL
                            mutemplate_cur["alpha0"] = 0.
                            self.warning('<alpha0> tag is empty; ' +
                                     'setting it to 0')
                        except ValueError:
                            raise ExpInputError(
                                self.filepath, 'value of the <alpha0> tag is not a number.')
                              
                    elif child.tag == "phi":
                        if ("phi") in mutemplate_cur:
                            self.warning('<phi> tag is being ' + 'redefined')
                        
                        try:
                            mutemplate_cur["phi"] = float(child.text)
                        except TypeError: # empty tag is of type NULL
                            mutemplate_cur["phi"] = 0.
                            self.warning('<phi> tag is empty; ' +
                                     'setting it to 0')
                        except ValueError:
                            raise ExpInputError(
                                self.filepath, 'value of the <phi> tag is not a number.')
                              
                    elif child.tag == "eta":
                        if "p" not in child.attrib or "pprime" not in child.attrib:
                            self.warning('attribute "p" or "pprime" is ' +
                                         'missing in <eta> tag')
                        elif child.attrib["p"] not in accepted_prod:
                            self.warning('<eta> tag with p="' +
                                         child.attrib["p"] + '" and pprime="' +
                                         child.attrib["pprime"] + '" has unknown p')
                        elif child.attrib["pprime"] not in accepted_prod:
                            self.warning('<eta> tag with p="' +
                                         child.attrib["p"] + '" and pprime="' +
                                         child.attrib["pprime"] + '" has unknown pprime')
                        else:
                            p = child.attrib["p"]
                            pprime = child.attrib["pprime"]
                            if (p,pprime) in mutemplate_cur:
                                self.warning('<eta> tag with p="' + p +
                                             '" and pprime="' + pprime + '" is being ' +
                                             'redefined')

                            try:
                                mutemplate_cur[p,pprime] = float(child.text)
                            except TypeError: # empty tag is of type NULL
                                mutemplate_cur[p, pprime] = 0.
                                self.warning('<eta> tag with p="' +
                                             p + '" and pprime="' +
                                             pprime + '" is empty; ' +
                                             'setting it to 0')
                            except ValueError:
                                raise ExpInputError(self.filepath,
                                    'value of the <eta> tag with p="' + p +
                                    '" and pprime="' + pprime + '" is not a number.')
                              
                if "alpha0" not in mutemplate_cur:
                    raise ExpInputError(self.filepath, "tag <alpha0> not defined")
                
                
                # ----------------------------------
                # checking consistency of the input
                # ----------------------------------

                # --- checking the multiparticle labels
                multiprod = {"VH": ["WH", "ZH"]}
                multiprod2 = {"VVH": ["VBF", "WH", "ZH"]}

                # -- first: p
                for multip, p_list in multiprod.items():
                    self.check_multiprod_mutemplate(mutemplate_cur, multip, p_list)

                for multip, p_list in multiprod2.items():
                    self.check_multiprod_mutemplate(mutemplate_cur, multip, p_list)

                # now, one can delete all multiprod labels
                for key,mu_value in mutemplate_cur.items():
                    if key == "extra" or key == "phi" or key == "alpha0":
                        continue
                    p,pprime = key
                    if p in multiprod or p in multiprod2 or pprime in multiprod or pprime in multiprod2:
                        del mutemplate_cur[p,pprime]


                # --- checking that all template coefficients are present in the
                #     input
                mandatory_prod = ["ggH", "VBF", "WH", "ZH", "ttH"]

                mandatory_mutemplate = []
                for p in mandatory_prod:
                    for pprime in mandatory_prod:
                        mandatory_mutemplate.append((p,pprime))

                for (p,pprime) in mandatory_mutemplate:
                    if (p,pprime) not in mutemplate_cur:
                        mutemplate_cur[p,pprime] = 0.

                if "phi" not in mutemplate_cur:
                    self.warning('phi value not specified; setting it to 0')
                    mutemplate_cur["phi"] = 0.

                mutemplate.append(mutemplate_cur)
                
        return mutemplate

    def check_multiprod_mutemplate(self, mutemplate_cur, multip, p_list):
        """for mu template"""
        
        for key,mu_value in mutemplate_cur.items():
            if key == "extra" or key == "phi" or key == "alpha0":
                continue
            p,pprime = key
            if pprime == multip:
                if set([(p,subprod) for subprod in p_list]).issubset(mutemplate_cur):
                    # if multiprod and individual pprime are all
                    # defined, check consistency
                    for subprod in p_list:
                        if mu_value != mutemplate_cur[p,subprod]:
                            self.warning('inconsistent definition of ' +
                                         'eta (p="' + p + '", pprime="' +
                                         pprime +'"); ' +
                                         'skipping this definition')
                else:
                    # only multiprod label is present, or multiprod
                    # label and tags for only part of the associated prod
                    given_mu = [(p,subprod) for subprod in p_list if
                                (p,subprod) in mutemplate_cur]
                    
                    # check if an eta tag is given inconsistently
                    for (p,subprod) in given_mu:
                        if mu_value != mutemplate_cur[p,subprod]:
                            raise ExpInputError(self.filepath,
                                'inconsistent definition of eta for p="' +
                                p + '", pprime ="' + pprime + '"')
            
                    if len(given_mu) == 0:
                        # only multiparticle tag is defined
                        for subprod in p_list:
                            mutemplate_cur[p,subprod] = mutemplate_cur[p,multip]
                    else:
                        # when multiprod tag and part of the prod tags
                        # are given, with equal values
                        other_mu = [(p,subprod) for subprod in p_list if
                                    (p,subprod) not in mutemplate_cur]
                        for (p,op) in other_mu:
                            mutemplate_cur[p,op] = mutemplate_cur[p,multip]
                        self.warning('eta for (p="' + p +
                                     '", pprime="' + pprime +'") and ' +
                                     str(given_mu) + ' are both defined and ' +
                                     'equal; assumed that the eta '+
                                     str(other_mu) + ' is the same"')
            elif p == multip:
                print [(subprod,pprime) for subprod in p_list]
                if set([(subprod,pprime) for subprod in p_list]).issubset(mutemplate_cur):
                    # if multiprod and individual prod are all
                    # defined, check consistency
                    for subprod in p_list:
                        if mu_value != mutemplate_cur[subprod,pprime]:
                            self.warning('inconsistent definition of ' +
                                         'eta (p="' + p + '", pprime="' +
                                         pprime +'"); ' +
                                         'skipping this definition')
                else:
                    # only multiprod label is present, or multiprod
                    # label and tags for only part of the associated prod
                    given_mu = [(subprod,pprime) for subprod in p_list if
                                (subprod,pprime) in mutemplate_cur]
                    
                    # check if a particle tag is given inconsistently
                    for (subprod,pprime) in given_mu:
                        if mu_value != mutemplate_cur[subprod,pprime]:
                            raise ExpInputError(self.filepath,
                                'inconsistent definition of eta for p="' +
                                p + '", pprime ="' + pprime + '"')
            
                    if len(given_mu) == 0:
                        # only multiparticle tag is defined
                        for subprod in p_list:
                            mutemplate_cur[subprod,pprime] = mutemplate_cur[multip,pprime]
                    else:
                        # when multiprod tag and part of the prod tags
                        # are given, with equal values
                        other_mu = [(subprod,pprime) for subprod in p_list if
                                    (subprod,pprime) not in mutemplate_cur]
                        for (op,pprime) in other_mu:
                            mutemplate_cur[op,pprime] = mutemplate_cur[multip,pprime]
                        self.warning('eta for (p="' + p +
                                     '", pprime="' + pprime + '") and ' +
                                     str(given_mu) + ' are both defined and ' +
                                     'equal; assumed that the eta '+
                                     str(other_mu) + ' is the same"')
