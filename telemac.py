import ppmodules.selafin_io_pp as pps


class Telemac(object):
    """Process Telemac output.

    Attributes:

    """

    def __init__(self, filename, vnames = None, step = None):
        """Import Telemac output file.

        Args:
            filename (str): Name of the file to import.
            vnames (list of str, optional): Names of the variables to import.
                Default to None (all variables are imported).
            step (int, optional): Time step to import (-1 for last time step).
                Default to None (all time steps are imported).

        Todo:
            Possibility to import a list of time steps

        """
        # Open file, read header and time steps
        slf = pps.ppSELAFIN(filename)
        slf.readHeader()
        slf.readTimes()

        # Store data into lists and arrays
        times = slf.getTimes()
        varnames = slf.getVarNames()
        varunits = slf.getVarUnits()
        float_type, float_size = slf.getPrecision()
        nelem, npoin, ndp, ikle, ipobo, x, y = slf.getMesh()

        # Geometry attributes
        self.float_type = float_type
        self.float_size = float_size
        self.nelem = nelem
        self.npoin = npoin
        self.ndp = ndp
        self.ikle = ikle
        self.ipobo = ipobo
        self.x = x
        self.y = y

        # If "times" is empty, the file only contains geometry and there is no
        # other attribute.

        # Variable attributes
        if len(times) > 0:

            # Times
            # Only the time steps given in "step" are imported. If "step" is not
            # provided (default), all time steps are imported.
            if step is None:
                self.times = times
            else:
                self.times = [times[step]]

            # Variable names and units
            # Only the variables given in "vname" are imported. If "vname" is
            # not provided (default), all variables are imported.
            if vnames is None:
                self.vnames = varnames
                self.vunits = varunits
            else:
                self.vnames = []
                self.vunits = []
                vids = []
                for vname in vnames: # variables to import
                    for varname in varnames: # variables in Telemac output file
                        if vname.lower().strip() == varname.lower().strip():
                            i = varnames.index(varname)
                            self.vnames.append(varnames[i])
                            self.vunits.append(varunits[i])

