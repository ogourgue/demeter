import ppmodules.selafin_io_pp as pps

################################################################################
class Telemac(object):
    """Process Telemac output.

    Attributes:

    """

    def __init__(self, filename, vnames = None, step = None):
        """Import Telemac geometry/output file.

        Args:
            filename (str): Name of the file to import.
            vnames (list of str, optional): Names of the variables to import.
                Default to None (all variables are imported).
            step (int, optional): Time step to import (-1 for last time step).
                Default to None (all time steps are imported).

        Todo:
            Possibility to import a list of time steps

        """
        # Open file
        slf = pps.ppSELAFIN(filename)

        # Read header
        self.read_header(slf)

        # Read data
        self.read_data(slf, vnames, step)

    ############################################################################
    def read_header(self, slf):
        """Read Telemac geometry/output file header.

        Args:
            slf (ppSELAFIN object): File to import.

        """
        # Read header
        slf.readHeader()
        vnames = slf.getVarNames()
        vunits = slf.getVarUnits()
        float_type, float_size = slf.getPrecision()
        nelem, npoin, ndp, ikle, ipobo, x, y = slf.getMesh()

        # Attributes
        self.vnames = vnames
        self.vunits = vunits
        self.float_type = float_type
        self.float_size = float_size
        self.nelem = nelem
        self.npoin = npoin
        self.ndp = ndp
        self.ikle = ikle
        self.ipobo = ipobo
        self.x = x
        self.y = y

    ############################################################################
    def read_data(self, slf, vnames, step):
        """Read Telemac geometry/output file data.

        Args:
            slf (ppSELAFIN object): File to import.
            vnames (list of str): Names of the variables to import (None to import all the variables).
            step (int): Time step to import (-1 for last time step; None to import all time steps ).

        """
        # Read times
        slf.readTimes()
        times = slf.getTimes()

        # If "times" is empty, the file only contains geometry.
        if len(times) == 0:
            self.times = []
            self.vnames = []
            self.vunits = []

        else:
            # Times
            # Only the time steps given in "step" are imported. If "step" is not
            # provided (default), all time steps are imported.
            if step is None:
                self.times = times
            else:
                self.times = [times[step]]

            # Variable names and units
            # Only the variables given in "vname" are imported. If "vname" is
            # not provided (default), all variables are imported (self.vnames and self.units are then kept as assigned in read_header)
            if vnames is not None:
                new_vnames = []
                new_vunits = []
                vids = []
                for vname in vnames: # variables to import
                    for vname_file in self.vnames: # variables in file
                        if vname.lower().strip() == vname_file.lower().strip():
                            i = self.vnames.index(vname_file)
                            new_vnames.append(self.vnames[i])
                            new_vunits.append(self.vunits[i])
                            vids.append(i)
                self.vnames = new_vnames
                self.vunits = new_vunits


        """# Number of time steps
        nt = len(self.times)

        # Number of variables
        nv = len(self.vnames)

        # Initialization
        data = np.zeros((nt, nv, npoin))

        # Read data
        for i in range(nt):
            slf.readVariables(step[i])
            data[i, :, :] = slf.getVarValues()[vid, :]"""



