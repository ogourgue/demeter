import numpy as np
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
        # Initialize header attributes.
        self.vnames = None
        self.vunits = None
        self.float_type = None
        self.float_size = None
        self.nelem = None
        self.npoin = None
        self.ndp = None
        self.ikle = None
        self.ipobo = None
        self.x = None
        self.y = None
        self.times = None

        # Initialize variable attributes (only Telemac variables used in
        # Demeter, the list will be updated depending on needs).
        self.u = None   # Velocity u
        self.v = None   # Velocity v
        self.s = None   # Free surface
        self.b = None   # Bottom
        self.t = None   # Suspended mud concentration
        self.r = None   # Rigid bed
        self.m = None   # Bottom mud mass

        # Open file.
        slf = pps.ppSELAFIN(filename)

        # Read header.
        self.read_header(slf)

        # Read data.
        self.read_data(slf, vnames, step)

    ############################################################################
    def read_header(self, slf):
        """Read Telemac geometry/output file header.

        Args:
            slf (ppSELAFIN object): File to import.

        """
        # Read header.
        slf.readHeader()
        vnames = slf.getVarNames()
        vunits = slf.getVarUnits()
        float_type, float_size = slf.getPrecision()
        nelem, npoin, ndp, ikle, ipobo, x, y = slf.getMesh()

        # Update attributes.
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
        # Read times.
        slf.readTimes()
        times = slf.getTimes()

        # No data to read if "times" is empty.
        if len(times) > 0:
            # Times.
            # Only the time steps given in "step" are imported. If "step" is not
            # provided (default), all time steps are imported.
            # Convert "step" into a list and treat last time step.
            if step is None:
                self.times = times
                step = range(len(times))
            else:
                self.times = [times[step]]
                if step == -1:
                    step = [len(times) - 1]
                else:
                    step = [step]

            # Variable names and units.
            # Only the variables given in "vname" are imported. If "vname" is
            # not provided (default), all variables are imported (self.vnames
            # and self.units are then kept as assigned in read_header).
            if vnames is None:
                vids = list(range(len(self.vnames)))
            else:
                new_vnames = []
                new_vunits = []
                vids = []
                # Loop on variables to import
                for vname in vnames:
                    # Loop on variables in the Telemac file
                    for vname_file in self.vnames:
                        if vname.lower().strip() == vname_file.lower().strip():
                            i = self.vnames.index(vname_file)
                            new_vnames.append(self.vnames[i])
                            new_vunits.append(self.vunits[i])
                            vids.append(i)
                self.vnames = new_vnames
                self.vunits = new_vunits

            # Initialize data array.
            data = np.zeros((len(self.times), len(self.vnames), self.npoin))

            # Read data.
            for i in range(len(self.times)):
                slf.readVariables(step[i])
                data[i, :, :] = slf.getVarValues()[vids, :]

            # Assign variables.
            self.assign_variables(data)

    ############################################################################
    def assign_variables(self, data):
        """Assign Telemac file data to variable attributes.

        Args:
            data (NumPy array): data imported by read_data

        """
        # Simple variables.
        for vname in self.vnames:
            if vname.lower().strip() == 'velocity u':
                self.u = data[:, self.vnames.index(vname), :]
            if vname.lower().strip() == 'velocity v':
                self.v = data[:, self.vnames.index(vname), :]
            if vname.lower().strip() == 'free surface':
                self.s = data[:, self.vnames.index(vname), :]
            if vname.lower().strip() == 'bottom':
                self.b = data[:, self.vnames.index(vname), :]
            if vname.lower().strip() == 'rigid bed':
                self.r = data[:, self.vnames.index(vname), :]

        # Determine the number of mud classes.
        nc = 0
        for vname in self.vnames:
            if vname.lower().strip()[:12] == 'coh sediment':
                nc = np.maximum(nc, int(vname[12:14]))

        # Determine the number of mud layers.
        nl = 0
        for vname in self.vnames:
            if vname.lower().strip()[5:13] == 'mass mud':
                nl = np.maximum(nl, int(vname[3:5]))

        # Suspended mud concentration array.
        self.t = np.zeros((nc, len(self.times), self.npoin))
        for vname in self.vnames:
            if vname.lower().strip()[:12] == 'coh sediment':
                i = int(vname[12:14]) - 1
                self.t[i, :, :] = data[:, self.vnames.index(vname), :]

        # Bottom mud mass array.
        self.m = np.zeros((nl, nc, len(self.times), self.npoin))
        for vname in self.vnames:
            if vname.lower().strip()[5:13] == 'mass mud':
                i = int(vname[3:5]) - 1
                j = int(vname[13:15]) - 1
                self.m[i, j,  :, :] = data[:, self.vnames.index(vname), :]

    ############################################################################
    def add_variable(self, v, vname):
        """Add variable to Telemac instance.

        Args:
            v (NumPy array): data

        """
        print('todo')







