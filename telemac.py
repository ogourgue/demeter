import numpy as np
import scipy.sparse
import scipy.sparse.linalg

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
            * Possibility to import a list of time steps.
            * Option to read or write Telemac file.
            * Update for more than one mud layer/class.

        """
        # Initialize header attributes.
        self.tel_vnames = None
        self.tel_vunits = None
        self.dem_vnames = None
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

        # Initialize parameter attributes
        self.rho = None

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
        tel_vnames = slf.getVarNames()
        tel_vunits = slf.getVarUnits()
        float_type, float_size = slf.getPrecision()
        nelem, npoin, ndp, ikle, ipobo, x, y = slf.getMesh()

        # Update attributes.
        self.tel_vnames = tel_vnames
        self.tel_vunits = tel_vunits
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
            # not provided (default), all variables are imported
            # (self.tel_vnames and self.tel_units are then kept as assigned in
            # read_header).
            if vnames is None:
                tel_vids = list(range(len(self.tel_vnames)))
            else:
                tel_vnames = []
                tel_vunits = []
                tel_vids = []
                # Loop on variables to import
                for vname in vnames:
                    # Loop on variables in the Telemac file
                    for tel_vname in self.tel_vnames:
                        if vname.lower().strip() == tel_vname.lower().strip():
                            i = self.tel_vnames.index(tel_vname)
                            tel_vnames.append(self.tel_vnames[i])
                            tel_vunits.append(self.tel_vunits[i])
                            tel_vids.append(i)
                self.tel_vnames = tel_vnames
                self.tel_vunits = tel_vunits

            # Initialize data array.
            data = np.zeros((len(self.times), len(self.tel_vnames), self.npoin))

            # Read data.
            for i in range(len(self.times)):
                slf.readVariables(step[i])
                data[i, :, :] = slf.getVarValues()[tel_vids, :]

            # Assign data.
            self.assign_data(data)

    ############################################################################
    def assign_data(self, data):
        """Assign Telemac file data to variable attributes.

        Args:
            data (NumPy array): Data imported by read_data

        """
        # Initialize list of variable names (Demeter style).
        self.dem_vnames = []

        # Simple variables.
        for tel_vname in self.tel_vnames:
            if tel_vname.lower().strip() == 'velocity u':
                self.u = data[:, self.tel_vnames.index(tel_vname), :]
                self.dem_vnames.append('velocity u')
            if tel_vname.lower().strip() == 'velocity v':
                self.v = data[:, self.tel_vnames.index(tel_vname), :]
                self.dem_vnames.append('velocity v')
            if tel_vname.lower().strip() == 'free surface':
                self.s = data[:, self.tel_vnames.index(tel_vname), :]
                self.dem_vnames.append('free surface')
            if tel_vname.lower().strip() == 'bottom':
                self.b = data[:, self.tel_vnames.index(tel_vname), :]
                self.dem_vnames.append('bottom')
            if tel_vname.lower().strip() == 'rigid bed':
                self.r = data[:, self.tel_vnames.index(tel_vname), :]
                self.dem_vnames.append('rigid bed')

        # Determine the number of mud classes and layers.
        nc = 0
        nl = 0
        for tel_vname in self.tel_vnames:
            if tel_vname.lower().strip()[:12] == 'coh sediment':
                nc = np.maximum(nc, int(tel_vname[12:14]))
            if tel_vname.lower().strip()[5:13] == 'mass mud':
                nc = np.maximum(nc, int(tel_vname[13:15]))
                nl = np.maximum(nl, int(tel_vname[3:5]))

        # Suspended mud concentration array.
        for tel_vname in self.tel_vnames:
            if tel_vname.lower().strip()[:12] == 'coh sediment':
                if self.t is None:
                    self.t = np.zeros((nc, len(self.times), self.npoin))
                    self.dem_vnames.append('coh sediment')
                i = int(tel_vname[12:14]) - 1
                self.t[i, :, :] = data[:, self.tel_vnames.index(tel_vname), :]

        # Bottom mud mass array.
        for tel_vname in self.tel_vnames:
            if tel_vname.lower().strip()[5:13] == 'mass mud':
                if self.m is None:
                    self.m = np.zeros((nl, nc, len(self.times), self.npoin))
                    self.dem_vnames.append('mass mud')
                i = int(tel_vname[3:5]) - 1
                j = int(tel_vname[13:15]) - 1
                self.m[i, j,  :, :] = data[:, self.tel_vnames.index(tel_vname),
                                           :]

    ############################################################################
    def add_times(self, times):
        """Add a list of time steps to the Telemac instance.

        Args:
            times (list of float): List of time steps.

        """
        self.times = times

    ############################################################################
    def add_parameter(self, param, pname):
        """Add a parameter to the Telemac instance.

        Args:
            param: (List of) Parameter value(s).
            pname: Parameter name.

        """
        # Layers mud concentration (means dry bulk density)
        if pname == 'layers mud concentration':
            if type(param) in [int, float]:
                self.rho = [param]
            else:
                self.rho = param

    ############################################################################
    def add_variable(self, v, vname):
        """Add a variable to the Telemac instance.

        Args:
            v (NumPy array):

        """
        # Initialize list of variable if needed.
        if self.dem_vnames is None:
            self.dem_vnames = []

        # Check if variable is not already in the Telemac instance.
        for dem_vname in self.dem_vnames:
            if vname == dem_vname:
                print('Error: "' + vname + '" already exists in the Telemac ' +
                      'instance.')
                sys.exit()

        # Check array shape.
        if v.shape[-1] != self.npoin:
            print('Error: The last dimension of the added variable must be ' +
                  'the number of Telemac grid nodes.')
            sys.exit()

        if v.shape[-2] != len(self.times):
            print('Error: The second last dimension of the added variable ' +
                  'must be the number of time steps.')
            sys.exit()

        # Check number of array dimensions.
        if vname == 'coh sediment':
            if v.ndim != 3:
                print('Error: "coh sediment" must be an array of shape ' +
                      '(number of mud classes, number of time steps, number ' +
                      'of Telemac grid nodes)')
                sys.exit()
        elif vname == 'mass mud':
            if v.ndim != 4:
                print('Error: "mass mud" must be an array of shape (number ' +
                      'of mud layers, number of mud classes, number of time ' +
                      'steps, number of Telemac grid nodes)')
                sys.exit()
        elif v.ndim != 2:
            print('Error: "' + vname + '" must be an array of shape (number ' +
                  'of time steps, number of Telemac grid nodes)')
            sys.exit()

        # Add variable name.
        self.dem_vnames.append(vname)

        # Add variable.
        if vname == 'velocity u':
            self.u = v
        elif vname == 'velocity v':
            self.v = v
        elif vname == 'free surface':
            self.s = v
        elif vname == 'bottom':
            self.b = v
        elif vname == 'rigid bed':
            self.r = v
        elif vname == 'coh sediment':
            self.t = v
        elif vname == 'mass mud':
            self.m =  v
        else:
            print('Error: "' + vname + '" is not implemented in the Telemac ' +
                  'class of Demeter')
            sys.exit()

    ############################################################################
    def append_times(self, time):
        print('Todo')


    ############################################################################
    def append_variable(self, v, vname):
        print('Todo')

    ############################################################################
    def diffuse_bottom(self, nu, dt, t = 1, step = -1, option = 'mass'):
        """Smooth the bottom surface by diffusion.

        Args:
            nu (float): Diffusion coefficient (m^2/s).
            dt (float): Time step (s).
            t (float, optional): Duration of diffusion (s). Default to 1 s.
            step (int, optional): Telemac time step on which the diffusion is
                applied. Default to -1 for last time step.
            option (str, optional): Option to determine priority when conserving
                both sediment mass and rigid bed surface is in conflict.
                Default to "mass" for priority given to mass conservation.
                "rigid" for priority given to rigid bed surface conservation.

        Todo:
            * MPI implementation.ls
            * Update for more than one mud layer/class.

        """
        # Check required variables.
        if self.b is None:
            print('Error: The variable "bottom" is needed to diffuse the ' +
                  'bottom.')
            sys.exit()
        if self.r is None:
            print('Error: The variable "rigid bed" is needed to diffuse the ' +
                  'bottom.')
            sys.exit()
        if self.m is None:
            print('Error: The variable "mass mud" is needed to diffuse the ' +
                  'bottom.')
            sys.exit()
        if (self.m.shape[0] > 1) or (self.m.shape[1] > 1):
            print('Error: Bottom diffusion for several sediment layers or '
                  'several sediment classes is not implemented.')
            sys.exit()

        # Class attributes.
        x = self.x
        y = self.y
        b0 = self.b[step, :]
        tri = self.ikle - 1
        r0 = self.r[step, :]
        rho = self.rho

        # Diffuse bottom surface.
        bi = diffusion(x, y, b0, tri, nu, dt, t)

        # Conservation.
        if option == 'mass':
            b1 = bi
            r1 = np.minimum(r0, b1)
        elif option == 'rigid':
            r1 = r0
            b0 = np.maximum(bi, r1)
        else:
            print('Error: Option must be either mass or rigid.')
            sys.exit()

        # Update class attributes
        self.b[step, :] = b1
        self.r[step, :] = r1
        self.m[0, 0, step, :] = rho * (b1 - r1)

################################################################################
def diffusion(x, y, f, tri, nu, dt, t = 1):
    """ Diffusion operator.

    Args:
        x (NumPy array): Grid node x-coordinates.
        y (NumPy array): Grid node y-coordinates.
        f (NumPy array): Grid node field values.
        tri (NumPy array): Triangle connectivity table.
        nu (float): Diffusion coefficient (m^2/s).
        dt (float): Time step (s)
        t (float, optional): Duration of diffusion (s). Default to 1 s.

    """
    # Number of grid nodes.
    npoin = len(x)

    # Number of triangles.
    ntri = len(tri)

    # Number of time steps.
    nt = np.int(np.ceil(t / dt))

    # Initialize Jacobian determinant matrix.
    jac = np.zeros(ntri)

    # Initialize matrix A.
    a = scipy.sparse.lil_matrix((npoin, npoin))

    # Initialize matrix b.
    b = scipy.sparse.lil_matrix((npoin, npoin))

    # Compute Jacobian determinant matrix.
    for i in range(len(tri)):
        # Local coordinates.
        x0 = x[tri[i, 0]]
        x1 = x[tri[i, 1]]
        x2 = x[tri[i, 2]]
        y0 = y[tri[i, 0]]
        y1 = y[tri[i, 1]]
        y2 = y[tri[i, 2]]
        # Jacobian determinant.
        jac[i] = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)

    # Feed matrix A.
    for i in range(ntri):
        # Local matrix.
        a_loc = jac[i] / 24 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        # Global indices of local nodes.
        i0 = tri[i, 0]
        i1 = tri[i, 1]
        i2 = tri[i, 2]
        # Feed matrix A.
        a[i0, i0] += a_loc[0, 0]
        a[i0, i1] += a_loc[0, 1]
        a[i0, i2] += a_loc[0, 2]
        a[i1, i0] += a_loc[1, 0]
        a[i1, i1] += a_loc[1, 1]
        a[i1, i2] += a_loc[1, 2]
        a[i2, i0] += a_loc[2, 0]
        a[i2, i1] += a_loc[2, 1]
        a[i2, i2] += a_loc[2, 2]

    # Feed matrix b.
    for i in range(ntri):
        # Local coordinates.
        x0 = x[tri[i, 0]]
        x1 = x[tri[i, 1]]
        x2 = x[tri[i, 2]]
        y0 = y[tri[i, 0]]
        y1 = y[tri[i, 1]]
        y2 = y[tri[i, 2]]
        # Shape function derivatives.
        phi0x = (y1 - y2) / jac[i]
        phi1x = (y2 - y0) / jac[i]
        phi2x = (y0 - y1) / jac[i]
        phi0y = (x2 - x1) / jac[i]
        phi1y = (x0 - x2) / jac[i]
        phi2y = (x1 - x0) / jac[i]
        # Local matrix b.
        b_loc = np.zeros((3, 3))
        b_loc[0, 0] = -.5 * jac[i] * (phi0x * phi0x + phi0y * phi0y)
        b_loc[0, 1] = -.5 * jac[i] * (phi0x * phi1x + phi0y * phi1y)
        b_loc[0, 2] = -.5 * jac[i] * (phi0x * phi2x + phi0y * phi2y)
        b_loc[1, 0] = -.5 * jac[i] * (phi1x * phi0x + phi1y * phi0y)
        b_loc[1, 1] = -.5 * jac[i] * (phi1x * phi1x + phi1y * phi1y)
        b_loc[1, 2] = -.5 * jac[i] * (phi1x * phi2x + phi1y * phi2y)
        b_loc[2, 0] = -.5 * jac[i] * (phi2x * phi0x + phi2y * phi0y)
        b_loc[2, 1] = -.5 * jac[i] * (phi2x * phi1x + phi2y * phi1y)
        b_loc[2, 2] = -.5 * jac[i] * (phi2x * phi2x + phi2y * phi2y)
        # Global indices of local nodes.
        i0 = tri[i, 0]
        i1 = tri[i, 1]
        i2 = tri[i, 2]
        # Feed matrix b.
        b[i0, i0] += b_loc[0, 0]
        b[i0, i1] += b_loc[0, 1]
        b[i0, i2] += b_loc[0, 2]
        b[i1, i0] += b_loc[1, 0]
        b[i1, i1] += b_loc[1, 1]
        b[i1, i2] += b_loc[1, 2]
        b[i2, i0] += b_loc[2, 0]
        b[i2, i1] += b_loc[2, 1]
        b[i2, i2] += b_loc[2, 2]

    # Convert to sparse matrices.
    a = scipy.sparse.csr_matrix(a)
    b = scipy.sparse.csr_matrix(b)

    # Initialize diffused array.
    f1 = f.copy()

    # Diffusion loop.
    for i in range(nt):
        # For each time step, the array to diffuse is the diffused array from
        # the previous time step.
        f0 = f1.copy()
        # Solve linear matrix equation.
        f1 = scipy.sparse.linalg.spsolve(a, a.dot(f0) + nu * dt * b.dot(f0))

    return f1

