#!/usr/bin/env python3
# -*- coding: utf-8 -*-

def _log(mss):
   sys.stdout.write(str(datetime.datetime.now())+": "+str(mss)+"\n")
   sys.stdout.flush()
   return

def _getScriptCall(sep=" "):
   cmd=sys.argv
   cmdtxt=""
   for cmd0 in cmd:
      cmdtxt=cmdtxt+sep+cmd0
   cmdtxt=cmdtxt.strip()
   return cmdtxt

# wrapping the crappy implementation of the vtf writer
def _vtf_pid_map(system, types='all'):
    """
    Generates a VTF particle index map to ESPResSo ``id``.
    This fills the gap for particle ID's as required by VMD

    Parameters
    ----------
    system: espressomd.System() object
    types : :obj:`str`
            Specifies the particle types. The id mapping depends on which
            particles are going to be printed. This should be the same as
            the one used in writevsf() and writevsf().
    Returns
    -------
    dict:   A dictionary where the values are the VTF indices and the keys are the ESPresSo particle ``id``
    """

    if not hasattr(types, '__iter__'):
        types = [types]
    if types == "all":
        types = [types]
    id_to_write = []
    for p in system.part:
        for t in types:
            if p.type == t or t == "all":
                id_to_write.append(p.id)
    return dict(zip(id_to_write, range(len(id_to_write))))


def _writevsf(system, fp, types='all', radiuses={}, dipoles = False):
    """
    writes a VST (VTF Structure Format) to a file.
    This can be used to write the header of a VTF file.

    Parameters
    ----------
    system: espressomd.System() object
    types : :obj:`str`
            Specifies the particle types. The string 'all' will write all particles
    fp : file
               File pointer to write to.

    """

    vtf_index = _vtf_pid_map(system, types)
    fp.write("unitcell {} {} {}\n".format(*(system.box_l)))
    if dipoles:
       dips = []
    for pid, vtf_id, in vtf_index.items():
        if system.part[pid].type in radiuses:
           rad=radiuses[system.part[pid].type]
        else:
           rad=1
        fp.write("atom {} radius {} name {} type {} \n".format(vtf_id, rad,
                                                              system.part[
                                                                  pid].type,
                                                              system.part[pid].type))
        if dipoles and system.part[pid].dipm > 0.:
           dips.append(rad)
    if dipoles:
      ndips = len(dips)
      for i in range(ndips):
         fp.write("atom {} radius {} name {} type {} \n".format(ndips + i, dips[i],
                                                              1000,
                                                              1000))
    for pid, vtf_id, in vtf_index.items():
        for b in system.part[pid].bonds:
            if (system.part[b[1]].id in vtf_index):
                fp.write("bond {}:{}\n".format(
                    vtf_id, vtf_index[system.part[b[1]].id]))


def _writevcf(system, fp, types='all',folded=False, dipoles = False):
    """
    writes a VCF (VTF Coordinate Format) to a file.
    This can be used to write a timestep to a VTF file.

    Parameters
    ----------
    system: espressomd.System() object
    types : :obj:`str`
            Specifies the particle types. The string 'all' will write all particles
    fp : file
               File pointer to write to.

    """
    vtf_index = _vtf_pid_map(system, types)
    fp.write("\ntimestep indexed\n")
    if dipoles:
       dips = []
    for pid, vtf_id, in vtf_index.items():
        pos=system.part[pid].pos
        if folded:
           pos=system.part[pid].pos_folded
        fp.write("{} {} {} {}\n".format(vtf_id, *(pos)))
        if dipoles and system.part[pid].dipm > 0.:
           dips.append(system.part[pid].pos + system.part[pid].director * 0.02)
    if dipoles:
       ndips = len(dips)
       for i in range(ndips):
          fp.write("{} {} {} {}\n".format(ndips + i, *(dips[i])))

def _saveposdip(s, fp, Hfield = False, ts = 1., normdip = False):
   for p in s.part:
      if normdip:
         m = p.director
      else:
         m = p.dip
      if Hfield:
         fp.write("%e %f %f %f %f %f %f %f %f %f\n"%(s.time * ts, p.pos[0], p.pos[1], p.pos[2], m[0], m[1], m[2], Hfield[0], Hfield[1], Hfield[2]))
      else:
         fp.write("%e %f %f %f %f %f %f\n"%(s.time *ts, p.pos[0], p.pos[1], p.pos[2], m[0], m[1], m[2]))
   fp.flush()
   return

def _saveposdipaxis(s, fp, mom, Hfield = False, ts = 1.):
   for p in s.part:
      a = p.director
      if Hfield:
         fp.write("%e %f %f %f %f %f %f %f %f %f %f %f %f\n"%(s.time * ts, p.pos[0], p.pos[1], p.pos[2], mom[0], mom[1], mom[2], a[0], a[1], a[2], Hfield[0], Hfield[1], Hfield[2]))
      else:
         fp.write("%e %f %f %f %f %f %f %f %f %f\n"%(s.time *ts, p.pos[0], p.pos[1], p.pos[2], mom[0], mom[1], mom[2], m[0], m[1], m[2]))
   fp.flush()
   return

class SW_polar(object):
  def __init__(self, eps_phi=1e-3):
    self.phi0 = 0.
    self.eps_phi = eps_phi

  def eta(self, phi, theta, h):
    ''' eta = E / (2*Ku*V) '''
    return 0.25 - 0.25*np.cos(2*(phi-theta)) - h*np.cos(phi)

  def deta(self, phi, theta, h):
    return 0.5*np.sin(2*(phi-theta))+h*np.sin(phi)
  
  def phi(self, theta, h):
    f = lambda phi: self.eta(phi, theta, h)
    df = lambda phi: self.deta(phi, theta, h)

    sol = minimize(f, x0=self.phi0+self.eps_phi, jac=df, method="BFGS", tol=1e-15, options={'gtol':1e-15})['x'][0]
    self.phi0 = sol
    return sol

class SW(SW_polar):
   def __init__(self, Hk, **kwargs): # , axis, **kwargs):
      self.Hk = Hk # Hk = 2*K1/Js
      #self.e_k = np.array(axis) / norm(axis)
      super(SW, self).__init__(**kwargs)

   def m(self, H, axis):
      self.e_k = axis
      H = np.array(H)
      normH = norm(H)
      e_h = H
      h = 0.
      if normH > 0.:
         e_h /= normH
         h = normH / self.Hk
      theta = np.arccos(e_h.dot(self.e_k))
      axis = np.cross(e_h, self.e_k)
      if theta > np.pi / 2.: 
         theta = np.pi - theta
         h = -h
         e_h = -e_h
      phi = self.phi(theta, h) % (2*np.pi)
      e_p = np.cross(np.cross(e_h, self.e_k), e_h)
      return e_h * np.cos(phi) + e_p * np.sin(phi)

   def momtau(self, H, axis): # torque in reduced units (*mu0*ms*V for SI-units)
      mom = self.m(H, axis)
      return mom, np.cross(mom, H)

class PhysParams:
   def __init__(self, mat = "magnetite", dm = 40.e-9, hs = 2.e-9, Hmax = 4.8e6, T = 298.15):
      # Constants
      self.mu0 = 4.e-7 * np.pi          # magnetic permittivity [kg m /(s^2 A^2)]
      self.kb  = 1.38064852e-23         # Boltzmann constant    [m^2 kg /(s^2 K)]
      
      # Background properties
      self.T     = T                    # system temp                                                          [K]
      self.Troom = 298.15               # room temp                                                            [K]
      self.etaw  = 0.89e-3              # dynamic viscosity of background fluid (let's assume water at room T) [kg / (m s)]
      self.densw = 997.0                # density of water at room T                                           [kg / m^3]
      self.Hmax  = Hmax                 # maximum (saturation) external field                                  [A / m]
      self.Bmax  = self.mu0 * self.Hmax # maximum (saturation) flux density                                    [kg / (A s^2)]
      
      # Particle's main intrinsic/intensive properties
      if mat == "magnetite":
         self.densm = 5170.0               # mass density of the magnetic core                      [kg / m^3]
         self.Ms    = 400.e3               # saturation magnetization                               [A / m]
         self.K1    = 3.e4                 # magnetic aisotropy constant                            [kg / (m s^2)]
      elif mat == "cobalt":
         self.densm = 5170.0               # mass density of the magnetic core                      [kg / m^3]
         self.Ms    = 400.e3               # saturation magnetization                               [A / m]
         self.K1    = 1.e4                 # magnetic aisotropy constant                            [kg / (m s^2)]
      
      # Particle's main extensive properties
      self.dm    = dm                   # magnetic core diameter                                 [m]
      self.hs    = hs                   # thickness of steric coating (i.e., citrate monolayer)  [m]
      
      # Particle's derived properties
      self.D      = self.dm + 2. * self.hs        # hydrodynamic diameter                [m]
      self.Vm     = np.pi * (self.dm**3.) / 6.    # magnetic volume                      [m^3]
      self.V      = np.pi * (self.D**3.) / 6.     # total volume                         [m^3]
      self.m      = self.densm * self.Vm          # total mass (neglecting steric layer) [kg]
      self.momI   = self.m * (self.dm**2.) / 10.  # moment of inertia                    [kg m^2]

      self.gammat = 3. * np.pi * self.etaw * self.D     # translational friction coefficient   [kg / s]
      self.gammar = np.pi * self.etaw * (self.D**3.)    # rotational friction ciefficient      [kg m^2 / s]

      self.Hk     = 2. * self.K1 / (self.mu0 * self.Ms) # anisotropy field                     [A / m]
      self.dip    = self.Ms * self.Vm                   # magnetic moment                      [A m^2]

      self.tau_rot = 3. * self.etaw * self.V
      if self.T > 0.:
         self.tau_rot /= (self.kb * self.T) # rotational diffusion relaxation time [s]
      else:
         self.tau_rot /= (self.kb * 298.15)

      # Unit scales
      self.d_ = self.D                                # length scale  [m]
      self.m_ = 10 * self.m                           # mass scale    [kg]
      self.U_ = self.kb * self.Troom                  # energy scale  [kg m^2 / s^2]
      self.t_ = self.d_ * np.sqrt(self.m_ / self.U_)  # time scale    [s]
      self.A_ = self.dip / (self.d_ * self.d_)        # Ampere scale  [A]

      # Reduced parameters
      self._d = self.D / self.d_
      self._m = self.m / self.m_
      self._I = self.momI / (self.m_ * self.d_ * self.d_)
      self._gammat = self.gammat * self.t_ / self.m_
      self._gammar = self.gammar * self.t_ / (self.m_ * self.d_ * self.d_)
      self._kT = self.kb * self.T / self.U_
      self._Hmax = self.Hmax * self.d_ / self.A_
      self._Bmax = self.Bmax * self.A_ * self.t_ * self.t_ / self.m_ #self._Bmax = self.mu0 * self.Hmax * self.A_ * self.t_ * self.t_ / self.m_
      self._tau_rot = self.tau_rot / self.t_
      self._dip = self.dip / (self.A_ * self.d_ * self.d_)

      #self.B   = self.mu0 * self.H
      #self._B  = self.B * self.A_ * self.t_ * self.t_ / self.m_
      #self.Oc  = self.dip * self.B / self.gammar

   def logParams(self):
      _log("System length scale: %e s"%(self.d_))
      _log("System time scale: %e s"%(self.t_))
      _log("System energy scale: %e s"%(self.U_))
      _log("System field scale: %e A/m"%(self.A_/self.d_))
      _log("System temperature: %f K (reduced thermal energy: %f)"%(self.T, self._kT))
      _log("Particle magnetic diameter: %e nm (%e)"%(self.dm / 1.e-9, self.dm / self.d_))
      _log("Particle hydrodynamic diameter: %e nm (%e)"%(self.D / 1.e-9, self.D / self.d_))
      _log("Particle rotational friction: %e kg m^2 / s (%e)"%(self.gammar, self._gammar))
      return

"""
def simRotH(model, H, Hfreq, Hturns = 20, suffix = "test", dm = 40., K1 = 1.e4, T = 298.15):
   
   def _setHfield(s, Hobj, log = True):
      while len(s.constraints) > 0:
         H0 = s.constraints[0]
         if log:
            _log("Removing existing external field "+str(H0.H))
         s.constraints.remove(H0)
      if log:
         _log("Applying external magnetic field %s"%(str(Hobj.H)))
      s.constraints.add(Hobj)
      return
   
   # MD parameters
   
   # Constants
   mu0 = 4.e-7 * np.pi          # magnetic permittivity [kg m /(s^2 A^2)]
   kb  = 1.38064852e-23         # Boltzmann constant    [m^2 kg /(s^2 K)]

   # Background properties
   Troom = 298.15               # room temp                                                            [K]
   etaw  = 0.89e-3              # dynamic viscosity of background fluid (let's assume water at room T) [kg / (m s)]
   densw = 997.0                # density of water at room T                                           [kg / m^3]
   Hmax  = 4.8e6                # maximum (saturation) external field                                  [A / m]

   # Particle's main properties (let's assume 40nm magnetite monodomain nanoparticles)
   dm    = dm * 1.e-9           # magnetic core diameter                                 [m]
   hs    = 2.e-9                # thickness of steric coating (i.e., citrate monolayer)  [m]
   densm = 5170.0               # mass density of the magnetic core                      [kg / m^3]
   Ms    = 400.e3               # saturation magnetization                               [A / m]
   K1    = 1. * K1              # magnetic aisotropy constant                            [kg / (m s^2)]
   tau_N = 1.e-7                # Neel relaxation time                                   [s]

   # Particle's derived properties
   D       = dm + 2. * hs             # hydrodynamic diameter                [m]
   Vm      = np.pi * (dm**3.) / 6.    # magnetic volume                      [m^3]
   V       = np.pi * (D**3.) / 6.     # total volume                         [m^3]
   m       = densm * Vm               # total mass (neglecting steric layer) [kg]
   momI    = m * (dm**2.) / 10.       # moment of inertia                    [kg m^2]

   gammat  = 3. * np.pi * etaw * D    # translational friction coefficient   [kg / s]
   gammar  = np.pi * etaw * (D**3.)   # rotational friction ciefficient      [kg m^2 / s]

   Hk      = 2. * K1 / (mu0 * Ms)     # anisotropy field                     [A / m]
   dip     = Ms * Vm                  # magnetic moment                      [A m^2]
   
   tau_rot = 3. * etaw * V / (kb * T) # rotational diffusion relaxation time [s]

   # Unit scales
   d_ = D                      # length scale  [m]
   m_ = m                      # mass scale    [kg]
   U_ = kb * Troom             # energy scale  [kg m^2 / s^2]
   t_ = d_ * np.sqrt(m_ / U_)  # time scale    [s]
   A_ = dip / (d_ * d_)        # Ampere scale  [A]

   # Reduced parameters
   _d = D / d_
   _m = m / m_
   _I = momI / (m_ * d_ * d_)
   _gammat = gammat * t_ / m_
   _gammar = gammar * t_ / (m_ * d_ * d_)
   _kT = kb * T / U_
   _Hmax = Hmax * d_ / A_
   _Bmax = mu0 * Hmax * A_ * t_ * t_ / m_
   _B    = mu0 * H * A_ * t_ * t_ / m_ #A_ * A_ * t_ * t_ / (d_ * m_)
   _Hk   = 2. * K1 / (mu0 * Ms) * (d_ / A_) 
   
   _log("System time scale: %e s"%(t_))
   _log("System field scale: %e A/m"%(A_/d_))
   _log("System temperature: %f K (reduced thermal energy: %f)"%(T, _kT))
   _log("Particle magnetic diameter: %e nm (%e)"%(dm / 1.e-9, dm / d_))
   _log("Particle hydrodynamic diameter: %e nm (%e)"%(D / 1.e-9, D / d_))
   _log("Particle rotational friction: %e kg m^2 / s (%e)"%(gammar, _gammar))
   
   # field rotation discretization
   tstep  = 1.e-11            # integration time step [s]
   _tstep = tstep / t_        # reduced integration time step
   int_st = 5                 # integration steps per field orientation
   nHs = int(round(1. /(Hfreq * int_st * tstep)))
   Hfreq0 = 1. /(nHs * int_st * tstep)
   
   _log("Integration time step: %e s (%f)"%(tstep, _tstep))
   _log("Integration steps per field discrete orientation: %d"%(int_st))
   _log("Number of discrete field orientations: %d"%(nHs))
   if nHs < 6:
      _log("WARNING! the discretization of the field orientations might be too coarse!")
   
   _log("Input field strength (H): %e A/m (%f)"%(H, H * d_ / A_))
   _log("Input field flux density (B): %f T (%f)"%(mu0 * H, _B))
   _log("Input field frequency: %f MHz"%(Hfreq/1.e6))
   
   _log("Actual sampled field frequency: %f MHz"%(Hfreq0 / 1.e6))
   
   # setting up the system
   boxl = [100., 100., 100.]
   s = system.System(box_l = [boxl[0], boxl[1], boxl[2]])
   rseed = os.getpid()
   s.set_random_state_PRNG()
   np.random.seed(seed = rseed)
   s.periodicity = [0, 0, 0]
   
   s.part.add(id = 0, type = 0, pos = [50., 50., 50.], rotation=(1,1,1), dip = [1., 0., 0.], mass=_m, rinertia=np.ones(3)*(_I), gamma=_gammat, gamma_rot=_gammar)
   
   s.time_step = _tstep
   s.cell_system.skin = 0.4
   
   s.thermostat.set_langevin(kT=_kT, gamma=_gammat, seed = rseed)
   
   # vtf trajectory file, can be visualized with VMD
   fp = open("trj-%s.vtf"%(suffix),"w")
   _writevsf(s, fp, types = "all",radiuses = {0 : _d / 2.}, dipoles = True)
   _writevcf(s, fp, types = "all",folded = False, dipoles = True)
   fp.flush()
   
   # flat trajectory file with field and particle orientations, to be analyzed
   if model == "brownian" or model == "sw-mom":
      fpdip = open("posdip-%s.dat"%(suffix),"w")
      fpdip.write("# %s\n"%(_getScriptCall()))
      fpdip.write("# t_exp posx posy posz mx my mz Hx Hy Hz\n")
   elif model == "sw":
      fpdip = open("posdipaxis-%s.dat"%(suffix),"w")
      fpdip.write("# %s\n"%(_getScriptCall()))
      fpdip.write("# t_exp posx posy posz mx my mz ax ay az Hx Hy Hz\n")
   fpdip.flush()
   
   # some warming up
   s.integrator.run(2000)
   if model == "brownian" or model == "sw-mom":
      s.part[0].dip=[1., 0., 0.]
   
   # cache of field objects, one per discrete field orientation
   # (this has better performance than creating/deleting field objects on the fly,
   # with the downside of forcing a periodic discretization of field orientations
   # as long as we keep fixed the time scale; as a result, actual sampled field frequency
   # may differ slightly from the input value)
   _log("Run field turns: %d"%(Hturns))
   Hw  = 2. * pi * Hfreq0
   Hx = _B
   Hy = 0.
   Hz = 0.
   th = 0.
   dth = 2. * pi / float(nHs)
   Hfields = []
   Hvecs = []
   turn = 0
   for i in range(nHs):
      Hx = _B * cos(th)
      Hy = _B * sin(th)
      if model == "brownian":
         Hfields.append(MagneticField(H=[Hx, Hy, Hz])) # note that what we set in Espresso is actually mu0 * H = B
      Hvecs.append(H * np.array([Hx, Hy, Hz]) / _B) # in A / m
      th += dth
   
   # we start spinning the field
   Hi = 0
   if model == "brownian":
      Hconstraint = _setHfield(s, Hfields[Hi], log = False)
   elif model == "sw":
      SWsolver = SW(Hk)
   
   s.integrator.run(0)
   wH = np.array([0., 0., 1.])
   Hang_st = 1 # for subsampling (maybe useful at very low frequencies)
   
   taupref = mu0 * Vm * Ms / (d_ * d_ * m_ / (t_ * t_))
   
   # main integration loop
   for i in range(nHs * Hturns):
      if model == "brownian":
         s.integrator.run(int_st)
      elif model == "sw":
         for j in range(int_st):
            mom, tau = SWsolver.momtau(Hvecs[Hi], s.part[0].director) # UNITS!!!!!
            _tau = tau * taupref
            s.part[0].ext_torque = _tau
            s.integrator.run()
      elif model == "sw-mom":
         for j in range(int_st):
            _tau = taupref * np.cross(s.part[0].director, Hvecs[Hi])
            s.part[0].ext_torque = _tau
            s.integrator.run()
      
      if i % Hang_st == 0 :
         # save data
         _writevcf(s, fp, types = "all",folded = False, dipoles = True)
         if model == "brownian":
            _saveposdip(s, fpdip, [Hfields[Hi].H[0], Hfields[Hi].H[1], Hfields[Hi].H[2]], ts = t_, normdip = True)
         elif model == "sw":
            Hvec = _B * Hvecs[Hi] / H
            _saveposdipaxis(s, fpdip, mom, [Hvec[0], Hvec[1], Hvec[2]], ts = t_)
         elif model == "sw-mom":
            _saveposdip(s, fpdip, [Hvecs[Hi][0], Hvecs[Hi][1], Hvecs[Hi][2]], ts = t_, normdip = True)
         # next field orientation
         Hi += 1
         Hi = Hi % len(Hvecs)
         if model == "brownian":
            Hconstraint = _setHfield(s, Hfields[Hi], log = False)
      if i % nHs == 0:
         turn += 1
         _log("turn %d / %d"%(turn, Hturns))
      pass
   return
"""

def simulate(model, mat, dm, hs, initmomaxis, initeasyaxis, H, Hfreq, Hcycles, Htype, T, suffix, Hvec_st = 1, dofix = False):
   
   p = PhysParams(mat = mat, dm = dm * 1.e-9, hs = hs * 1.e-9, Hmax = H, T = T)
   p.logParams()
   
   # field discretization
   tstep  = 1.e-11            # integration time step [s]
   _tstep = tstep / p.t_      # reduced integration time step
   int_st = 5                 # integration steps per field
   nHs = int(round(1. /(Hfreq * int_st * tstep)))
   Hfreq0 = 1. /(nHs * int_st * tstep)
   
   _log("Integration time step: %e s (%e)"%(tstep, _tstep))
   _log("Integration steps per discrete field vector: %d"%(int_st))
   _log("Number of discrete field vectors: %d"%(nHs))
   if nHs < 6:
      _log("WARNING! the discretization of the field might be too coarse!")
   
   _log("Input field strength (H): %e A/m (%f)"%(p.Hmax, p._Hmax))
   _log("Input field flux density (B): %f T (%f)"%(p.Bmax, p._Bmax))
   _log("Input field frequency: %f MHz"%(Hfreq/1.e6))
   _log("Input thermal energy: %e kg m^2 / s^2 (%f)"%(p.kb * p.T, p._kT))
   
   _log("Actual sampled field frequency: %f MHz"%(Hfreq0 / 1.e6))
   
   # setting up the system
   boxl = [100., 100., 100.]
   s = system.System(box_l = [boxl[0], boxl[1], boxl[2]])
   rseed = os.getpid()
   s.set_random_state_PRNG()
   np.random.seed(seed = rseed)
   s.periodicity = [0, 0, 0]
   
   if initmomaxis == "x":
      initdip = [p._dip, 0., 0.]
   elif initmomaxis == "y":
      initdip = [0., p._dip, 0.]
   else:
      initdip = [0., 0., p._dip]
   
   if model == "sw-fast":
      if initeasyaxis == "x":
         initdip = [1., 0., 0.]
      elif initeasyaxis == "y":
         initdip = [0., 1., 0.]
      else:
         initdip = [0., 0., 1.]
   
   rot = [1, 1, 1]
   fix = [0, 0, 0]
   if dofix:
      _log("Constraining dynamics to rotations around y-axis")
      rot = [0, 1, 0]
      fix = [1, 1, 1]
   
   s.part.add(id = 0, type = 0, pos = [50., 50., 50.], rotation = rot, dip = initdip, fix = fix, mass = p._m, rinertia = np.ones(3)*(p._I), gamma = p._gammat, gamma_rot = p._gammar)
   
   s.time_step = _tstep
   s.cell_system.skin = 0.4
   
   s.thermostat.set_langevin(kT = p._kT, gamma = p._gammat, seed = rseed)
   
   # flat trajectory file with field and particle orientations, to be analyzed
   if model == "ideal" or model == "ideal-fast":
      fpdip = open("posdip-%s.dat"%(suffix),"w")
      fpdip.write("# %s\n"%(_getScriptCall()))
      fpdip.write("# t_exp posx posy posz mx my mz Hx Hy Hz\n")
   elif model == "sw-fast":
      fpdip = open("posdipaxis-%s.dat"%(suffix),"w")
      fpdip.write("# %s\n"%(_getScriptCall()))
      fpdip.write("# t_exp posx posy posz mx my mz ax ay az Hx Hy Hz\n")
   fpdip.flush()
   
   # some warming up
   s.integrator.run(500)
   if model == "ideal" or model == "ideal-fast":
      s.part[0].dip = initdip
   
   # cache of discretized field vectors
   # (this has better performance than creating/deleting field objects on the fly,
   # with the downside of forcing a periodic discretization of field orientations
   # while we keep fixed the time scale; as a result, actual sampled field frequency
   # may differ slightly from the input value)
   _log("Run field turns: %d"%(Hcycles))
   Hw  = 2. * pi * Hfreq0
   Hfields = []
   Hvecs = []
   turn = 0
   dth = 2. * pi / float(nHs)
   Hy = 0.
   Hz = 0.
   th = 0.
   if Htype == "rot":
      for i in range(nHs):
         Hx = p._Bmax * cos(th)
         Hy = p._Bmax * sin(th)
         if model == "ideal":
            Hfields.append(MagneticField(H=[Hx, Hy, Hz])) # note that what we set in Espresso is actually mu0 * H = B
         Hvecs.append(H * np.array([Hx, Hy, Hz]) / p._Bmax) # in A / m
         th += dth
   elif Htype == "osc":
      for i in range(nHs):
         Hx = p._Bmax * sin(th)
         if model == "ideal":
            Hfields.append(MagneticField(H=[Hx, Hy, Hz])) # note that what we set in Espresso is actually mu0 * H = B
         Hvecs.append(H * np.array([Hx, Hy, Hz]) / p._Bmax) # in A / m
         th += dth
   
   # we start looping over the field vectors
   Hi = 0
   if model == "ideal":
      Hconstraint = _setHfield(s, Hfields[Hi], log = False)
   elif model == "sw-fast":
      SWsolver = SW(p.Hk)
   
   s.integrator.run(0)
   wH = np.array([0., 0., 1.])
   
   taupref = p.mu0 * p.Vm * p.Ms / (p.d_ * p.d_ * p.m_ / (p.t_ * p.t_))
   
   # main integration loop
   for i in range(nHs * Hcycles):
      if model == "ideal":
         s.integrator.run(int_st)
      elif model == "sw-fast":
         for j in range(int_st):
            mom, tau = SWsolver.momtau(Hvecs[Hi], s.part[0].director) # UNITS!!!!!
            _tau = tau * taupref
            s.part[0].ext_torque = _tau
            s.integrator.run()
      elif model == "ideal-fast":
         for j in range(int_st):
            _tau = taupref * np.cross(s.part[0].director, Hvecs[Hi])
            s.part[0].ext_torque = _tau
            s.integrator.run()
      
      if i % Hvec_st == 0 :
         # save data
         if model == "ideal":
            _saveposdip(s, fpdip, [Hfields[Hi].H[0], Hfields[Hi].H[1], Hfields[Hi].H[2]], ts = p.t_, normdip = True)
         elif model == "sw-fast":
            Hvec = p._Bmax * Hvecs[Hi] / H
            _saveposdipaxis(s, fpdip, mom, [Hvec[0], Hvec[1], Hvec[2]], ts = p.t_)
         elif model == "ideal-fast":
            _saveposdip(s, fpdip, [Hvecs[Hi][0], Hvecs[Hi][1], Hvecs[Hi][2]], ts = p.t_, normdip = True)
      
      # next field vector
      Hi += 1
      Hi = Hi % len(Hvecs)
      if model == "ideal":
         Hconstraint = _setHfield(s, Hfields[Hi], log = False)
      if i % nHs == 0:
         turn += 1
         _log("turn %d / %d"%(turn, Hcycles))
   return

def simulateEggModel(mat, dm, hs, initmomaxis, initeasyaxis, H, Hfreq, Hcycles, Htype, T, suffix, Hvec_st = 1, dofix = False):
   assert_features(['DIPOLES', 'EGG_BOND', 'EXTERNAL_FORCES', 'LANGEVIN_PER_PARTICLE', 'MASS', 'PARTICLE_ANISOTROPY', 'ROTATION', 'ROTATIONAL_INERTIA'])
   from espressomd.interactions import EggBond
   
   def printEnergy():
      print("total:",s.analysis.energy()["total"], ", kinetic:",s.analysis.energy()["kinetic"], ", bonded:", s.analysis.energy()["bonded"], ", dipolar:", s.analysis.energy()["dipolar"])

   tstep  = 1.e-11            # integration time step [s] (=10 ps = 0.01 ns; attempt time for Neel relaxation is ~1e-10; Brownian relaxation time is ~1.e-7)
   
   # first, to set proper units and to fit parameters to get proper relaxation times when both mechanisms (Neel and Brown) work together. For a 10nm magnetite MNP the dipole switching time is ~4e-10 s and the Brownian is given by the rotational friction
   # next test is to apply a perpendicular field to a relaxed egg bond and see how the yellow forces the turn of the white (basically, SW)

   rseed=os.getpid()
   s=system.System(box_l = [100.,100.,100.])
   s.set_random_state_PRNG()
   s.periodicity = [0, 0, 0]

   s.part.add(id=0, type=0, pos=(50., 50., 50), rotation=(0, 0, 0), fix=(1, 1, 1), gamma=1., gamma_rot=1., temp=0.) # white part, carries the anisotropy axis, initially with a default orientation (0,0,1); for a different initial orientation use s.part[id].rotate
   print(s.part[0].director)
   s.part[0].rotate(axis=(1,0,0), angle=pi/4.)
   print(s.part[0].director)


   I = 0.1
   gamma_r = 10.
   s.part.add(id=1, type=100, pos=(50., 50., 50), rotation=(1,1,1), fix=(1,1,1), gamma=1., gamma_rot= gamma_r, temp=0., dip=[0., 0., 1.], rinertia = np.ones(3)*(I)) # yellow part, carries the dipole; set a different type for this so that no other interactions than the bond and the magnetic ones are calculated on it; fixing this particle might improve a bit the performance as the calculation of its translational motion, that at the end of each integration step is anyway overriden by the bond, is suposed to be avoided; use real particles only, behavior with virtual particles is unknown

   s.thermostat.set_langevin(kT=0.0, gamma=1.0, gamma_rotation=1.0, seed=rseed)
   s.time_step=0.01
   s.cell_system.skin=0.3

   egg = EggBond(k = 1000.)
   s.bonded_inter.add(egg)

   s.part[0].add_bond((egg, 1))  # the particle that stores the bond is the white part, the other one is the yellow part and has to carry the dipole

   #field=HomogeneousMagneticField(H=(5.0,0.,0.0))
   #s.constraints.add(field)
   
   """
   s.integrator.run(0)
   print("part 1, pos:",s.part[0].pos,"torque:",s.part[0].torque_lab, ", director:", s.part[0].director)
   print("part 2, pos:",s.part[1].pos,"torque:",s.part[1].torque_lab, ", director:", s.part[1].director)
   printEnergy()

   # let's test that the yellow stays in the center of the white
   s.part[1].pos = [90, 90, 90]

   s.integrator.run(0)
   print("part 1, pos:",s.part[0].pos,"torque:",s.part[0].torque_lab, ", director:", s.part[0].director)
   print("part 2, pos:",s.part[1].pos,"torque:",s.part[1].torque_lab, ", director:", s.part[1].director)
   printEnergy()
   
   s.part[0].pos = [90, 90, 90]

   s.integrator.run(0)
   print("part 1, pos:",s.part[0].pos,"torque:",s.part[0].torque_lab, ", director:", s.part[0].director)
   print("part 2, pos:",s.part[1].pos,"torque:",s.part[1].torque_lab, ", director:", s.part[1].director)
   printEnergy()
   """
   
   s.part[1].dip = [1., 0., 0.]
   s.integrator.run(0)
   # the torque coming from the egg bond doesn't seem to be reported by p.torque_lab
   print("part 1, pos:",s.part[0].pos,"torque:",s.part[0].torque_lab, ", director:", s.part[0].director)
   print("part 2, pos:",s.part[1].pos,"torque:",s.part[1].torque_lab, ", director:", s.part[1].director)
   printEnergy()
   
   for i in range(1000):
      s.integrator.run(1)
      print(np.dot(s.part[0].director, s.part[1].director))
   
   return

if __name__ == "__main__":
   from espressomd import system, assert_features
   assert_features(["EXTERNAL_FORCES", "DIPOLES", "PARTICLE_ANISOTROPY", "LANGEVIN_PER_PARTICLE", "ROTATION", "ROTATIONAL_INERTIA", "MASS"])
   from espressomd.constraints import HomogeneousMagneticField as MagneticField
   
   # libraries for the SW solver
   from scipy.linalg import norm, expm
   from scipy.optimize import minimize
   
   import argparse
   import glob
   import numpy as np
   import random
   import sys
   from math import *
   import gzip
   import os
   
   import datetime
   
   parser = argparse.ArgumentParser()
   parser.add_argument('--simulate', choices = ['ideal', 'ideal-fast', 'sw-fast'], help = "Simulate single particle under field (ideal=infinite anisotropy; sw=Stoner-Wohlfarth)")
   parser.add_argument('--mat', choices = ["magnetite", "cobalt"], help="Magnetic material (default=magnetite)", default="magnetite")
   parser.add_argument('--dm', help = "Magnetic diameter of the particle in nm (default=40.0)", default = 40., type = float)
   parser.add_argument('--hs', help = "Thickness of the nonmagnetic outer layer (slip plane) in nm (default=2.0)", default = 2., type = float)
   parser.add_argument('--initmomaxis', choices = ["x", "y", "z"], help="Initial axis orientation of the dipole moment (default=x)", default="x")
   parser.add_argument('--initeasyaxis', choices = ["x", "y", "z"], help="Option for sw-fast simulations only. Initial orientation of the easy axis (default=x)", default="x")
   parser.add_argument('--Htype', choices = ['rot', 'osc'], help = "Driving field type of change. rot=rotating field; osc=oscillating field (default=rot)", default = "rot")
   parser.add_argument('--Hfreq', help = "Driving field changing frequency (default=1.e7 Hz)", default = 10.e6, type = float)
   parser.add_argument('--H', help = "Applied field (maximum) strength (default=4.8e6 A/m)", default = 4.8e6, type = float)
   parser.add_argument('--Hcycles', help = "Field cycles to run (default=20)", default = 20 ,type = int)
   parser.add_argument('--Hvecst', help = "Subsample measures step rate (default=1)", default = 1 ,type = int)
   parser.add_argument('--T', help = "System temperature in K (default=298.15)", default = 298.15, type = float)
   parser.add_argument('--fix', help = "Fix position and allow rotations around y only", action='store_true')
   parser.add_argument('--suffix', help = "Suffix for output files", default = "test")
   
   parser.add_argument('--simegg', action='store_true')
   
   args = parser.parse_args()
   
   if not args.simulate and not args.simegg:
      parser.print_help()
      sys.exit()
   
   _log(_getScriptCall())
   
   if args.simulate:
      simulate(model = args.simulate, mat = args.mat, dm = args.dm, hs = args.hs, initmomaxis= args.initmomaxis, initeasyaxis = args.initeasyaxis, dofix = args.fix, H = args.H, Hfreq = args.Hfreq, Hcycles = args.Hcycles, Htype = args.Htype, T = args.T, suffix = args.suffix, Hvec_st = args.Hvecst)

   if args.simegg:
      simulateEggModel(mat = args.mat, dm = args.dm, hs = args.hs, initmomaxis= args.initmomaxis, initeasyaxis = args.initeasyaxis, dofix = args.fix, H = args.H, Hfreq = args.Hfreq, Hcycles = args.Hcycles, Htype = args.Htype, T = args.T, suffix = args.suffix, Hvec_st = args.Hvecst)
