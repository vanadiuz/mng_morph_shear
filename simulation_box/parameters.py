import argparse
import numpy as np
from math import pi,sqrt
import os
from dataclasses import dataclass, field
import warnings
from typing import List
from pathlib import Path

@dataclass
class Parameters: 
    init_lambda     : float = 3
    _N_part         : float = 89   #specific for this phi_m and box_len (futher in this script)
    _box_l          : float = 1000
    _timesteps      : float = 50
    _hmf            : float = 0
    particle        : str   = 'mng'  # or 'raspberry', 'cluster'

    # _m_scale  : float = 1.6
    # _t_scale  : float = 0.01
    # _e_scale  : float = 0.179*1000
    # _v_factor : float  = 1

    #MACF
    _m_scale  : float = 1.6
    _t_scale  : float = 0.01 #setted in system
    _e_scale  : float = 0.179*1000 #*0.032
    _v_factor : float  = 1*0.11

    _tot_sim_time : int = 71 #hours

    #ts for dynamic sim. step
    ts_mult : int = 1
    tmstep_size_lb : float = 0.02

    #system parameters
    periodicity             : List[bool] = field(default_factory=list)
    skin                    : float       = 1
    domain_decomposition_VL : bool        = True
    rnd_seed                : int         = 42

    _rseed        : int   = field(init=False)
    _kT           : float = field(init=False)
    t_            : float = field(init=False)
    d_            : float = field(init=False)
    mass_         : float = field(init=False)
    _gap          : float = field(init=False)
    _p_type       : int   = field(init=False)
    _mnp_type     : int   = field(init=False)
    _p_length     : int   = field(init=False)
    _p_number     : int   = field(init=False)
    _Nlinks       : int   = field(init=False)
    _Rg           : float = field(init=False)
    _phi_m        : float = field(init=False)
    _Kbond        : float = field(init=False)
    _max_bond_ext: float  = field(init=False)

    _W: float = field(init=False)
    F_: float = field(init=False)

    nub_of_int_steps_for_rot_field: float = field(init=False)
    _nHs: int = field(init=False) #num of H change per rotation

    _sigma_m   : float = field(init=False)
    _mass_m    : float = field(init=False)
    _momI_m    : float = field(init=False)
    _mu        : float = field(init=False)
    _alpha     : float = field(init=False)
    _gamma_t_m: float  = field(init=False)
    _gamma_r_m: float  = field(init=False)
    lambda_m   : float = field(init=False)

    _sigma_gel   : float = field(init=False)
    _mass_gel    : float = field(init=False)
    _momI_gel    : float = field(init=False)
    _gamma_t_gel: float  = field(init=False)

    _eta_w: float = field(init=False)

    lb_dens       : float = field(init=False)
    lb_fric       : float = field(init=False)
    shear_rate  : float = field(init=False)
    poiseuille  : float = field(init=False)
    lb_visc       : float = field(init=False)
    lb_agrid      : float = field(init=False)
    lb_tau        : float = field(init=False)
    lb_force_dens: list   = field(init=False)
    _wall_type    : int   = field(init=False)

    _lj_eps   : float = field(init=False)
    _lj_sigma: float  = field(init=False)
    _lj_cut   : float = field(init=False)
    _lj_shift: float  = field(init=False)
    _epsWCA   : float = field(init=False)

    _ddSumCpu_prefactor : float = field(init=False)
    _p3m_prefactor      : float = field(init=False)
    _p3m_cao            : float = field(init=False)
    _p3m_mesh           : float = field(init=False)
    _p3m_r_cut          : float = field(init=False)
    _p3m_accuracy       : float = field(init=False)
    _p3m_tune           : bool  = field(init=False)
    _dlc_maxPWerror     : float = field(init=False)
    _dlc_gap_size       : float = field(init=False)
    _min_skin           : float = field(init=False)
    _max_skin           : float = field(init=False)
    _tol                : float = field(init=False)
    _int_steps          : float = field(init=False)

    _timesteps_init : int   = field(init=False)
    _t_step_init    : float = field(init=False)


    def __post_init__(self): 

                
        #   ██▓███   ▄▄▄       ██▀███   ▄▄▄       ███▄ ▄███▓▓█████▄▄▄█████▓ ██▀███   ██▓▒███████▒ ▄▄▄     ▄▄▄█████▓ ██▓ ▒█████   ███▄    █ 
        #  ▓██░  ██▒▒████▄    ▓██ ▒ ██▒▒████▄    ▓██▒▀█▀ ██▒▓█   ▀▓  ██▒ ▓▒▓██ ▒ ██▒▓██▒▒ ▒ ▒ ▄▀░▒████▄   ▓  ██▒ ▓▒▓██▒▒██▒  ██▒ ██ ▀█   █ 
        #  ▓██░ ██▓▒▒██  ▀█▄  ▓██ ░▄█ ▒▒██  ▀█▄  ▓██    ▓██░▒███  ▒ ▓██░ ▒░▓██ ░▄█ ▒▒██▒░ ▒ ▄▀▒░ ▒██  ▀█▄ ▒ ▓██░ ▒░▒██▒▒██░  ██▒▓██  ▀█ ██▒
        #  ▒██▄█▓▒ ▒░██▄▄▄▄██ ▒██▀▀█▄  ░██▄▄▄▄██ ▒██    ▒██ ▒▓█  ▄░ ▓██▓ ░ ▒██▀▀█▄  ░██░  ▄▀▒   ░░██▄▄▄▄██░ ▓██▓ ░ ░██░▒██   ██░▓██▒  ▐▌██▒
        #  ▒██▒ ░  ░ ▓█   ▓██▒░██▓ ▒██▒ ▓█   ▓██▒▒██▒   ░██▒░▒████▒ ▒██▒ ░ ░██▓ ▒██▒░██░▒███████▒ ▓█   ▓██▒ ▒██▒ ░ ░██░░ ████▓▒░▒██░   ▓██░
        #  ▒▓▒░ ░  ░ ▒▒   ▓▒█░░ ▒▓ ░▒▓░ ▒▒   ▓▒█░░ ▒░   ░  ░░░ ▒░ ░ ▒ ░░   ░ ▒▓ ░▒▓░░▓  ░▒▒ ▓░▒░▒ ▒▒   ▓▒█░ ▒ ░░   ░▓  ░ ▒░▒░▒░ ░ ▒░   ▒ ▒ 
        #  ░▒ ░       ▒   ▒▒ ░  ░▒ ░ ▒░  ▒   ▒▒ ░░  ░      ░ ░ ░  ░   ░      ░▒ ░ ▒░ ▒ ░░░▒ ▒ ░ ▒  ▒   ▒▒ ░   ░     ▒ ░  ░ ▒ ▒░ ░ ░░   ░ ▒░
        #  ░░         ░   ▒     ░░   ░   ░   ▒   ░      ░      ░    ░        ░░   ░  ▒ ░░ ░ ░ ░ ░  ░   ▒    ░       ▒ ░░ ░ ░ ▒     ░   ░ ░ 
        #                 ░  ░   ░           ░  ░       ░      ░  ░           ░      ░    ░ ░          ░  ░         ░      ░ ░           ░ 
        #                                                                               ░                                                  
        

        #espressomd system parameters
        self.periodicity = [True, True, True]

        #scales for LB
        if self.particle == 'cluster':
            self._m_scale  : float = 0.8
            self._t_scale  : float = 0.01
            self._e_scale  : float = 1.4*0.179*1000
            self._v_factor : float  = 1

        # Physical Constants (SI units)
        mu0 = pi*4.e-7        # kg m /(s^2 A^2)
        kb  = 1.38064852e-23  # m^2 kg /(s^2 K)


        # Physical parameters (SI units)s

        ## Background properties
        T = 298.15 # room temp. K
        #⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
        eta_w  = self._v_factor*0.89e-3  # water dynamic viscosity at room T, kg/(m s)
        #⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
        # eta_w    = 100e-3        # water dynamic viscosity at room T, kg/(m s)
        dens_w   = 1*997.0         # density of water at room T, kg/m^3
        mu_water = 1.256627e10-6  # kg m /(s^2 A^2)

        f_velocity = 0  #10.e-2   # typical velocity of the flow, m / s


        ## 🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲
        ## 🧲🧲🧲 MNP particle properties (magnetite core) 🧲🧲🧲
        ## 🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲🧲

        if self.init_lambda == 3:
        ### CoFe2O4 | lambda = 3
            ### Intrinsic properties (SI units)
            M_m       = 420.e3        # saturation magnetization of CoFe2O4, A/m 
            dens_m    = 5300.0        # density of CoFe2O4, kg/m^3
            dens_m_sh = 1.6 * dens_w  # density of dna, oating and crosslinking the CoFe2O4 core, kg/m^3

            ### Arbitrary properties (SI units)
            D_m    = 14.e-9  # diameter of CoFe2O4 core, m
            R_m_sh = 2.e-9   # thickness of polymer coating, m

        elif self.init_lambda == 5:
        ### CoFe2O4 | lambda = 5
            ### Intrinsic properties (SI units)
            M_m       = 420.e3        # saturation magnetization of CoFe2O4, A/m 
            dens_m    = 5300.0        # density of CoFe2O4, kg/m^3
            dens_m_sh = 1.6 * dens_w  # density of dna, oating and crosslinking the CoFe2O4 core, kg/m^3

            ### Arbitrary properties (SI units)
            D_m    = 16.1 * 1.e-9  # diameter of CoFe2O4 core, m
            R_m_sh = 2.e-9   # thickness of polymer coating, m

        elif self.init_lambda == 6:
        ###  Co | lambda = 6
            ### Intrinsic properties (SI units)
            M_m       = 1440.e3       # saturation magnetization of Co, A/m 
            dens_m    = 8900.0        # density of Co, kg/m^3
            dens_m_sh = 1.6 * dens_w  # density of dna, coating and crosslinking the Co core, kg/m^3

            ### Arbitrary properties (SI units)
            D_m    = 9e-9     # diameter of magnetite core, m
            R_m_sh = 2.23e-9  # thickness of polymer coating, m


        R_m     = D_m/2.        # radius of the magnetic core, m
        R_f     = R_m + R_m_sh  # total radius, m (we assume this as the hydrodynamic radius)
        R_h     = R_f           # hydrodynamic radius
        sigma_m = 2.*R_h        # total diameter, m

        # derived properties (SI units)
        dip           = M_m*pi*(D_m**3.)/6.                          # dipole moment, A m^2
        self.lambda_m = 2.*mu0*dip*dip/(4.*pi*kb*T*((sigma_m)**3.))  # lambda (dimensionless dipolar coupling parameter)

        mass_m=(4./3.)*pi*(dens_m*R_m**3.+ dens_m_sh *
                                (R_f**3. - R_m**3.))         # mass, kg
        momI_m=(8./15.)*pi*(dens_m*(R_m**5.) + dens_m_sh *
                                ((R_f**5.) -
                                (R_m**5.))) # moment of inertia, kg m^2

        gamma_t_m = 6.*pi*eta_w*R_h          # translational friction coefficient, kg/s
        gamma_r_m = 8.*pi*eta_w*((R_h)**3.)  # rotational friction coefficient (not used in LB), kg m^2 /s



        ## 🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇
        ## 🍇🍇🍇 PNIPAM _gel (single bead properties) 🍇🍇🍇🍇🍇
        ## 🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇🍇

        ### Intrinsic properties (SI units)
        #⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️⚠️
        dens_gel = 1.3*dens_w if self.particle == 'mng' else 1.0*dens_w # 1300 density of PNIPAM at room T, kg/m^3 1.3*dens_w

        ### Arbitrary properties (SI units)
        sigma_gel = sigma_m  # total diameter, m

        # derived properties (SI units)
        mass_gel = dens_gel * pi * (sigma_gel**3.) / 6.  # mass, kg
        momI_gel = mass_gel * (sigma_gel**2.) / 10.      # moment of inertia, kg m^2

        gamma_t_gel = gamma_t_m  # translational friction coefficient, kg/s
        gamma_r_gel = gamma_r_m  # rotational friction coefficient (not used in LB), kg m^2 /s

        ## nanogel properties
        self._Rg       = 5.98
        self._p_length = 100
        self._p_number = 6
        self._Nlinks   = 100
        self._phi_m    = 0.01

        self._Kbond        = 10 #10.
        self._max_bond_ext = 1.5#1.5
        self._r0 = 1 #1.5

        if self.particle == 'cluster':
            self._max_bond_ext = 1.02
            self._Kbond        = 100
            self._r0 = 1 #not used

        # you should set the boundaries after you introduced the lb fluid (I not
        # sure if this is a must but definitely saver)
        # The viscosity of 33.22 is too high, the frictions is ok.
        # The problem with LB is that is only reproduces the correct HIs in a certain
        # parameter range and you have to rescale the necessary parameters to be in
        # that range. This includes the density of the fluid as well. I would
        # recommend to use a friction between 1-50, viscosity 0.5-5 and density
        # 0.1-5 (for the density range I'm not 100% sure but 0.042 looks too small
        # for me). You should check your parameter set by investigating the motion
        # of a single particle in the fluid. This gives you the best way to check
        # for suitable parameters.

        # Scales of the MD units (SI units)
        self.d_          = sigma_gel                                                   # length scale, m
        scale_on_mass_of = mass_gel if self.particle in ['mng', 'raspberry'] else mass_m
        self.mass_       = self._m_scale*scale_on_mass_of                              # mass scale, kg 1.6 1.4 20 0.8 | Re  = 01 : 5 !!!
        U_               = self._e_scale*kb*T                                          # energy scale, m^2 kg / s^2 (J) 1000 20 100000 1000000 | Re = 01 : 100
        self.t_          = self.d_*sqrt(self.mass_/U_)                                 # time scale, s
        v_               = self.d_/self.t_                                             # velocity scale, m/s

        dip_ = np.sqrt(4.*pi*U_*(self.d_**3.)/mu0)  # dipole scale, A m^2

        A_      = dip_/(self.d_*self.d_)           # Ampere scale,    A
        B_      = self.mass_/(A_*self.t_*self.t_)  # B-field scale,   kg/ A s^2 (T)
        self.F_ = self.mass_*self.d_/(self.t_*self.t_) # Force scale, kg m/s^2
        
        self.nub_of_int_steps_for_rot_field = 7

        # Reduced parmeters (MD units)
        ## relevant particle properties
        self._gamma_t_gel = gamma_t_gel*self.t_/self.mass_
        self._sigma_gel   = sigma_gel/self.d_
        self._mass_gel    = mass_gel/self.mass_
        self._momI_gel    = momI_gel/(self.mass_*self.d_*self.d_)

        self._gamma_t_m = gamma_t_m*self.t_/self.mass_
        self._sigma_m   = sigma_m/self.d_
        self._mass_m    = mass_m/self.mass_
        self._momI_m    = momI_m/(self.mass_*self.d_*self.d_)
        self._mu        = dip/dip_
        self._gamma_r_m = gamma_r_m*self.t_/(self.mass_*self.d_*self.d_)


        _dens_w     = dens_w * self.d_**3 / self.mass_
        self._eta_w = eta_w * self.d_ * self.t_ / self.mass_
        _visc       = self._eta_w / _dens_w

        #Magnetic field
        self._hmf /= (1000*B_)
        # self._nHs = int(round(1. /(self.hmf_freq * self.nub_of_int_steps_for_rot_field * self.t_)))

        ## LB parameters
        self.lb_agrid      = self._sigma_gel
        self.lb_dens       = _dens_w
        self.lb_visc       = _visc
        self.lb_tau        = self.tmstep_size_lb #self._t_scale                                                            #0.001 | different Re 0.3
        lb_g               = 25.0                                                                       #Ahlrichs-Dünweg geometric factor
        #!!!1💥 R_h_factor = 1; R_h_factor was 0.125
        R_h_factor = 1 if self.particle in ['mng', 'cluster'] else 0.5 #1
        self.lb_fric       = 1. / (1. / (self._gamma_t_gel*R_h_factor) - 1. / (lb_g * self._eta_w * self.lb_agrid))
        lb_velocity        = 0
        self.lb_force_dens = [0,0.,0.]
        self.shear_rate  = f_velocity/self.d_*self.t_
        # if self.args.shear: 
        # lb_velocity = float(self.args.shear)
        # elif self.args.poiseuille: 
        # lb_velocity        = float(self.args.poiseuille)
        # self.lb_force_dens = [lb_velocity,0.,0.]

        # background and channel properties
        self._kT         = kb*T/U_
        self._W          = (self._box_l)*self._sigma_gel
        self._box_l     *= self._sigma_gel                # WARNING! factor should be an integer
        self._gap        = 5.                             #better don't change
        self._wall_type  = 1000
        # _Bz             = float(self.args.Bz)

        #magnetic parameter
        self._alpha = self._hmf * self._mu / self._kT 

        # fix problems with bonds
        self._max_bond_ext *= self._sigma_m
        self._r0           *= self._sigma_m
        self._Kbond        *= self._kT

        # repulsive Lennard-Jones
        self._epsWCA   = 1.
        self._lj_eps   = 1 * self._kT
        self._lj_sigma = 16*self._sigma_m if self.particle in ['sphere'] else self._sigma_m
        self._lj_cut   = 1.12246 * self._lj_sigma
        self._lj_shift = 0.25 #* self._lj_eps

        if self.particle == 'cluster':
            self._epsWCA   *= 10*self._lj_eps

        # auxiliary and integration parameters
        self._p_type   = 0
        self._mnp_type = 1
        self._t_step   = self._t_scale  #0.001
        self._rseed    = self.rnd_seed#os.getpid() #42

        self._ddSumCpu_prefactor = 1
        self._p3m_prefactor      = 1
        self._p3m_cao            = 3
        self._p3m_mesh           = [20,20,20]
        # self._p3m_mesh           = [16,16,16]
        self._p3m_r_cut          = 5
        self._p3m_r_cut_iL       = 3
        self._p3m_accuracy       = 8E-3
        # self._p3m_accuracy       = 5E-4
        self._p3m_alpha          = 1.09903
        self._p3m_alpha_L        = 2.06695e+01
        self._p3m_tune           = True
        self._dlc_maxPWerror     = 1E-3
        self._dlc_gap_size       = 2*self._gap
        self._min_skin           = 0.2
        self._max_skin           = 4
        self._tol                = 0.2
        self._int_steps          = 1000

        self._timesteps_init = self._timesteps
        self._t_step_init    = self._t_step


@dataclass
class TMP_Parameters: 

    #ts for dynamic sim. step
    ts_mult: int = 0

    def __post_init__(self): 
        pass

        

def read_param_from_args(self):
    parser = argparse.ArgumentParser()

    # parser.add_argument("--shear", action="store", dest="shear_rate", type=float)
    parser.add_argument("--project_name", action="store", dest="project_name", type=str)
    parser.add_argument("--script_path", action="store", dest="script_folder", type=str)
    parser.add_argument("--global_full_path", action="store", dest="project_path", type=str)
    parser.add_argument("--tmstep", action="store", dest="timesteps", type=int)
    parser.add_argument("--tmstep_size", action="store", dest="timestep_size", type=float)
    parser.add_argument("--tmstep_size_lb", action="store", dest="tmstep_size_lb", type=float)
    parser.add_argument("--particle", action="store", dest="particle", type=str)
    parser.add_argument("--config", action="store", dest="conf", type=str)
    parser.add_argument("--field", action="store", dest="hmf", type=float)
    parser.add_argument("--field_direction", action="store", dest="hmf_direct", type=str)
    parser.add_argument("--field_freq", action="store", dest="hmf_freq", type=float)
    parser.add_argument("--lmbda", action="store", dest="lmbda", type=int)
    parser.add_argument("--N_part", action="store", dest="N_part", type=int)
    parser.add_argument("--box_len", action="store", dest="box", type=int)
    parser.add_argument("--new_continue_sim", action="store", dest="new_continue_sim", type=str)
    parser.add_argument("--extra_mnps", action="store", dest="extra_mnps", type=int)
    parser.add_argument("--is_cpu", type=bool, default=False)
    parser.add_argument("--steps", action="store", dest="num_of_steps", type=int)
    parser.add_argument("--corr_tau_max", action="store", dest="corr_tau_max", type=int)
    parser.add_argument("--thermostat", action="store", dest="thermostat", type=str)
    parser.add_argument("--experiment", action="store", dest="experiment", type=str)
    parser.add_argument("--worker", action="store", dest="worker", type=str)
    parser.add_argument("--pypresso", action="store", dest="pypresso", type=str)
    parser.add_argument("--prefix", action="store", dest="prefix", type=str)
    parser.add_argument("--no_dipdip", action="store", dest="no_dipdip", type=int)
    parser.add_argument("--fixed_mnps", action="store", dest="fixed_mnps", type=int)
    parser.add_argument("--z_coord", action="store", dest="z_coord", type=int)
    parser.add_argument("--x_coord", action="store", dest="x_coord", type=int)
    parser.add_argument("--fix_rasp", action="store", dest="fix_rasp", type=int)
    parser.add_argument("--zero_coupling", action="store", dest="zero_coupling", type=int)
    parser.add_argument("--_m_scale", action="store", dest="_m_scale", type=float)
    parser.add_argument("--_t_scale", action="store", dest="_t_scale", type=float)
    parser.add_argument("--_e_scale", action="store", dest="_e_scale", type=float)
    parser.add_argument("--_v_factor", action="store", dest="_v_factor", type=float)
    parser.add_argument("--add_init_vel_z", action="store", dest="add_init_vel_z", type=float)
    parser.add_argument("--g", action="store", dest="g", type=float)
    parser.add_argument("--lmbda_scale", action="store", dest="lmbda_scale", type=float)
    parser.add_argument("--write_vtf", action="store", dest="write_vtf", type=int)
    parser.add_argument("--write_vtk", action="store", dest="write_vtk", type=int)
    parser.add_argument("--custom_id", action="store", dest="custom_id", type=int)
    parser.add_argument("--tracer_size", action="store", dest="tracer_size", type=float)
    parser.add_argument("--rnd_seed", action="store", dest="rnd_seed", type=int)
    group2 = parser.add_mutually_exclusive_group()
    group2.add_argument('--shear_rate', action="store", dest="shear_rate", type=float)
    group2.add_argument('--poiseuille', action="store", dest="poiseuille", type=float)


    self.args = vars(parser.parse_args())
    print(self.args)

    if self.args["project_name"] != None:
        self.project_name = self.args["project_name"]
    else:
        warnings.warn("project_name wasn't given")

    if self.args["script_folder"] != None:
        self.script_folder = self.args["script_folder"]
    else:
        warnings.warn("script_folder wasn't given")

    if self.args["project_path"] != None:
        self.project_path = self.args["project_path"] + self.args["project_name"] + "/"
        Path(self.project_path).mkdir(parents = True, exist_ok = True)
    else:
        warnings.warn("project_path wasn't given")

    if self.args["timesteps"] != None:
        self.timesteps = self.args["timesteps"]
    else:
        warnings.warn("timesteps wasn't given")

    if self.args["timestep_size"] != None:
        self.timestep_size = self.args["timestep_size"]
    else:
        warnings.warn("timestep_size wasn't given")

    if self.args["tmstep_size_lb"] != None:
        self.tmstep_size_lb = self.args["tmstep_size_lb"]
    else:
        warnings.warn("tmstep_size_lb wasn't given")

    if self.args["particle"] != None:
        self.particle = self.args["particle"]
    else:
        warnings.warn("particle wasn't given")

    if self.args["conf"] != None:
        self.conf = self.args["conf"]
    else:
        warnings.warn("conf wasn't given")

    if self.args["lmbda"] != None:
        self.lmbda = self.args["lmbda"]
    else:
        warnings.warn("lmbda wasn't given")

    if self.args["hmf"] != None:
        self.hmf = self.args["hmf"]
    else:
        warnings.warn("hmf wasn't given")

    if self.args["hmf_direct"] in ['x', 'y', 'z', 'o']:
        self.hmf_direct = self.args["hmf_direct"]
    else:
        warnings.warn("hmf direction is wrong or not given")

    if self.args["hmf_freq"] != None:
        self.hmf_freq = self.args["hmf_freq"]
    else:
        warnings.warn("hmf_freq wasn't given")

    if self.args["N_part"] != None:
        self.N_part = self.args["N_part"]
    else:
        warnings.warn("N_part wasn't given")

    if self.args["box"] != None:
        self.box_len = self.args["box"]
    else:
        warnings.warn("box wasn't given")

    if self.args["new_continue_sim"] != None:
        self.new_continue_sim = self.args["new_continue_sim"]
    else:
        warnings.warn("new_continue_sim wasn't given")

    if self.args["extra_mnps"] != None:
        self.extra_mnps = self.args["extra_mnps"]
    else:
        warnings.warn("extra_mnps wasn't specified")

    if self.args["corr_tau_max"] != None:
        self.corr_tau_max = self.args["corr_tau_max"]
    else:
        warnings.warn("corr_tau_max wasn't given")

    if self.args["thermostat"] != None:
        self.thermostat = self.args["thermostat"]
    else:
        warnings.warn("thermostat wasn't given")

    if self.args["num_of_steps"] != None:
        self.num_of_steps = self.args["num_of_steps"]
    else:
        warnings.warn("num_of_steps wasn't given")

    if self.args["no_dipdip"] != None:
        self.no_dipdip = bool(self.args["no_dipdip"])
    else:
        warnings.warn("no_dipdip wasn't given and was set to False")
        self.no_dipdip = False

    if self.args["fixed_mnps"] != None:
        self.fixed_mnps = bool(self.args["fixed_mnps"])
    else:
        warnings.warn("fixed_mnps wasn't given and was set to False")
        self.fixed_mnps = False

    if self.args["z_coord"] != None:
        self.z_coord = self.args["z_coord"]
    else:
        warnings.warn("z_coord wasn't given")
        self.z_coord = None

    if self.args["x_coord"] != None:
        self.x_coord = self.args["x_coord"]
    else:
        warnings.warn("x_coord wasn't given")
        self.x_coord = None

    if self.args["fix_rasp"] != None:
        self.fix_rasp = bool(self.args["fix_rasp"])
    else:
        warnings.warn("fix_rasp wasn't given")
        self.fix_rasp = False

    if self.args["zero_coupling"] != None:
        self.zero_coupling = bool(self.args["zero_coupling"])
    else:
        warnings.warn("zero_coupling wasn't given and was set to False")
        self.zero_coupling = False

    if self.args["_m_scale"] != None:
        self._m_scale = self.args["_m_scale"]
    else:
        warnings.warn("_m_scale wasn't given")
        self._m_scale = None

    if self.args["_e_scale"] != None:
        self._e_scale = self.args["_e_scale"]
    else:
        warnings.warn("_e_scale wasn't given")
        self._e_scale = None

    if self.args["_t_scale"] != None:
        self._t_scale = self.args["_t_scale"] 
    else:
        warnings.warn("_t_scale wasn't given")
        self._t_scale = None

    if self.args["_v_factor"] != None:
        self._v_factor = self.args["_v_factor"] 
    else:
        warnings.warn("_v_factor wasn't given")
        self._v_factor = None
    
    if self.args["add_init_vel_z"] != None:
        self.add_init_vel_z = float(self.args["add_init_vel_z"])
    else:
        warnings.warn("add_init_vel_z wasn't given and was set to 0")
        self.add_init_vel_z = 0

    if self.args["lmbda_scale"] != None:
        self.lmbda_scale = float(self.args["lmbda_scale"])
    else:
        warnings.warn("lmbda_scale wasn't given")
        self.lmbda_scale = 1

    if self.args["prefix"] != None:
        self.prefix = str(self.args["prefix"])
    else:
        warnings.warn("prefix wasn't given")
        self.prefix = ''

    if self.args["pypresso"] != None:
        self.pypresso = str(self.args["pypresso"])
    else:
        warnings.warn("pypresso wasn't given")
        self.pypresso = ''

    if self.args["write_vtf"] != None:
        self.write_vtf_ = bool(self.args["write_vtf"])
    else:
        warnings.warn("write_vtf wasn't given and was set to False")
        self.write_vtf_ = False

    if self.args["write_vtk"] != None:
        self.write_vtk_ = bool(self.args["write_vtk"])
    else:
        warnings.warn("write_vtk wasn't given and was set to False")
        self.write_vtk_ = False

    if self.args["custom_id"] != None:
        self.custom_id = int(self.args["custom_id"])
    else:
        warnings.warn("custom_id wasn't given and was set to False")
        self.custom_id = 0

    if self.args["tracer_size"] != None:
        self.tracer_size = self.args["tracer_size"]
    else:
        warnings.warn("tracer_size wasn't specified")
        self.tracer_size = 1

    if self.args["rnd_seed"] != None:
        self.rnd_seed = self.args["rnd_seed"]
    else:
        warnings.warn("rnd_seed wasn't specified")
        self.rnd_seed = 42


    
    # # #GEL BUMP
    self.project_name = "test"
    self.script_folder = "/home/fs70806/Vania/worker_8/magnetic_nanogels/scripts/all_in_flow/"
    self.project_path = "/gpfs/data/fs70806/Vania/all_in_flow/mng/test/"
    self.timesteps = 100# 155000
    self.particle = 'mng'
    self.conf = 'A_0'#'1_0'#0
    self.lmbda = 5
    self.hmf = 0
    self.hmf_direct = 'x'
    self.hmf_freq = 0#.5*2895408
    self.N_part = 1
    self.box_len = 50 #216
    self.args["shear_rate"] = 0#9e5 #*1.51
    self.new_continue_sim = "new"
    self.num_of_steps = 20
    self.corr_tau_max = 0#10000
    self.extra_mnps = 0
    self.thermostat = 'brownian' #or 'lb'
    self.no_dipdip = 0
    self.fixed_mnps = 1
    self.timestep_size = 0.001
    self.tmstep_size_lb = 0.001*16
    self._m_scale = 1.6
    self._t_scale = 0.001 #setted in system
    self.part_radius = 7
    self.zero_coupling = False
    self.add_init_vel_z = 0
    self.g = 0#3.6e-12
    self.lmbda_scale = 1
    # self.write_vtf_ = True
    self.write_vtk_ = True

    v_trick = 1

    self._e_scale = 0.179*1000  / v_trick**2
    self._v_factor = 1/v_trick
    self.prefix = "taskset -c 1 mpirun -np 1 "


    self.write_vtf_ = True
    self.write_vtk_ = True


    


# %%
