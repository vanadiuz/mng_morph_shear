import types
import espressomd
from espressomd.interactions import FeneBond
from espressomd.interactions import HarmonicBond
from espressomd.interactions import RigidBond
import espressomd.magnetostatics as magnetostatics
from espressomd.constraints import HomogeneousMagneticField, Gravity
import espressomd.actors
from espressomd.observables import MagneticDipoleMoment, ParticleForces, ParticleVelocities
import espressomd.accumulators
import espressomd.observables
from espressomd.virtual_sites import VirtualSitesRelative

# LB
from espressomd import shapes
from espressomd import lb#, lbboundaries

import numpy as np
import pandas as pd
import json

def init_system(self):
    """
    creates minimal MD system

    """

    # if self.particle == 'raspberry':
    #     self.box_len = np.array([
    #         int(self.param._box_l*2),
    #         self.param._box_l,
    #         self.param._box_l+2*self.param._gap
    #     ]) 
    # else:
    #     self.box_len = np.array([
    #         self.param._box_l,
    #         self.param._box_l,
    #         self.param._box_l
    #         ]) 

    self.box_len = np.array([
            self.param._box_l,
            self.param._box_l,
            self.param._box_l
            ]) 

    self.s = espressomd.System(box_l=self.box_len)

    self.s.virtual_sites = VirtualSitesRelative() #(have_velocity=True, have_quaternion=False)



def configure_system(self):
    """
    configures MD system based on passed and calculated parameters

    """
    np.random.seed(seed=self.param._rseed)
    self.s.periodicity = self.param.periodicity
    self.s.time_step = self.param._t_step
    self.s.cell_system.skin = self.param.skin
    self.s.cell_system.set_regular_decomposition(use_verlet_lists=self.param.domain_decomposition_VL)
    self.s.min_global_cut = 1.05 * self.param._max_bond_ext

def reset_seed(self):
    """
    reset seed for the case of brocken bonds 

    """
    self.param._rseed += 1
    np.random.seed(seed=self.param._rseed)


def free_up_resources(self):
    """
    deletes all simulation related stuff to reload previous checkpoint

    """

    try:
        del self.dpm_corr
        del self.pickled_dpm_corr
    except:
        print('del self.dpm_corr and self.pickled_dpm_corr failed')

    try:
        del self.direct_sum
    except:
        try:
            del self.dlc
            del self.p3m
        except:
            print('del self.direct_sum or del self.p3m failed')
    
    # del self.param
    # del self.lbf

    self.s.thermostat.turn_off()
    self.s.part.clear()
    self.s.constraints.clear()
    self.s.actors.clear()
    self.s.lbboundaries.clear()
    self.s.bonded_inter.clear()
    self.s.auto_update_accumulators.clear()
    del self.lbf

    del self.s
    try:
        del self.checkpoint
    except:
        print('del self.checkpoint failed')

def add_fene_and_harm(self):
    """
    add bonded interractions (for gels)

    """

    # interactions
    fene = FeneBond(k=self.param._Kbond*2.25, d_r_max=self.param._max_bond_ext)
    hb = HarmonicBond(k=self.param._Kbond*1, r_0=self.param._lj_sigma)
    # fene_mnps = FeneBond(k=self.param._Kbond*2, d_r_max=self.param._max_bond_ext-.95*self.param._sigma_m) #-0.5*self.param._sigma_m
    fene_mnps = FeneBond(k=self.param._Kbond*2.25, d_r_max=0.95*self.param._sigma_m)
    # hb_tmp = HarmonicBond(k=self.param._Kbond*50, r_0=self.param._lj_sigma*0.0) #for equilibrating frozen mnps
    # hb_tmp = HarmonicBond(k=self.param._Kbond*50, r_0=self.param._lj_sigma*0.0) #for equilibrating frozen mnps
    # hb_tmp = HarmonicBond(k=self.param._Kbond*1, r_0=self.param._lj_sigma*0.1) #for equilibrating frozen mnps
    # rig = RigidBond(r=self.param._lj_sigma*0.1, ptol=0.001, vtol=10)

    self.s.bonded_inter.add(fene)
    self.s.bonded_inter.add(hb)
    self.s.bonded_inter.add(fene_mnps)
    # self.s.bonded_inter.add(hb_tmp)



def add_magnetostatics_DSCpu(self):
    """
    for counting dip-dip interractions

    """

    if not self.no_dipdip:
        # ⚠️ ok for small num. of dipoles
        self.direct_sum = magnetostatics.DipolarDirectSumCpu(prefactor=self.param._ddSumCpu_prefactor)
        # direct_sum = magnetostatics.DipolarDirectSumWithReplicaCpu(prefactor=1, n_replica=0)
        self.s.actors.add(self.direct_sum)

def add_magnetostatics_DSGpu(self):
    """
    for counting dip-dip interractions

    """

    if not self.no_dipdip:
        # ⚠️ ok for small num. of dipoles
        self.direct_sum = magnetostatics.DipolarDirectSumGpu(prefactor=self.param._ddSumCpu_prefactor)
        # direct_sum = magnetostatics.DipolarDirectSumWithReplicaCpu(prefactor=1, n_replica=0)
        self.s.actors.add(self.direct_sum)



def add_magnetostatics_P3M_DLC(self, add_actor=True):
    """
    for counting dip-dip interractions in case of a suspension

    """
    self.p3m = magnetostatics.DipolarP3M(
        prefactor=self.param._p3m_prefactor, 
        cao=self.param._p3m_cao, 
        mesh=self.param._p3m_mesh, 
        r_cut=self.param._p3m_r_cut, 
        alpha = self.param._p3m_alpha_L,
        accuracy=self.param._p3m_accuracy, 
        tune=self.param._p3m_tune
    )
    # self.s.actors.add(p3m)

    self.dlc = magnetostatics.DLC(
        actor=self.p3m,
        maxPWerror=self.param._dlc_maxPWerror,
        gap_size=self.param._dlc_gap_size,
    )


    if add_actor:
        self.s.actors.add(self.dlc)


def add_magnetostatics_P3M(self, add_actor=True):
    """
    for counting dip-dip interractions in case of a suspension

    """

    # self.p3m = magnetostatics.DipolarP3M(
    #     prefactor=self.param._p3m_prefactor, 
    #     cao=self.param._p3m_cao, 
    #     alpha = self.param._p3m_alpha,
    #     mesh=self.param._p3m_mesh, 
    #     r_cut=self.param._p3m_r_cut, 
    #     accuracy=self.param._p3m_accuracy, 
    #     tune=self.param._p3m_tune
    # )

    
    # self.s.cell_system.tune_skin(
    #     min_skin=self.param._min_skin, 
    #     max_skin=self.param._max_skin,
    #     tol=self.param._tol, 
    #     int_steps=self.param._int_steps
    # )

    self.p3m = magnetostatics.DipolarP3M(
        prefactor=self.param._p3m_prefactor, 
        cao=self.param._p3m_cao, 
        alpha = self.param._p3m_alpha,
        mesh=self.param._p3m_mesh, 
        r_cut=self.param._p3m_r_cut, 
        accuracy=self.param._p3m_accuracy, 
        tune=self.param._p3m_tune
    )

    # self.s.cell_system.tune_skin(
    #     min_skin=self.param._min_skin, 
    #     max_skin=self.param._max_skin,
    #     tol=self.param._tol, 
    #     int_steps=self.param._int_steps
    # )

    # if len(self.s.actors.active_actors) == 1:
    #     self.s.actors.clear()
    #     self.s.actors.add(self.p3m)

    if add_actor:
        self.s.actors.add(self.p3m)


#TODO add frequency
def add_magnetic_field(self):
    """
    HMF is applied along Z direction!

    """

    # self.hmf_freq

    if self.hmf != 0:
        if self.hmf_freq == 0:
            if self.hmf_direct == 'x':
                H_constraint = espressomd.constraints.HomogeneousMagneticField(
                    H=np.array([self.param._hmf, 0.0, 0.0])
                )
            elif self.hmf_direct == 'y':
                H_constraint = espressomd.constraints.HomogeneousMagneticField(
                    H=np.array([0.0, self.param._hmf, 0.0])
                )
            elif self.hmf_direct == 'z':
                H_constraint = espressomd.constraints.HomogeneousMagneticField(
                    H=np.array([0.0, 0.0, self.param._hmf])
                )

            self.s.constraints.add(H_constraint)
        else:
            self.gen_rot_magnetic_field()


def gen_rot_magnetic_field(self):
    """
    rotational magnetic field

    """
    int_st = self.param.nub_of_int_steps_for_rot_field

    self.Hfields = []
    self.Hvecs = []
    dth = 2. * np.pi / float(int_st)
    Hy = 0.
    Hz = 0.
    th = 0.

    for _ in range(int_st):
        Hx = self.param._hmf * np.cos(th)
        Hz = self.param._hmf * np.sin(th)
        self.Hfields.append(espressomd.constraints.HomogeneousMagneticField(H=[Hx, Hy, Hz]))
        th += dth

def _setHfield(self, Hobj, log = True):

    s = self.s
    while len(s.constraints) > 0:
        H0 = s.constraints[0]
        if log:
            print("Removing existing external field "+str(H0.H))
        s.constraints.remove(H0)
    if log:
        print("Applying external magnetic field %s"%(str(Hobj.H)))
    s.constraints.add(Hobj)
    return



def enable_brownian_thermostat_for_mnps(self, zero_kbT=False):
    """
    setup brownian dynamics for rotations of mnp's
    ⚠️ use it very careful ⚠️ 

    """
    # self.s.integrator.set_vv()
    self.s.thermostat.turn_off()
    self.s.thermostat.set_brownian(
        kT=self.param._kT if not zero_kbT else 0,
        gamma=self.param._gamma_t_m,
        gamma_rotation=self.param._gamma_r_m,
        act_on_virtual=False,
        seed=self.param._rseed,
    )  # , act_on_virtual=False
    self.s.integrator.set_brownian_dynamics()
    # self.s.integrator.run(0)
    self.s.thermostat.turn_off()
    self.s.integrator.set_vv()


def enable_lb_thermostat(self, zero_kbT = False, act_on_virt = False):
    """
    for fun

    """
    if self.args["is_cpu"]:
        self.lbf = lb.LBFluid(
            kT=self.param._kT if not zero_kbT else 0,# if self.param.particle != 'raspberry' else 0,
            seed=self.param._rseed,
            agrid=self.param.lb_agrid,
            dens=self.param.lb_dens,
            visc=self.param.lb_visc,
            tau=self.param.lb_tau,
            ext_force_density=self.param.lb_force_dens,
        )
    else:
        # self.lbf = lb.LBFluidGPU(
        #     kT=self.param._kT if not zero_kbT else 0,# if self.param.particle != 'raspberry' else 0,
        #     seed=self.param._rseed,
        #     agrid=self.param.lb_agrid,
        #     dens=self.param.lb_dens,
        #     visc=self.param.lb_visc,
        #     tau=self.param.lb_tau,
        #     ext_force_density=self.param.lb_force_dens,
        # )

        self.lbf = lb.LBFluidWalberlaGPU(
            kT=self.param._kT if not zero_kbT else 0,# if self.param.particle != 'raspberry' else 0,
            seed=self.param._rseed,
            agrid=self.param.lb_agrid,
            dens=self.param.lb_dens,
            visc=self.param.lb_visc,
            tau=self.param.lb_tau,
            ext_force_density=self.param.lb_force_dens,
        )

    self.s.actors.add(self.lbf)

    self.s.thermostat.set_lb(
        LB_fluid=self.lbf, 
        seed=self.param._rseed, 
        gamma=self.param.lb_fric if self.zero_coupling == False else 0, 
        act_on_virtual=act_on_virt
    )

    # self.lbf.write_vtk_boundary(self.project_path + "/vtk/boundary.vtk")

def enable_lang_thermostat(self):
    """
    for fun. 

    """
    # TODO: uncommit and be very careful doing merge commit
    # self.s.thermostat.turn_off()
    # self.s.thermostat.set_brownian(
    #     kT=0,
    #     gamma=0,
    #     gamma_rotation=0,
    #     act_on_virtual=False,
    #     seed=self.param._rseed,
    # )  # , act_on_virtual=False
    # self.s.thermostat.turn_off()

    self.s.thermostat.set_langevin(kT=self.param._kT, gamma=self.param._gamma_t_m, gamma_rotation=self.param._gamma_r_m, seed=self.param._rseed, act_on_virtual=False)

def enable_brownian_thermostat(self):
    """
    for fun. 

    """
    self.s.thermostat.set_brownian(kT=1*self.param._kT, gamma=self.param._gamma_t_m, gamma_rotation=self.param._gamma_r_m, seed=self.param._rseed, act_on_virtual=False)
    self.s.integrator.set_brownian_dynamics()


def add_walls_and_shear_flow(self, shear_flow=False):
    """
    one of LB method limitations

    """

    wall_bottom = shapes.Wall(normal=[0, 0, 1.0], dist=self.param._gap ) #self.param._gap
    wall_top = shapes.Wall(normal=[0, 0, -1.0], dist=-(self.param._box_l - 2*self.param._gap))

    self.s.constraints.add(
        shape=wall_bottom,
        particle_type=self.param._wall_type,
        penetrable=False,
        only_positive=True,
    )
    self.s.constraints.add(
        shape=wall_top,
        particle_type=self.param._wall_type,
        penetrable=False,
        only_positive=True,
    )

    lbwall = lbboundaries.LBBoundary(shape=wall_bottom, velocity=[0, 0, 0])
    self.s.lbboundaries.add(lbwall)

    if shear_flow == True:
        v = self.param.shear_rate*self.param._box_l

        shear_profile = np.linspace(0, v, int(self.param._box_l - 3*self.param._gap))

        for idx, sh in enumerate(shear_profile[:]):
            for _ in range(int(self.param._box_l)):
                for __ in range(int(self.param._box_l)):
                    self.lbf[_,__,idx+int(self.param._gap)].velocity = np.array([sh, 0, 0])
    else:
        v = 0

    ltwall = lbboundaries.LBBoundary(shape=wall_top, velocity=[v, 0, 0]) 
    self.s.lbboundaries.add(ltwall)


def add_walls(self):
    """
    walls for langevin dynamics

    """

    wall_bottom = shapes.Wall(normal=[0, 0, 1.0], dist=self.param._gap ) #self.param._gap
    wall_top = shapes.Wall(normal=[0, 0, -1.0], dist=-(self.param._box_l - 2*self.param._gap))

    self.s.constraints.add(
        shape=wall_bottom,
        particle_type=self.param._wall_type,
        penetrable=False,
        only_positive=True,
    )
    self.s.constraints.add(
        shape=wall_top,
        particle_type=self.param._wall_type,
        penetrable=False,
        only_positive=True,
    )

def add_sphere_around_particle(self, radius=12, center=[False, False, False]):
    """
    creates spherical constraint abound passed particle (i.e. gel, cluster)

    """
    if center[0] == False:
        cm = self.calc_cm()
    else:
        cm = center

    sphere = espressomd.shapes.Sphere(center=cm, radius=radius, direction=-1)

    self.s.constraints.add(
        shape=sphere,
        particle_type=self.param._wall_type,
        penetrable=False,
    )

def remove_all_constraits(self):
    """
    removes all constraints

    """
    for c in self.s.constraints:
        self.s.constraints.remove(c)


def gen_observ_dip_moment(self):
    """
    generatges observator for dipole moment of all MNPs

    """

    mnp_s = []
    for p in self.s.part.all():
        if p.dip[0] != 0:
            mnp_s.append(p.id)

    self.part_dip = espressomd.observables.MagneticDipoleMoment(ids=mnp_s)


def gen_observ_forces(self):
    """
    generatges force observator for rasp

    """
    if self.particle == 'raspberry':
        part_force = ParticleForces(ids=[0])
        self.part_force = part_force
    else:
        self.part_force = None
    

def run(self, timesteps):
    self.s.integrator.run(timesteps)
    self.pos_dip_id += 1

def run_rot_H(self, timesteps):
    self.s.integrator.run(timesteps)


def init_vtf(self, types="all"):
    """
    runs only once at the beginning

    """

    fp = open(self.project_path + self.base + ".vtf", "w")
    self._writevsf(fp, types=types)
    fp.flush()

def print_parameters(self):

    # Print some parameters
    # -------------
    print("param.shear_rate:" + str(self.param.shear_rate))
    print("project_name:" + str(self.project_name))
    print("script_folder:" + str(self.script_folder))
    print("timesteps:" + str(self.timesteps))
    print("timestep_size:" + str(self.param._t_scale))
    print("box_len:" + str(self.box_len))

    # finally, let's log what we have
    print("\nSystem parameters")
    print("-----------------")
    print("rseed:" + str(self.param._rseed))
    print("kT:" + str(self.param._kT))
    print("sigma:" + str(self.param._sigma_m))
    print("mass_m:" + str(self.param._mass_m))
    print("momI_m:" + str(self.param._momI_m))
    print("mass_gel:" + str(self.param._mass_gel))
    print("momI_gel:" + str(self.param._momI_gel))
    print("timescale:" + str(self.param.t_))
    print("massscale:" + str(self.param.mass_))
    print("distancescale:" + str(self.param.d_))
    print("mu:" + str(self.param._mu))
    print("lambda2:" + str(self.param.lambda_m))
    print("N_part:" + str(self.param._N_part))
    print("lb_dens:" + str(self.param.lb_dens))
    print("lb_visc:" + str(self.param.lb_visc))
    print("lb_fric:" + str(self.param.lb_fric) + " (gamma_t:" + str(self.param._gamma_t_gel) + ")")

    if self.args["poiseuille"] is None:
        print(
            "shear_rate:"
            + str(self.param.shear_rate)
            + " ("
            + str('{:.1e}'.format(self.param.shear_rate / self.param.t_))
            + " 1/s)"
        )
        print("wall_velocity:" + str(self.param.shear_rate*self.param._box_l))

        try:
            rad = self.part_radius 
        except:
            rad = 10

        print(
            "Estimated Reynolds number:"
            + str(self.param.lb_dens * self.param.shear_rate * rad**2 / self.param.lb_visc)
        )

    else:
        print(
            "poiseuille:"
            + str(self.param.poiseuille)
            + " ("
            + str('{:.1e}'.format(self.param.poiseuille / self.param.t_))
            + " 1/s)"
        )
        print(
            "Estimated Reynolds number:"
            + str(self.param.lb_dens * self.param.poiseuille * 10**2 / self.param.lb_visc)
        )
    

    print("gamma tr. gel:", self.param._gamma_t_gel)
    print("gamma tr. m.:", self.param._gamma_t_m)

    print("mass gel:", self.param._mass_gel)
    print("mass m.:", self.param._mass_m)

    print("H:", self.param._hmf)
    print("alpha:", self.param._alpha)


def write_parameters_to_json(self):
    """
    writes parameters to json file

    """
    print('write json')
    with open(self.project_path + self.base + ".json", "w") as fp:
        #concatenate two dictionaries
        param_dict = {**self.param.__dict__, **self.args} 
        json.dump(param_dict, fp, indent=4)

def split_vol(self, nogap=False):
    #a,b,c are the sides of the box
    #N is the number of subvolumes
    #returns list of centers of subvolumes
    if nogap == False:
        gap = self.param._gap
    else:
        gap = 0
    N = int(self.N_part**(1/3))+1
    a = self.box_len[0] - 1*gap
    b = self.box_len[1] - 1*gap
    c = self.box_len[2] - 5*gap
    self.centers = []

    for i in range(N):
        for j in range(N):
            for k in range(N):
                self.centers.append([
                    a/N*(i+0.5)+gap, 
                    b/N*(j+0.5)+gap, 
                    c/N*(k+0.5) + 2*gap
                ])

    np.random.shuffle(self.centers)
    return self.centers

def pack_spheres_into_sphere(self, r_outer, r, N):
    #r_outer is the radius of the outer sphere
    #r is the radius of the inner spheres
    #N is the number of spheres
    #returns list of centers of spheres
    centers = []
    while len(centers) < N:
        phi = np.random.uniform(0, 2*np.pi)
        theta = np.random.uniform(0, np.pi)
        r_inner = np.random.uniform(0, r_outer-r*1.1)

        x = r_inner*np.sin(theta)*np.cos(phi)
        y = r_inner*np.sin(theta)*np.sin(phi)
        z = r_inner*np.cos(theta)
        new_pos = np.array([x,y,z])

        to_add = True
        for c in centers:
            if np.linalg.norm(new_pos-c) < r*2.2:
                to_add = False
                break
        if to_add:
            centers.append(new_pos)

    shift = np.array([r_outer, r_outer, r_outer])
    for idx, c in enumerate(centers):
        centers[idx] = c + shift

    return centers


def shift_particle_to_center(self, gap=0):
    """
    shift particle to the center of the box
    or to the provided coordinates as args: z_coord and x_coord

    """

    if self.z_coord:
        pos_z = self.z_coord
    else:
        pos_z = (self.box_len[2]+2*gap)/2

    if self.x_coord:
        pos_x = self.x_coord
    else:
        pos_x = self.box_len[0]/2

    cm = [0]*3
    p_len = len(self.s.part.all())
    for p in self.s.part.all():
        for i, pp in enumerate(p.pos):
            cm[i] += pp/p_len
    
    delta_x = cm[0] - pos_x
    delta_y = cm[1] - self.box_len[1]/2
    delta_z = cm[2] - pos_z

    for p in self.s.part.all():
        p.pos = (p.pos[0] - delta_x, p.pos[1] - delta_y, p.pos[2] - delta_z)

def shift_particle_to_pos(self, pos, ids):
    """
    shifts particle to the provided position
    """

    cm = [0]*3
    p_len = len(ids)
    for p_id in ids:
        for i, pp in enumerate(self.s.part.by_id(p_id).pos):
            cm[i] += pp/p_len
    
    delta_x = cm[0] - pos[0]
    delta_y = cm[1] - pos[1]
    delta_z = cm[2] - pos[2]

    for p_id in ids:
        p = self.s.part.by_id(p_id)
        p.pos = (p.pos[0] - delta_x, p.pos[1] - delta_y, p.pos[2] - delta_z)

def shift_particle_to_positive_octant(self):
    """
    shift particle to the center of the box

    """

    cm = [0]*3
    p_len = len(self.s.part)
    for p in self.s.part.all():
        for i, pp in enumerate(p.pos):
            cm[i] += pp/p_len
    
    delta_x = cm[0] - self.box_len/2
    delta_y = cm[1] - self.box_len/2
    delta_z = cm[2] - (self.box_len+2*self.param._gap)/2

    for p in self.s.part.all():
        p.pos = (p.pos[0] - delta_x, p.pos[1] - delta_y, p.pos[2] - delta_z)


def init_dip_obs_and_corr(self, state):
    """
    init correlator for raspberries central bead to measure MACF

    """

    if state == 'new':

        self.dpm_corr = espressomd.accumulators.Correlator(
            obs1=self.part_dip,
            tau_lin=16, 
            delta_N=10,
            tau_max=self.corr_tau_max,
            corr_operation="scalar_product", compress1="discard1"
        )

    self.s.auto_update_accumulators.add(self.dpm_corr)


def add_vel_z(self):
    """
    add z-component of velocity to all particles

    """

    for p in self.s.part.all():

        p.v = np.array([p.v[0], p.v[1], p.v[2] + self.add_init_vel_z]) 

def add_gravity(self):
    """
    add gravity to the system
    """

    gravity_constraint = espressomd.constraints.Gravity(
        g=np.array([0.0, 0.0, self.g/self.param.F_])
    )

    self.s.constraints.add(gravity_constraint)

    
def map_pid_conf_type(self, list_of_ids_to_add, conf, it):
    """
    create pandas df with part_id, conf_type, part_type and [(connected_to, bonds_type)]
    """

    #create df with part_id, conf_type, part_type, bonds
    try:
        self.pid_conf_type 
    except:
        self.pid_conf_type = pd.DataFrame(columns=['part_id', 'conf_type', 'gel_id', 'part_type', 'bonds'])

    data = []
    for p in self.s.part.by_ids(list_of_ids_to_add):
        #add row to df
        data.append({
                'part_id': p.id,
                'conf_type': conf,
                'gel_id': it,
                'part_type': p.type,
                'bonds': p.bonds
            })
       

    self.pid_conf_type = self.pid_conf_type.append(data)
    

def add_ext_force(self, part_ids, force):
    """
    add ext_force
    """

    for p in self.s.part.by_ids(part_ids):
        if p.type != 2:
            p.ext_force = force

def remove_ext_force(self, part_ids=None):
    """
    remove ext_force
    """

    if part_ids != None:
        for p in self.s.part.by_ids(part_ids):
            p.ext_force = (0, 0, 0)
    else:
        for p in self.s.part.all():
            p.ext_force = (0, 0, 0)


def equilibrate_hard_spheres(self, num_of_part, positions, rg):
    """
    fill box with hard spheres and equilibrate them
    remove spheres and return their coordinates
    """

    for where_to_place in positions:
        #add particles
        particles = self.s.part.add(type=111, pos=where_to_place)

        
    LJ_EPS = 200
    Rg = rg
    LJ_SIG = 2*Rg
    LJ_CUT = 1.5 * LJ_SIG

    self.s.non_bonded_inter[111, 111].lennard_jones.set_params(
        epsilon=LJ_EPS, 
        sigma=LJ_SIG, 
        cutoff=LJ_CUT, 
        shift=0
    )

    self.s.non_bonded_inter[1000, 111].lennard_jones.set_params(
        epsilon=LJ_EPS, 
        sigma=LJ_SIG*0.6, 
        cutoff=LJ_CUT*0.6, 
        shift=0
    )
    print('start equilibration of spheres')
    self.s.integrator.run(steps=200)
    # cap = 1
    # sb.s.force_cap = cap
    min_dist = self.s.analysis.min_dist()
    print('initial min_dist ', min_dist)
    while min_dist < (Rg-1)*2:
        print(min_dist)
        self.s.integrator.run(steps=100)
        min_dist = self.s.analysis.min_dist()
        # print(' min_dist ', min_dist)
        # cap += 1
        # sb.s.force_cap = cap

    # system.force_cap = 0

    pos = []

    for p in self.s.part.all():
        pos.append(p.pos)
    
    for p in self.s.part.all():
        p.remove()

    np.random.shuffle(pos)

    return pos


def update_ts(self):
    """
    update simulation speed
    """

    self.param._timesteps = int(self.param._timesteps_init/2**self.tmp_param.ts_mult)
    self.s.time_step = 2**self.tmp_param.ts_mult * self.param._t_step_init

    print(self.param._timesteps, self.s.time_step)


def fix_mnps_along_field(self, field_direction = 0):
    for p in self.s.part.all():
        if p.dip[0] != 0:
            #get dipole moment of the particle
            dip = p.dip
            #get norm of dipole moment
            dip_norm = np.linalg.norm(dip)
            #set dipole moment to zero

            if field_direction == 0:
                p.dip = (dip_norm, 0, 0)
            elif field_direction == 1:
                p.dip = (0, dip_norm, 0)
            elif field_direction == 2:
                p.dip = (0, 0, dip_norm)

            p.rotation = [False, False, False]


def freeup_mnps_rot(self):
    for p in self.s.part.all():
        if p.dip[0] != 0 or p.dip[1] != 0 or p.dip[2] != 0:
            p.rotation = [True, True, True]


def rotate_mng(self, angle = 0):

    from scipy.spatial.transform import Rotation
    rot = Rotation.random()

    com = np.array(self.s.analysis.center_of_mass(1)) * 0.1 + np.array(self.s.analysis.center_of_mass(0)) * 0.9


    for p in self.s.part.all():

        pos = p.pos - com
        pos = rot.apply(pos) + com
        p.pos = pos

        director = rot.apply(p.director)
        p.director = director

    