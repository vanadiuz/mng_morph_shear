import sys
import io
import os
import warnings
import random
import re
import functools
import argparse
from datetime import datetime, timedelta
from copy import deepcopy, copy
import pickle
import numpy as np
import json
import time
import subprocess
import sys

from .parameters import Parameters, TMP_Parameters

from espressomd import checkpointing


class SimulationBox(object):

    def __init__(self):
        print('Hi from SimulationBox')

        self.pos_dip_id = 0

        self.read_param_from_args()
        self.compose_base()

        if self.new_continue_sim == "new":
            self.init_parameters()
            self.init_system()
            self.configure_system()
            self.add_fene_and_harm()
            # self.add_magnetostatics_DSCpu()
            # self.add_magnetostatics_P3M_DLC()

        elif self.new_continue_sim == "continue":
            self.find_checkpoint()
            self.load_system_from_checkpoint()
            self.reinit_thermostats() #self.thermostat

    def find_checkpoint(self):
        try:
            path, dirs, files = next(os.walk(self.project_path + "/" + self.base + "/"))
            if len(files) == 0:
                raise ValueError('No checkpoints/files in given dir!')

            self.checkpoint = checkpointing.Checkpoint(
                checkpoint_id=self.base, checkpoint_path=self.project_path
            )
        except:
            raise ValueError('No checkpoints/files in given dir!')


    def reinit_thermostats(self):

        if self.thermostat == 'lb':

            if self.param.particle == 'raspberry':
                act_on_virtual = True
            else:
                act_on_virtual = False

            self.s.thermostat.turn_off()
            self.s.thermostat.set_brownian(
                kT=self.param._kT,
                gamma=self.param._gamma_t_m,
                gamma_rotation=self.param._gamma_r_m,
                act_on_virtual=False,
                seed=self.param._rseed,
            )  # , act_on_virtual=False
            self.s.thermostat.turn_off()

            # self.s.actors.remove(self.lbf)
            # self.s.actors.add(self.lbf)

            self.s.thermostat.set_lb(
                LB_fluid=self.lbf, 
                seed=self.param._rseed, 
                gamma=self.param.lb_fric if self.zero_coupling == False else 0, 
                act_on_virtual=act_on_virtual
            )

            self.s.integrator.run(steps=0, reuse_forces=True)

        elif self.thermostat == 'lang':

            if self.param.particle == 'raspberry':
                act_on_virtual = True
            else:
                act_on_virtual = False

            self.s.thermostat.turn_off()
            self.s.thermostat.set_brownian(
                kT=0, #self.param._kT,
                gamma=self.param._gamma_t_m,
                gamma_rotation=self.param._gamma_r_m,
                act_on_virtual=False,
                seed=self.param._rseed,
            )  # , act_on_virtual=False
            self.s.thermostat.turn_off()

            self.s.thermostat.set_langevin(kT=self.param._kT, gamma=self.param._gamma_t_m, gamma_rotation=self.param._gamma_r_m, seed=self.param._rseed, act_on_virtual=False)

            self.s.integrator.run(steps=0, reuse_forces=True)

        elif self.thermostat == 'brownian':
            self.init_thermostat_brownian()


    def load_system_from_checkpoint(self):
        
        self.checkpoint.load()

        if self.thermostat == 'lb':
            lbf.load_checkpoint(self.project_path + "/lb_checkpoint", 1)
        
        self.s = s

        if self.thermostat == 'lb':
            self.lbf = lbf
        
        # self.param = param
        self.init_parameters()

        if not self.no_dipdip:
            try:
                self.direct_sum = direct_sum
            except:
                try:
                    if len(self.s.lbboundaries) > 0:
                        self.add_magnetostatics_P3M_DLC(add_actor=False)
                    else:
                        self.add_magnetostatics_P3M(add_actor=False)
                    # self.
                    # self.p3m = p3m
                    # self.dlc = dlc
                except:
                    pass
            
        self.pos_dip_id = pos_dip_id

        #load correlator for the rasp. director
        try:
            self.pickled_dpm_corr = pickled_dpm_corr
            self.dpm_corr = pickle.loads(pickled_dpm_corr)
        except:
            pass

    def preinit_thermostat(self, zero_kbT=False):
        self.enable_brownian_thermostat_for_mnps(zero_kbT)

    def init_thermostat(self, zero_kbT_B = False, zero_kbT_lb = False, act_on_virtual = False):
        # self.enable_brownian_thermostat_for_mnps()
        # self.preinit_thermostat(zero_kbT = zero_kbT_B)
        self.enable_lb_thermostat(zero_kbT = zero_kbT_lb, act_on_virt = act_on_virtual)

    def init_thermostat_rasp(self):
        self.enable_lb_thermostat(act_on_virt = True)

    def init_thermostat_lang(self):
        self.enable_lang_thermostat()

    def init_thermostat_brownian(self):
        self.enable_brownian_thermostat()

    def init_tmp_parameters(self):
        self.tmp_param = TMP_Parameters()

    def init_parameters(self):

        # _t_scale_mult = 1
        # if self.fixed_mnps:
        #     self.timesteps = int(self.timesteps*10)
        #     _t_scale_mult = 0.1


        self.param = Parameters(
            init_lambda    = self.lmbda,
            _N_part        = self.N_part,
            _box_l         = self.box_len,
            _timesteps     = self.timesteps,
            _hmf           = self.hmf,
            particle       = self.particle,
            _t_scale       = self._t_scale,
            tmstep_size_lb = self.tmstep_size_lb,
            _m_scale       = self._m_scale,
            _e_scale       = self._e_scale,
            _v_factor      = self._v_factor,
            rnd_seed       = self.rnd_seed,
        )

        self.tmp_param = TMP_Parameters()

        if self.args["shear_rate"]:
            self.param.shear_rate = self.args["shear_rate"] * self.param.t_ 
        else: 
            self.param.shear_rate = 0

        if self.args["poiseuille"]:
            self.param.poiseuille = self.args["poiseuille"] * self.param.t_ 
            self.param.lb_force_dens = [self.param.poiseuille, 0., 0.]
        else: 
            self.param.poiseuille = 0


    def init_tmp_containers_for_output_data(self):
        self.energies = []
        self.pos_and_dips = []
        self.pos_and_dips_hdf5 = []
        self.pos_dip_cms = []
        self.rgs = []
        self.vtfs = []
        self.momentas = []
        self.forces = []
        self.force_onto_wall = []

        if self.param.particle in ['raspberry', 'mng'] and self.corr_tau_max != 0:
            self.init_dip_obs_and_corr(self.new_continue_sim)

    def init_clocks(self):
        self.time_start = datetime.now()
        self.time_end = self.time_start + timedelta(hours=self.param._tot_sim_time)
        self.time_btw_checkpoints = 0
        self.time_checkpoint = datetime.now()

    def init_data_containers_and_clock(self):
        self.gen_observ_dip_moment()
        self.gen_observ_forces()
        self.init_clocks()
        self.init_tmp_containers_for_output_data()

    def write_sim_data(self, step, checkpoint_step=10):

        if self.fixed_mnps:
            checkpoint_step=checkpoint_step
        
        self.energies.append(self.get_energy())
        # self.pos_and_dips.append(self.get_pos_and_dip())
        self.pos_and_dips_hdf5.append(self.get_pos_and_dip_hdf5())
        self.pos_dip_cms.append(self.get_pos_dip_cm())
        self.rgs.append(self.get_rg(p_type=0))
        if self.write_vtf_ == True:
            self.vtfs.append(self.get_vcf(types=[0,1,2,3]))
        # self.momentas.append(self.get_momenta())
        # self.force_onto_wall.append(self.get_force_onto_wall())

        if self.particle == 'raspberry':
            self.forces.append(self.get_forces())

        if step % checkpoint_step == 0:
            # self.pos_and_dips_hdf5.append(self.get_pos_and_dip_hdf5())

            # if self.write_vtk_ == True:
            #     self.write_vtk(particles=False)

            self.write_energy()
            self.write_pos_dip_hdf5()
            # self.write_pos_and_dip()
            self.write_pos_dip_cm()
            self.write_rg()
            if self.write_vtf_ == True:
                self.write_vtf()
            # self.write_momenta()
            # self.write_force_onto_wall()

            if self.particle == 'raspberry':
                self.write_forces()
            
            self.make_checkpoint(self.thermostat)
            self.del_all_but_last_checkpoint()

            self.time_btw_checkpoints = (datetime.now() - self.time_checkpoint).total_seconds()
            self.time_checkpoint = datetime.now()

            #stop close to time limit
            approx_time_of_next_chkpnt = self.time_checkpoint + timedelta(seconds=1.22*self.time_btw_checkpoints)
            if approx_time_of_next_chkpnt > self.time_end:
                sys.exit()


    def write_sim_data_test(self, step, checkpoint_step=10):
        # self.energies.append(self.get_energy())
        self.vtfs.append(self.get_vcf(types=[0,1,2]))
        # self.force_onto_wall.append(self.get_force_onto_wall())
        if step % checkpoint_step == 0:
            # self.write_energy()
            # self.write_force_onto_wall()
            self.write_vtf()
            # self.write_vtk()

            self.make_checkpoint(self.thermostat)
            self.del_all_but_last_checkpoint()

            self.time_btw_checkpoints = (datetime.now() - self.time_checkpoint).total_seconds()
            self.time_checkpoint = datetime.now()

            #stop close to time limit
            approx_time_of_next_chkpnt = self.time_checkpoint + timedelta(seconds=1.22*self.time_btw_checkpoints)
            if approx_time_of_next_chkpnt > self.time_end:
                sys.exit()


    def write_sim_data_susp(self, step, checkpoint_step=10):

        self.energies.append(self.get_energy())
        self.pos_and_dips_hdf5.append(self.get_pos_and_dip_hdf5())
        self.momentas.append(self.get_momenta())

        if step % checkpoint_step == 0:
            self.write_energy()
            self.write_pos_dip_hdf5()

            self.vtfs.append(self.get_vcf())
            self.write_vtf()
            self.write_momenta()
            
            self.make_checkpoint(self.thermostat)
            self.del_all_but_last_checkpoint()

            self.time_btw_checkpoints = (datetime.now() - self.time_checkpoint).total_seconds()
            self.time_checkpoint = datetime.now()

            #stop close to time limit
            approx_time_of_next_chkpnt = self.time_checkpoint + timedelta(seconds=1.22*self.time_btw_checkpoints)
            if approx_time_of_next_chkpnt > self.time_end:
                sys.exit()


    def write_sim_data_vtk_only(self, step, checkpoint_step=10):
        if step % checkpoint_step == 0:
            self.write_vtk(noise = self.no_part)

    def write_sim_data_vtk_cm_only(self, step, checkpoint_step=10):
        self.pos_dip_cms.append(self.get_pos_dip_cm())
        self.pos_and_dips_hdf5.append(self.get_pos_and_dip_hdf5())
        if step % checkpoint_step == 0:
            self.write_pos_dip_hdf5()
            self.write_pos_dip_cm()
            self.write_vtk(noise = self.no_part)


    def finalize(self):
        if self.param.particle in ['raspberry', 'mng'] and self.corr_tau_max != 0:
            self.finalize_correlator_and_write_result()

    def warm_up(self):
        self.s.force_cap = 0.1
        gt = self.s.galilei


        # self.lbf.tau = 0.002
        # self.s.time_step = 0.001


        print("Warming up..")
        for step in range(50):
            print(step)

            self.s.integrator.run(100)

            gt.kill_particle_forces()
            gt.kill_particle_motion()

            # sb.remove_all_constraits()
            # sb.add_sphere_around_particle(radius=6-step*0.025)
            self.s.force_cap += 0.1

        self.s.force_cap = 0
        gt.kill_particle_forces()

        # self.lbf.tau = 0.001

    def short_warm_up(self):
        self.s.force_cap = 1
        gt = self.s.galilei

        # self.s.time_step = 0.0001
        # self.lbf.tau = 0.0002
        # self.s.time_step = 0.0001
        gt.kill_particle_forces()
        gt.kill_particle_motion()

        print("Short warming up..")
        for step in range(50):
            print(step)

            self.s.integrator.run(1)

            # gt.kill_particle_forces()
            # gt.kill_particle_motion()

            # sb.remove_all_constraits()
            # sb.add_sphere_around_particle(radius=6-step*0.025)
            self.s.force_cap += 0.1

        self.s.force_cap = 0
        # gt.kill_particle_forces()

        # self.lbf.tau = 0.002
        # self.s.time_step = 0.001

    def kill_motion_and_forces(self):
        gt = self.s.galilei
        gt.kill_particle_forces()
        gt.kill_particle_motion()

    def squize(self, radius, center, shift_particle_to_center=True):
        
        if shift_particle_to_center:
            self.shift_particle_to_center()

        # self.s.force_cap = 10#

        if self.param._hmf == 0:
            # gt = self.s.galilei
            self.add_sphere_around_particle(radius=radius, center=np.array(center))


            print("Running..")
            for step in range(510):
                print(step)

                try:
                    self.run(100)
                except:
                    print("Brock")
                    self.write_pos_dip_bonds()
                    self.write_sim_data_test(step)
                    sys.exit()

                # gt.kill_particle_forces()
                # gt.kill_particle_motion()


                self.remove_all_constraits()
                self.add_sphere_around_particle(radius=radius-step*0.025, center=np.array(center))
                # sb.s.force_cap += 0.1
                
                # if step >= 10:
                #     sb.write_pos_dip_bonds()

                # sb.write_sim_data(step)

                self.write_sim_data_test(step)

            self.write_pos_dip_bonds()

    def be_faster(self, max_ts = 0.001):
        
        if self.pos_dip_id % 1 == 0:
            if self.s.time_step < max_ts: 
                self.tmp_param.ts_mult += 1
                self.update_ts()

    def process_sim_except(self):
        
        print("Brocken bonds")
        self.tmp_param.ts_mult -= 1 #decrease simulation speed
        self.save_tmp_parameters()
        for index, arg in enumerate(sys.argv):
            if 'new_continue_sim' in arg:
                sys.argv[index+1] = 'continue'

        autorestart = True
        try:
            if "mpi" in self.param['prefix']:
                autorestart = False
        except:
            if "mpi" in self.prefix:
                autorestart = False

        if autorestart:
            os.execv(sys.executable, [self.pypresso] + sys.argv)
        else:
            try:
                fl = self.param['script_folder'].split('all_in_flow')[0] + 'utilities/jobs_manager/restartme.csv'

                with open(fl, 'a') as f:
                    f.write(self.param['project_name'] + '\n')
            except:
                fl = self.script_folder.split('all_in_flow')[0] + 'utilities/jobs_manager/restartme.csv'

                with open(fl, 'a') as f:
                    f.write(self.project_name + '\n')

            sys.exit()


    def int_rot_field(self):
        int_st = self.param.nub_of_int_steps_for_rot_field
        to_integrate = int(1 / (self.hmf_freq * self.param.t_) / (int_st * self.s.time_step))

        Hi = self.pos_dip_id % len(self.Hfields)
        self._setHfield(self.Hfields[Hi], log = False)
        self.run_rot_H(to_integrate)

        self.pos_dip_id += 1
        self.write_sim_data( self.pos_dip_id,  checkpoint_step=10)

    def init_part_vel_form_interp_lb(self):
        for p in self.s.part.all():
            if p.type != 2:
                p.v = self.lbf.get_interpolated_velocity(p.pos)


    def add_mngs(self, mng, version = 'equalibrated'):
        #TODO: refactor  TDRY CONTINUE
        spheres = '/gpfs/data/fs70806/Vania/all_in_flow/sphere/for_mng_susps'

        self.generate_list_of_configs_to_load(version)
        # self.split_vol()

        if self.N_part != 1:
            loaded = False
            print("try to load positions")
            while(not loaded):
                try:
                    positions = self.load_pos_of_equlibrated_spheres(spheres, self.N_part, self.box_len)
                    loaded = True
                except:
                    #sleep 10 sec
                    print("sleep 10 sec")
                    time.sleep(10)
        else:
            positions = [[int(self.box_len[0]/2),int(self.box_len[1]/2),int(self.box_len[2]/2)]]

        it = 0
        for conf, where_to_place in zip(self.list_of_configs_to_load, positions):
        # for conf in self.list_of_configs_to_load:
            print(it)

            
            if version == 'equalibrated':	
                #order of calls is very important
                mng.load(conf, self, version = version)
                
                # num_of_mnps = int((0.1 * (1 + self.extra_mnps / 100))*mng.tot_num_of_beads)
                # mng.randomly_assigne_mnps(num_of_mnps=num_of_mnps, tot_num_of_beads=mng.tot_num_of_beads) 
                mng.initialize(self, self.new_continue_sim) 

                # self.shift_particle_to_pos([50,50,50], mng.new_ids)
                self.shift_particle_to_pos(where_to_place, mng.new_ids)
            elif version == 'magnetized':
                #order of calls is very important
                mng.load(conf, self, version = version)

                mng.initialize(self, self.new_continue_sim)
                self.shift_particle_to_pos(where_to_place, mng.new_ids)

            else:
                mng.load(conf, self, version = version)
                
                num_of_mnps = int((0.1 * (1 + self.extra_mnps / 100))*mng.tot_num_of_beads)
                # mng.randomly_assigne_mnps(num_of_mnps=num_of_mnps, tot_num_of_beads=mng.tot_num_of_beads) 
                mng.initialize(self, self.new_continue_sim) 

                self.shift_particle_to_pos([50,50,50], mng.new_ids)
            
            self.map_pid_conf_type(mng.new_ids, conf, it)
            it+=1

    def add_mngs_into_sphere(self, mng, equalibrated = False):
        #TODO: refactor  TDRY CONTINUE
        spheres = '/gpfs/data/fs70806/Vania/all_in_flow/sphere/for_viscoelasticity'

        self.generate_list_of_configs_to_load()
        positions = self.pack_spheres_into_sphere(self.box_len[0]/2, self.part_radius, self.N_part)

        it = 0
        for conf, where_to_place in zip(self.list_of_configs_to_load, positions):
        # for conf in self.list_of_configs_to_load:
            print(it)

            #order of calls is very important
            mng.load(conf, self, equalibrated = equalibrated)

            mng.initialize(self, self.new_continue_sim)
            self.shift_particle_to_pos(where_to_place, mng.new_ids)

            self.map_pid_conf_type(mng.new_ids, conf, it)
            it+=1

    def add_raspberies(self, rasp, r = 3):
        spheres = '/gpfs/data/fs70806/Vania/all_in_flow/sphere/for_viscoelasticity'

        if self.N_part != 1:
            loaded = False
            while(not loaded):
                try:
                    positions = self.load_pos_of_equlibrated_spheres(spheres, self.N_part)
                    loaded = True
                except:
                    #sleep 10 sec
                    time.sleep(10)
        else:
            positions = [[int(self.box_len[0]/2),int(self.box_len[1]/2),int(self.box_len[2]/2)]]

        it = 0
        for where_to_place in positions[:]:

            print(it)
            rasp.load(self.conf)
            rasp.initialize(self, self.new_continue_sim)

            self.shift_particle_to_pos(where_to_place, rasp.new_ids)

            self.map_pid_conf_type(rasp.new_ids, self.conf, it)
            it+=1

    def fix_mnps(self, mng, fix_all = True):
        init_periodicity = self.s.periodicity
        self.s.periodicity = [0, 0, 0]
        self.s.integrator.run(1)
        self.kill_motion_and_forces()


        if fix_all:
            mng.freeze_all(self)
        else:
            mng.freeze_mnps(self)

        self.kill_motion_and_forces()

        # self.s.time_step = 0.001
        def shortest_bond_virt():
            for indx, p in enumerate(self.s.part.all()):
                if indx > int(self.N_part*mng.tot_num_of_beads) and indx%2 == 1:
                    if p.type == 2:
                        dist = np.linalg.norm(self.s.part.by_id(p.id).pos-self.s.part.by_id(p.id-1).pos)
                        if dist > 0.5 or dist < 0.05:
                            print(dist, p.id)
                            return True

            return False

        # self.warm_up()
        
        while shortest_bond_virt():
            self.kill_motion_and_forces()
            self.s.integrator.run(1000)
            print('1k')

        self.s.periodicity = init_periodicity

        mng.freeze_mnps_change_bonds_warmup(self)

        

    def generate_list_of_configs_to_load(self, version):
        self.list_of_configs_to_load = []
        if self.N_part == 1:
            self.list_of_configs_to_load.append(self.conf)
        elif self.N_part > 1:
            if self.conf != 'X':
                if version == 'magnetized':
                    #get list of files in /home/fs70806/Vania/worker_8/magnetic_nanogels/scripts/all_in_flow/src/particle/mng/configurations/magnetized
                    path, dirs, files = next(os.walk(self.script_folder + "src/particle/mng/configurations/magnetized/"))
                    num_of_conf = len(files)
                    for i in range(self.N_part):
                        self.list_of_configs_to_load.append(files[i%num_of_conf])

                else:
                    num_of_conf = int(self.conf.split('_')[-1])
                    for i in range(self.N_part):
                        cnfg = random.sample(range(0, num_of_conf), 1)
                        print("rnd conf = " + str(cnfg))
                        self.list_of_configs_to_load.append(self.conf.split('_')[0] + '_' + str(cnfg[0]))
            else:
                for i in range(self.N_part):
                    self.list_of_configs_to_load.append(self.conf)


    def additional_afterload_init(self, particle):
        particle.initialize(self, self.new_continue_sim)
        particle.add_calc_cm(self, self.new_continue_sim)

        try:
            self.load_tmp_parameters()
            self.update_ts()
        except:
            self.init_tmp_parameters()
        
        if self.hmf_freq != 0:
            self.param._nHs = int(round(1. /(self.hmf_freq * self.param.nub_of_int_steps_for_rot_field * self.param.t_)))
            self.add_magnetic_field()


    
    def add_tracers(self, n_tracers = 5):
        tracer_type = 3
        cm = self.calc_cm()
        for _ in range(n_tracers):
            self.s.part.add(
                pos = cm,
                type = tracer_type,
                mass = self.param._mass_m,
            )

        p = self.param
        t_sz = self.tracer_size
        self.s.non_bonded_inter[p._mnp_type, tracer_type].lennard_jones.set_params(
            epsilon=p._lj_eps, sigma=t_sz*p._lj_sigma, cutoff=t_sz*p._lj_cut, shift=p._lj_shift
        )
        self.s.non_bonded_inter[p._p_type, tracer_type].lennard_jones.set_params(
            epsilon=p._lj_eps, sigma=t_sz*p._lj_sigma, cutoff=t_sz*p._lj_cut, shift=p._lj_shift
        )


                
    from .parameters import read_param_from_args

    from .read_write import foldCoordinates, fold_coordinates_back, \
        write_pos_and_dip, write_pos_dip_cm, write_rg, write_momenta, \
        _vtf_pid_map, _writevsf, get_vcf, preset_xyz_writer, write_xyz, \
        compose_base, write_energy, write_vtk, write_vtf, make_checkpoint, \
        del_all_but_last_checkpoint, get_momenta, get_rg, get_pos_dip_cm, \
        get_pos_and_dip, get_energy, write_pos_dip_bonds, finalize_correlator_and_write_result, \
        write_pos_dip_hdf5, get_pos_and_dip_hdf5, write_forces, get_forces,\
        write_map_pid_conf_type, save_tmp_parameters, load_tmp_parameters, \
        get_force_onto_wall, write_force_onto_wall, load_pos_of_equlibrated_spheres, \
        write_ts_mult_history, read_ts_mult_history

    from .system import init_system, configure_system, add_fene_and_harm, \
        add_magnetostatics_DSCpu, add_magnetostatics_P3M_DLC, add_magnetic_field, \
        enable_brownian_thermostat_for_mnps, enable_lb_thermostat, \
        add_walls_and_shear_flow, gen_observ_dip_moment, run, print_parameters, \
        init_vtf, shift_particle_to_center, enable_lang_thermostat, \
        add_sphere_around_particle, remove_all_constraits, gen_rot_magnetic_field, \
        init_dip_obs_and_corr, _setHfield, write_parameters_to_json, gen_observ_forces, \
        split_vol, shift_particle_to_pos, add_magnetostatics_DSGpu, add_vel_z, add_gravity, \
        map_pid_conf_type, add_ext_force, remove_ext_force, equilibrate_hard_spheres, \
        reset_seed, free_up_resources, update_ts, add_walls, add_magnetostatics_P3M, \
        run_rot_H, enable_brownian_thermostat, pack_spheres_into_sphere, fix_mnps_along_field, \
        rotate_mng, freeup_mnps_rot