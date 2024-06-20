from particle import Particle

import settings
settings.init()  

from ast import literal_eval
import numpy as np
import random
from random import gauss
import os
import math
import sys

from espressomd import checkpointing
import espressomd.virtual_sites

class MNG(Particle):

    def __init__(self):
        print("A new MNG has been created!")

    def load(self, conf, sb, version = 'equalibrated'):
        """
        [loads gel (with bonds)]

        Args:
            conf ([int]): [0-9 configuration id]
        """

        if version == 'equalibrated':
            system = sb.s

            conf_name = 'config_' + conf + '_extra_mnps_' + \
                str(sb.extra_mnps) + '_field_0' + \
                '_fixed_mnps_' + str(int(sb.fixed_mnps))
            conf = '/equilibrated/' + conf_name

            with open(settings.script_path + '/particle/mng/configurations/' + conf, 'r') as file:
                lines = file.readlines()
        elif version == 'magnetized' and sb.N_part == 1:
            system = sb.s

            conf_name = 'config_' + conf + '_custom_id_' + str(sb.custom_id) 
            conf = '/magnetized/' + conf_name

            with open(settings.script_path + '/particle/mng/configurations/' + conf, 'r') as file:
                lines = file.readlines()
        elif version == 'magnetized' and sb.N_part != 1:
            system = sb.s
            conf_name = conf
            conf = '/magnetized/' + conf_name

            with open(settings.script_path + '/particle/mng/configurations/' + conf, 'r') as file:
                lines = file.readlines()
        else:
            with open(settings.script_path + '/particle/mng/configurations/conf_' + str(conf), 'r') as file:
                    lines = file.readlines()

        self.ids = []
        self.positions = []
        self.dips = []
        self.bonds = []
        self.types = []

        for line_index, line in enumerate(lines):
            if line == '\n':
                self.tot_num_of_beads = line_index
                break

        for line in lines[:self.tot_num_of_beads]: 
            data = line.split()
            id = int(data[0])
            # x, y, z = map(lambda x: float(x)%sb.param._box_l, data[1:4])
            x, y, z = map(lambda x: float(x), data[1:4])

            p_type = float(data[4])

            if p_type == 2:
                rel_to = int(data[5])
                self.types.append([p_type,rel_to])
                self.dips.append([0, 0, 0])
            else:
                dip_x, dip_y, dip_z = map(lambda x: float(x), data[5:8])
                if math.isnan(dip_x) or math.isnan(dip_y) or math.isnan(dip_z):
                    dip_x, dip_y, dip_z = 0, 0, 0
                if dip_x != 0:
                    scale_dip = sb.lmbda_scale * sb.param._mu / np.sqrt(dip_x**2 + dip_y**2 + dip_z**2)
                else:
                    scale_dip = 0
                self.dips.append([
                    dip_x*scale_dip, 
                    dip_y*scale_dip, 
                    dip_z*scale_dip, 
                    ])
                self.types.append(p_type)

            self.ids.append(id)
            self.positions.append([x, y, z])

            
        for line in lines[self.tot_num_of_beads+1:]: 
            data = line.replace(';', '').split(' ')

            cur_bonds = []

            for idx, d in enumerate(data[1:-1:2]):
                cur_bonds.append((int(data[1+idx*2]), int(data[2+idx*2])))

            # text = " ".join(data)
            self.bonds.append(cur_bonds)


    def randomly_assigne_mnps(self, num_of_mnps=60, tot_num_of_beads=600):
        #change types of mnps from 0 to 1

        mnp_s = random.sample(range(0, int(tot_num_of_beads)), num_of_mnps)
        self.types = [0] * tot_num_of_beads
        for i in mnp_s:
            self.types[i] = 1

    def add_calc_cm(self, sb, new_continue_sim):

        system = sb.s 
        p = sb.param

        tot_mass = p._mass_m + p._mass_gel

        def calc_cm(self):
            cm_mnp = system.analysis.center_of_mass(p._mnp_type)
            cm_bead = system.analysis.center_of_mass(p._p_type)

            cm = [0]*3

            #TODO: calculate CM more adequate and do it for many particles

            for idx, (mnp, b) in enumerate(zip(cm_mnp, cm_bead)):
                cm[idx] = (mnp*p._mass_m + b*p._mass_gel)/tot_mass

            return cm

        sb.calc_cm = calc_cm.__get__(sb)


    def initialize(self, sb, new_continue_sim):

        def make_rand_vector(dims):
            vec = [gauss(0, 1) for i in range(dims)]
            mag = sum(x**2 for x in vec) ** .5
            return [x/mag for x in vec]

        system = sb.s 
        p = sb.param

        self.new_ids = []

        if new_continue_sim == "new":

            #creates gel (with bonds) in system

            max_id = system.part.highest_particle_id

            if max_id == -1:

                system.non_bonded_inter[p._p_type, p._p_type].lennard_jones.set_params(
                    epsilon=p._lj_eps, sigma=p._lj_sigma, cutoff=p._lj_cut, shift=p._lj_shift)

                system.non_bonded_inter[p._mnp_type, p._p_type].lennard_jones.set_params(
                    epsilon=p._lj_eps, sigma=p._lj_sigma, cutoff=p._lj_cut, shift=p._lj_shift
                )

                system.non_bonded_inter[p._mnp_type, p._mnp_type].lennard_jones.set_params(
                    epsilon=p._lj_eps, sigma=p._lj_sigma, cutoff=p._lj_cut, shift=p._lj_shift
                )

                # system.non_bonded_inter[p._p_type, p._p_type].wca.set_params(
                #     epsilon=p._lj_eps, sigma=p._lj_sigma
                # )

                # system.non_bonded_inter[p._mnp_type, p._p_type].wca.set_params(
                #     epsilon=p._lj_eps, sigma=p._lj_sigma
                # )

                # system.non_bonded_inter[p._mnp_type, p._mnp_type].wca.set_params(
                #     epsilon=p._lj_eps, sigma=p._lj_sigma
                # )
                
                system.non_bonded_inter[p._mnp_type, p._wall_type].wca.set_params(
                    epsilon=p._lj_eps, sigma=p._lj_sigma
                )

                system.non_bonded_inter[p._p_type, p._wall_type].wca.set_params(
                    epsilon=p._lj_eps, sigma=p._lj_sigma
                )


                #how to calculate CM of this particle

                tot_mass = p._mass_m + p._mass_gel

                def calc_cm(self):
                    cm_mnp = system.analysis.center_of_mass(p._mnp_type)
                    cm_bead = system.analysis.center_of_mass(p._p_type)

                    cm = [0]*3

                    #TODO: calculate CM more adequate and do it for many particles

                    for idx, (mnp, b) in enumerate(zip(cm_mnp, cm_bead)):
                        cm[idx] = (mnp*p._mass_m + b*p._mass_gel)/tot_mass

                    return cm

                sb.calc_cm = calc_cm.__get__(sb)


            for id, pos in enumerate(self.positions):
                sys_id = id + max_id + 1
                system.part.add(pos=(pos[0], pos[1], pos[2]), id=sys_id)
                # system.part.by_id(sys_id).pos = (pos[0], pos[1], pos[2])
                self.new_ids.append(sys_id)
                dip = self.dips[id]
                p_type = self.types[id]
                
                #check if P_type is a list
                if type(p_type) == list:
                    rel_to = int(p_type[1])
                    p_type = int(p_type[0])

                if p_type == 1:
                    system.part.by_id(sys_id).type = p._mnp_type
                    system.part.by_id(sys_id).rotation = (True, True, True)
                    system.part.by_id(sys_id).rinertia = np.ones(3) *p._momI_m
                    system.part.by_id(sys_id).mass = p._mass_m

                    # r = random.choice([-1, 1])
                    r = random.choice([-1, 1])

                    if dip[0] != 0:
                        system.part.by_id(sys_id).dip = dip
                    else:
                        system.part.by_id(sys_id).dip =  np.array([r , 0, 0]) #* sb.lmbda_scale * p._mu
                    # system.part.by_id(sys_id).dip = np.array(make_rand_vector(3)) * p._mu# np.array([r * p._mu, 0, 0])
                elif p_type == 0:
                    system.part.by_id(sys_id).type = p._p_type
                    system.part.by_id(sys_id).rinertia = np.ones(3) * p._momI_gel
                    system.part.by_id(sys_id).mass = p._mass_gel
                    system.part.by_id(sys_id).dip = (0.0, 0.0, 0.0)
                    system.part.by_id(sys_id).rotation = (True, True, True)
                    # system.part.by_id(sys_id).quat = [ 0.5, -0.5,  0.5, -0.5]
                elif p_type == 2:


                    system.part.by_id(sys_id).type = p_type
                    system.part.by_id(sys_id).pos = np.array([i%100 for i in system.part.by_id(sys_id).pos])
                    # system.part.by_id(sys_id).rinertia = (0,0,0)#np.ones(3) * p._momI_gel
                    # system.part.by_id(sys_id).mass =1
                    # system.part.by_id(sys_id).dip = (0.0, 0.0, 0.0)
                    system.part.by_id(sys_id).rotation = (False, False, False)
                    # system.part.by_id(sys_id).fix = [1, 1, 1]
                    system.part.by_id(sys_id).virtual = True

                    system.part.by_id(sys_id).vs_auto_relate_to(rel_to + max_id + 1)
                    # system.part.by_id(sys_id).quat = [ 0.5, -0.5,  0.5, -0.5]

                

                for b in self.bonds[id]:
                    #TODO REMOVE RESTRICTION ON BONDING!!!
                    # if b[0] != 1:
                    system.part.by_id(sys_id).add_bond((b[0], b[1]+max_id+1))

            #add LJ(WCA)



    def freeze_mnps(self, sb, chain_size=100):

        # fix for the case of adjasting MNPs

        virt_type = sb.param._mnp_type + 1
        eps = 0.1*sb.param._sigma_m

        def add_virts_and_bond(v1_pos, v1_rel_to_id, v2_pos, v2_rel_to_id, bond=3):


            # sb.s.part[v1_rel_to_id].quat = [ 0.5, -0.5,  0.5, -0.5]
            # sb.s.part[v2_rel_to_id].quat = [ 0.5, -0.5,  0.5, -0.5]

            id_1 = len(sb.s.part.all())
            sb.s.part.add(id=id_1, pos=v1_pos, dip=(0,0,0), type=virt_type, rotation=(0,0,0),fix = [False, False, False])
            sb.s.part.by_id(id_1).virtual = True
            sb.s.part.by_id(id_1).vs_auto_relate_to(v1_rel_to_id)

            # print(sb.s.part[id_1].quat)
            # print(sb.s.part[v1_rel_to_id].quat)


            id_2 = len(sb.s.part.all())
            sb.s.part.add(id=id_2, pos=v2_pos, dip=(0,0,0), type=virt_type, rotation=(0,0,0),fix = [False, False, False])
            sb.s.part.by_id(id_2).virtual = True
            sb.s.part.by_id(id_2).vs_auto_relate_to(v2_rel_to_id)

            sb.s.part.by_id(id_2).add_bond((bond, id_1))

        def free_anchor_point(pos, type, eps=eps):
            for p in sb.s.part.all():
                if p.type == type:
                    if np.linalg.norm(pos - p.pos) < eps:
                        return False
            return True

        # assign quats to beads without dipoles for virts
        for p in sb.s.part.all(): 
            if p.id % chain_size != 0:
                if sb.s.part.by_id(p.id).dip[0] == 0 and sb.s.part.by_id(p.id-1).dip[0] != 0:
                    sb.s.part.by_id(p.id).quat = [ 0.5, -0.5,  0.5, -0.5]
                elif sb.s.part.by_id(p.id).dip[0] != 0 and sb.s.part.by_id(p.id-1).dip[0] == 0:
                    sb.s.part.by_id(p.id-1).quat = [ 0.5, -0.5,  0.5, -0.5]


        
        for p in sb.s.part.all():
            if p.type == sb.param._mnp_type: #if part is MNP
                nghbrs = []
                if p.id % chain_size == 0:
                    nghbrs.append(p.id+1)
                    sb.s.part.by_id(p.id+1).delete_bond((0, p.id))
                elif p.id % chain_size == chain_size-1:
                    if sb.s.part.by_id(p.id-1).type != sb.param._mnp_type:
                        nghbrs.append(p.id-1)
                        sb.s.part.by_id(p.id).delete_bond((0, p.id-1))
                else:
                    if sb.s.part.by_id(p.id-1).type != sb.param._mnp_type:
                        nghbrs.append(p.id-1)
                        sb.s.part.by_id(p.id).delete_bond((0, p.id-1))
                    nghbrs.append(p.id+1)
                    sb.s.part.by_id(p.id+1).delete_bond((0, p.id))

                v1 = p.pos + p.dip/np.linalg.norm(p.dip)*0.5*sb.param._sigma_m
                v1 = v1 if free_anchor_point(v1, virt_type, eps=0.1*sb.param._sigma_m) else np.array([-9999, -9999, -9999])
                v2 = p.pos - p.dip/np.linalg.norm(p.dip)*0.5*sb.param._sigma_m
                v2 = v2 if free_anchor_point(v2, virt_type, eps=0.1*sb.param._sigma_m) else np.array([-9999, -9999, -9999])

                if np.linalg.norm(v1 - v2) < eps:
                    print("something wrong in freeze_mnps!!!")

                for n in nghbrs:
                    v = v1 - sb.s.part.by_id(n).pos
                    v = v if np.linalg.norm(v) < np.linalg.norm(v1 - sb.s.part.by_id(n).pos) else (v2 - sb.s.part.by_id(n).pos)
                    v = v / np.linalg.norm(v) * 0.5*sb.param._sigma_m #be careful that sigma_gel = sigma_m
                    v_n = sb.s.part.by_id(n).pos + v

                    if sb.s.part.by_id(n).type == sb.param._mnp_type: #feint with the ears, in case the next neighbor is MNP
                        pp = sb.s.part.by_id(n)
                        v_n1 = pp.pos + pp.dip/np.linalg.norm(pp.dip)*0.5*sb.param._sigma_m
                        v_n1 = v_n1 if free_anchor_point(v_n1, virt_type, eps=0.1*sb.param._sigma_m) else np.array([-9999, -9999, -9999])
                        v_n2 = pp.pos - pp.dip/np.linalg.norm(pp.dip)*0.5*sb.param._sigma_m
                        v_n2 = v_n2 if free_anchor_point(v_n2, virt_type, eps=0.1*sb.param._sigma_m) else np.array([-9999, -9999, -9999])

                        dst_0 = np.linalg.norm(v_n - v_n1)
                        if  dst_0 > np.linalg.norm(v_n - v_n2):
                            v_n = v_n2
                        else:
                            v_n = v_n1

                    dst = np.linalg.norm(v_n - v1)

                    if dst > np.linalg.norm(v_n - v2):
                        add_virts_and_bond(v_n, n, v2, p.id)
                        v2 = np.array([-1000*sb.param._sigma_m,-1000*sb.param._sigma_m,-1000*sb.param._sigma_m]) #to insure that no unconnected site left
                    else:
                        add_virts_and_bond(v_n, n, v1, p.id)
                        v1 = np.array([-1000*sb.param._sigma_m,-1000*sb.param._sigma_m,-1000*sb.param._sigma_m])

    def freeze_all(self, sb, chain_size=100):

        # fix for the case of adjasting MNPs

        virt_type = sb.param._mnp_type + 1
        eps = 0.1*sb.param._sigma_m

        def add_virts_and_bond(v1_pos, v1_rel_to_id, v2_pos, v2_rel_to_id, bond=3):


            # sb.s.part[v1_rel_to_id].quat = [ 0.5, -0.5,  0.5, -0.5]
            # sb.s.part[v2_rel_to_id].quat = [ 0.5, -0.5,  0.5, -0.5]

            id_1 = len(sb.s.part.all())
            sb.s.part.add(id=id_1, pos=v1_pos, dip=(0,0,0), type=virt_type, rotation=(0,0,0),fix = [False, False, False])
            sb.s.part.by_id(id_1).virtual = True
            sb.s.part.by_id(id_1).vs_auto_relate_to(v1_rel_to_id)

            # print(sb.s.part[id_1].quat)
            # print(sb.s.part[v1_rel_to_id].quat)


            id_2 = len(sb.s.part.all())
            sb.s.part.add(id=id_2, pos=v2_pos, dip=(0,0,0), type=virt_type, rotation=(0,0,0),fix = [False, False, False])
            sb.s.part.by_id(id_2).virtual = True
            sb.s.part.by_id(id_2).vs_auto_relate_to(v2_rel_to_id)

            sb.s.part.by_id(id_2).add_bond((bond, id_1))

        def free_anchor_point(pos, type, eps=eps):
            for p in sb.s.part.all():
                if p.type == type:
                    if np.linalg.norm(pos - p.pos) < eps:
                        return False
            return True

        # assign quats to beads without dipoles for virts
        for p in sb.s.part.all(): 
            if p.type != sb.param._mnp_type:
                sb.s.part.by_id(p.id).quat = [ 0.5, -0.5,  0.5, -0.5]
                sb.s.part.by_id(p.id).dip = [ 1,0,0]


        
        for p in sb.s.part.all():
            nghbrs = []
            if not (p.id % chain_size == chain_size-1):
                nghbrs.append(p.id+1)
                sb.s.part.by_id(p.id+1).delete_bond((0, p.id))
            
            v1 = p.pos + p.dip/np.linalg.norm(p.dip)*0.5*sb.param._sigma_m
            v1 = v1 if free_anchor_point(v1, virt_type, eps=0.01*sb.param._sigma_m) else np.array([-9999, -9999, -9999])
            v2 = p.pos - p.dip/np.linalg.norm(p.dip)*0.5*sb.param._sigma_m
            v2 = v2 if free_anchor_point(v2, virt_type, eps=0.01*sb.param._sigma_m) else np.array([-9999, -9999, -9999])

            if np.linalg.norm(v1 - v2) < eps:
                print("something wrong in freeze_mnps!!!")

            for n in nghbrs:
                v = v1 - sb.s.part.by_id(n).pos
                v = v if np.linalg.norm(v) < np.linalg.norm(v1 - sb.s.part.by_id(n).pos) else (v2 - sb.s.part.by_id(n).pos)
                v = v / np.linalg.norm(v) * 0.5*sb.param._sigma_m #be careful that sigma_gel = sigma_m
                v_n = sb.s.part.by_id(n).pos + v
            
                pp = sb.s.part.by_id(n)
                v_n1 = pp.pos + pp.dip/np.linalg.norm(pp.dip)*0.5*sb.param._sigma_m
                v_n1 = v_n1 if free_anchor_point(v_n1, virt_type, eps=0.01*sb.param._sigma_m) else np.array([-9999, -9999, -9999])
                v_n2 = pp.pos - pp.dip/np.linalg.norm(pp.dip)*0.5*sb.param._sigma_m
                v_n2 = v_n2 if free_anchor_point(v_n2, virt_type, eps=0.01*sb.param._sigma_m) else np.array([-9999, -9999, -9999])

                if v_n2[0] == -9999 or v_n1[0] == -9999:
                    print("something wrong in freeze_mnps!!!")
                    return

                dst_0 = np.linalg.norm(v_n - v_n1)
                if  dst_0 > np.linalg.norm(v_n - v_n2):
                    v_n = v_n2
                else:
                    v_n = v_n1

                dst = np.linalg.norm(v_n - v1)

                if dst > np.linalg.norm(v_n - v2):
                    add_virts_and_bond(v_n, n, v2, p.id)
                    v2 = np.array([-1000*sb.param._sigma_m,-1000*sb.param._sigma_m,-1000*sb.param._sigma_m]) #to insure that no unconnected site left
                else:
                    add_virts_and_bond(v_n, n, v1, p.id)
                    v1 = np.array([-1000*sb.param._sigma_m,-1000*sb.param._sigma_m,-1000*sb.param._sigma_m])

        for p in sb.s.part.all(): 
            if p.type != sb.param._mnp_type:
                sb.s.part.by_id(p.id).dip = [ 0,0,0]


    def freeze_mnps_change_bonds_warmup(self, sb):      
        virt_type = sb.param._mnp_type + 1
        for p in sb.s.part.all():
            if p.type == virt_type:
                if len(p.bonds) == 1:
                    bonded_with = p.bonds[0][1]
                    p.delete_bond((3, bonded_with))
                    p.add_bond((2, bonded_with))

                
    def crosslink(self, sb, max_harm_dist, asymmetricity=0.5, num_of_bonds=60, upsidedown_bell=False, uniform=False):

        def invNormal(low, high, mu=0, sd=1, size=1, block_size=1024):
            remain = size
            result = []
            
            mul = -0.5 * sd**-2

            while remain:
                # draw next block of uniform variates within interval
                x = np.random.uniform(low, high, size=min((remain+5)*2, block_size))
                
                # reject proportional to normal density
                x = x[np.exp(mul*(x-mu)**2) < np.random.rand(*x.shape)]
                
                # make sure we don't add too much
                if remain < len(x):
                    x = x[:remain]

                result.append(x)
                remain -= len(x)

            return np.concatenate(result)

        ids = [p.id for p in sb.s.part.select(lambda p: p.type < 2)]
        N = len(ids)

        dist = np.zeros((N,N))
        for r in range(N):
            for c in range(r, N):
                dist[r,c] = sb.s.distance(sb.s.part.by_id(ids[r]), sb.s.part.by_id(ids[c]))
        dist = dist + dist.T

        max_dist = int(np.ceil(np.amax(dist)))

        grid = np.empty((max_dist*2+1, max_dist*2+1, max_dist*2+1), object)

        #put particle in grid
        cm = [0]*3
        p_len = len(sb.s.part)
        for p in sb.s.part.all():
            for i, pp in enumerate(p.pos):
                cm[i] += pp/p_len
        delta_x = cm[0] - max_dist/2
        delta_y = cm[1] - max_dist/2
        delta_z = cm[2] - max_dist/2
        for p in sb.s.part.all():
            p.pos = (p.pos[0] - delta_x, p.pos[1] - delta_y, p.pos[2] - delta_z)
        # for p in sb.s.part[:]:
        #     p.pos = list(map(lambda x: x%max_dist, p.pos))

        #fill grid with sorted bond lengths; use bond midpoint as index of matrix element

        for r in range(N):
            for c in range(r, N):
                p1 = sb.s.part.by_id(ids[r])
                p2 = sb.s.part.by_id(ids[c])
                if r // 100 != c // 100:
                    d = dist[r,c]
                    if d < max_harm_dist:
                        indices = np.rint(list(map(lambda x, y: (x+y), p1.pos, p2.pos))).astype(int)
                        if grid[tuple(indices)] == None:
                            grid[tuple(indices)] = []
                            grid[tuple(indices)].append([(ids[r],ids[c]), d])
                        else:
                            for index, value in enumerate(grid[tuple(indices)]):
                                if value[1] > d:
                                    grid[tuple(indices)].insert(index, [(ids[r],ids[c]), d])
                                    break
        if uniform:
            #create uniform distribution
            norm_x = np.random.randint(0, max_dist*2, size=10000)
            norm_y = np.random.randint(0, max_dist*2, size=10000)
            norm_z = np.random.randint(0, max_dist*2, size=10000)

        elif not upsidedown_bell:                               
            #generate 3 norm. dist; x is shifted                         
            norm_x = np.rint(np.random.normal(
                loc=max_dist*(1+asymmetricity), 
                scale=max_dist/3,
                size=1000)).astype(int)

            norm_y = np.rint(np.random.normal(
                loc=max_dist, 
                scale=max_dist/3,
                size=1000)).astype(int)

            norm_z = np.rint(np.random.normal(
                loc=max_dist, 
                scale=max_dist/3,
                size=1000)).astype(int)
        else:
            d_x = invNormal(0, max_dist*2, mu=max_dist, sd=max_dist/3, size=10000, block_size=1024)
            norm_x = np.rint(d_x).astype(int)

            d_y = invNormal(0, max_dist*2, mu=max_dist, sd=max_dist/3, size=10000, block_size=1024)
            norm_y = np.rint(d_y).astype(int)

            d_z = invNormal(0, max_dist*2, mu=max_dist, sd=max_dist/3, size=10000, block_size=1024)
            norm_z = np.rint(d_z).astype(int)

        #randomly assign harmonic bonds
        already_bounded = []
        for n_x, n_y, n_z in zip(norm_x, norm_y, norm_z):
            if num_of_bonds != 0:
                if all(list(map(lambda x: 0<=x<=max_dist*2, [n_x, n_y, n_z]))):
                    if grid[tuple([n_x, n_y, n_z])] != None and grid[tuple([n_x, n_y, n_z])] != []:
                        b = grid[tuple([n_x, n_y, n_z])][0]
                        if (b[0][0] not in already_bounded) and (b[0][1] not in already_bounded):
                            bond = grid[tuple([n_x, n_y, n_z])].pop(0)
                            sb.s.part.by_id(bond[0][0]).add_bond((1, bond[0][1]))
                            already_bounded.append(bond[0][0])
                            already_bounded.append(bond[0][1])
                        
                            num_of_bonds -= 1
            else:
                break

        if num_of_bonds != 0:
            print('can\'t install all bonds')
            sys.exit()

        print('crosslinking done')

