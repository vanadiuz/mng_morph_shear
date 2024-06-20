import os  
import sys
import shutil
import pickle

from espressomd import checkpointing

# import espressomd.io.writer
import numpy as np
import h5py
import json
import zipfile
import pandas as pd

from .parameters import TMP_Parameters

# #for xyz trajectory | throws an error
# import MDAnalysis as mda
# from espressomd import MDA_ESP
# from MDAnalysis.coordinates.XYZ import XYZWriters


def foldCoordinates(self):
    """
    Fold coordinates in sim_box for vtf
    """
    list_of_coordinates = []
    for p in self.s.part:
        list_of_coordinates.append((p.pos[0], p.pos[1], p.pos[2]))
        p.pos = (p.pos_folded[0], p.pos_folded[1], p.pos_folded[2])
    return list_of_coordinates


def fold_coordinates_back(self, list_of_coordinates):
    """
    Unfold coordinates after vtf is written
    """
    for i, p in enumerate(self.s.part):
        p.pos = (
            list_of_coordinates[i][0],
            list_of_coordinates[i][1],
            list_of_coordinates[i][2],
        )

def get_pos_and_dip_hdf5(self):
    """[returns pos_and_dip for further usage in write_pos_dip_hdf5]

    Returns:
        [id; list of values]: [pos_dip_id; x,y,z type dip's]
    """

    cur_pos_dip = []

    for i in self.s.part:
        
        part_pos_dip = (
            i._id, 
            i.pos[0], i.pos[1], i.pos[2], 
            i.type, 
            i.dip[0], i.dip[1], i.dip[2]
        )

        cur_pos_dip.append(part_pos_dip)

    return [self.pos_dip_id, cur_pos_dip]


def get_pos_and_dip(self):
    """[returns pos_and_dip for further usage in write_pos_and_dip]

    Returns:
        [id; list of strings]: [pos_dip_id; x,y,z type dip's]
    """

    cur_pos_dip = []

    for i in self.s.part:

        part_pos_dip = ("{} "
           "{:0.2f} {:0.2f} {:0.2f} "
           "{} "
           "{:0.4f} {:0.4f} {:0.4f}").format(i._id, 
                    i.pos[0], i.pos[1], i.pos[2], 
                    i.type, 
                    i.dip[0], i.dip[1], i.dip[2])

        cur_pos_dip.append(part_pos_dip)

    return [self.pos_dip_id, cur_pos_dip]


def write_pos_dip_hdf5(self):
    """
    Write custom pos/dip to hdf5 file from tmp list (pos_and_dips)
    and cleans that list
    """

    ps_dp = np.dtype([
        ('id','int'), 
        ('x',float), 
        ('y',float), 
        ('z',float), 
        ('type',int), 
        ('dip-x',float), 
        ('dip-y',float), 
        ('dip-z',float)
        ])
        
    ds_dtype = np.dtype([
        ('step', 'int'),
        ('pos_dip',ps_dp, len(self.s.part))
        ])

    ds_arr = np.rec.array(self.pos_and_dips_hdf5, dtype=ds_dtype)

    if not os.path.exists(self.project_path + "/pos_dip.h5"):
        with h5py.File(self.project_path + "/pos_dip.h5", 'w') as h5f:
            dset = h5f.create_dataset('pos_dip', data=ds_arr, maxshape=(None,), chunks=True)

    else:
        with h5py.File(self.project_path + "/pos_dip.h5", 'a') as h5f:
            # dset = h5f.create_dataset('pos_dip', data=ds_arr, maxshape=(None))

            h5f["pos_dip"].resize((h5f["pos_dip"].shape[0] + ds_arr["step"].shape[0]), axis = 0)
            h5f["pos_dip"][-ds_arr["step"].shape[0]:] = ds_arr

            # h5f["pos_dip"].resize((h5f["pos_dip"].shape[0] + ds_arr["pos_dip"].shape[0]), axis = 0)
            # h5f["pos_dip"][-ds_arr["pos_dip"].shape[0]:] = ds_arr["pos_dip"]

    del self.pos_and_dips_hdf5[:]


def write_pos_and_dip(self):
    """
    Write custom pos/dip files from tmp list (pos_and_dips)
    and cleans that list
    """

    if not os.path.exists(self.project_path + "/pos_dip"):
        os.makedirs(self.project_path + "/pos_dip")

    for id, data in self.pos_and_dips:

        file = open(self.project_path + "/pos_dip/" + str(id), mode="a")

        for item in data:
            file.write("%s\n" % item)
         
        file.close()

    del self.pos_and_dips[:]


def get_pos_dip_cm(self, p_type=None):
    """[returns pos_dip_cm for further usage in write_pos_dip_cm]

    Returns:
        [string]: [cm_pos time dip's]
    """

    if self.N_part == 1:
        if p_type == None:
            pos = self.calc_cm()
            p_type = 'all'
        else:
            pos = self.s.analysis.center_of_mass(p_type)


        cur_pos_dip_cm = str(p_type) + " " \
            + "{:0.4f} {:0.4f} {:0.4f} {:0.4f} {:0.4f} {:0.4f} {:0.4f}".format(
                pos[0], pos[1], pos[2], self.s.time, *self.part_dip.calculate())
                # self.s.part.by_id(0).dip[0], self.s.part.by_id(0).dip[1], self.s.part.by_id(0).dip[2])
            # + str(self.part_dip.calculate()) 
    else:
        cur_pos_dip_cm = str(p_type) + " " \
        + "{:0.4f} {:0.4f} {:0.4f} {:0.4f}".format(
            self.s.time, *self.part_dip.calculate())

    return cur_pos_dip_cm


def get_force_onto_wall(self):
    """[returns force onto top wall in LB from the fluid]

    Returns:
        [list]: [force 3d vector]
    """
    return self.s.lbboundaries[1].get_force()


def write_pos_dip_bonds(self):
    """
    Write custom pos/dip file and bonds
    """

    if not os.path.exists(self.project_path):
        os.makedirs(self.project_path)

    file = open(self.project_path + "/pos_dip_bonds", mode="w")

    for i in self.s.part.all():
        if i.type != 2:
            file.write(
                str(i._id)
                + " "
                + str(i.pos[0])
                + " "
                + str(i.pos[1])
                + " "
                + str(i.pos[2])
                + " "
                + str(i.type)
                + " "
                + str(i.dip[0])
                + " "
                + str(i.dip[1])
                + " "
                + str(i.dip[2])
            )
        else:
            file.write(
                str(i._id)
                + " "
                + str(i.pos[0])
                + " "
                + str(i.pos[1])
                + " "
                + str(i.pos[2])
                + " "
                + str(i.type)
                + " "
                + str(i.vs_relative[0])
            )
        file.write("\n")

    file.write("\n")

    for i in self.s.part.all():
        str_to_write = str(i._id) + " "
        for index, b in enumerate(i.bonds):
            str_to_write += str(b[0]._bond_id)+" " + str(b[1]) + "; "

            # if i.type == 2:
            #     str_to_write += "2 " + str(b[1]) + "; "
            # elif i.id % 100 != 0:
            #     str_to_write +=  str(index) + " " + str(b[1]) + "; "
            # else:
            #     str_to_write +=  "1 " + str(b[1]) + "; "
        file.write(str_to_write + "\n")

def write_pos_dip_cm(self):
    """
    Write custom cm pos/dip files from tmp list (pos_dip_cms)
    and cleans that list
    """
    file = open(self.project_path + "pos_dip_cm_" + self.base, mode="a")

    for item in self.pos_dip_cms:
        file.write("%s\n" % item)

    file.close()

    del self.pos_dip_cms[:]

def write_force_onto_wall(self):
    """
    Write force onto top wall for LB from tmp list
    and cleans that list
    """
    file = open(self.project_path + "force_onto_top_wall_" + self.base, mode="a")

    for item in self.force_onto_wall:
        file.write("%s\n" % item)

    file.close()

    del self.force_onto_wall[:]


def get_rg(self, p_type=None):
    """[returns rg for further usage in write_rg]

    Returns:
        [string]: [gyration_tensor + other stuff]
    """

    if p_type == None:
        
        types = []
        for p in self.s.part.all():
            if p.type not in types:
                types.append(p.type)

        rg_dict = self.s.analysis.gyration_tensor(types)
        p_type = 'all'
    else:
        rg_dict = self.s.analysis.gyration_tensor(p_type)

    cur_rg = str(p_type) + " " + str(rg_dict) + " " + str(self.s.time)
    
    return cur_rg


def write_rg(self):
    """
    Write custom rg file
    """

    file = open(self.project_path + "rg_" + self.base, mode="a")

    for item in self.rgs:
        file.write("%s\n" % item)

    file.close()

    del self.rgs[:]

def get_momenta(self):
    """
    Returns total linear momentum of the particles and of the fluid
    """

    linmomp = self.s.analysis.linear_momentum(
        include_particles=True, include_lbfluid=False
    )
    linmomf = self.s.analysis.linear_momentum(
        include_particles=False, include_lbfluid=True
    )

    cur_momenta = [
        str(linmomp) + " " + str(self.s.time), 
        str(linmomf) + " " + str(self.s.time)
    ]

    return cur_momenta

def write_momenta(self):
    """
    Write total linear momentum of the particles and of the fluid
    """

    file_linmomp = open(self.project_path + "linmomp_" + self.base, mode="a")
    file_linmomf = open(self.project_path + "linmomf_" + self.base, mode="a")

    for item in self.momentas:
        file_linmomp.write("%s\n" % item[0])
        file_linmomf.write("%s\n" % item[1])

    file_linmomp.close()
    file_linmomf.close()

    del self.momentas[:]


def _vtf_pid_map(self, types="all"):
    """
    wrapping the crappy implementation of the vtf writer

    Generates a VTF particle index map to ESPResSo ``id``.
    This fills the gap for particle ID's as required by VMD

    Parameters
    ----------
    self.s: espressomd.self.s() object
    types : :obj:`str`
            Specifies the particle types. The id mapping depends on which
            particles are going to be printed. This should be the same as
            the one used in writevsf() and writevsf().
    Returns
    -------
    dict:   A dictionary where the values are the VTF indices and the keys are the ESPresSo particle ``id``
    """

    if not hasattr(types, "__iter__"):
        types = [types]
    if types == "all":
        types = [types]
    id_to_write = []

    for p in self.s.part:
        for t in types:
            if p.type == t or t == "all":
                id_to_write.append(p.id)
    return dict(zip(id_to_write, range(len(id_to_write))))


def _writevsf(self, fp, types="all", radiuses={}):
    """
    writes a VST (VTF Structure Format) to a file.
    This can be used to write the header of a VTF file.

    Parameters
    ----------
    self.s: espressomd.self.s() object
    types : :obj:`str`
            Specifies the particle types. The string 'all' will write all particles
    fp : file
            File pointer to write to.

    """

    vtf_index = _vtf_pid_map(self, types)
    fp.write("unitcell {} {} {}\n".format(*(self.s.box_l)))

    for (
        pid,
        vtf_id,
    ) in vtf_index.items():
        if self.s.part.by_id(pid).type in radiuses:
            rad = radiuses[self.s.part.by_id(pid).type]
        else:
            rad = 1
        fp.write(
            "atom {} radius {} name {} type {} \n".format(
                vtf_id, rad, self.s.part.by_id(pid).type, self.s.part.by_id(pid).type
            )
        )
    for (
        pid,
        vtf_id,
    ) in vtf_index.items():
        for b in self.s.part.by_id(pid).bonds:
            if self.s.part.by_id(b[1]).id in vtf_index:
                fp.write("bond {}:{}\n".format(vtf_id, vtf_index[self.s.part.by_id(b[1]).id]))


def get_vcf(self, types="all", folded=False):
    """
    returns vtk configs for a single timestep for futhers
    writing in .vtk file.

    Parameters
    ----------
    self.s: espressomd.self.s() object
    types : :obj:`str`
            Specifies the particle types. The string 'all' will write all particles
    """
    cur_vtf = []

    vtf_index = _vtf_pid_map(self, types)

    cur_vtf.append("\ntimesteps indexed\n")

    for (
        pid,
        vtf_id,
    ) in vtf_index.items():
        pos = self.s.part.by_id(pid).pos
        if folded:
            pos = self.s.part.by_id(pid).pos_folded
        cur_vtf.append("{} {} {} {}\n".format(vtf_id, *(pos)))

    return cur_vtf


def preset_xyz_writer(self):
    """
    very useful for Blender

    """

    names = []

    for p in self.s.part:
        if p.dip[0] != 0:
            names.append("Fm")
        else:
            names.append("Md")

    eos = MDA_ESP.Stream(self.s)  # create the stream
    W = XYZWriter(
        project_path + "traj_" + self.base + ".xyz", n_atoms=len(self.s.part), atoms=names
    )  # open the trajectory file
    u = mda.Universe(eos.topology, eos.trajectory)  # create the MDA universe

    return eos, W, u


def write_xyz(self, eos, W, u):
    """
    very useful for Blender

    """
    coords = foldCoordinates(self.s)
    u.load_new(eos.trajectory)
    W.write_next_timesteps(u.trajectory.ts)
    fold_coordinates_back(self.s, coords)


def get_energy(self):
    """[returns energy for further usage in write_energy]

    Returns:
        [str]: [energies]
    """

    e = self.s.analysis.energy()

    cur_energy = str(self.pos_dip_id) + " " \
        + "total: " + str(e["total"]) \
        + " bonded: " + str(e["bonded"]) \
        + " non_bonded: " + str(e["non_bonded"]) \
        + " dipolar: " + str(e["dipolar"])

    return cur_energy

def write_energy(self):
    """[writes batch of strings and cleans tmp list]

    Args:
        energies ([list of strings]): [with energies]
    """

    with open(self.project_path + "energy_eq_" + self.base, 'a') as f:
        for item in self.energies:
            f.write("%s\n" % item)

    del self.energies[:]

def write_vtk(self, noise = False, particles = True):

    base = self.project_path + "/vtk/"
    if noise:
        fl = "vel" + str(self.pos_dip_id) + "_noise"
        self.lbf.write_vtk_velocity(base + fl+ ".vtk")
    else:
        fl ="vel" + str(self.pos_dip_id)

        self.lbf.write_vtk_velocity(base + fl + ".vtk")

        if particles:
            self.s.part.writevtk(
                base + "part_beads" + str(self.pos_dip_id) + ".vtk", self.param._p_type
            )
            self.s.part.writevtk(
                base + "part_mnps" + str(self.pos_dip_id) + ".vtk", self.param._mnp_type
            )

    #zip file fl.vtk
    

    with zipfile.ZipFile(base + fl + ".zip", 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(base + fl + ".vtk", fl + ".vtk")

    os.remove(base + fl + ".vtk")


def write_vtf(self):

    fp = open(self.project_path + self.base + ".vtf", "a")

    for ts in self.vtfs:
        for item in ts:
            fp.write("%s" % item)

    fp.flush()

    del self.vtfs[:]

def make_checkpoint(self, thermo):
    """
    MD and LB checkpoints

    """

    checkpoint = checkpointing.Checkpoint(
        checkpoint_id=self.base, checkpoint_path=self.project_path
    )


    thismodule = sys.modules[__name__]

    if not self.no_dipdip:
        try:
            setattr(thismodule, "direct_sum", self.direct_sum)
            checkpoint.register("direct_sum")
        except:
            pass

        # try:
        #     setattr(thismodule, "p3m", self.p3m)
        #     checkpoint.register("p3m")
        #     setattr(thismodule, "dlc", self.dlc)
        #     checkpoint.register("dlc")
        # except:
        #     pass


    setattr(thismodule, "param", self.param)
    checkpoint.register("param")

    setattr(thismodule, "pos_dip_id", self.pos_dip_id)
    checkpoint.register("pos_dip_id")

    correlator = False
    try:
        setattr(thismodule, "dpm_corr", self.dpm_corr)
        self.pickled_dpm_corr = pickle.dumps(self.dpm_corr)
        setattr(thismodule, "pickled_dpm_corr", self.pickled_dpm_corr)
        correlator = True
    except:
        pass
    if correlator:
        checkpoint.register("pickled_dpm_corr")

    if thermo == 'lb':
        try:
            setattr(thismodule, "lbf", self.lbf)
            checkpoint.register("lbf")
            # special checkpoint for LB
            self.lbf.save_checkpoint(self.project_path + "lb_checkpoint", 1)
        except:
            'lb checkpoint failed'

    setattr(thismodule, "s", self.s)
    checkpoint.register("s")

    checkpoint.save()

  


def del_all_but_last_checkpoint(self):
    """
    MD - checkpoints

    """

    checkpoints_dir = self.project_path + '/' + self.base
    checkpoints = os.listdir(checkpoints_dir)
    if len(checkpoints) != 0:
        max_el = max([int(i.split('.')[0]) for i in checkpoints])
        for chpk in checkpoints:
            if int(chpk.split('.')[0]) != max_el:
                pass
                os.remove(checkpoints_dir + '/' + chpk)
        os.rename(checkpoints_dir + '/' + str(max_el) + ".checkpoint", checkpoints_dir + '/0.checkpoint')


def compose_base(self):
    """
    suffix used in output files

    """

    self.base = (
        "part_"
        + str(self.particle)
        + "_lambda_"
        + str(self.lmbda)
        + "_hmf_"
        + str(self.hmf).replace(".", "-")
        + "_H_direct_"
        + self.hmf_direct
        + "_freq_" 
        + str(self.hmf_freq).replace(".", "-")
        + "_shear_"
        + str(self.args["shear_rate"])
        + "_box_"
        + str(self.args["box"])
        + "_tmstep_"
        + str(self.timesteps)
        + "_conf_"
        + str(self.conf)
    )

    self.base = self.project_name


def finalize_correlator_and_write_result(self):
    """
    write MACF for central bead of raspberry

    """

    self.dpm_corr.finalize()
    # dpm = np.array(self.dpm_corr.result())

    # np.savetxt( self.project_path + "MACF_" + self.base, np.array([dpm[:,0], dpm[:,1], dpm[:,2]]).T)

    # uncomment for latest version of Espresso
    rslt = [i[0] for i in self.dpm_corr.result()]

    # # print( len(self.dpm_corr.sample_sizes()), len(self.dpm_corr.result()), len(self.dpm_corr.lag_times()))

    np.savetxt( self.project_path + "MACF_" + self.base, np.array([
        self.dpm_corr.lag_times(), 
        [i for i in self.dpm_corr.sample_sizes() if i != 0], 
        rslt
    ]).T)



def get_forces(self):
    """[returns forces acting on rasp and bottomn wall]

    Returns:
        [pos_dip_id; and two lists with forces]
    """

    pf = self.part_force.calculate()
    wall_force = self.s.constraints[0].total_force()
    return [self.pos_dip_id, pf, wall_force]



def write_forces(self):
    """
    Write force onto rasp file from tmp list (forces)
    and cleans that list
    """

    file = open(self.project_path + "forces_" + self.base, mode="a")

    for item in self.forces:
        file.write("{} {:0.6f} {:0.6f} {:0.6f}\n".format(
            item[0], 
            *(item[1].flatten().tolist())
        ))

    file.close()

    del self.forces[:]


def write_map_pid_conf_type(self):
    """
    Write map of pid and conf_type and bonds
    """

    #write pandas df to csv
    self.pid_conf_type.to_csv(self.project_path + "map_pid_conf_type_" + self.base + ".csv")


def save_tmp_parameters(self):
    """
    Write latest ts_mult to "time_step" 
    """

    with open(self.project_path + "tmp_parameters.json", "w") as fp:
        param_dict = {**self.tmp_param.__dict__} 
        json.dump(param_dict, fp, indent=4)

def load_tmp_parameters(self):
    """
    Read latest ts_mult to "time_step" 
    """
    
    #read .json into @dataclass
    with open(self.project_path + "tmp_parameters.json", "r") as fp:
        param_dict = json.load(fp)
        self.tmp_param = TMP_Parameters(**param_dict)


def load_pos_of_equlibrated_spheres(self, path, num_of_spheres, box_len):
    """
    Read pos from .h5 file
    """
    
    path += "/box_len_"+str(int(box_len[0]))+"/pos_dip.h5"
    with h5py.File(path, "a") as f:
        data = f["pos_dip"][f["pos_dip"].shape[0] - 1][1]

    d1 = []
    for d in data:
        d1.append((d[1]%int(self.box_len[0]), d[2]%int(self.box_len[1]), d[3]%int(self.box_len[2])))
    
    return d1


def write_ts_mult_history(self):
    """
    Write ts_mult history to file
    """

    if self.pos_dip_id == 0:
        self.ts_mult_history_file = open(self.project_path + "/ts_mult_history.csv", "w")
        self.ts_mult_history_file.write("step_id, ts_mult\n")
    else:
        self.ts_mult_history_file = open(self.project_path + "/ts_mult_history.csv", "a")

    self.ts_mult_history_file.write(str(self.pos_dip_id-1) + ", " + str(self.tmp_param.ts_mult) + "\n")
    self.ts_mult_history_file.close()

def read_ts_mult_history(self):
    """
    Write ts_mult history to file
    """

    #read ts_mult_history.csv into pandas df where the column names are the first row: step_id, ts_mult
    self.ts_mult_history = pd.read_csv(self.project_path.replace('_noise', '') + "/ts_mult_history.csv", index_col=None, header=None)
    