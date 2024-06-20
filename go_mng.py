#Playground for naughty kids
from simulation_box import SimulationBox
from particle.mng import MNG
import espressomd
import numpy as np
import functools
import sys 
import io
import os
import time


# For utf-8!
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding="utf-8")

# Immediate output!
print = functools.partial(print, flush=True)

sb = SimulationBox()
mng = MNG()

if sb.new_continue_sim == "new":

    print("Adding particles..")

    sb.add_mngs(mng, version = 'equalibrated') #magnetized

    if sb.thermostat == 'lb':
        sb.init_thermostat(zero_kbT_B = False, zero_kbT_lb = False, act_on_virtual=False)
        sb.add_walls_and_shear_flow(shear_flow=False)
    elif sb.thermostat == 'lang':
        sb.init_thermostat_lang()
    elif sb.thermostat == 'brownian':
        sb.init_thermostat_brownian()

    if sb.no_dipdip == False:
        sb.add_magnetostatics_DSCpu()
    sb.add_magnetic_field()
    sb.print_parameters()

    # sb.add_tracers()

    sb.init_vtf()
    sb.write_map_pid_conf_type()
    sb.write_parameters_to_json()

    sb.warm_up()
    sb.s.time = 0


elif sb.new_continue_sim == "continue":
    sb.additional_afterload_init(mng)

sb.init_data_containers_and_clock()

print("Running..")
while(sb.num_of_steps - sb.pos_dip_id > 0):
    print(sb.pos_dip_id, sb.s.time)

    if sb.hmf_freq != 0:
        sb.int_rot_field()
    else:
        sb.run(sb.param._timesteps)
        sb.write_sim_data(sb.pos_dip_id, checkpoint_step=1) 

print("Done 👍")
