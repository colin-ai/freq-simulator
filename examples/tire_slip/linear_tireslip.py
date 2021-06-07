from sysipy.generator import linear_interp, seq_generator_tire_slip, to_csv
from sysipy.model import magic_formula
from sysipy.viz import subplots_dataset_tireslip

# System character
m = 300  # mass
g = 9.81  # gravity constant
p = m * g  # weight

# Vehicle position pos_breakpoints
freq = 1  # Number of measure / second
step_breakpoint = 10  # DT between two v_vehicle breakpoint
# fmt: off
v_vehicle_breaks = (10, 10, 110, 20, 40, 40, 100, 70, 10, 110, 50, 5, 5, 60, 100, 100,
                    130, 130, 60, 70, 0, 0, 75, 10, 10, 10, 10, 110, 20, 40, 40, 100,
                    70, 10, 110, 50, 5, 5,60, 100, 100, 130, 130, 60, 70, 0, 0, 75, 10,
                    10)
# fmt: on
F_long, alpha = magic_formula(p)

n_train_seq = 200
velocities = seq_generator_tire_slip(
    v_vehicle_breaks,
    n_seq=n_train_seq,
    xp_rand=True,
    func_interp=linear_interp,
    m=m,
    freq=freq,
    step_breakpoint=step_breakpoint,
    F_long=F_long,
    alpha=alpha,
)

to_csv(velocities, f"./dataset/linear_n_seq_{n_train_seq}_step_{step_breakpoint}_freq_{freq}.csv")

subplots_dataset_tireslip(
    f"./dataset/linear_n_seq_{n_train_seq}_step_{step_breakpoint}_freq_{freq}.csv",
    n_rows=4,
    n_cols=4,
)
