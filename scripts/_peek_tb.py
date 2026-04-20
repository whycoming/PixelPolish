from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob, sys
path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/runs/borda_v5_target_exp"
d = sorted(glob.glob(path + "/events*"))[-1]
ea = EventAccumulator(d); ea.Reload()
want = [
    "borda/score_clipiqa", "borda/score_musiq", "borda/score_l_exposure",
    "borda/xk_clipiqa_mean", "borda/xk_musiq_mean", "borda/xk_l_exposure_mean",
    "loss/kl_ref", "grpo/return_mean",
]
keys = [k for k in want if k in ea.Tags()["scalars"]]
steps = sorted(set(e.step for k in keys for e in ea.Scalars(k)))
if not steps:
    print("no data yet"); sys.exit(0)
pick = sorted(set([steps[0]] + steps[::max(1, len(steps) // 6)] + [steps[-1]]))
short = [k.split("/")[-1][:16] for k in keys]
hdr = "step | " + " | ".join(f"{s:>16s}" for s in short)
print(hdr); print("-" * len(hdr))
for s in pick:
    row = []
    for k in keys:
        v = [e.value for e in ea.Scalars(k) if e.step == s]
        row.append(f"{v[0]:16.4f}" if v else " " * 16)
    print(f"{s:4d} | " + " | ".join(row))
