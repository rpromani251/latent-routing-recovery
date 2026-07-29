"""Parallel driver: runs remaining sim1d cells across processes. Resumable."""
import multiprocessing as mp
import sim1d_known_regimes as s1


def work(job):
    kind, arg = job
    if kind == "noise":
        tau, model = arg
        if s1.run_noise_cell(tau, model):
            return f"tau={tau:g}/{model}"
    else:
        if s1.run_rob_cell(arg):
            return f"rob {arg}"
    return None


if __name__ == "__main__":
    from dip import preload_null_table
    preload_null_table(s1.M_DIP, f"null_table_{s1.M_DIP}.npz")
    jobs = [("noise", (t, m)) for t in s1.NOISES for m in ("gated", "honest")]
    jobs += [("rob", n) for n, _, _ in s1.GP_SETTINGS] + [("rob", "kink_control")]
    with mp.Pool(4) as p:
        for r in p.imap_unordered(work, jobs):
            if r:
                print("done:", r, flush=True)
    print("ALL DONE")
