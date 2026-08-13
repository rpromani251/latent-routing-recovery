"""Parallel resumable driver for the E1-E4 extension experiments."""
import multiprocessing as mp
import ext1_fullvec as e1
import ext2_kgt2 as e2
import ext3_vector_output as e3
import ext4_nongeo as e4


def work(job):
    tag, args = job
    try:
        ran = {"e1": lambda a: e1.run_cell(*a),
               "e2": lambda a: e2.run_block(*a),
               "e3": lambda a: e3.run_cell(*a),
               "e4": lambda a: e4.run_cell(*a)}[tag](args)
        return f"{tag} {args}" if ran else None
    except Exception as ex:
        return f"ERROR {tag} {args}: {ex!r}"


if __name__ == "__main__":
    from dip import preload_null_table
    preload_null_table(1000, "null_table_1000.npz")
    jobs = ([("e1", c) for c in e1.CELLS] + [("e2", (b,)) for b in range(6)]
            + [("e3", c) for c in e3.CELLS] + [("e4", c) for c in e4.CELLS])
    with mp.Pool(4) as p:
        for r in p.imap_unordered(work, jobs):
            if r:
                print(r, flush=True)
    print("EXT ALL DONE")
