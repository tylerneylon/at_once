""" at_once.py

    Usage:

        import at_once

        def check_inp(inp):

            # ... Parallel-friendly code.

            # In a map-only situation, return:
            return [is_done, None]

            # In a map-reduce setup, return:
            return [is_done, value_if_is_done]

        def process_inp(inp):

            # ... Parallel-friendly code.

            # Save result to disk; return nothing.

        at_once.run(inp_list, check_inp, process_inp)

    ----------------------------------------------------------------------

    This spawns one process per local cpu core, and runs
    those processes in parallel, keeping each one busy with
    continuous process_inp() calls over all of inp_list,
    skipping over any `inp` values for which
    check_inp(inp) returns True.

    The purpose of this little library is to assist in
    running crash-robust steps of a large data pipeline.

    ----------------------------------------------------------------------
    Reduce functions

    You can send in an optional `reduce_fn` parameter to at_once.run().
    If you do, it will receive a list of consolidated values per key
    that is output by process_inp(). The output of reduce_fn() then
    becomes the value of the same key, and the dictionary of all these
    keys and all these reduced values is the return value from
    at_once.run(). If this is super-confusing: This is a standard
    map-reduce setup. I recommend reading online about map-reduce in
    general.

    ----------------------------------------------------------------------
    Logging output

    I recommend setting the value

        at_once.logfile = my_filepath

    and then using calls to

        at_once.log(msg)

    in order to record errors; don't print anything to
    stdout or stderr.

    ----------------------------------------------------------------------

    The optional keyword parameter do_shuffle to at_once.run()
    is False by default; if True, it will shuffle inp_list
    randomly before it's used. This can be helpful if there
    is some order to the underlying data, and you want to
    ensure that partial progress is representative of the
    overall dataset.
"""
# Future ideas:
# * Support a dry run, somehow (not sure how yet).
# * Support jsonl output, and automatically include which inp was
#   being processed by the process giving the error message.


# ______________________________________________________________________
# Imports

import os
from itertools import repeat
from multiprocessing import Pool

import numpy as np
from tqdm import tqdm


# ______________________________________________________________________
# Globals and Constants

logfile = 'at_once_log.txt'


# ______________________________________________________________________
# Functions

def log(msg):
    with open(logfile, 'a') as f:
        f.write(msg + '\n')

def _manage_inp(triple):

    inp, check_inp, process_inp = triple

    is_done, value = check_inp(inp)

    if is_done:
        return value
    else:
        return process_inp(inp)

def run(
        inp_list,
        check_inp,
        process_inp,
        do_shuffle = False,
        reduce_fn = None,
        combine_fn = None,
        postprocess_fn = None,
        do_flatten_values = False
        ):

    assert not (reduce_fn and combine_fn)

    if do_shuffle:
        np.random.shuffle(inp_list)

    length = len(inp_list)
    task_list = zip(inp_list, repeat(check_inp), repeat(process_inp))
    with Pool(os.cpu_count()) as p:
        data = list(tqdm(p.imap(_manage_inp, task_list), total=length))

    if reduce_fn is None and combine_fn is None:

        return data

    else:

        # Consolidate the data.
        kv_to_reduce = {}
        for kv_map in data:
            for key, value in kv_map.items():
                if do_flatten_values:
                    kv_to_reduce.setdefault(key, []).extend(value)
                else:
                    kv_to_reduce.setdefault(key, []).append(value)

        # Collect the reductions.
        if reduce_fn:

            return {
                    key: reduce_fn(key, values)
                    for key, values in kv_to_reduce.items()
            }

        else:

            print('Combining results ..')
            combined_results = {}
            for key, values in tqdm(kv_to_reduce.items()):
                combined = values.pop(0)
                while len(values) > 0:
                    combined = combine_fn(key, combined, values.pop(0))
                combined_results[key] = combined
            if postprocess_fn is None:
                return combined_results
            return {
                k: postprocess_fn(k, v)
                for k, v in combined_results.items()
            }
