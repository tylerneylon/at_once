""" at_once.py

    Usage:

        import at_once

        def is_inp_done(inp):

            # ... Parallel-friendly code.

            return is_done  # True or False.

        def handle_inp(inp):

            # ... Parallel-friendly code.

            # Save result to disk; return nothing.

        at_once.run(inp_list, is_inp_done, handle_inp)

    This spawns one process per local cpu core, and runs
    those processes in parallel, keeping each one busy with
    continuous handle_inp() calls over all of inp_list,
    skipping over any `inp` values for which
    is_inp_done(inp) returns True.

    The purpose of this little library is to assist in
    running crash-robust steps of a large data pipeline.

    I recommend setting the value

        at_once.logfile = my_filepath

    and then using calls to

        at_once.log(msg)

    in order to record errors; don't print anything to
    stdout or stderr.

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

    inp, is_inp_done, handle_inp = triple

    if is_inp_done(inp):
        return

    handle_inp(inp)

def run(inp_list, is_inp_done, handle_inp, do_shuffle=False):

    if do_shuffle:
        np.random.shuffle(inp_list)

    length = len(inp_list)
    task_list = zip(inp_list, repeat(is_inp_done), repeat(handle_inp))
    with Pool(os.cpu_count()) as p:
        list(tqdm(p.imap_unordered(_manage_inp, task_list), total=length))
