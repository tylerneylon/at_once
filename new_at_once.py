""" new_at_once.py

    Usage:

        import new_at_once as at_once

    TODO

"""


# ______________________________________________________________________
# Imports

# TODO: Check if I use all this.
import base64
import fcntl
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
import uuid

from collections import defaultdict
from glob        import glob
from hashlib     import sha256
from itertools   import repeat
from pathlib     import Path

import numpy as np
from tqdm import tqdm


# ______________________________________________________________________
# Queues

map_pbar_q   = mp.Queue()
cache_pbar_q = mp.Queue()
avg_times_q  = mp.Queue()


# ______________________________________________________________________
# Classes


class Job(object):

    # ______________________________
    # Public interface

    # The parameters are all named to encourage sending in arguments that
    # are also named when the constructor is called.
    def __init__(
            self,
            inputs     = None,
            outputs    = None,
            map_fn     = None,
            combine_fn = None,
            **kwargs
    ):
        # Work with our primary arguments.

        assert inputs  is not None
        assert map_fn  is not None

        self.inputs     = inputs
        self.outputs    = outputs
        self.map_fn     = map_fn
        self.combine_fn = combine_fn

        # Set up default values for other parameters.
        self.num_processes  = os.cpu_count()
        self.reader         = self._default_reader
        self.do_silent_mode = False

        self._prep_output()  # Ensure output directories exist.

        # By assigning these last, they will override any default values set
        # above.
        for key, value in kwargs.items():
            self.__dict__[key] = value

    def run(self):

        # Log the start and end time.
        start_time = time.time()
        self._log_beginning(start_time)
        def be_done():
            self._log_ending(start_time)

        input_list  = self._get_input_list()
        n_inputs    = len(input_list)
        # If self.inputs was a lambda, then we need to drop it so that self
        # becomes pickleable.
        self.inputs = None
        if n_inputs == 0:
            print('Warning: input list is empty; nothing to do.')
            return be_done()

        pbar_args = (map_pbar_q, n_inputs, 'map')
        progress  = pass_thru if self.do_silent_mode else tqdm

        # Create a process to show the progress bar.
        def show_pbar(q, t, desc):
            pbar = tqdm(total=t, desc=desc)
            for j in iter(q.get, 'STOP'):
                pbar.update(1)
            pbar.update(n=pbar.total - pbar.n)
            pbar.close()
        mp.Process(target=show_pbar, args=pbar_args).start()

        N = self.num_processes
        if N == 1:
            for inp in input_list:
                self._handle_input(inp)
        else:
            with mp.Pool(N) as p:
                list(p.map(self._handle_input, input_list))
                map_pbar_q.put('STOP')
                p.close()
                p.join()

        be_done()  # This logs the ending.

    def log(self, msg_obj):
        if self.outputs is None:
            return
        with self.logfile.open('a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(json.dumps(msg_obj) + '\n')

    # ______________________________
    # Internal interface

    def _default_reader(self, filepath):
        ''' For a pickle file, this returns the file's pickled value. For a
            jsonl file, this returns a dict of all the JSON objects in the file.
            If there are key clashes in the input, only a single value will be
            kept, but a warning will be printed.
        '''

        if str(filepath).endswith('.pickle'):
            with open(filepath, 'rb') as f:
                return pickle.load(f)

        # There may be duplication of (input, key) pairs that we should ignore.
        # This can happen if an earlier map run had crashed, and the job was
        # re-run, which we support.
        seen_line_keys = set()
        items = {}
        if str(filepath).endswith('.jsonl'):
            with open(filepath) as f:
                for line in f:
                    obj = self._decode_jsonl_line(line)
                    key = obj['key']
                    line_key = (key, obj['input'])
                    if line_key in seen_line_keys: continue
                    if key in items:
                        print(f'Warning: Key {key} duplicated in input {inp}')
                        print( '    Only one value for this key will be kept')
                    items[key] = obj['value']
                    seen_line_keys.add(line_key)
            return items
        
        raise ValueError(f'Unknown input file type for path {filepath}')

    def _get_input_list(self):

        # The plan is that we will internally have a list of values that are
        # either file paths or raw values. We'll call self.reader() on each
        # value and then pass on the return value to the map function.

        # Case 1: Our input is a directory given as a string.
        if type(self.inputs) is str:
            dirpath = Path(self.inputs)
            assert dirpath.is_dir()
            pickles = list(dirpath.glob('*.pickle'))
            jsonls  = list(dirpath.glob('*.jsonl'))
            assert len(pickles) == 0 or len(jsonls) == 0
            return pickles if len(pickles) > 0 else jsonls

        # TODO: Eventually support generators.

        # Case 2: Our input is a function that will return a list of values;
        #         each value will be sent to map_fn().
        if callable(self.inputs):
            self.reader = pass_thru
            return self.inputs()

        # Case 3: Our input is a list of a file paths.
        if type(self.inputs) is list:
            return self.inputs

    def _handle_input(self, inp):

        if self._input_is_done(inp):
            map_pbar_q.put(1)
            return

        out = self.map_fn(inp, self.reader(inp))
        self._save_output(inp, out)

        map_pbar_q.put(1)

    def _input_is_done(self, inp):

        if self.outputs is None:
            return False

        receipts = self._get_output_receipts()
        outpath  = self.out_dir / Path(inp).name
        if outpath.name not in receipts:
            return False
        # TODO: Test that this works as I'd like it to.
        #       Might we have problems due to small errors?
        out_time = outpath.stat().st_ctime
        rct_time = receipts[outpath.name]
        assert out_time <= rct_time
        return out_time == rct_time

    def _prep_output(self):

        # Ensure the output dir and metadata dir both exist.
        if type(self.outputs) is str:

            self.out_dir = Path(self.outputs)
            self.out_dir.mkdir(parents=True, exist_ok=True)

            self.metadata_dir = self.out_dir / 'metadata'
            self.metadata_dir.mkdir(parents=True, exist_ok=True)

            self.logfile = self.metadata_dir / 'log.jsonl'

    # TODO: Warn if it looks like we've already saved here.
    def _save_output(self, inp, out):

        # Case 1: No saved output. Then we do nothing.
        if self.outputs is None:
            return

        # Case 2: self.outputs is a string, interpreted as a pickle directory.
        outpath = self.out_dir / Path(inp).name
        if outpath.is_file():
            print(f'Warning: Got multiple outputs for input {inp}')

        with outpath.open('wb') as f:
            pickle.dump(out, f)

        self._save_output_receipt(outpath)

    # ______________________________
    # Internal logging functions

    def _log_beginning(self, start_time):
        self.log({'system_msg': {
            'event': 'start',
            'time' : time.ctime(start_time),
            'timestamp': start_time
        }})

    def _log_ending(self, start_time):
        end_time = time.time()
        self.log({'system_msg': {
            'event': 'stop',
            'time' : time.ctime(end_time),
            'timestamp': end_time,
            'duration': end_time - start_time
        }})

    # ______________________________
    # Internal data receipt code

    def _save_output_receipt(self, outpath):
        data_receipt = self.metadata_dir / 'data_receipt.jsonl'
        with data_receipt.open('a') as f:
            ctime = outpath.stat().st_ctime
            receipt = {'saved_file': outpath.name, 'ctime': ctime}
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(json.dumps(receipt) + '\n')

    def _get_output_receipts(self):
        ''' Output receipts come in the form of a dict.
            Each key is a saved output file name.
            Each value is a ctime for that file.
        '''
        if 'output_receipts' in self.__dict__:
            return self.output_receipts

        self.output_receipts = {}
        data_receipt = self.metadata_dir / 'data_receipt.jsonl'
        if not data_receipt.is_file(): return self.output_receipts
        with (self.metadata_dir / 'data_receipt.jsonl').open() as f:
            for line in f:
                obj = json.loads(line)
                fname = obj['saved_file']
                ctime = obj['ctime']
                self.output_receipts[fname] = ctime
        return self.output_receipts


# ______________________________________________________________________
# Functions

def pass_thru(x, **kwargs):
    return x

def hash(s):
    ''' Return a (usually quite large) integer hash of the given
        string s. '''
    assert type(s) is str
    return int.from_bytes(sha256(s.encode()).digest(), byteorder='big')

def encode(value):
    b = pickle.dumps(value)
    # The final .decode() below converts a bytes object to a str.
    return base64.b64encode(b).decode()

def decode(value):
    """ This expects value to be a `str` that was encoded by a call
        to encode(). """
    b = value.encode()  # Convert a str to a bytes object.
    return pickle.loads(base64.b64decode(b))

def run(**kwargs):
    """ This function accepts the union of keyword arguments that
        can be sent to Job.__init__() and Job.run(). """
    job = Job(**kwargs)
    return job.run()
