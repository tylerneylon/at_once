""" at_once.py

    TODO -- Completely re-write this.

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

import fcntl
import json
import os
import pickle
import sys
import time
import uuid

from   collections     import defaultdict
from   glob            import glob
from   hashlib         import sha256
from   itertools       import repeat
from   multiprocessing import Manager, Pool
from   pathlib         import Path

import numpy as np
from   tqdm import tqdm

# XXX
import multiprocessing as mp


# ______________________________________________________________________
# Classes

class Job(object):

    def __init__(
            self,
            input_files      = None,
            input_list       = None,
            do_shuffle       = False,
            output_dir       = None,
            num_output_files = None,
            cache_root       = None,
            num_processes    = None,
            is_one_value     = False
    ):

        if input_list is not None:
            assert input_files is None
            self.input      = list(input_list)  # The `list` is to copy it.
            self.input_type = 'list'
        else:
            assert input_list is None
            self.input      = list(input_files)  # The `list` is to copy it.
            self.input_type = 'files'
        assert self.input is not None

        if do_shuffle:
            np.random.shuffle(self.input)

        # Set up the temporary directory.
        assert cache_root is not None
        cache_root = Path(cache_root)
        hash_num = hash(sys.argv[0]) % 100000
        g = glob(f'{cache_root}/*{hash_num}')
        if len(g) == 0:
            time_str = time.strftime('%Y_%m_%d_%H_%M')
            self.cache_dir = Path(cache_root) / f'{time_str}_{hash_num}'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            print(f'Using the new cache directory {self.cache_dir}')
        else:
            if len(g) > 1:
                print(f'WARNING: I see multiple cache directories I might use.')
            self.cache_dir = Path(g[0])
            print(f'Using the pre-existing cache directory {self.cache_dir}')

        self.logfile = self.cache_dir / 'logfile.jsonl'

        self.output_dir = None
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            print(f'Output will be saved to {self.output_dir}')

        self.num_output_files = len(self.input)
        if num_output_files:
            self.num_output_files = num_output_files

        self.num_processes = num_processes or os.cpu_count()

        self.run_id_of_input = {}
        self.saved_inp_filepath = self.cache_dir / 'saved_inputs.jsonl'
        if self.saved_inp_filepath.is_file():
            with self.saved_inp_filepath.open() as f:
                for line in f:
                    obj = json.loads(line)
                    self.run_id_of_input[obj['input']] = obj['run_id']

        self.run_id = str(uuid.uuid4())[-8:]

        self.is_one_value = is_one_value

    def log(self, msg_obj):
        with self.logfile.open('a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(json.dumps(msg_obj) + '\n')

    def _append_to_cache(self, inp, key, value):
        n = self.num_output_files
        i = hash(str(key)) % n
        num_digits = len(str(n - 1))
        outfile = self.cache_dir / f'{str(i).zfill(num_digits)}_of_{n}.jsonl'
        with outfile.open('a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            s = json.dumps({
                'key': key,
                'value': value,
                'run_id': self.run_id,
                'input': inp
            })
            f.write(s + '\n')
    # XXX
    def _print(self, s):
        proc = mp.current_process()
        t = time.asctime()
        # print(f'{proc.name} {t}: {s}', flush=True)

    def _handle_input(self, inp):

        self._print(f'About to handle input: {inp}')

        # Check to see if `inp` already has output in our cache_dir.
        if inp in self.run_id_of_input:
            self._print(f'    Skipping bc its cached.')
            return

        # Process the input.
        if self.input_type == 'list':
            self._print(f'About to call map_fn on {inp}')
            results = self.map_fn(inp)
            self._print(f'Just finished map_fn on {inp}')
            if not self.is_one_value:
                assert all(type(v) is list for v in results.values())
        else:
            # TODO
            results = self.map_fn(key, value)

        self._print(f'About to append to cache.')
        for key, value in results.items():
            self._append_to_cache(inp, key, value)
        self._print(f'Just appended to cache; about to record inp saved.')
        # TODO: Eventually, I may want to put a mutex around these
        #       writes. In the meantime, it looks like writes under ~1k
        #       will be atomic on my os/fs, so this is ok for now.
        with self.saved_inp_filepath.open('a') as f:
            f.write(json.dumps({'input': inp, 'run_id': self.run_id}) + '\n')
        self.run_id_of_input[inp] = self.run_id
        self._print(f'Just recorded inp saved.')

    def _handle_cache(self, cache_file):

        assert self.output_dir is not None

        # Load in the full cache file, filtering out possible
        # partial data from a crashed-out run.
        cache = defaultdict(list)
        with open(cache_file) as f:
            for i, line in enumerate(f):
                obj = json.loads(line)

                # XXX
                if 'input' not in obj:
                    print('\n\n\n')
                    print(f'Problem in {cache_file}')
                    print(f'On 1-based line number {i + 1}')

                inp_run_id = self.run_id_of_input.get(obj['input'], None)
                if inp_run_id != obj['run_id']:
                    continue
                if self.is_one_value:
                    cache[obj['key']] = obj['value']
                else:
                    cache[obj['key']].extend(obj['value'])
        cache = dict(cache)

        # TODO: This is a good place to run combine / reduce.

        # Save to the official output directory.
        filename = cache_file.name.split('.')[0] + '.pickle'
        with (self.output_dir / filename).open('wb') as f:
            pickle.dump(cache, f)

    def run(
            self,
            map_fn     = None,
            combine_fn = None,
            reduce_fn  = None
    ):
        assert map_fn is not None
        self.map_fn = map_fn

        t = len(self.input)

        # XXX
        print(f'self.input has length {len(self.input)}.')

        with Pool(self.num_processes) as p:
            list(tqdm(p.imap(self._handle_input, self.input), total=t))

        if self.output_dir:
            n = self.num_output_files
            cache_data = list(self.cache_dir.glob(f'*_of_{n}.jsonl'))
            t = len(cache_data)
            with Pool(self.num_processes) as p:
                list(tqdm(p.imap(self._handle_cache, cache_data), total=t))


# ______________________________________________________________________
# Functions

def hash(s):
    ''' Return a (usually quite large) integer hash of the given
        string s. '''
    assert type(s) is str
    return int.from_bytes(sha256(s.encode()).digest(), byteorder='big')

def _manage_inp(triple):

    inp, check_inp, process_inp = triple

    is_done, value = check_inp(inp)

    if is_done:
        return value
    else:
        return process_inp(inp)

# XXX Working notes.
#
#  * Expect either input_files (a list of filenames, glob-friendly), or
#  * input_list, which is passed directly to the handler (no caching).
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
