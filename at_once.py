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
# Debug

do_dbg_timing = True


# ______________________________________________________________________
# Constants

# Constants for map types.
ARBITRARY  = 'arbitrary'
KEEP_KEYS  = 'keep keys'
MAP_CHUNKS = 'map chunks'


# ______________________________________________________________________
# Queues

map_pbar_q   = mp.Queue()
cache_pbar_q = mp.Queue()
avg_times_q  = mp.Queue()


# ______________________________________________________________________
# Classes

# TODO: Organize where this goes.
def pass_thru(x, **kwargs):
    return x

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
            is_one_value     = False,
            do_map_per_chunk = False,
            job_key          = None,
            do_silent_mode   = False,
            map_type         = ARBITRARY,
            **kwargs
    ):
        self.do_silent_mode = do_silent_mode

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
        hash_key = sys.argv[0]
        if job_key is not None:
            hash_key += str(job_key)
        hash_num = hash(hash_key) % 100000
        g = glob(f'{cache_root}/*{hash_num}')
        if len(g) == 0:
            time_str = time.strftime('%Y_%m_%d_%H_%M')
            self.cache_dir = Path(cache_root) / f'{time_str}_{hash_num}'
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._say(f'Using the new cache directory {self.cache_dir}')
        else:
            if len(g) > 1:
                print(f'WARNING: multiple cache directories I might use.')
            self.cache_dir = Path(g[0])
            self._say(f'Using pre-existing cache directory {self.cache_dir}')

        self.logfile = self.cache_dir / 'logfile.jsonl'

        self.output_dir = None
        if output_dir:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self._say(f'Output will be saved to {self.output_dir}')

        self.num_output_files = len(self.input)
        if num_output_files:
            self.num_output_files = num_output_files

        self.num_processes = num_processes or os.cpu_count()

        self.run_id_of_input = {}
        self.data_receipt = self.cache_dir / 'data_receipt.jsonl'
        if self.data_receipt.is_file():
            with self.data_receipt.open() as f:
                for line in f:
                    obj = json.loads(line)
                    self.run_id_of_input[obj['input']] = obj['run_id']

        self.run_id = str(uuid.uuid4())[-8:]

        self.is_one_value = is_one_value

        self.do_map_per_chunk = do_map_per_chunk

        assert map_type in [ARBITRARY, KEEP_KEYS, MAP_CHUNKS]
        self.map_type = map_type

        # Set up debug log.
        self.last_event = None
        self.last_time  = None
        self.dbg_log = self.cache_dir / 'dbg_log.jsonl'

        self.dbg_timing = {}

    def _say(self, s):
        if self.do_silent_mode:
            return
        print(s)

    def _dbg_log_event_start(self, eventname):

        worker_id = mp.current_process().name
        t = time.time()
        if self.last_event is not None:
            delta = t - self.last_time
            with self.dbg_log.open('a') as f:
                f.write(json.dumps({
                    'worker_id': worker_id,
                    'event'    : self.last_event,
                    'took'     : f'{delta:.4f}'
                }) + '\n')
        self.last_event = eventname
        self.last_time  = t

    def log(self, msg_obj):
        with self.logfile.open('a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.write(json.dumps(msg_obj) + '\n')

    def _append_to_cache(self, inp, key, value, out_chunk=None):
        n = self.num_output_files
        i = hash(str(key)) % n
        if out_chunk is not None:
            i = out_chunk
        num_digits = len(str(n - 1))
        outfile = self.cache_dir / f'{str(i).zfill(num_digits)}_of_{n}.jsonl'
        with outfile.open('a') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            s = json.dumps({
                'key'   : key,
                'value' : encode(value),
                'run_id': self.run_id,
                'input' : str(inp)
            })
            f.write(s + '\n')

    # TODO: Factor out the commonalities between this and handle_cache().
    def _handle_chunked_input(self, chunked_inp):
        """ This expects chunked_inp = (chunk_num, input_list).
            This maps all inputs at once, saving them directly to the output
            directory.
        """

        assert self.output_dir is not None

        chunk_num, inp_list = chunked_inp
        if self._is_output_done(chunk_num):
            map_pbar_q.put(1)
            return

        chunk_result = defaultdict(list)
        for inp in inp_list:
            result = self._handle_input(inp, do_save_to_cache=False)
            # TODO: Check if chunk is already done in the output dir.
            assert result is not None
            if self.is_one_value:
                chunk_result.update(result)
            else:
                for key, value in result.items():
                    chunk_result[key].extend(value)
            map_pbar_q.put(1)
        chunk_result = dict(chunk_result)

        # Save to disk with a data receipt.
        self._save_output(chunk_result, chunk_num)

    def _handle_input(self, inp, do_save_to_cache=True):
        """ This expects a single input `inp`, and calls out to the user's map
            function.
        """

        self._dbg_log_event_start('start of input')

        # Check to see if `inp` already has output in our cache_dir.
        if inp in self.run_id_of_input:
            map_pbar_q.put(1)
            return

        # Process the input.
        out_chunk = None
        if self.input_type == 'list':
            self._dbg_log_event_start('call map_fn()')
            result = self.map_fn(inp)
            self._dbg_log_event_start('verify map_fn() output type')
            if not self.is_one_value:
                assert all(type(v) is list for v in result.values())
            if not do_save_to_cache:
                return result
        else:
            # Process input files.
            inp = Path(inp)
            if not inp.is_file():
                # TODO: Log that we're skipping this.
                return
            chunk_num = int(inp.stem.split('_')[0])
            self._dbg_log_event_start('load from pickle file')

            with inp.open('rb') as f:
                data = pickle.load(f)

            self._dbg_log_event_start('call map_fn')
            if self.do_map_per_chunk:
                result = self.map_fn(chunk_num, data)
                assert type(result) is dict
                out_chunk = chunk_num
            else:
                # Run map_fn() once per (key, value) pair.
                result = defaultdict(list)
                for in_k, in_v in data.items():
                    out = self.map_fn(in_k, in_v)
                    if out is None:
                        continue
                    for out_k, out_v in out.items():
                        assert type(out_v) is list
                        self._dbg_log_event_start('append out to result')
                        result[out_k].extend(out_v)

        self._dbg_log_event_start('mark input as saved')
        for key, value in result.items():
            self._append_to_cache(inp, key, value, out_chunk)

        # TODO: Eventually, I may want to put a mutex around these
        #       writes. In the meantime, it looks like writes under ~1k
        #       will be atomic on my os/fs, so this is ok for now.
        with self.data_receipt.open('a') as f:
            f.write(json.dumps({
                'input': str(inp),
                'run_id': self.run_id
            }) + '\n')
        self.run_id_of_input[inp] = self.run_id
        self._dbg_log_event_start('ready for next input')

        self._dbg_log_event_start('XXX')  # XXX
        map_pbar_q.put(1)

    def _start_dbg_time(self, time_type):
        if do_dbg_timing:
            self.dbg_timing[time_type] = time.time()

    def _end_dbg_time(self, time_type):
        if do_dbg_timing:
            duration = time.time() - self.dbg_timing[time_type]
            avg_times_q.put((time_type, duration))

    def _get_output_path(self, chunk_num):
        """ This returns the output path, as a Path instance, for the given
            chunk_num. """
        n           = self.num_output_files
        num_digits  = len(str(n - 1))
        filename    = f'{str(chunk_num).zfill(num_digits)}_of_{n}.pickle'
        return self.output_dir / filename

    def _is_output_done(self, chunk_num):
        """ This returns True iff the given chunk number is already saved to
            disk and in the data receipt in the output directory.
        """
        output_path = self._get_output_path(chunk_num)

        # Check that the output file exists.
        if not output_path.is_file():
            return False

        # Check that the output file has a receipt.
        output_receipts = self.get_output_receipts()
        receipt_ctime = output_receipts.get(output_path.name, None)
        if receipt_ctime is None:
            return False

        # Verify that the output receipt is for this data. This is a heuristic
        # that, in practice, I expect to essentially always work.
        file_ctime = output_path.stat().st_ctime
        return abs(file_ctime - receipt_ctime) < 0.001

    def _save_output(self, output, chunk_num):
        """ This expects `output` to be mapping from keys to values. This
            applies any appropriate combine_fn to the values, saves the result
            to disk, and records this in the output data receipt. """

        # Determine the output file path.

        # Apply combine_fn as appropriate.
        if self.combine_fn and not self.is_one_value:
            for key, values in output.items():
                v = values.pop()
                while len(values) > 0:
                    v = self.combine_fn(key, v, values.pop())
                output[key] = v

        output_path = self._get_output_path(chunk_num)
        with output_path.open('wb') as f:
            pickle.dump(output, f)
        self.save_output_receipt(output_path)

    def _handle_cache(self, cache_file):

        assert self.output_dir is not None

        # If a data receipt exists, skip out early.
        # We send a message to the appropriate queue for progress.
        chunk_num = int(Path(cache_file).name.split('_')[0])
        if self._is_output_done(chunk_num):
            cache_pbar_q.put(1)
            return

        # Load in the full cache file, filtering out possible
        # partial data from a crashed-out run.
        cache = defaultdict(list)
        with open(cache_file) as f:
            for i, line in enumerate(f):
                self._start_dbg_time('json_decode')
                obj = json.loads(line)
                self._end_dbg_time('json_decode')

                self._start_dbg_time('post_json_decode_checks')

                inp_run_id = self.run_id_of_input.get(obj['input'], None)
                if obj['run_id'] not in [inp_run_id, self.run_id]:
                    # TODO: Log that we are skipping an output that
                    #       appears to be corrupt or outdated.
                    continue
                self._end_dbg_time('post_json_decode_checks')

                self._start_dbg_time('b64_unpickling')
                value = decode(obj['value'])
                self._end_dbg_time('b64_unpickling')

                if self.is_one_value:
                    cache[obj['key']] = value
                else:
                    cache[obj['key']].extend(value)
        cache = dict(cache)

        # Save to the official output directory.
        self._start_dbg_time('write_to_file')
        self._save_output(cache, chunk_num)
        self._end_dbg_time('write_to_file')

        cache_pbar_q.put(1)

    def get_output_receipts(self):
        if 'output_receipts' not in self.__dict__:
            self.output_receipts = {}
            metadata_dir = self.get_metadata_dir()
            with (metadata_dir / 'data_receipt.jsonl').open() as f:
                for line in f:
                    obj = json.loads(line)
                    fname = obj['saved_file']
                    ctime = obj['ctime']
                    self.output_receipts[fname] = ctime
        return self.output_receipts

    def get_metadata_dir(self):
        if 'metadata_dir' not in self.__dict__:
            self.metadata_dir = self.output_dir / 'metadata'
            self.metadata_dir.mkdir(parents=True, exist_ok=True)
        return self.metadata_dir

    def save_output_receipt(self, filepath):
        metadata_dir = self.get_metadata_dir()
        data_receipt = metadata_dir / 'data_receipt.jsonl'
        with data_receipt.open('a') as f:
            ctime = filepath.stat().st_ctime
            receipt = {'saved_file': filepath.name, 'ctime': ctime}
            f.write(json.dumps(receipt) + '\n')

    def run(
            self,
            map_fn     = None,
            combine_fn = None,
            reduce_fn  = None,
            **kwargs
    ):
        assert map_fn is not None
        self.map_fn     = map_fn
        self.combine_fn = combine_fn
        self.reduce_fn  = reduce_fn

        if len(self.input) == 0:
            print('Nothing to do since input length = 0.')
            return

        end_fn = lambda: 0

        t = len(self.input)
        pbar_args = (map_pbar_q, t, 'Map to cache')

        # XXX TODO
        progress = pass_thru if self.do_silent_mode else tqdm

        self.is_input_pre_chunked = False
        input_handler = self._handle_input

        if self.map_type == KEEP_KEYS and self.input_type == 'list':
            self.is_input_pre_chunked = True
            # Pull together same-chunk inputs.
            n = self.num_output_files
            chunked_inputs = defaultdict(list)
            for x in self.input:
                chunked_inputs[hash(str(x)) % n].append(x)
            self.input = list(chunked_inputs.items())
            pbar_args = (map_pbar_q, t, 'Map to output dir')
            input_handler = self._handle_chunked_input

        if self.num_processes == 1:
            for inp in progress(self.input):
                input_handler(inp)
        else:

            # Set up the progress bar.
            def show_pbar(q, t, desc):
                pbar = tqdm(total=t, desc=desc)
                for j in iter(q.get, 'STOP'):
                    pbar.update(1)
                pbar.update(n=pbar.total - pbar.n)
                pbar.close()

            def avg_times(q):
                times = defaultdict(list)
                for update in iter(q.get, 'STOP'):
                    time_type, time = update
                    times[time_type].append(time)
                for k, v in times.items():
                    print(f'{k}: {sum(v) / len(v):.4f}')

            mp.Process(target=show_pbar, args=pbar_args).start()

            if do_dbg_timing:
                end_fn = lambda: avg_times_q.put('STOP')
                mp.Process(target=avg_times, args=(avg_times_q,)).start()

            N = self.num_processes
            with mp.Pool(N) as p:
                list(p.map(input_handler, self.input))
                map_pbar_q.put('STOP')

        if self.output_dir:
            cache_data = list(self.cache_dir.glob(f'*_of_*.jsonl'))
            if len(cache_data) == 0:
                end_fn()
                return
            if self.num_processes == 1:
                for cache_item in progress(cache_data):
                    self._handle_cache(cache_item)
            else:
                pbar_args = (cache_pbar_q, len(cache_data), 'Cache to output')
                mp.Process(target=show_pbar, args=pbar_args).start()
                with mp.Pool(N) as p:
                    list(p.map(self._handle_cache, cache_data))
                    cache_pbar_q.put('STOP')
        end_fn()


# ______________________________________________________________________
# Functions

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
    return job.run(**kwargs)
