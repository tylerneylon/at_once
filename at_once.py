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
            map_type         = 'arbitrary',
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

        # Set up debug log.
        self.last_event = None
        self.last_time  = None
        self.dbg_log = self.cache_dir / 'dbg_log.jsonl'

        self.dbg_timing = {}

    def _assign_name(self, i):
        proc = mp.current_process()
        proc.name = str(i)
        # XXX
        # print(f'My name is {proc.name}')

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
                'input' : inp
            })
            f.write(s + '\n')

    # XXX
    def _print(self, s):
        proc = mp.current_process()
        t = time.asctime()
        # print(f'{proc.name} {t}: {s}', flush=True)

    def _handle_input(self, inp):

        # self._print(f'About to handle input: {inp}')
        self._dbg_log_event_start('start of input')

        # Check to see if `inp` already has output in our cache_dir.
        if inp in self.run_id_of_input:
            # self._print(f'    Skipping bc its cached.')
            map_pbar_q.put(1)
            return

        # Process the input.
        out_chunk = None
        if self.input_type == 'list':
            # self._print(f'About to call map_fn on {inp}')
            self._dbg_log_event_start('call map_fn()')
            result = self.map_fn(inp)
            # self._print(f'Just finished map_fn on {inp}')
            self._dbg_log_event_start('verify map_fn() output type')
            if not self.is_one_value:
                assert all(type(v) is list for v in result.values())
        else:
            # Process input files.
            chunk_num = int(Path(inp).stem.split('_')[0])
            self._dbg_log_event_start('load from pickle file')

            with open(inp, 'rb') as f:
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
            f.write(json.dumps({'input': inp, 'run_id': self.run_id}) + '\n')
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

    def _handle_cache(self, cache_file):

        assert self.output_dir is not None

        # If a data receipt exists, skip out early.
        # We send a message to the appropriate queue for progress.
        if self.does_cache_have_output(cache_file):
            cache_pbar_q.put(1)
            return

        # Load in the full cache file, filtering out possible
        # partial data from a crashed-out run.
        cache = defaultdict(list)
        with open(cache_file) as f:
            for i, line in enumerate(f):
                self._start_dbg_time('json_decode')
                try:
                    obj = json.loads(line)
                except json.decoder.JSONDecodeError as e:
                    # XXX
                    print(e)
                    print('Exception in _handle_cache()')
                    print(f'cache_file = {cache_file}')
                    print(f'reading in 0-based line {i}')
                    print(f'line has length {len(line)}')
                    print()
                    print('Beginning of line is below:')
                    print(line[:1000])
                    print()
                    print('Ending of line is next:')
                    print(line[-1000:])
                    raise
                self._end_dbg_time('json_decode')

                self._start_dbg_time('post_json_decode_checks')
                # XXX
                if 'input' not in obj:
                    print('\n\n\n')
                    print(f'Problem in {cache_file}')
                    print(f'On 1-based line number {i + 1}')

                inp_run_id = self.run_id_of_input.get(obj['input'], None)
                if obj['run_id'] not in [inp_run_id, self.run_id]:
                    print('\n\n\n')
                    print(f'In cache file {cache_file}:')
                    print(f'   Skipping item for input {obj["input"]}')
                    print(f'   Its run_id={obj["run_id"]} but ', end='')
                    print(f'run_id on file is {inp_run_id}')
                    print(f'   My run id is {self.run_id}')
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

        self._print(f'Before pickling, cache has size {len(cache)}')

        if self.combine_fn:
            for key, values in cache.items():
                v = values.pop()
                while len(values) > 0:
                    v = self.combine_fn(key, v, values.pop())
                cache[key] = v

        # TODO: Call reduce_fn() if it's present.

        # Save to the official output directory.
        self._start_dbg_time('write_to_file')
        output_path = self.get_output_path(cache_file)
        with output_path.open('wb') as f:
            pickle.dump(cache, f)
        self.save_output_receipt(output_path)
        self._end_dbg_time('write_to_file')

        # print(f'Sending out 2 to pbar_q.')  # XXX
        cache_pbar_q.put(1)

    def get_output_path(self, cache_file):
        name = cache_file.name
        return self.output_dir / (name.split('.')[0] + '.pickle')

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

    def does_cache_have_output(self, cache_file):

        # Check that the output file exists.
        output_path     = self.get_output_path(cache_file)
        if not output_path.is_file():
            return False

        # Check that the output file has a receipt.
        output_receipts = self.get_output_receipts()
        receipt_ctime = output_receipts.get(output_path.name, None)
        if receipt_ctime is None:
            return False

        # Verify that the output receipt is for this data.
        # This is a heuristic that, in practice, I expect to essentially always
        # work.
        file_ctime = output_path.stat().st_ctime
        return abs(file_ctime - receipt_ctime) < 0.001

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

        t = len(self.input)

        # XXX TODO
        progress = pass_thru if self.do_silent_mode else tqdm

        if self.num_processes == 1:
            for inp in progress(self.input):
                self._handle_input(inp)
        else:

            # Set up the progress bar.
            def show_pbar(q, t, desc):
                # print(f'Starting show_pbar(), i={i}')
                n = 0
                pbar = tqdm(total=t, desc=desc)
                for j in iter(q.get, 'STOP'):
                    pbar.update(1)
                    n += 1
                    # XXX
                    if False:
                        if (n % 10) == 0:
                            print(f'In show_pbar(), n={n}')
                pbar.update(n=pbar.total - n)
                pbar.close()

            def avg_times(q):
                times = defaultdict(list)
                for update in iter(q.get, 'STOP'):
                    time_type, time = update
                    times[time_type].append(time)
                for k, v in times.items():
                    print(f'{k}: {sum(v) / len(v):.4f}')

            pbar_args = (map_pbar_q, t, 'Map to cache')
            mp.Process(target=show_pbar, args=pbar_args).start()

            if do_dbg_timing:
                mp.Process(target=avg_times, args=(avg_times_q,)).start()

            N = self.num_processes
            # print(f'Starting map_fn() calls, t={t}')  # XXX
            with mp.Pool(N) as p:
                list(p.map(self._assign_name, range(N)))
                # inp = zip(self.input, repeat(q))
                list(p.map(self._handle_input, self.input))
                map_pbar_q.put('STOP')

        if self.output_dir:
            cache_data = list(self.cache_dir.glob(f'*_of_*.jsonl'))
            t = len(cache_data)
            if self.num_processes == 1:
                for cache_item in progress(cache_data):
                    self._handle_cache(cache_item)
            else:
                pbar_args = (cache_pbar_q, t, 'Cache to output')
                mp.Process(target=show_pbar, args=pbar_args).start()
                with mp.Pool(N) as p:
                    list(p.map(self._handle_cache, cache_data))
                    cache_pbar_q.put('STOP')
                avg_times_q.put('STOP')


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
