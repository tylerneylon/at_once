# at_once

A Python library to assist with large data pipelines.
In particular, this library assumes you have some big list of
data items that you want to process. It helps achieve these
two goals:

* Processing data more quickly by using all your cpu cores, and
* Doing so in a crash-friendly way.

By "crash-friendly," I mean that if this process explodes, you
can see what the problem was, fix it (or you know just throw
in some temporary `try-except` clause to skip it), and then
run the exact same process without losing your partial work from
all the previous runs. Of course, if you have a bug that ouputs
bad data, then (sadly) you will have to erase any potentially
bad output and start again.

## Using `at_once`

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
