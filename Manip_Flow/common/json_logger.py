from typing import Optional, Callable, Any, Sequence
import os
import copy
import json
import numbers
import pandas as pd


def read_json_log(path: str, 
        required_keys: Sequence[str]=tuple(), 
        **kwargs) -> pd.DataFrame:
    """
    Read json-per-line file, with potentially incomplete lines.
    kwargs passed to pd.read_json
    """
    lines = list()
    with open(path, 'r') as f:
        while True:
            # one json per line
            line = f.readline()
            if len(line) == 0:
                # EOF
                break
            elif not line.endswith('\n'):
                # incomplete line
                break
            is_relevant = False
            for k in required_keys:
                if k in line:
                    is_relevant = True
                    break
            if is_relevant:
                lines.append(line)
    if len(lines) < 1:
        return pd.DataFrame()  
    json_buf = f'[{",".join([line for line in (line.strip() for line in lines) if line])}]'
    df = pd.read_json(json_buf, **kwargs)
    return df

class NullJsonLogger:
    """No-op stand-in for ``JsonLogger`` on non-zero ranks.

    Every rank shares ``output_dir`` (typically NAS), so if all of them append to
    the same ``logs.json.txt`` their line-buffered writes interleave mid-record and
    the file becomes unparseable -- which then makes the NEXT run of the same
    ``hydra.run.dir`` die inside ``JsonLogger.start()`` on ``json.loads``.
    """

    def __init__(self, *args, **kwargs):
        self.last_log = None

    def start(self):
        pass

    def stop(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def log(self, data: dict):
        self.last_log = data


class JsonLogger:
    def __init__(self, path: str,
            filter_fn: Optional[Callable[[str,Any],bool]]=None):
        if filter_fn is None:
            filter_fn = lambda k,v: isinstance(v, numbers.Number)

        # default to append mode
        self.path = path
        self.filter_fn = filter_fn
        self.file = None
        self.last_log = None
    
    def start(self):
        # use line buffering
        try:
            self.file = file = open(self.path, 'r+', buffering=1)
        except FileNotFoundError:
            self.file = file = open(self.path, 'w+', buffering=1)

        # Move the pointer (similar to a cursor in a text editor) to the end of the file
        pos = file.seek(0, os.SEEK_END)

        # Read each character in the file one at a time from the last
        # character going backwards, searching for a newline character
        # If we find a new line, exit the search
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        # now the file pointer is at one past the last '\n'
        # and pos is at the last '\n'.
        last_line_end = file.tell()
        
        # find the start of second last line
        pos = max(0, pos-1)
        file.seek(pos, os.SEEK_SET)
        while pos > 0 and file.read(1) != "\n":
            pos -= 1
            file.seek(pos, os.SEEK_SET)
        # now the file pointer is at one past the second last '\n'
        last_line_start = file.tell()

        if last_line_start < last_line_end:
            # has last line of json
            last_line = file.readline()
            try:
                self.last_log = json.loads(last_line)
            except json.JSONDecodeError:
                # A pre-existing log can be truncated mid-write or (from older
                # runs) interleaved by several ranks. last_log is advisory only,
                # so degrade instead of killing a job that hasn't trained a step.
                print(f'[JsonLogger] ignoring unparseable last line of {self.path}')
                self.last_log = None


        # remove the last incomplete line
        file.seek(last_line_end)
        file.truncate()
    
    def stop(self):
        self.file.close()
        self.file = None
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
    
    def log(self, data: dict):
        filtered_data = dict(
            filter(lambda x: self.filter_fn(*x), data.items()))
        # save current as last log
        self.last_log = filtered_data
        for k, v in filtered_data.items():
            if isinstance(v, numbers.Integral):
                filtered_data[k] = int(v)
            elif isinstance(v, numbers.Number):
                filtered_data[k] = float(v)
        buf = json.dumps(filtered_data)
        # ensure one line per json
        buf = buf.replace('\n','') + '\n'
        self.file.write(buf)
    
    def get_last_log(self):
        return copy.deepcopy(self.last_log)
