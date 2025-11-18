import os
import pathlib
import re
import ast
import numpy as np
from datetime import datetime
import codecs
import pandas as pd


def cls_find_optimal_model(command, df, output_dir, executeParameters):
    print(f"{command}")

    launch_time = datetime.now()

    # tc_time = "0:00:00"
    # return f"NG.  {command} failed.", tc_time  # debug

    porec = "@@@"

    # create result file
    result_file = os.path.join(output_dir, "result.txt")

    if pathlib.Path(result_file).exists():
        pathlib.Path(result_file).unlink()

    result_str = ""
    for line in porec:
        result_str += line + "\n"

    with codecs.open(result_file, "w", "utf-8") as rf:
        print(result_str, file=rf)

    tc_time = "{}".format(datetime.now() - launch_time)
    match = re.search('[0-9]{1,2}:[0-9]{2}:[0-9]{2}', tc_time)  # 0:00:00
    tc_time = match.group(0)

    alert = "OK", "Done."

    return alert, tc_time
