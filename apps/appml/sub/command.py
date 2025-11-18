import ast
import io
import logging
import os
import shutil
# import algo.time_series_analysis
import subprocess
from pathlib import Path

import pandas as pd
from django.conf import settings

import apps.appml.sub.algo.cls_compare_algorithms as cca
import apps.appml.sub.algo.cls_find_optimal_model as cfom
import apps.appml.sub.algo.reg_compare_algorithms as rca
import apps.appml.sub.algo.reg_find_optimal_model as rfom
import apps.appml.sub.algo.statistics_analysis as sa
from apps.appmain.sub import command as appmain_cmd

mid = Path(__file__).name
ftime = "%Y-%m-%d %H:%M:%S"


def get_a_list_of_forms_for_machine_learning(current_username):
    print(f"[{mid}] get_a_list_of_forms_for_machine_learning")

    items = appmain_cmd.get_a_list_of_forms(current_username)

    items_filtered = list()
    for item in items:
        selected_form = item[1]

        # create instance object and read item(s)
        db_f, group = appmain_cmd.create_db_instance(selected_form)
        key, _ = db_f.read_form_items("show_the_first_item")

        if "class" in key or "target" in key:
            items_filtered.append(item)

    items = [(i+1, *item[1:]) for i, item in enumerate(items_filtered)]  # reset number column

    return items


def prepare_working_dir(current_username):
    working_dir = os.path.join(settings.BASE_DIR, "static/wdir", current_username)

    if not Path(working_dir).exists():
        Path(working_dir).mkdir(parents=True)
        Path(working_dir, "output").mkdir(parents=True)
    else:
        [p.unlink() for p in Path(working_dir).resolve().iterdir() if p.is_file()
         and "user.log" not in p.name]  # clean up

    return working_dir


def prepare_data(selected_form):
    # prepare data
    command = "show_all_the_items"
    keys, items = appmain_cmd.read_form_items(command, selected_form)

    string = ",".join(keys) + "\n"  # header line

    with io.StringIO() as fw:
        for item in items:
            for s in item[:-1]:
                print(s, file=fw, end=",")  # ObjectID, datetime.datetime -> pure string
            print(item[-1], file=fw)  # the last item

        string += fw.getvalue()

    try:
        df = pd.read_csv(io.StringIO(string))
    except Exception as err:
        alert = ("NG", f"error: {err}")
        return alert, ""

    # remove object type columns. ex.) "_id", "update_at", "modified_by" column
    object_column = df.select_dtypes("object").columns
    df = df.drop(object_column, axis=1)
    alert = ("OK", "Done.")
    return alert, df


def make_images_and_urls(output_dir, rename_option=None):
    if rename_option:
        pass
    else:
        # ブラウザキャッシュによる画像更新不可対応のためファイル名称変更
        for p in Path(output_dir).resolve().iterdir():
            if p.suffix in [".png", ".html"]:
                mtime = int(os.stat(p).st_mtime)  # last update
                x = str(p.parent) + "/" + str(mtime) + "_p_" + p.name  # prefix
                os.rename(p, x)

    images = list()
    urls = list()

    # generate result area info: images and urls
    for file_path in Path(output_dir).iterdir():
        if ".png" in file_path.suffix:
            images.append(file_path.name)
        elif ".html" in file_path.suffix:
            urls.append(file_path.name)
        else:
            pass

    return images, urls


def make_results(result_file, output_dir):
    results = list()

    try:
        with open(Path(output_dir, result_file), "r", encoding="utf-8") as f:
            for line in f:
                line = line.split("\n")[:-1]
                line = line[0].split(",")
                results.append(line)
    except Exception as err:
        print(f"{err= }")

    return results


def make_payload(working_dir, output_dir):
    # digit = 15  # length
    # rnd = random.randrange(10 ** (digit - 1), 10 ** digit)  # to clear browser cash
    rnd = ""  # debug code

    shutil.make_archive(working_dir + "/payload" + str(rnd), "zip", root_dir=output_dir)

    filename = "payload" + str(rnd) + ".zip"

    return filename


def machine_learning_dispatcher(command, selected_form, executeParameters, current_username):
    print(f"{command= } {selected_form= } {executeParameters= }")

    tc_time = "0:00:00"

    # prepare data
    alert, df = prepare_data(selected_form)
    if "NG" in alert[0]:
        return alert, ""

    # prepare working directory and output file name
    working_dir = prepare_working_dir(current_username)
    output_dir = os.path.join(str(working_dir), "output")

    # clear output_dir
    for path in Path(output_dir).iterdir():
        if path.is_file():
            path.unlink()

    if "statistics_analysis" in command:
        # result is to be saved under output_dir
        # alert, tc_time = appml.algo.statistics_analysis.statistics_analysis(command, df, output_dir)

        alert, tc_time = sa.statistics_analysis(command, df, output_dir)

        if "NG" in alert[0]:
            return alert, [], [], [], "", tc_time  # alert, images, urls, results, filename, tc_time

    elif "cls_compare_algorithms" in command:
        alert, tc_time = cca.cls_compare_algorithms(command, df, output_dir)
        if "NG" in alert[0]:
            return alert, [], [], [], "", tc_time  # alert, images, urls, results, filename, tc_time

    elif "cls_find_optimal_model" in command:
        alert, tc_time = cfom.cls_find_optimal_model(command, df, output_dir, executeParameters)
        if "NG" in alert[0]:
            return alert, [], [], [], "", tc_time  # alert, images, urls, results, filename, tc_time

    elif "reg_compare_algorithms" in command:
        alert, tc_time = rca.reg_compare_algorithms(command, df, output_dir)
        if "NG" in alert[0]:
            return alert, [], [], [], "", tc_time  # alert, images, urls, results, filename, tc_time

    elif "reg_find_optimal_model" in command:
        alert, tc_time = rfom.reg_find_optimal_model(command, df, output_dir, executeParameters)
        if "NG" in alert[0]:
            return alert, [], [], [], "", tc_time  # alert, images, urls, results, filename, tc_time

    elif "time_series_analysis" in command:
        # alert, tc_time = execute.appml.time_series_analysis.time_series_analysis(command,
        #                                                                          df,
        #                                                                          output_dir,
        #                                                                          executeParameters)

        """ 2021.9.16
        tensorflowロードエラー対策（※）のため，subprocess化
        （※）Apache環境でログインダンマリ（ローカル環境では発生せず
             メモリの問題か？
        (補足)tensorflow plot_modelも機能せず（コメントアウト中）
        """
        # save temporary interface files
        BASE_DIR = settings.BASE_DIR
        user_log_file = Path(BASE_DIR, "appml/static/wdir", current_username, "user.log")

        d = {"selected_form": selected_form,
             "executeParameters": executeParameters,
             "user_log_file": str(user_log_file)}

        with open(Path(working_dir, "subprocess_call.txt"), "w", encoding="utf-8") as f:
            f.write(str(d))

        df.to_csv(Path(working_dir, "subprocess_call.csv"), encoding="utf-8", index=False, header=True)

        # create command
        cmd_str = settings.PYTHON_EXE_FILE
        BASE_DIR = settings.BASE_DIR
        subprocess_file = BASE_DIR + r"/appml/algo/time_series_analysis.py"
        cmd_str += ' "' + subprocess_file + '" '
        cmd_str += '"' + str(working_dir) + '"'
        print(f"{cmd_str=}")

        # subprocess call
        return_code = subprocess.call(cmd_str, timeout=3600)

        if return_code == 0:
            with open(Path(working_dir, "subprocess_call.txt"), "r", encoding="utf-8") as f:
                text = f.read()
            d = ast.literal_eval(text)
            tc_time = d["tc_time"]
            alert = "OK", "Done."
        else:
            alert = "NG", "error: subprocess call failed."

        # clean up temporary interface files
        try:
            Path(Path(working_dir, "subprocess_call.txt")).unlink()
            Path(Path(working_dir, "subprocess_call.csv")).unlink()
        except Exception as err:
            print(f"[{mid}] subprocess_call.txt/.csv unlink error occurred.  {err= }")

        if "NG" in alert[0]:
            return alert, [], [], [], "", tc_time  # alert, images, urls, results, filename, tc_time

    else:
        pass

    # make_images_and_urls
    images, urls = make_images_and_urls(output_dir)

    # make results list through result file
    results = make_results("result.txt", output_dir)

    # make payload.zip
    filename = make_payload(working_dir, output_dir)

    return alert, images, urls, results, filename, tc_time


def restore_the_result(current_username):
    working_dir = prepare_working_dir(current_username)
    output_dir = Path(str(working_dir), "output")

    # check output_dir
    if not Path(output_dir).exists():
        return ("NG", "Nothing to show now."), [], [], [], ""  # alert, images, urls, results, filename

    # make_images_and_urls
    images, urls = make_images_and_urls(output_dir, rename_option=True)

    # make results list through result file
    results = make_results("result.txt", output_dir)

    # make payload.zip
    filename = make_payload(working_dir, output_dir)

    alert = "OK", "Done."

    return alert, images, urls, results, filename
