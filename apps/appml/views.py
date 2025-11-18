from django.conf import settings
import logging
from datetime import datetime as dt
from pathlib import Path

from django.shortcuts import render

# import appml.command as cmd
from .sub import command as cmd

# from django.views.decorators.csrf import csrf_exempt


mid = Path(__file__).name
ftime = "%Y/%m/%d %H:%M:%S"


# Create your views here.


# @csrf_exempt
def appml(request):  # methods=["POST"]
    logger = logging.getLogger(__name__)

    current_username = request.user.email  # username

    msg = f'[{mid}] ----- "/appml" {current_username} -----'
    print(dt.now().strftime(ftime) + " " + msg)
    logger.info(msg)

    # initialize
    command = request.POST.get("command")  # or request.form["command"]
    selected_form = request.POST.get("selected_form")
    groupname = request.POST.get("groupname")
    print(f"[{mid}] {command= } {selected_form= } {groupname= }")

    record_to_be_processed = request.POST.get("record_to_be_processed")
    print(f"[{mid}] {record_to_be_processed= }")
    executeParameters = request.POST.get("executeParameters")
    print(f"[{mid}] {executeParameters= }")

    # appml
    if command in ["get_a_list_of_forms_for_machine_learning"]:
        print(f"[{mid}] {command= }")

        items = cmd.get_a_list_of_forms_for_machine_learning(current_username)

        return render(request,
                      'appml/area4Inquire_appml.html',
                      {'user': current_username,
                       'command': command,
                       'items': items,
                       'message': "Hello There!",
                       'alert': "OK"},
                      )
    else:
        pass

    if command in ["statistics_analysis",
                   "cls_compare_algorithms",
                   "cls_find_optimal_model",
                   "reg_compare_algorithms",
                   "reg_find_optimal_model",
                   "time_series_analysis",
                   ]:
        print(f"[{mid}] {command= }")

        alert, images, urls, results, filename, tc_time = \
            cmd.machine_learning_dispatcher(
                command, selected_form, executeParameters, current_username)
        print(f"{alert= } {tc_time= }")
        return render(request,
                      'appml/area4Result_appml.html',
                      {'user': current_username,
                       'images': images,
                       'urls': urls,
                       'results': results,
                       'filename': filename,
                       'tc_time': tc_time,  # total computation time
                       'message': alert[1],
                       'alert': alert[0]},
                      )

    # elif command in ["time_series_analysis_parameters",
    #                  # "cls_find_optimal_model_parameters",
    #                  # "reg_find_optimal_model_parameters",
    #                  ]:
    #     print(f"[{mid}] {command= }")
    #
    #     if "time_series_analysis_parameters" in command:
    #         command = "time_series_analysis"
    #         executeParameters = '{"model_name": "GRU_Model", "units":32,' \
    #                             ' "lookback": 1440, "step": 6, "delay": 144, "batch_size": 128, "epochs": 5}'
    #     # elif "cls_find_optimal_model_parameters" in command:
    #     #     command = "cls_find_optimal_model"
    #     #     executeParameters = '{"max_evals": 10}'
    #     # elif "reg_find_optimal_model_parameters" in command:
    #     #     command = "reg_find_optimal_model"
    #     #     executeParameters = '{"max_evals": 10}'
    #     else:
    #         pass
    #
    #     return render_template('area4Parameters_appml.html',
    #                            user=current_user.username,
    #                            message="Hello There!",
    #                            command=command,
    #                            selected_form=selected_form,
    #                            executeParameters=executeParameters,
    #                            alert="OK")

    elif command in ["restore_the_result"]:
        print(f"[{mid}] {command= }")

        alert, images, urls, results, filename = cmd.restore_the_result(
            current_username)

        return render(request,
                      'appml/area4Result_appml.html',
                      {'user': current_username,
                       'images': images,
                       'urls': urls,
                       'results': results,
                       'filename': filename,
                       'tc_time': "0:00:00",  # total computation time
                       'message': alert[1],
                       'alert': alert[0]},
                      )
    else:
        pass
