import io
import logging
import zipfile
from datetime import datetime as dt
from pathlib import Path

from django.conf import settings
from django.shortcuts import render

from .models import FormList
from .sub import command as cmd
from .sub import utilities as utl

# from django.http import HttpResponse
# from django.views.generic import TemplateView
# from django.views.decorators.csrf import csrf_exempt




mid = Path(__file__).name  # module id
ftime = "%Y/%m/%d %H:%M:%S"

# Create your views here.


def prepare_working_dir(current_username):
    # working_dir = Path(settings.BASE_DIR, "appmain/static/wdir", current_username)
    working_dir = Path(settings.BASE_DIR, "static/wdir", current_username)

    if not Path(working_dir).exists():
        Path(working_dir).mkdir(parents=True)
        Path(working_dir, "output").mkdir(parents=True)
    else:
        [p.unlink() for p in Path(working_dir).resolve().iterdir() if p.is_file()
         and "user.log" not in p.name]  # clean up

    return working_dir


# @csrf_exempt
def appmain(request):  # methods=["POST"]
    logger = logging.getLogger(__name__)

    current_username = request.user.email  # username

    msg = f'[{mid}] ----- "/appmain" {current_username} -----'
    print(dt.now().strftime(ftime) + " " + msg)
    logger.info(msg)

    # initialize
    command = request.POST.get('command')
    selected_form = request.POST.get("selected_form")
    groupname = request.POST.get("groupname")
    print(f"[{mid}] {command= } {selected_form= } {groupname= }")

    record_to_be_processed = request.POST.get("record_to_be_processed")
    print(f"[{mid}] {record_to_be_processed= }")
    executeParameters = request.POST.get("executeParameters")
    print(f"[{mid}] {executeParameters= }")

    # set current username
    current_username = request.user.email  # username

    # appmain
    if command in ["submit_a_form"]:
        print(f"[{mid}] {command= }")

        file = request.FILES["data_file"]
        file_name = file.name
        if ".xlsx" in file_name[-5:]:
            file_suffix = ".xlsx"
        elif ".csv" in file_name[-4:]:
            file_suffix = ".csv"
        else:
            alert = ("NG", "error: unsupported file type.")
            return render(request,
                          'appmain/area4Submit.html',
                          {'message': alert[1],
                           'alert': alert[0]},
                          )

        data = file.read()

        # create control_data_dict
        if file_suffix == ".xlsx":
            alert, control_data_dict = utl.gen_control_data_dict(
                data, file_name)
        elif file_suffix == ".csv":
            alert, control_data_dict = utl.gen_control_data_dict_for_csv(
                data, file_name)
        else:
            control_data_dict = dict()
            alert = ("NG", "error: unsupported file type.")
        # print(f"[{mid}] {control_data_dict= }")

        if "NG" in alert[0]:
            return render(request,
                          'appmain/area4Submit.html',
                          {'message': alert[1],
                           'alert': alert[0]},
                          )

        # create input_data
        alert, input_data = utl.gen_input_data(data, control_data_dict)

        if "NG" in alert[0]:
            return render(request,
                          'appmain/area4Submit.html',
                          {'message': alert[1],
                           'alert': alert[0]},
                          )

        # dispatch command
        if "database_info_entry" in file_name:
            alert = cmd.database_info_entry(input_data, current_username)
        elif "group_info_entry" in file_name:
            alert = cmd.group_info_entry(input_data, current_username)
        elif "form_info_entry" in file_name:
            alert = cmd.form_info_entry(input_data, current_username)
        else:
            alert = cmd.write_form_items(
                input_data, control_data_dict, current_username)

        return render(request,
                      'appmain/area4Submit.html',
                      {'message': alert[1],
                       'alert': alert[0]},
                      )

    elif command in ["get_a_list_of_forms"]:
        print(f"[{mid}] {command= }")

        # current_username = request.user.username

        items = cmd.get_a_list_of_forms(current_username)

        return render(request,
                      'appmain/area4Inquire.html',
                      {'user': current_username,
                       'command': command,
                       'items': items,
                       'message': "",
                       'alert': "OK"},
                      )

    elif command in ["privilege"]:
        print(f"[{mid}] {command= }")

        items = cmd.get_a_list_of_forms_for_privilege(current_username)

        return render(request,
                      'appmain/area4Inquire.html',
                      {'user': current_username,
                       'command': command,
                       'items': items,
                       'message': "",
                       'alert': "OK"},
                      )

    elif command in ["show_privilege_items"]:
        key, items = cmd.read_form_items_for_privilege(
            command, selected_form, current_username)
        items = [(i + 1, *item) for i, item in enumerate(items)]

        return render(request,
                      'appmain/area4InquireItems.html',
                      {'user': current_username,
                       'command': command,
                       'keys': key,
                       'items': items,
                       'selected_form': selected_form,
                       'groupname': groupname,
                       'message': "",
                       'alert': "OK"},
                      )

    elif command in ["signupusers"]:
        print(f"[{mid}] {command= }")

        items = cmd.get_a_list_of_signup_users(current_username)

        return render(request,
                      'appmain/area4Inquire.html',
                      {'user': current_username,
                       'command': command,
                       'items': items,
                       'message': "",
                       'alert': "OK"},
                      )

    elif command in ["show_signup_users"]:
        key, items = cmd.read_signup_users(current_username)
        items = [(i + 1, *item) for i, item in enumerate(items)]

        return render(request,
                      'appmain/area4InquireItems.html',
                      {'user': current_username,
                       'command': command,
                       'keys': key,
                       'items': items,
                       'selected_form': selected_form,
                       'groupname': groupname,
                       'message': "",
                       'alert': "OK"},
                      )

    elif command in ["show_the_first_item",
                     "show_items_updated_today",
                     "show_items_updated_in_a_week",
                     "show_all_the_items"]:
        print(f"[{mid}] {command= }")

        key, items = cmd.read_form_items(command, selected_form)

        # cope with too large items
        items_volume = len(key) * len(items)
        print(f"[{mid}]{items_volume= } ")
        if items_volume > 300_000:
            items = [(i + 1, *item) for i, item in enumerate(items[:500])]
            alert = (
                "NG", "error: Too large to show.  Show 500 items.  Please try to download csv file.")
        else:
            items = [(i + 1, *item) for i, item in enumerate(items)]
            alert = ("OK", "Done.")

        return render(request,
                      'appmain/area4InquireItems.html',
                      {'user': current_username,
                       'command': command,
                       'keys': key,
                       'items': items,
                       'selected_form': selected_form,
                       'groupname': groupname,
                       'message': alert[1],
                       'alert': alert[0]},
                      )

    elif command in ["drop_form_confirm"]:
        print(f"[{mid}] {command= }")

        items = list()
        for i, obj in enumerate(FormList.objects.filter().all()):
            items.append([i + 1, obj.formname, obj.form_size,
                         obj.groupname, obj.update_at.strftime(ftime), obj.modified_by])
        items = [item for item in items if item[1] ==
                 selected_form and item[3] == groupname]

        return render(request,
                      'appmain/area4DeleteForm.html',
                      {'user': current_username,
                       'items': items,
                       'message': "",
                       'alert': "OK"},
                      )

    elif command in ["drop_form"]:
        print(f"[{mid}] {command= }")

        alert = cmd.drop_form(selected_form)

        if "OK" in alert[0]:
            items = cmd.get_a_list_of_forms(current_username)
            alert = ("OK", "Done.")
        else:
            pass
        return render(request,
                      'appmain/area4Inquire.html',
                      {'user': current_username,
                       'command': command,
                       'items': items,
                       'message': alert[1],
                       'alert': alert[0]},
                      )

    elif command in ["delete_item"]:
        print(f"[{mid}] {command= }")

        cmd.delete_item(selected_form, record_to_be_processed,
                        current_username)

        key, items = cmd.read_form_items("show_all_the_items", selected_form)

        # cope with too large items
        items_volume = len(key) * len(items)
        print(f"[{mid}]{items_volume= } ")
        if items_volume > 300_000:
            items = [(i + 1, *item) for i, item in enumerate(items[:500])]
            alert = (
                "NG", "error: Too large to show.  Show 500 items.  Please try to download csv file.")
        else:
            items = [(i + 1, *item) for i, item in enumerate(items)]
            alert = ("OK", "Done.")

        return render(request,
                      'appmain/area4InquireItems.html',
                      {'user': current_username,
                       'command': command,
                       'keys': key,
                       'items': items,
                       'selected_form': selected_form,
                       'groupname': groupname,
                       'message': alert[1],
                       'alert': alert[0]},
                      )
#
#     elif command in ["delete_user"]:
#         print(f"[{mid}] {command= }")
#
#         cmd.delete_user(selected_form, record_to_be_processed)
#
#         key, items = cmd.read_signup_users()
#         items = [(i + 1, *item) for i, item in enumerate(items)]
#         alert = "OK"
#         return render_template('area4InquireItems.html',
#                                user=current_user.username,
#                                message="",
#                                command=command,
#                                keys=key,
#                                items=items,
#                                selected_form="",
#                                groupname="",
#                                alert=alert)
#
    elif command in ["download_all_the_items"]:
        print(f"[{mid}] {command= }")

        command = "show_all_the_items"
        key, items = cmd.read_form_items(command, selected_form)

        # prepare working directory and output file name
        working_dir = prepare_working_dir(current_username)

        filename = "payload" + ".zip"

        output_file_name = Path(working_dir, filename)
        # output_file_name = Path(working_dir, "download", filename)

        # compress
        key = key[:-2]  # trim update_at and modified_by
        string = ",".join(key) + "\n"  # header line

        with io.StringIO() as fw:
            for item in items:
                for s in item[:-3]:  # trim update_at and modified_by
                    # ObjectID, datetime.datetime -> pure string
                    print(s, file=fw, end=",")
                print(item[-3], file=fw)  # write the last column without comma

            string += fw.getvalue()

        with zipfile.ZipFile(output_file_name, 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
            # archive file name and string data
            new_zip.writestr(selected_form + '.txt', string)

        return render(request,
                      'appmain/area4Download.html',
                      {'user': current_username,
                       'filename': filename,
                       'message': "Ready to download.",
                       'alert': "OK"}
                      )
