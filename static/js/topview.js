/*------------------------------
 load html
 topview.html - .dropdown-item
------------------------------*/
$(function () {
    $('.dropdown-item').off().on("click", function () {
        $("#home").hide();

        // get dropdown-menu and dropdown-item
        dropdown_menu = $(this).parent().attr("id");  // "Entry", "Inquiry", "ML" or "Help"
        dropdown_item = $(this).attr("id");  // ex.) "Submit_a_form"
        console.log("[topview.js] dropdown_menu= ", dropdown_menu);
        console.log("[topview.js] dropdown_item= ", dropdown_item);

        // breadcrumb list
        str = dropdown_menu + " > " + dropdown_item;
        $("#label_request_area").text(str);

        // load html
        console.log("[topview.js] +++++ THE CAUSE OF DEPRECATION WARNING");

        if (dropdown_menu == "Entry") {
            if (dropdown_item == "submit_a_form") {
                $("#selected_dropdown_item").val("submit_a_form");
                $('#area4Request #area1').load("../static/html/submit.html");
            } else {
                // nothing to do
            };

            $('#label_request_area').show();
            $('#area4Request #area1').show();
            $("#area4Result").hide();

        } else if (dropdown_menu == "Inquiry") {
            $("#selected_dropdown_item").val(dropdown_item);  // get_a_list_of_forms, Show privilege, Show signup users
            $('#area4Request #area1').load("../static/html/inquire.html");
            $("#area4Result").empty();  // clear result area

            // formData for POST
            var formData = new FormData($('#uploadForm')[0]);  // FormData object
            formData.append("command", dropdown_item);
            formData.append("requested_form", "");  // unused

            // deactivate button
            $('#btn_execute').prop("disabled", true);
            $('#btn_execute').text("Executing...");

            const csrftoken = getCookie('csrftoken');
            console.log('[topview.js] csrftoken= ', csrftoken);

            $.ajax({
                url: '/appmain/',
                type: 'POST',
                headers: { 'X-CSRFToken': csrftoken },
                mode: 'same-origin',  // Do not send CSRF token to another domain.
                data: formData,
                processData: false,
                contentType: false,
            })
                .done(function (output, status, xhr) {
                    $('#area4Result').html(output);
                    $('#area4Result').show();

                    $('#label_request_area').show();
                    $('#area4Request #area1').show();

                    // activate button
                    $('#btn_execute').prop("disabled", false);
                    $('#btn_execute').text("Execute");
                })
                .fail(function (error) {
                    console.log(error);
                });  // ajax

        } else if (dropdown_menu == "machine_learning") {  // ML
            $("#selected_dropdown_item").val(dropdown_item);  // get_a_list_of_forms_for_machine_learning, restore_the_result
            $('#area4Request #area1').load("../static/html/inquire_appml.html");
            $("#area4Result").empty();  // clear result area

            // formData for POST
            var formData = new FormData($('#uploadForm')[0]);  // FormData object
            formData.append("command", dropdown_item);
            formData.append("requested_form", "");  // unused

            // deactivate button
            $('#btn_execute').prop("disabled", true);
            $('#btn_execute').text("Executing...");

            const csrftoken = getCookie('csrftoken');
            console.log('[topview.js] csrftoken= ', csrftoken);

            $.ajax({
                url: '/appml/',
                type: 'POST',
                headers: { 'X-CSRFToken': csrftoken },
                mode: 'same-origin',  // Do not send CSRF token to another domain.
                data: formData,
                processData: false,
                contentType: false,
            })
                .done(function (output, status, xhr) {
                    $('#area4Result').html(output);
                    $('#area4Result').show();

                    $('#label_request_area').show();
                    $('#area4Request #area1').show();

                    // activate button
                    $('#btn_execute').prop("disabled", false);
                    $('#btn_execute').text("Execute");
                })
                .fail(function (error) {
                    console.log(error);
                });  // ajax

        } else if (dropdown_menu == "Help") {
            if (dropdown_item == "FAQ") {
                $('#area4Result').load("../static/html/faq.html");
            } else if (dropdown_item == "Contact") {
                $('#area4Result').load("../static/html/contact.html");
            } else {
                // nothing to do
            };

            $('#label_request_area').show();
            $('#area4Request #area1').hide();
            $('#area4Result').show();

        } else {
            // nothing to do
        };
    });  // function
});  // function


$(function () {
    $('.btn_faqlanguage').off().on("click", function () {
        $('#area4Result').load("../static/html/faq_j.html");
        $('#area4Result').show();

    });
});

/*-------------------
 show home tab again
 topview.html
--------------------*/
$(function () {
    $('#home_tab').off().on("click", function () {

        $('#home').show();

        $('#label_request_area').hide();
        $('#area4Request #area1').hide();
        $('#area4Result').hide();
    });
});


function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Does this cookie string begin with the name we want?
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}
