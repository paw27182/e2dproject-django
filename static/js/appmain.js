/*------------------------------
 appmain/static/html/
   submit.html - #btn_execute
   inquire.html - #btn_execute
------------------------------*/
$(function () {
    $('#btn_execute').off().on("click", function () {

        $("#area4Result").empty();  // clear result area

        var command = $("#selected_dropdown_item").val();  // Submit_a_form or Get a list of forms ...
        console.log("[appmain.js] command= ", command);

        // formData for POST
        var formData = new FormData($('#uploadForm')[0]);  // FormData object
        formData.append("command", command);

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
                // console.log(xhr.getResponseHeader("Content-Type"));
                // console.log(xhr.getResponseHeader("Results"));
                // console.log(output)

                $('#area4Result').html(output);
                $('#area4Result').show();

                // activate button
                $('#btn_execute').prop("disabled", false);
                $('#btn_execute').text("Execute");

            })
            .fail(function (error) {
                console.log(error);
            });  // ajax
    });  // function
});  // function

/*---------------------------------------
 appmain/templates/appmain/
   area4DeleteForm.html - .btn_result
   area4Inquire.html - .btn_result
   area4InquireItems.html - .btn_result
---------------------------------------*/
$(function () {
    $('.btn_result').off().on("click", function () {

        $("#area4Result").empty();  // clear result area

        var command = $(this).attr("id");
        console.log("[appmain.js] command= ", command);

        var selected_form = $(this).attr("value1");
        console.log("[appmain.js] selected_form= ", selected_form);

        var groupname = $(this).attr("value2");
        console.log("[appmain.js] groupname= ", groupname);

        if (command == "delete_item") {  // FYI. item, not form
            var record_to_be_processed = $(this).attr("value3");
            console.log("[appmain.js] record_to_be_processed= ", record_to_be_processed);
        };

        if (command == "delete_user") {
            var record_to_be_processed = $(this).attr("value3");
            console.log("[appmain.js] record_to_be_processed= ", record_to_be_processed);
        };

        // formData for POST
        var formData = new FormData($('#uploadForm')[0]);  //FormData object
        formData.append("command", command);
        formData.append("selected_form", selected_form);
        formData.append("groupname", groupname);
        formData.append("record_to_be_processed", record_to_be_processed);

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
              	// console.log(xhr.getResponseHeader("Content-Type"));
              	// console.log(xhr.getResponseHeader("Results"));
              	// console.log(output)

                $('#area4Result').html(output);
                $('#area4Result').show();

                // activate button
                $('#btn_execute').prop("disabled", false);
                $('#btn_execute').text("Execute");

            })
            .fail(function (error) {
                console.log(error);
            });  // ajax
    });  // function
});  // function
