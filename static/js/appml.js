/*------------------------------
 appml/static/html/
   inquire.html - #btn_execute
------------------------------*/
$(function () {
    $('#btn_execute').off().on("click", function () {

        $("#area4Result").empty();  // clear result area

        var command = $("#selected_dropdown_item").val();  // Submit_a_form or Get a list of forms ...
        console.log("[appml.js] command= ", command);

        // formData for POST
        var formData = new FormData($('#uploadForm')[0]);  // FormData object
        formData.append("command", command);

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


/*----------------------------------------
 appml/templates/appml/
   area4Inquire_appml.html - .btn_result
----------------------------------------*/
$(function () {
    $('.btn_result').off().on("click", function () {

        $("#area4Result").empty();  // clear result area

        var command = $(this).attr("id");
        console.log("[appml.js] command= ", command);

        var selected_form = $(this).attr("value1");
        console.log("[appml.js] selected_form= ", selected_form);

        var groupname = $(this).attr("value2");
        console.log("[appml.js] groupname= ", groupname);

        // formData for POST
        var formData = new FormData($('#uploadForm')[0]);  //FormData object
        formData.append("command", command);
        formData.append("selected_form", selected_form);
        formData.append("groupname", groupname);

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
